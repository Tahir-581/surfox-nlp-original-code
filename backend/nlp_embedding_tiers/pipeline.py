"""Orchestrate the full embedding consensus tiering pipeline."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from NLP_Extraction_and_Ranking.bge_client import BGETritonClient
from NLP_Extraction_and_Ranking.deduplicate_nlps import _l2_normalize
from NLP_Extraction_and_Ranking.nlp_serving_urls import BGE_MODEL_ID

from .anchored_clustering import assign_anchor_tiers_by_sim_quota
from .artifacts import RunArtifacts
from .config import ClusterConfig
from .consensus import build_final_clusters_from_instances
from .evaluation import build_evaluation_report
from .fuzzy_dedup import normalize_competitor_hostname, normalize_competitor_hostnames
from .query_generator import generate_query_variants
from .validation import ensure_minimum_anchors

log = logging.getLogger(__name__)

bge_client = BGETritonClient()


def _entity_key(text: str) -> str:
    return (text or "").strip().casefold()


def _competitor_domains_from_merge_output(
    merge_output: dict[str, Any] | None,
    entities: list[dict[str, Any]],
) -> list[str]:
    """Resolve competitor hostnames for offline artifact runs."""
    if merge_output:
        stored = merge_output.get("competitor_domains")
        if stored:
            return normalize_competitor_hostnames(stored)

    seen: set[str] = set()
    raw_domains: list[str] = []
    for entity in entities:
        for domain in entity.get("found_in_files") or []:
            normalized = normalize_competitor_hostname(str(domain))
            if normalized and normalized not in seen:
                seen.add(normalized)
                raw_domains.append(normalized)
    return raw_domains


def load_nimra_input(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    keyword = (data.get("keyword") or "").strip()
    merge_output = data.get("merge_output") or {}
    entities = merge_output.get("entities") or []
    return {
        "keyword": keyword,
        "entities": entities,
        "merge_output": merge_output,
        "session_id": data.get("session_id"),
        "source_file": str(path),
    }


def run_pipeline(
    *,
    keyword: str,
    entities: list[dict[str, Any]],
    output_dir: Path,
    config: ClusterConfig | None = None,
    merge_output: dict[str, Any] | None = None,
    save_embeddings: bool = True,
) -> dict[str, Any]:
    """Execute full pipeline and write all JSON artifacts."""
    config = config or ClusterConfig.from_env()
    artifacts = RunArtifacts(output_dir)

    entities_by_key: dict[str, dict[str, Any]] = {}
    for e in entities:
        key = _entity_key(e.get("text", ""))
        if key:
            entities_by_key[key] = e

    nlp_texts = [e.get("text", "") for e in entities if e.get("text")]

    run_config = {
        "version": "proportional_instance_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "keyword": keyword,
        "entity_count": len(entities),
        "bge_model": BGE_MODEL_ID,
        "config": config.public_dict(),
    }
    artifacts.write("run_config.json", run_config)
    artifacts.write(
        "00_input.json",
        {
            "keyword": keyword,
            "entity_count": len(entities),
            "entities_preview": nlp_texts[:20],
        },
    )

    # Step 1: Query variants (GLiNER entities from keyword)
    query_gen = generate_query_variants(keyword, config)
    artifacts.write(
        "01_query_instances.json",
        {
            **query_gen,
            "gliner_variants": sorted(query_gen.get("gliner_variants") or []),
        },
    )

    # Step 2: Validate anchors
    anchor_texts, validation = ensure_minimum_anchors(
        keyword,
        query_gen.get("variants") or [],
        config,
        gliner_variants=set(query_gen.get("gliner_variants") or []),
    )
    artifacts.write("02_query_instance_validation.json", validation)

    # Step 3: NLP list
    artifacts.write(
        "03_generated_nlps.json",
        {
            "count": len(entities),
            "entities": entities,
        },
    )

    stats = (merge_output or {}).get("average_statistics") or {}
    avg_word_count = float(stats.get("avg_word_count") or 0.0)
    competitor_domains = _competitor_domains_from_merge_output(merge_output, entities)

    if not nlp_texts:
        empty_final = {
            "green_nlps": [],
            "orange_nlps": [],
            "white_nlps": [],
            "consensus_metadata": {"anchors_used": 0, "method": "proportional_instance_v1"},
        }
        artifacts.write("08_final_clusters.json", empty_final)
        artifacts.write("09_evaluation_report.json", {"error": "no entities"})
        return {"output_dir": str(output_dir), "final_clusters": empty_final}

    # Steps 4–8: embeddings, clustering, consensus (with artifact capture)
    query_embeddings = bge_client.encode(anchor_texts, is_query=True)
    nlp_embeddings = bge_client.encode(nlp_texts, is_query=False)
    query_unit = _l2_normalize(query_embeddings.astype(np.float32))
    nlp_unit = _l2_normalize(nlp_embeddings.astype(np.float32))

    manifest = {
        "bge_model": BGE_MODEL_ID,
        "embedding_dim": int(nlp_unit.shape[1]) if nlp_unit.size else 0,
        "query_anchors": anchor_texts,
        "nlp_texts": nlp_texts,
        "query_is_query": True,
        "nlp_is_query": False,
    }
    artifacts.write("04_embeddings_manifest.json", manifest)

    if save_embeddings:
        np.save(str(output_dir / "embeddings_queries.npy"), query_unit)
        np.save(str(output_dir / "embeddings_nlps.npy"), nlp_unit)

    sim_matrix: dict[str, dict[str, float]] = {}
    sim_records: list[dict[str, Any]] = []
    for j, anchor in enumerate(anchor_texts):
        sims = (nlp_unit @ query_unit[j]).astype(np.float32)
        for i, text in enumerate(nlp_texts):
            sim_val = round(float(sims[i]), 4)
            sim_matrix.setdefault(text, {})[anchor] = sim_val
            sim_records.append({"nlp": text, "anchor": anchor, "similarity": sim_val})
    artifacts.write(
        "05_similarity_matrix.json",
        {"by_nlp": sim_matrix, "records": sim_records},
    )

    per_query_results: list[dict[str, Any]] = []
    for idx, anchor in enumerate(anchor_texts):
        result = assign_anchor_tiers_by_sim_quota(
            nlp_unit,
            nlp_texts,
            query_unit[idx],
            anchor,
            config,
        )
        per_query_results.append(result)
        fname = "original_query.json" if idx == 0 else f"query_instance_{idx}.json"
        artifacts.write_per_query(fname, result)

    artifacts.write(
        "07_proportional_merge.json",
        {
            "anchors_used": len(per_query_results),
            "per_query_tier_counts": [
                {
                    "query_text": r.get("query_text"),
                    "tier_counts": r.get("tier_counts"),
                }
                for r in per_query_results
            ],
        },
    )

    final_clusters = build_final_clusters_from_instances(
        per_query_results,
        entities_by_key,
        keyword,
        config,
        avg_word_count=avg_word_count,
        competitor_domains=competitor_domains,
    )
    artifacts.write("08_final_clusters.json", final_clusters)

    eval_report = build_evaluation_report(
        keyword=keyword,
        entities=entities,
        final_clusters=final_clusters,
        consensus={},
        merge_output=merge_output,
    )
    artifacts.write("09_evaluation_report.json", eval_report)

    log.info(
        "Pipeline done — green=%d orange=%d white=%d → %s",
        len(final_clusters.get("green_nlps") or []),
        len(final_clusters.get("orange_nlps") or []),
        len(final_clusters.get("white_nlps") or []),
        output_dir,
    )

    return {
        "output_dir": str(output_dir),
        "final_clusters": final_clusters,
        "evaluation": eval_report,
    }
