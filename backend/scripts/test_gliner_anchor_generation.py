#!/usr/bin/env python3
"""
Test GLiNER-based anchor generation for embedding consensus tiering.

Usage:
    python scripts/test_gliner_anchor_generation.py "best dog breeds"
    python scripts/test_gliner_anchor_generation.py "best dog breeds for apartment owners" --json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from nlp_embedding_tiers.config import ClusterConfig  # noqa: E402
from nlp_embedding_tiers.query_generator import generate_query_variants  # noqa: E402
from nlp_embedding_tiers.service import prepare_anchor_bundle  # noqa: E402
from nlp_embedding_tiers.validation import ensure_minimum_anchors  # noqa: E402

log = logging.getLogger(__name__)


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "shape"):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def run(keyword: str, config: ClusterConfig) -> dict:
    query_gen = generate_query_variants(keyword, config)
    gliner_variants = set(query_gen.get("gliner_variants") or [])
    anchor_texts, validation = ensure_minimum_anchors(
        keyword,
        query_gen.get("variants") or [],
        config,
        gliner_variants=gliner_variants,
    )
    bundle = prepare_anchor_bundle(keyword, config)
    return {
        "keyword": keyword,
        "query_generation": query_gen,
        "validation": validation,
        "anchor_texts": anchor_texts,
        "bundle": {
            "anchor_texts": bundle["anchor_texts"],
            "query_generation_method": bundle.get("query_generation_method"),
            "query_unit_shape": list(bundle["query_unit"].shape),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Test GLiNER keyword anchor generation")
    parser.add_argument("keyword", help="Search keyword / query to extract GLiNER anchors from")
    parser.add_argument("--json", action="store_true", help="Print full JSON output")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    keyword = (args.keyword or "").strip()
    if not keyword:
        print("Error: keyword must not be empty", file=sys.stderr)
        return 1

    config = ClusterConfig.from_env()

    try:
        result = run(keyword, config)
    except Exception as exc:
        log.exception("Anchor generation failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_default))
        return 0

    query_gen = result["query_generation"]
    validation = result["validation"]
    bundle = result["bundle"]

    print(f"Keyword: {keyword}")
    print(f"GLiNER keyword threshold: {query_gen.get('gliner_keyword_threshold')}")
    print(f"Generation method: {query_gen.get('generation_method')}")

    raw_gliner = query_gen.get("raw_gliner") or []
    print(f"GLiNER raw entities ({len(raw_gliner)}):")
    for ent in raw_gliner:
        if isinstance(ent, dict):
            label = ent.get("label", "")
            score = ent.get("score", "")
            text = ent.get("text", "")
            print(f"  - {text!r} label={label!r} score={score}")
        else:
            print(f"  - {ent!r}")

    variants = query_gen.get("variants") or []
    print(f"GLiNER entity anchors ({len(variants)}):")
    for v in variants:
        print(f"  - {v}")

    print(f"Accepted anchors ({validation.get('final_anchor_count', len(result['anchor_texts']))}):")
    for item in validation.get("accepted_anchors") or []:
        role = item.get("role", "")
        sim = item.get("sim_to_original", "")
        gliner = item.get("gliner_entity", False)
        print(f"  - [{role}] {item.get('text')} (sim={sim}, gliner={gliner})")

    rejected = validation.get("rejected") or []
    if rejected:
        print(f"Rejected ({len(rejected)}):")
        for item in rejected:
            print(f"  - {item.get('text')}: {item.get('reason')} (sim={item.get('sim_to_original')})")

    print(f"Final anchor_texts: {bundle['anchor_texts']}")
    print(f"Query unit shape: {bundle['query_unit_shape']}")
    print(f"Bundle generation method: {bundle.get('query_generation_method')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
