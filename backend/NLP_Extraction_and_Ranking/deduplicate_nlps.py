import json
import csv
import math
import numpy as np
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .bge_client import BGETritonClient
from nlp_tier_utils import (
    exceeds_max_nlp_words,
    filter_entities_by_max_words,
    is_exempt_nlp_text,
)

bge_client = BGETritonClient()

def _l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def _cosine_sim_matrix_to_vector(x: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between each row of x and vector v.
    Works even if x/v are not normalized (normalizes internally).
    """
    x_n = _l2_normalize(x)
    v_n = _l2_normalize(v.reshape(1, -1))[0]
    return (x_n @ v_n).astype(np.float32)


def _cosine_sim_max_to_seeds(x_unit: np.ndarray, seeds_unit: np.ndarray) -> np.ndarray:
    """
    For each row in x_unit, return max cosine similarity to any seed vector.
    Assumes inputs are already L2-normalized (unit length).
    """
    if x_unit.ndim != 2 or seeds_unit.ndim != 2:
        raise ValueError("x_unit and seeds_unit must be 2D arrays")
    sims = x_unit @ seeds_unit.T
    return np.max(sims, axis=1).astype(np.float32)


def _build_anchor_seed_texts(anchor_title: str) -> list[str]:
    """
    Build "seed" phrases from the anchor title by removing stopwords and splitting
    into meaningful chunks of consecutive non-stopwords.

    Example:
      "Dog Breeds to Deter Intruders and Keep You Safe"
      -> [
          full title,
          "Dog Breeds",
          "Deter Intruders",
          "Keep You Safe",
      ]
    """
    title = (anchor_title or "").strip()
    if not title:
        return []

    # Keep "you/your/keep" because phrases like "Keep You Safe" are meaningful.
    stop = set(ENGLISH_STOP_WORDS) - {"you", "your", "keep"}

    # Tokenize on words while preserving original casing per token.
    tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", title)
    if not tokens:
        return [title]

    chunks: list[str] = []
    cur: list[str] = []
    for tok in tokens:
        if tok.lower() in stop:
            if cur:
                chunks.append(" ".join(cur))
                cur = []
            continue
        cur.append(tok)
    if cur:
        chunks.append(" ".join(cur))

    # Dedupe while keeping order; always include full title first.
    seeds: list[str] = []
    for s in [title] + chunks:
        s = s.strip()
        if not s:
            continue
        if s.lower() in {x.lower() for x in seeds}:
            continue
        seeds.append(s)
    return seeds


def _spherical_kmeans(
    x_unit: np.ndarray,
    init_centroids_unit: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-6,
    pinned_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    True spherical k-means:
    - assumes x_unit rows are L2-normalized
    - assigns by cosine similarity (dot product on unit sphere)
    - updates centroids as mean direction then re-normalizes each centroid
    - pinned_indices: centroid row indices that are never updated (e.g. query anchor)
    """
    if x_unit.ndim != 2:
        raise ValueError("x_unit must be 2D (n_samples, n_features)")
    if init_centroids_unit.ndim != 2:
        raise ValueError("init_centroids_unit must be 2D (k, n_features)")
    if x_unit.shape[1] != init_centroids_unit.shape[1]:
        raise ValueError("x_unit and init_centroids_unit feature dims must match")

    k = init_centroids_unit.shape[0]
    pinned = set(pinned_indices or [])
    centroids = _l2_normalize(init_centroids_unit.astype(np.float32, copy=True))
    labels = np.full((x_unit.shape[0],), -1, dtype=np.int32)

    for _ in range(max_iter):
        sims = x_unit @ centroids.T  # cosine similarities on the unit sphere
        new_labels = np.argmax(sims, axis=1).astype(np.int32)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centroids = centroids.copy()
        for j in range(k):
            if j in pinned:
                continue
            mask = labels == j
            if not np.any(mask):
                # Keep previous centroid if cluster becomes empty
                continue
            c = x_unit[mask].mean(axis=0, dtype=np.float64)
            c_norm = np.linalg.norm(c)
            if c_norm < tol:
                continue
            new_centroids[j] = (c / c_norm).astype(np.float32)

        # Stop if centroids stop moving
        if np.max(np.linalg.norm(new_centroids - centroids, axis=1)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    return labels, centroids


def deduplicate_nlps(input_file, output_file, similarity_threshold=0.9):
    """
    Iteratively deduplicate NLP entities based on semantic similarity using BAAI model.
    Repeats until no more entities with similarity > threshold are found.
    """
    
    # Load the input JSON
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Use BAAI model via API on port 6005
    print("Using BAAI/bge-m3 embeddings (hosted API)...")

    ranked_entities = data.get('ranked_entities', [])
    initial_count = len(ranked_entities)
    print(f"Found {initial_count} entities\n")
    
    iteration = 0
    total_merged = 0
    
    while True:
        iteration += 1
        print(f"=== ITERATION {iteration} ===")
        print(f"Current entity count: {len(ranked_entities)}")
        
        # Extract entity texts
        entities_text = [entity['text'] for entity in ranked_entities]
        
        # Compute embeddings via API
        print("Computing embeddings...")
        embeddings = bge_client.encode(entities_text, is_query=False)

        # Compute similarity matrix (cosine similarity)
        print("Computing similarity matrix...")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-9)
        embeddings_n = embeddings / norms
        similarity_matrix = (embeddings_n @ embeddings_n.T).astype(np.float64)
        
        # Find duplicates
        print("Finding duplicate entities...")
        to_keep = set(range(len(ranked_entities)))
        merged_in_iteration = 0
        
        for i in range(len(ranked_entities)):
            if i not in to_keep:
                continue
                
            for j in range(i + 1, len(ranked_entities)):
                if j not in to_keep:
                    continue
                
                similarity = similarity_matrix[i][j]
                
                if similarity > similarity_threshold:
                    # Keep the entity with higher count, remove the other
                    count_i = ranked_entities[i]['count']
                    count_j = ranked_entities[j]['count']
                    
                    if count_i >= count_j:
                        to_keep.discard(j)
                        print(f"  Merging: '{ranked_entities[j]['text'][:50]}...' -> '{ranked_entities[i]['text'][:50]}...' (similarity: {similarity:.4f})")
                        merged_in_iteration += 1
                        total_merged += 1
                    else:
                        to_keep.discard(i)
                        print(f"  Merging: '{ranked_entities[i]['text'][:50]}...' -> '{ranked_entities[j]['text'][:50]}...' (similarity: {similarity:.4f})")
                        merged_in_iteration += 1
                        total_merged += 1
                        break
        
        # Create deduplicated list
        ranked_entities = [ranked_entities[i] for i in sorted(to_keep)]
        
        print(f"Merged in this iteration: {merged_in_iteration}")
        print(f"Remaining entities: {len(ranked_entities)}\n")
        
        # If no merges happened, we're done
        if merged_in_iteration == 0:
            print("✓ No more similar entities found. Deduplication complete!")
            break

    anchor_title = (
        data.get("title")
        or data.get("anchor_title")
        or "Dog Breeds to Deter Intruders and Keep You Safe"
    )
    exempt_texts = [anchor_title] if anchor_title else ()
    ranked_entities = filter_entities_by_max_words(
        ranked_entities,
        exempt_texts=exempt_texts,
    )

    # Cluster deduplicated NLPs into 3 groups using Spherical K-Means (cosine)
    print("\nClustering deduplicated NLPs into 3 clusters (Spherical K-Means)...")
    entities_text = [entity["text"] for entity in ranked_entities]

    # We want cluster_1 to be centered on this title
    anchor_seed_texts = _build_anchor_seed_texts(anchor_title)
    if not anchor_seed_texts:
        anchor_seed_texts = [anchor_title]

    # Encode anchor seed(s) + all entities together so they share the same embedding space
    all_embeddings = bge_client.encode(anchor_seed_texts + entities_text, is_query=False)

    seed_count = len(anchor_seed_texts)
    seed_embs_raw = all_embeddings[:seed_count]
    embeddings = all_embeddings[seed_count:]

    # Use mean of seed embeddings as the "anchor center"
    anchor_emb_raw = seed_embs_raw.mean(axis=0)

    # Normalize embeddings for cosine similarity (spherical k-means).
    anchor_emb = _l2_normalize(anchor_emb_raw.reshape(1, -1))[0]
    embeddings = _l2_normalize(embeddings)
    seed_embs = _l2_normalize(seed_embs_raw)
    # Normalized embedding of the full anchor title (first seed)
    title_emb_unit = seed_embs[0]

    # Build custom initial centroids so that cluster_1 (label 0) is anchored on the title
    n_clusters = 3
    rng = np.random.default_rng(42)
    init_centroids = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float32)
    init_centroids[0] = anchor_emb
    if embeddings.shape[0] >= n_clusters - 1:
        # Pick remaining centroids from entity embeddings
        indices = rng.choice(embeddings.shape[0], size=n_clusters - 1, replace=False)
        init_centroids[1:] = embeddings[indices]
    else:
        # Fallback: just duplicate anchor if too few points (very unlikely here)
        init_centroids[1:] = anchor_emb

    labels, _centroids = _spherical_kmeans(
        embeddings,
        init_centroids,
        max_iter=200,
        tol=1e-6,
    )

    # -------------------------------------------------------------------------
    # Make cluster_1 a "relevance bucket" to the anchor *seed phrases* (cosine).
    # Using max similarity to any seed is typically tighter than using the mean
    # anchor embedding, which can become too broad and pull everything into cluster_1.
    # -------------------------------------------------------------------------
    # Much tighter relevance settings:
    # - exclude the full title seed for matching (it can be too broad)
    # - require similarity to >=2 distinct seed phrases to be considered relevant
    per_seed_threshold = 0.55   # each seed hit must be at least this similar
    min_seed_hits = 2           # require at least N seed phrases to match
    max_sim_override = 0.70    # OR allow very strong match to any single seed
    drop_irrelevant = False     # set True to remove non-relevant items entirely

    # embeddings/seed_embs may not be unit length on the kmeans path, normalize defensively.
    emb_unit = _l2_normalize(embeddings)
    seed_unit_all = _l2_normalize(seed_embs)
    # Prefer matching against chunk seeds only (exclude full-title seed at index 0)
    seed_unit = seed_unit_all[1:] if seed_unit_all.shape[0] > 1 else seed_unit_all

    sims_matrix = emb_unit @ seed_unit.T
    sims_max = np.max(sims_matrix, axis=1).astype(np.float32)
    seed_hits = (sims_matrix >= per_seed_threshold).sum(axis=1).astype(np.int32)
    is_relevant = (seed_hits >= min_seed_hits) | (sims_max >= max_sim_override)

    # Quick stats to help tune threshold
    try:
        p50, p75, p90, p95 = np.percentile(sims_max, [50, 75, 90, 95]).tolist()
        print(f"\nRelevance(max_sim) stats vs chunk seeds: p50={p50:.3f} p75={p75:.3f} p90={p90:.3f} p95={p95:.3f}")
        print(
            "Relevance rule: "
            f"(seed_hits >= {min_seed_hits} with per_seed_threshold={per_seed_threshold:.2f}) "
            f"OR (max_sim >= {max_sim_override:.2f})"
        )
        print(f"cluster_1 gets {int(is_relevant.sum())}/{len(is_relevant)} entities (+ seed phrases)")
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # Build final clusters:
    # - cluster_1: all "relevant" items (based on seeds) – kept as is
    # - remaining items: reclustered (no anchor) into small spherical clusters
    #   with at most ~5 items each, then ordered by similarity of their centroid
    #   to cluster_1's centroid.
    # -------------------------------------------------------------------------
    idx_all = np.arange(len(entities_text))
    idx_relevant = idx_all[is_relevant]
    idx_rest = idx_all[~is_relevant]

    # cluster_1 texts (before seed cleanup)
    cluster1_texts = [entities_text[i] for i in idx_relevant]

    clusters: dict[str, list[str]] = {}
    clusters["cluster_1"] = cluster1_texts.copy()

    # If we choose to drop irrelevant entirely, skip reclustering them.
    if not drop_irrelevant and len(idx_rest) > 0:
        rest_embeddings = embeddings[idx_rest]
        rest_texts = [entities_text[i] for i in idx_rest]

        # Determine k so that we have at most ~5 items per cluster.
        k_rest = max(1, int(math.ceil(len(rest_texts) / 5.0)))
        print(f"\nReclustering {len(rest_texts)} non-cluster_1 items into {k_rest} spherical sub-clusters (~<=5 per cluster).")

        # Initialize centroids for spherical k-means from random rest points.
        rng_local = np.random.default_rng(123)
        if rest_embeddings.shape[0] >= k_rest:
            init_idx = rng_local.choice(rest_embeddings.shape[0], size=k_rest, replace=False)
        else:
            init_idx = np.arange(rest_embeddings.shape[0])
        init_centroids_rest = rest_embeddings[init_idx]

        labels_rest, centroids_rest = _spherical_kmeans(
            rest_embeddings,
            init_centroids_rest,
            max_iter=100,
            tol=1e-6,
        )

        # Compute centroid for cluster_1 (using entity embeddings, not seeds).
        if len(idx_relevant) > 0:
            c1 = embeddings[idx_relevant].mean(axis=0, keepdims=True)
            c1 = _l2_normalize(c1)[0]
        else:
            c1 = anchor_emb  # fallback

        # Group texts by rest-cluster label and compute similarity to cluster_1.
        rest_clusters_text: dict[int, list[str]] = {}
        rest_clusters_center: dict[int, np.ndarray] = {}
        for i_local, (lbl, emb_vec) in enumerate(zip(labels_rest, rest_embeddings)):
            rest_clusters_text.setdefault(int(lbl), []).append(rest_texts[i_local])

        for lbl in rest_clusters_text.keys():
            mask = labels_rest == lbl
            center = rest_embeddings[mask].mean(axis=0, keepdims=True)
            center = _l2_normalize(center)[0]
            rest_clusters_center[int(lbl)] = center

        # Rank rest clusters by cosine similarity to cluster_1 centroid,
        # and also compute similarity to the anchor title embedding.
        scored = []
        for lbl, center in rest_clusters_center.items():
            sim_c1 = float(c1 @ center)
            sim_anchor = float(anchor_emb @ center)
            scored.append((lbl, sim_c1, sim_anchor))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Add them as cluster_2, cluster_3, ... according to similarity,
        # and collect/print their relevance scores.
        cluster_scores: dict[str, dict[str, float]] = {}

        # score for cluster_1 itself
        cluster_scores["cluster_1"] = {
            "sim_to_cluster_1_centroid": 1.0,
            "sim_to_anchor_title": float(title_emb_unit @ c1),
        }

        next_cluster_idx = 2
        print("\nRelevance scores for new sub-clusters (vs cluster_1 centroid and anchor):")
        for lbl, sim_c1, sim_anchor in scored:
            key = f"cluster_{next_cluster_idx}"
            clusters[key] = rest_clusters_text[lbl]
            cluster_scores[key] = {
                "sim_to_cluster_1_centroid": sim_c1,
                "sim_to_anchor_title": sim_anchor,
            }
            print(
                f"  {key}: size={len(rest_clusters_text[lbl])}, "
                f"sim(cluster_1_centroid)={sim_c1:.4f}, sim(anchor_title)={sim_anchor:.4f}"
            )
            next_cluster_idx += 1

    # Final clean-up for anchor seeds in output:
    # - Keep only the full anchor title in the final JSON
    # - Use seed phrases only for shaping clusters / relevance, not as separate outputs
    full_title = anchor_seed_texts[0] if anchor_seed_texts else anchor_title
    sub_seeds = {s.lower() for s in anchor_seed_texts[1:]} if len(anchor_seed_texts) > 1 else set()

    # Remove all sub-seed phrases from every cluster, and also remove any existing
    # copies of the full title so we can re-insert it exactly once at the top.
    full_title_lc = full_title.lower()
    for key, items in list(clusters.items()):
        filtered = []
        for t in items:
            tl = t.lower()
            if tl == full_title_lc:
                continue
            if tl in sub_seeds:
                continue
            filtered.append(t)
        clusters[key] = filtered

    # Ensure cluster_1 exists and prepend the full anchor title once.
    clusters.setdefault("cluster_1", [])
    clusters["cluster_1"].insert(0, full_title)

    # Simple console summary
    print("\nCluster sizes:")
    for key in sorted(clusters.keys()):
        print(f"  {key}: {len(clusters[key])} NLPs")

    # Update data
    data['ranked_entities'] = ranked_entities
    data['unique_count'] = len(ranked_entities)
    data['deduplicated'] = True
    data['similarity_threshold'] = similarity_threshold
    data['iterations'] = iteration
    data['clusters'] = clusters
    # Attach per-cluster similarity scores (if we computed them during reclustering)
    if 'cluster_scores' in locals():
        data['cluster_scores'] = cluster_scores
    
    # Save output
    print(f"\nSaving final deduplicated entities (with clusters) to {output_file}...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Also save clusters-only JSON, e.g. cluster_1: [nlp1, nlp2, ...]
    clusters_only_path = output_path.with_name(output_path.stem + "_clusters.json")
    with open(clusters_only_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"Saved clusters JSON to {clusters_only_path}")

    # ------------------------------------------------------------------
    # Build final JSON of top NLPs ranked by similarity to anchor title.
    # If cluster_1 (excluding the title itself) has < 40 NLPs, try to
    # augment it by concatenating nearest clusters until we reach up to
    # ~50 NLPs. Then score each selected NLP by cosine similarity to the
    # anchor title and save to a separate JSON file.
    # ------------------------------------------------------------------
    final_nlps: list[str] = []
    seen_texts: set[str] = set()

    # cluster_1 texts excluding the title (first item)
    c1_items = clusters.get("cluster_1", [])
    c1_entity_texts = c1_items[1:] if c1_items else []

    for t in c1_entity_texts:
        if t not in seen_texts:
            seen_texts.add(t)
            final_nlps.append(t)

    # If fewer than 40, try to pull from nearest clusters (2,3,...) until ~50.
    target_min = 40
    target_max = 50
    if len(final_nlps) < target_min and 'cluster_scores' in locals():
        # Build ordered list of clusters by similarity to cluster_1 (excluding cluster_1).
        scored_other = []
        for key, scores in cluster_scores.items():
            if key == "cluster_1":
                continue
            scored_other.append((key, scores["sim_to_cluster_1_centroid"]))
        scored_other.sort(key=lambda x: x[1], reverse=True)

        for key, _sim in scored_other:
            for t in clusters.get(key, []):
                if t in seen_texts:
                    continue
                seen_texts.add(t)
                final_nlps.append(t)
                if len(final_nlps) >= target_max:
                    break
            if len(final_nlps) >= target_max:
                break

    final_nlps = [
        text
        for text in final_nlps
        if not exceeds_max_nlp_words(text)
        or is_exempt_nlp_text(text, exempt_texts)
    ]

    # Map entity text -> embedding row (normalized); note title is not in entities_text.
    text_to_emb: dict[str, np.ndarray] = {}
    for txt, emb_vec in zip(entities_text, embeddings):
        text_to_emb[txt] = emb_vec

    # Compute similarity of each final NLP to the title embedding.
    final_entries: list[dict[str, float | str]] = []
    for txt in final_nlps:
        emb_vec = text_to_emb.get(txt)
        if emb_vec is None:
            continue
        sim_title = float(title_emb_unit @ emb_vec)
        final_entries.append({"nlp": txt, "sim_to_title": sim_title})

    # Sort by similarity descending.
    final_entries.sort(key=lambda x: x["sim_to_title"], reverse=True)

    # Save to a fixed final result file name next to the main output.
    final_json_path = output_path.with_name("final_result.json")
    with open(final_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_entries, f, indent=2, ensure_ascii=False)
    print(f"Saved final ranked NLPs JSON to {final_json_path}")
    
    # Save CSV with both ranks (no effect on order; same order as JSON)
    csv_file = output_path.with_suffix('.csv')
    has_cross = any(e.get('crossencoder_score') is not None for e in ranked_entities)
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        if has_cross:
            writer = csv.DictWriter(f, fieldnames=['Rank', 'Entity Text', 'Count', 'BiEncoder Score', 'CrossEncoder Score'])
        else:
            writer = csv.DictWriter(f, fieldnames=['Rank', 'Entity Text', 'Count', 'BiEncoder Score'])
        writer.writeheader()
        for i, entity in enumerate(ranked_entities, 1):
            row = {
                'Rank': i,
                'Entity Text': entity['text'],
                'Count': entity['count'],
                'BiEncoder Score': entity.get('biencoder_score'),
            }
            if has_cross:
                row['CrossEncoder Score'] = entity.get('crossencoder_score')
            writer.writerow(row)
    print(f"Saved CSV to {csv_file}")
    
    # Print remaining NLPs with both ranks
    print(f"\n{'='*50}")
    print(f"✓ FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  Initial unique count: {initial_count}")
    print(f"  Final deduplicated count: {len(ranked_entities)}")
    print(f"  Total removed duplicates: {initial_count - len(ranked_entities)}")
    print(f"  Total iterations: {iteration}")
    print(f"  Total merges: {total_merged}")
    print(f"{'='*50}")
    
    print(f"\n[Remaining NLPs with BiEncoder & CrossEncoder ranks]")
    print("-" * 100)
    for i, entity in enumerate(ranked_entities[:50], 1):
        be = entity.get('biencoder_score')
        ce = entity.get('crossencoder_score')
        be_str = f"{be:.4f}" if be is not None else "—"
        ce_str = f"{ce:.4f}" if ce is not None else "—"
        print(f"{i:3}. {entity['text'][:48]:48} | BiEnc: {be_str} | CrossEnc: {ce_str} | Count: {entity['count']}")
    if len(ranked_entities) > 50:
        print(f"  ... and {len(ranked_entities) - 50} more (see JSON/CSV)")
    print("-" * 100)

if __name__ == "__main__":
    # Input: ranked_final.json (from rank_entities.py) has both biencoder_score and crossencoder_score
    input_file = "outputs/gliner_output_ranked_final.json"
    output_file = "outputs/gliner_output_final.json"

    deduplicate_nlps(input_file, output_file, similarity_threshold=0.85)
