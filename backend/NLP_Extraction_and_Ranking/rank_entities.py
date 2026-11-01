import json
import os
import csv
import numpy as np

from .nlp_serving_urls import BGE_MODEL_ID, RERANK_MODEL_ID, RERANK_URL, USE_RERANKER
from .bge_client import BGETritonClient
from .reranker_client import RerankerClient

bge_client = BGETritonClient()
reranker_client = RerankerClient()


def rank_entities_by_similarity(input_json_path, output_dir, title):
    """
    Rank extracted entities by similarity. Primary ranking uses BGE reranker when enabled;
    BiEncoder (BAAI/bge-m3) scores are also computed for reference.
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = data.get("entities", [])

    if not entities:
        print("No entities found in input file")
        return

    print(f"\n{'='*70}")
    print(f"--- BiEncoder: {BGE_MODEL_ID} ---")
    print(f"{'='*70}")

    texts_to_encode = [title] + [entity["text"] for entity in entities]
    embeddings = bge_client.encode(texts_to_encode, is_query=False)
    title_emb = embeddings[0:1]
    entity_embeddings = embeddings[1:]
    norms_title = np.linalg.norm(title_emb, axis=1, keepdims=True)
    norms_ent = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
    norms_ent = np.maximum(norms_ent, 1e-9)
    sims = (entity_embeddings / norms_ent) @ (title_emb / np.maximum(norms_title, 1e-9)).T
    similarity_scores = sims.ravel()

    ranked_entities = []
    for i, entity in enumerate(entities):
        ranked_entities.append({
            "text": entity["text"],
            "count": entity["count"],
            "biencoder_score": round(float(similarity_scores[i]), 4),
            "crossencoder_score": None,
        })

    print("\n[BiEncoder Similarity Scores]")
    print("-" * 80)
    by_biencoder = sorted(ranked_entities, key=lambda x: x["biencoder_score"], reverse=True)
    for i, entity in enumerate(by_biencoder[:20], 1):
        print(f"{i:2}. {entity['text']:40} | BiEncoder: {entity['biencoder_score']:.4f} | Count: {entity['count']}")
    print("-" * 80)

    crossencoder_available = False
    crossencoder_model_used = RERANK_MODEL_ID

    if USE_RERANKER:
        print(f"\n{'='*70}")
        print(f"--- Reranker: {RERANK_MODEL_ID} @ {RERANK_URL} ---")
        print(f"{'='*70}")
        try:
            documents = [e["text"] for e in ranked_entities]
            scores = reranker_client.rerank(title, documents)
            for i, entity in enumerate(ranked_entities):
                entity["crossencoder_score"] = round(float(scores[i]), 4)
            crossencoder_available = True
        except Exception as e:
            print(f"Warning: Reranker API failed: {e}")
            print("Continuing with BiEncoder results only...\n")

    if crossencoder_available:
        ranked_entities = sorted(ranked_entities, key=lambda x: x["crossencoder_score"], reverse=True)
        print("\n[Final Ranking by Reranker (Primary)]")
        print("-" * 110)
        for i, entity in enumerate(ranked_entities[:20], 1):
            print(
                f"{i:2}. {entity['text']:40} | BiEncoder: {entity['biencoder_score']:.4f} "
                f"| Reranker: {entity['crossencoder_score']:.4f} | Count: {entity['count']}"
            )
    else:
        ranked_entities = sorted(ranked_entities, key=lambda x: x["biencoder_score"], reverse=True)
        print("\n[Final Ranking by BiEncoder (Reranker unavailable)]")
        print("-" * 80)
        for i, entity in enumerate(ranked_entities[:20], 1):
            print(f"{i:2}. {entity['text']:40} | BiEncoder: {entity['biencoder_score']:.4f} | Count: {entity['count']}")

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.splitext(os.path.basename(input_json_path))[0] + "_ranked_final.json"
    output_json_path = os.path.join(output_dir, filename)

    result = {
        "source": data.get("source"),
        "title": title,
        "unique_count": len(ranked_entities),
        "total_instances": data.get("total_instances"),
        "ranking_by": "crossencoder" if crossencoder_available else "biencoder",
        "biencoder_model": BGE_MODEL_ID,
        "crossencoder_model": crossencoder_model_used if crossencoder_available else "Not available",
        "ranked_entities": ranked_entities,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Ranked results saved to JSON: {output_json_path}")

    csv_filename = os.path.splitext(os.path.basename(input_json_path))[0] + "_ranked_combined.csv"
    output_csv_path = os.path.join(output_dir, csv_filename)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as csvfile:
        if crossencoder_available:
            fieldnames = ["Rank (Reranker)", "Entity Text", "Count", "BiEncoder Score", "Reranker Score"]
        else:
            fieldnames = ["Rank (BiEncoder)", "Entity Text", "Count", "BiEncoder Score"]

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i, entity in enumerate(ranked_entities, 1):
            rank_key = "Rank (Reranker)" if crossencoder_available else "Rank (BiEncoder)"
            row = {
                rank_key: i,
                "Entity Text": entity["text"],
                "Count": entity["count"],
                "BiEncoder Score": entity["biencoder_score"],
            }
            if crossencoder_available:
                row["Reranker Score"] = entity["crossencoder_score"]
            writer.writerow(row)

    print(f"Ranked results saved to CSV: {output_csv_path}")


def main():
    INPUT_FILE = "outputs/gliner_output.json"
    OUTPUT_DIR = "outputs"
    TITLE = "Dog Breeds to Deter Intruders and Keep You Safe"

    if not os.path.isfile(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    rank_entities_by_similarity(INPUT_FILE, OUTPUT_DIR, TITLE)


if __name__ == "__main__":
    main()
