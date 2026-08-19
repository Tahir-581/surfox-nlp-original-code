"""
Serve all models over FastAPI: GLiNER (6000), BiEncoder (6005), CrossEncoder (6010).
Run: python model_server.py
Then run Gliner_.py, rank_entities.py, deduplicate_nlps.py as usual; they call these endpoints.
"""
import multiprocessing
import os
import uvicorn


def create_gliner_app():
    """Factory for the GLiNER FastAPI app (used by uvicorn with workers)."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    from gliner import GLiNER
    import torch
    import flair

    from .nlp_serving_urls import GLINER_THRESHOLD

    MODEL_NAME = "urchade/gliner_multi-v2.1"
    DEVICE = "cuda:1"
    flair.device = torch.device(DEVICE)

    app = FastAPI()
    print(f"Loading GLiNER: {MODEL_NAME} on {DEVICE}...")
    model = GLiNER.from_pretrained(MODEL_NAME)
    model = GLiNER.from_pretrained(MODEL_NAME)
    model.model.to(torch.device(DEVICE))
    print(f"GLiNER ready on port 6000")

    class PredictRequest(BaseModel):
        text: str
        labels: list[str]
        threshold: float = GLINER_THRESHOLD

    class PredictBatchRequest(BaseModel):
        texts: list[str]
        labels: list[str]
        threshold: float = GLINER_THRESHOLD

    @app.post("/predict_entities")
    def predict_entities(req: PredictRequest):
        with torch.inference_mode():
            entities = model.predict_entities(req.text, req.labels, threshold=req.threshold)
            return {"entities": entities}

    @app.post("/predict_entities_batch")
    def predict_entities_batch(req: PredictBatchRequest):
        """
        Batch version to reduce HTTP overhead. Tries to call GLiNER in batch mode if
        supported by the library; otherwise falls back to per-text inference while
        keeping a single request/response.
        """
        texts = [t for t in (req.texts or []) if t and t.strip()]
        if not texts:
            return {"entities_per_text": []}

        with torch.inference_mode():
            try:
                # GLiNER 0.1.9+: true batched inference
                if hasattr(model, "batch_predict_entities"):
                    out = model.batch_predict_entities(
                        texts,
                        req.labels,
                        flat_ner=True,
                        threshold=req.threshold,
                    )
                    if isinstance(out, list) and len(out) == len(texts):
                        return {"entities_per_text": out}
            except Exception:
                pass

            entities_per_text = [
                model.predict_entities(t, req.labels, threshold=req.threshold)
                for t in texts
            ]
            return {"entities_per_text": entities_per_text}

    return app


def run_gliner():
    # Run GLiNER with multiple workers (processes) for parallel requests.
    # Default to 3 as requested; can be overridden via GLINER_WORKERS env var.
    workers = int(os.getenv("GLINER_WORKERS", "3"))
    # IMPORTANT: use import string + factory so uvicorn can safely spawn workers without warnings.
    uvicorn.run(
        "model_server:create_gliner_app",
        host="0.0.0.0",
        port=6000,
        workers=workers,
        factory=True,
    )


def run_biencoder():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from sentence_transformers import SentenceTransformer

    MODEL_NAME = "BAAI/bge-m3"
    DEVICE = "cuda:1"

    app = FastAPI()
    print(f"Loading BiEncoder: {MODEL_NAME} on {DEVICE}...")
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"BiEncoder ready on port 6005")

    class EncodeRequest(BaseModel):
        texts: list[str]

    @app.post("/encode")
    def encode(req: EncodeRequest):
        embeddings = model.encode(req.texts, convert_to_tensor=False)
        return {"embeddings": embeddings.tolist()}

    uvicorn.run(app, host="0.0.0.0", port=6005)


def run_crossencoder():
    from fastapi import FastAPI
    from pydantic import BaseModel
    from sentence_transformers import CrossEncoder

    CROSSENCODER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
    DEVICE = "cuda:1"

    app = FastAPI()
    print(f"Loading CrossEncoder: {CROSSENCODER_MODEL_ID} on {DEVICE}...")
    ce_model = CrossEncoder(CROSSENCODER_MODEL_ID, device=DEVICE)
    print(f"CrossEncoder ({CROSSENCODER_MODEL_ID}) ready on port 6010")

    class PredictRequest(BaseModel):
        pairs: list[list[str]]

    @app.get("/model_info")
    def model_info():
        return {"model": CROSSENCODER_MODEL_ID}

    @app.post("/predict")
    def predict(req: PredictRequest):
        scores = ce_model.predict(req.pairs)
        return {"scores": [float(s) for s in scores]}

    uvicorn.run(app, host="0.0.0.0", port=6010)


if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_gliner)
    p2 = multiprocessing.Process(target=run_biencoder)
    p3 = multiprocessing.Process(target=run_crossencoder)
    p1.start()
    p2.start()
    p3.start()
    print("All model servers starting (6000, 6005, 6010). Press Ctrl+C to stop.")
    p1.join()
    p2.join()
    p3.join()
