"""
Local GLiNER FastAPI server for Surfox NLP extraction.

Run from backend/:
    python NLP_Extraction_and_Ranking/gliner_server.py

Endpoints:
    POST /predict                  — production client contract
    POST /predict_entities         — legacy alias
    POST /predict_entities_batch   — batched inference
    GET  /health
"""
from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from NLP_Extraction_and_Ranking.nlp_serving_urls import GLINER_THRESHOLD

log = logging.getLogger(__name__)

MODEL_NAME = os.getenv("SURF_GLINER_MODEL_ID", "urchade/gliner_multi-v2.1").strip() or "urchade/gliner_multi-v2.1"
PORT = int(os.getenv("GLINER_PORT", "8081"))
WORKERS = max(1, int(os.getenv("GLINER_WORKERS", "1")))
MAX_TEXT_CHARS = max(1, int(os.getenv("GLINER_MAX_TEXT_CHARS", "4000")))
MAX_LEN = max(1, int(os.getenv("GLINER_MAX_LEN", "384")))

_inference_lock = threading.Lock()


class PredictRequest(BaseModel):
    text: str
    labels: list[str]
    threshold: float = GLINER_THRESHOLD


class PredictBatchRequest(BaseModel):
    texts: list[str]
    labels: list[str]
    threshold: float = GLINER_THRESHOLD


def _resolve_device() -> str:
    explicit = os.getenv("GLINER_DEVICE", "").strip()
    if explicit:
        return explicit
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _validate_predict_inputs(text: str, labels: list[str]) -> str:
    if not labels:
        raise HTTPException(status_code=400, detail="labels must not be empty")
    if not (text or "").strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if len(text) > MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"text exceeds maximum length of {MAX_TEXT_CHARS} characters",
        )
    return text


def create_app():
    from nlp_compat import patch_torch_jit_for_py313

    patch_torch_jit_for_py313()
    import flair
    from gliner import GLiNER

    device = _resolve_device()
    flair.device = torch.device(device)

    app = FastAPI(title="Surfox GLiNER Server")
    model_holder: dict = {"model": None, "device": device, "error": None}

    @app.on_event("startup")
    async def load_model():
        print(f"Loading GLiNER: {MODEL_NAME} on {device}...")
        try:
            model = GLiNER.from_pretrained(MODEL_NAME)
            model.model.to(torch.device(device))
            model_holder["model"] = model
            print(f"GLiNER ready on port {PORT}")
        except Exception as exc:
            model_holder["error"] = str(exc)
            print(f"Failed to load GLiNER: {exc}")
            raise

    def _run_inference(fn):
        model = model_holder["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="GLiNER model not loaded")

        wait_start = time.perf_counter()
        with _inference_lock:
            wait_seconds = time.perf_counter() - wait_start
            if wait_seconds > 0.5:
                log.info("GLiNER inference lock wait %.2fs", wait_seconds)
            try:
                with torch.inference_mode():
                    return fn(model)
            except HTTPException:
                raise
            except Exception as exc:
                log.exception("GLiNER inference failed")
                raise HTTPException(status_code=500, detail=str(exc)) from exc

    def _predict_one(text: str, labels: list[str], threshold: float) -> list:
        text = _validate_predict_inputs(text, labels)

        def _infer(model):
            return model.predict_entities(
                text,
                labels,
                threshold=threshold,
                max_length=MAX_LEN,
            )

        return _run_inference(_infer)

    def _predict_batch(texts: list[str], labels: list[str], threshold: float) -> list:
        if not labels:
            raise HTTPException(status_code=400, detail="labels must not be empty")
        validated = [_validate_predict_inputs(t, labels) for t in texts]

        def _infer(model):
            try:
                if hasattr(model, "batch_predict_entities"):
                    out = model.batch_predict_entities(
                        validated,
                        labels,
                        flat_ner=True,
                        threshold=threshold,
                        max_length=MAX_LEN,
                    )
                    if isinstance(out, list) and len(out) == len(validated):
                        return out
            except Exception:
                log.exception("GLiNER batch_predict_entities failed; falling back to sequential")

            return [
                model.predict_entities(
                    t,
                    labels,
                    threshold=threshold,
                    max_length=MAX_LEN,
                )
                for t in validated
            ]

        return _run_inference(_infer)

    @app.post("/predict")
    def predict(req: PredictRequest):
        entities = _predict_one(req.text, req.labels, req.threshold)
        return {"entities": entities}

    @app.post("/predict_entities")
    def predict_entities(req: PredictRequest):
        entities = _predict_one(req.text, req.labels, req.threshold)
        return {"entities": entities}

    @app.post("/predict_entities_batch")
    def predict_entities_batch(req: PredictBatchRequest):
        texts = [t for t in (req.texts or []) if t and t.strip()]
        if not texts:
            return {"entities_per_text": []}
        entities_per_text = _predict_batch(texts, req.labels, req.threshold)
        return {"entities_per_text": entities_per_text}

    @app.get("/health")
    def health():
        ready = model_holder["model"] is not None
        return {
            "status": "online" if ready else "loading",
            "model": MODEL_NAME,
            "device": model_holder["device"],
            "ready": ready,
            "error": model_holder["error"],
        }

    return app


def main():
    logging.basicConfig(level=logging.INFO)
    if WORKERS > 1:
        uvicorn.run(
            "NLP_Extraction_and_Ranking.gliner_server:create_app",
            host="0.0.0.0",
            port=PORT,
            workers=WORKERS,
            factory=True,
        )
    else:
        uvicorn.run(create_app(), host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()

