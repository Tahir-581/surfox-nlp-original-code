import os
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from gliner import GLiNER
from sentence_transformers import SentenceTransformer, util

from NLP_Extraction_and_Ranking.nlp_serving_urls import GLINER_THRESHOLD

# Configuration
GLINER_MODEL = "urchade/gliner_multi-v2.1"
ST_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
PORT = int(os.getenv("MODEL_SERVER_PORT", "8011"))

app = FastAPI(
    title="Surfox Model Server", 
    description="Standalone server for GLiNER and SentenceTransformer models"
)

# Global model storage
MODELS = {
    "gliner": None,
    "st": None
}

class EntityRequest(BaseModel):
    text: str
    labels: List[str] = [
        "Person", "Organization", "Location", "City", "Country", "Address",
        "Date", "Time", "Event", "Work of Art", "Consumer Good", "Other",
        "Price", "Phone Number", "Law", "Language", "Percentage", 
        "Scientific Term", "Title", "Position", "Product", "Brand",
        "Concept", "Theory", "Medical Condition", "Chemical", "Award",
        "Animal Breed", "Nature", "Substance", "Vehicle", "Facility"
    ]
    threshold: float = GLINER_THRESHOLD

class SimilarityRequest(BaseModel):
    source_text: str
    target_texts: List[str]

@app.on_event("startup")
async def load_models():
    """Load models into memory on startup"""
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    print(f"--- Loading Models on {device} ---")
    
    try:
        # Load GLiNER
        print(f"Loading GLiNER: {GLINER_MODEL}...")
        MODELS["gliner"] = GLiNER.from_pretrained(GLINER_MODEL).to(device)
        
        # Load SentenceTransformer
        print(f"Loading SentenceTransformer: {ST_MODEL}...")
        MODELS["st"] = SentenceTransformer(ST_MODEL, device=device)
        
        print("--- Models Loaded Successfully ---")
    except Exception as e:
        print(f"Error loading models: {e}")
        raise e

@app.post("/extract_entities")
async def extract_entities(request: EntityRequest):
    """Extract entities using GLiNER"""
    if MODELS["gliner"] is None:
        raise HTTPException(status_code=503, detail="GLiNER model not loaded")
    
    try:
        entities = MODELS["gliner"].predict_entities(
            request.text, 
            request.labels, 
            threshold=request.threshold
        )
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/similarity")
async def calculate_similarity(request: SimilarityRequest):
    """Calculate cosine similarity using SentenceTransformer"""
    if MODELS["st"] is None:
        raise HTTPException(status_code=503, detail="SentenceTransformer model not loaded")
    
    try:
        # Encode source and targets
        source_emb = MODELS["st"].encode(request.source_text, convert_to_tensor=True)
        target_embs = MODELS["st"].encode(request.target_texts, convert_to_tensor=True)
        
        # Calculate similarity
        cosine_scores = util.cos_sim(source_emb, target_embs)[0]
        
        results = [
            {"text": text, "score": round(float(score), 4)}
            for text, score in zip(request.target_texts, cosine_scores)
        ]
        
        return {"similarity_scores": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.get("/health")
async def health_check():
    """Check if the server and models are ready"""
    return {
        "status": "online",
        "models_loaded": {
            "gliner": MODELS["gliner"] is not None,
            "st": MODELS["st"] is not None
        },
        "device": str(next(MODELS["st"].parameters()).device) if MODELS["st"] else "n/a"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
