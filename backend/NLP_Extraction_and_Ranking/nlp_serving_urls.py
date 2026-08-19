"""NLP serving URLs and model IDs.

Default mode uses hosted HTTP APIs:
  - GLiNER: urchade/gliner_multi-v2.1 (FastAPI /predict)
  - Embeddings: BAAI/bge-m3 (OpenAI-compatible /v1/embeddings)
  - Reranker: BAAI/bge-reranker-v2-m3 (/v2/rerank)

Triton KServe v2 remains available when SURF_MODEL_API_MODE=triton.
"""
import os
from urllib.parse import urlparse, urlunparse


def _normalize_client_url(url: str) -> str:
    """Rewrite bind-all addresses to localhost for outbound HTTP clients."""
    parsed = urlparse(url)
    if parsed.hostname in {"0.0.0.0", "::"}:
        port = parsed.port
        netloc = f"localhost:{port}" if port else "localhost"
        return urlunparse(parsed._replace(netloc=netloc)).rstrip("/")
    return url.rstrip("/")


def _compose(host_or_base: str, port: str) -> str:
    h = (host_or_base or "localhost").strip() or "localhost"
    p = (port or "").strip()
    if h.startswith("http://") or h.startswith("https://"):
        return f"{h.rstrip('/')}:{p}"
    return f"http://{h}:{p}"


def _join_url(base: str, path: str) -> str:
    return f"{base.rstrip('/')}/{path.strip('/')}"


def _service_url(explicit_env: str, default_url: str) -> str:
    explicit = os.getenv(explicit_env, "").strip().rstrip("/")
    if explicit:
        return _normalize_client_url(explicit)
    return _normalize_client_url(default_url)


# Model service defaults (override via SURF_*_API_URL in backend/.env)
_DEFAULT_GLINER_URL = "http://localhost:8081"
_DEFAULT_BGE_URL = "https://bg3.limeox.org"
_DEFAULT_RERANK_URL = "https://bge.limeox.org"

GLINER_API_URL = _service_url("SURF_GLINER_API_URL", _DEFAULT_GLINER_URL)
BIENCODER_API_URL = _service_url("SURF_BIENCODER_API_URL", _DEFAULT_BGE_URL)
CROSSENCODER_API_URL = _service_url("SURF_CROSSENCODER_API_URL", _DEFAULT_RERANK_URL)

# HuggingFace model IDs (for API payloads and logging)
GLINER_MODEL_ID = os.getenv("SURF_GLINER_MODEL_ID", "urchade/gliner_multi-v2.1").strip() or "urchade/gliner_multi-v2.1"


def _parse_gliner_threshold() -> float:
    raw = os.getenv("GLINER_THRESHOLD", "0.15").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.15


GLINER_THRESHOLD = _parse_gliner_threshold()
BGE_MODEL_ID = os.getenv("SURF_BGE_MODEL_ID", "BAAI/bge-m3").strip() or "BAAI/bge-m3"
RERANK_MODEL_ID = os.getenv("SURF_RERANK_MODEL_ID", "BAAI/bge-reranker-v2-m3").strip() or "BAAI/bge-reranker-v2-m3"

# API paths
GLINER_PREDICT_URL = _join_url(
    GLINER_API_URL,
    os.getenv("SURF_GLINER_PREDICT_PATH", "/predict"),
)
GLINER_BATCH_URL = _join_url(
    GLINER_API_URL,
    os.getenv("SURF_GLINER_BATCH_PATH", "/predict_entities_batch"),
)


def is_local_gliner_url(url: str | None = None) -> bool:
    """True when GLiNER is served on localhost (single-process local server)."""
    target = url or GLINER_API_URL
    try:
        host = (urlparse(target).hostname or "localhost").lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}
BGE_ENCODE_URL = _join_url(
    BIENCODER_API_URL,
    os.getenv("SURF_BGE_ENCODE_PATH", "/v1/embeddings"),
)
RERANK_URL = _join_url(
    CROSSENCODER_API_URL,
    os.getenv("SURF_RERANK_PATH", "/v2/rerank"),
)

# BGE client style: openai (vLLM /v1/embeddings), legacy (/encode), triton (KServe)
BGE_API_STYLE = os.getenv("SURF_BGE_API_STYLE", "openai").strip().lower() or "openai"

# Reranker enabled in live pipeline by default
USE_RERANKER = os.getenv("SURF_USE_RERANKER", "true").strip().lower() in {"1", "true", "yes", "on"}

# Triton opt-in fallback
TRITON_HTTP_URL = (
    os.getenv("TRITON_HTTP_URL", "").strip().rstrip("/")
    or _compose(os.getenv("SURF_NLP_MODELS_HOST", "localhost"), os.getenv("TRITON_HTTP_PORT", "8010"))
)

TRITON_GLINER_MODEL_NAME = os.getenv("TRITON_GLINER_MODEL_NAME", "gliner_ner").strip() or "gliner_ner"
TRITON_BGE_MODEL_NAME = os.getenv("TRITON_BGE_MODEL_NAME", "bge_embeddings").strip() or "bge_embeddings"

TRITON_GLINER_INFER_URL = f"{TRITON_HTTP_URL}/v2/models/{TRITON_GLINER_MODEL_NAME}/infer"
TRITON_BGE_INFER_URL = f"{TRITON_HTTP_URL}/v2/models/{TRITON_BGE_MODEL_NAME}/infer"

MODEL_API_MODE = os.getenv("SURF_MODEL_API_MODE", "fastapi").strip().lower() or "fastapi"
USE_TRITON = MODEL_API_MODE == "triton" or BGE_API_STYLE == "triton"
USE_FASTAPI_MODEL_WRAPPERS = (
    not USE_TRITON
    and MODEL_API_MODE in {"fastapi", "wrapper", "served", ""}
)
