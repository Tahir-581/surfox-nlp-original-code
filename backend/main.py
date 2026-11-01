import os
import json
import secrets
import asyncio
import logging
import re
import sys
import csv
import math
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Optional, Set, Any
from datetime import datetime, timedelta
from collections import defaultdict
import time
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
from urllib.parse import urlparse, quote_plus
from playwright.async_api import async_playwright
from playwright.sync_api import sync_playwright
import aiohttp
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Load environment variables
load_dotenv()

# Playwright needs subprocess support on Windows; Proactor loop provides it.
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from serp_backends import fetch_serp
from serp_backends.page_backends import page_backend, parse_page_html, scrape_url, scrape_url_sync

# Import from google_search module
from google_search import (
    BASE_CHROMIUM_ARGS,
    get_hardened_fingerprint,
    apply_stealth,
    resolve_browser_headless,
    normalize_url,
    is_organic_host,
    load_dr_data,
    get_authority,
    GOOGLE_URL,
    SESSION_PATH,
    EXTRACT_JS,
)

from serp_captcha_recovery import (
    SerpBlockedError,
    SerpCaptchaError,
    apply_recovery_rotation,
    can_recover_in_process,
    clear_hard_restart_flag,
    clear_recovery_state,
    get_recovery_overrides,
    is_hard_restart_pending,
    load_pending_recovery,
    max_captcha_attempts,
    prepare_recovery,
    restart_backend,
    should_hard_restart,
    update_recovery_state,
)

# Import from process_entities module (used in merge/other logic if needed)
from nlp_compat import patch_torch_jit_for_py313

patch_torch_jit_for_py313()
from process_entities import (
    calculate_tfidf_scores,
    calculate_weightage,
)

from nlp_tier_utils import exceeds_max_nlp_words, is_exempt_nlp_text

# NLP extraction pipeline: hosted GLiNER -> BGE-M3 -> reranker -> dedup -> cluster
from NLP_Extraction_and_Ranking.pipeline import run_pipeline
from NLP_Extraction_and_Ranking.nlp_serving_urls import (
    BGE_API_STYLE,
    BGE_ENCODE_URL as NLP_BGE_ENCODE_URL,
    BGE_MODEL_ID as NLP_BGE_MODEL_ID,
    BIENCODER_API_URL as NLP_BIENCODER_URL,
    CROSSENCODER_API_URL as NLP_CROSSENCODER_URL,
    GLINER_API_URL as NLP_GLINER_URL,
    GLINER_MODEL_ID as NLP_GLINER_MODEL_ID,
    GLINER_PREDICT_URL as NLP_GLINER_PREDICT_URL,
    GLINER_THRESHOLD as NLP_GLINER_THRESHOLD,
    MODEL_API_MODE as NLP_MODEL_API_MODE,
    RERANK_MODEL_ID as NLP_RERANK_MODEL_ID,
    RERANK_URL as NLP_RERANK_URL,
    TRITON_HTTP_URL as NLP_TRITON_URL,
    USE_RERANKER as NLP_USE_RERANKER,
    USE_TRITON as NLP_USE_TRITON,
)

# Import merge functionality
from nlp_word_buckets import (
    build_tier_word_buckets,
    ensure_word_buckets_in_merge_output,
)
from surfer_nlp_service import list_keyword_nlp_outputs, load_keyword_nlp
from merge_service import (
    aggregate_entities_from_items,
    build_merge_response,
    build_session_tiering_prep,
    ensure_keyword_instances_in_merge_output,
    ensure_gliner_labels_in_merge_output,
    ensure_proportional_tiers_in_merge_output,
    load_merge_cache,
    load_tiering_prep,
    save_merge_cache,
    save_tiering_prep,
    selection_cache_key,
)
from nlp_embedding_tiers.service import prepare_anchor_bundle
from database import (
    assign_article_permission,
    authenticate_user,
    create_search_session,
    create_user,
    delete_search_session,
    get_article_history,
    get_article_for_user,
    get_search_session,
    get_user_by_id,
    import_json_outputs,
    init_db,
    list_keyword_json_outputs,
    list_articles_for_user,
    delete_keyword_json_output,
    list_search_sessions,
    update_search_session_merge,
    update_search_session_merge_output,
    update_user_role,
    upsert_article,
    delete_article,
    upsert_keyword_json_output,
    verify_email_token,
    set_reset_token,
    reset_password_with_token,
)
from email_service import send_verification_email, send_reset_password_email
import jwt

SECRET_KEY = os.getenv("JWT_SECRET", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 14

security = HTTPBearer()

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


def get_current_admin(current_user: dict = Depends(get_current_user)):
    user_id = current_user.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = get_user_by_id(user_id)
    if not user or user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRACK_CSV_PATH = Path(__file__).resolve().parent.parent / "time-track.csv"
TRACK_CSV_LOCK = threading.Lock()
TRACK_CSV_COLUMNS = [
    "sr_no",
    "timestamp",
    "session_id",
    "keyword",
    "status",
    "error",
    "requested_k",
    "google_urls_found",
    "urls_selected_for_scraping",
    "scrape_success_count",
    "scrape_failed_count",
    "total_results_returned",
    "use_proxy",
    "use_browser",
    "headless",
    "device",
    "total_time_seconds",
    "google_search_seconds",
    "content_scraping_seconds",
    "save_result_files_seconds",
    "autosave_json_seconds",
    "nlp_total_seconds",
    "nlp_preprocess_seconds",
    "nlp_gliner_seconds",
    "nlp_ranking_seconds",
    "nlp_dedup_seconds",
    "nlp_clustering_seconds",
    "avg_page_scrape_seconds",
    "avg_page_nlp_seconds",
]


def _round_seconds(value: float) -> float:
    return round(float(value or 0.0), 4)


def _ensure_tracking_csv() -> None:
    TRACK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (not TRACK_CSV_PATH.exists()) or TRACK_CSV_PATH.stat().st_size == 0
    if needs_header:
        with open(TRACK_CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACK_CSV_COLUMNS)
            writer.writeheader()


def _next_sr_no() -> int:
    if not TRACK_CSV_PATH.exists() or TRACK_CSV_PATH.stat().st_size == 0:
        return 1
    try:
        with open(TRACK_CSV_PATH, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            return len(rows) + 1
    except Exception:
        return 1


def append_time_track_row(row: Dict) -> None:
    with TRACK_CSV_LOCK:
        _ensure_tracking_csv()
        row_out = {col: row.get(col, "") for col in TRACK_CSV_COLUMNS}
        row_out["sr_no"] = row_out.get("sr_no") or _next_sr_no()
        with open(TRACK_CSV_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRACK_CSV_COLUMNS)
            writer.writerow(row_out)

_recovery_resume_started = False
_serp_operation_lock = asyncio.Lock()


def _search_request_to_dict(request: "SearchRequest") -> Dict[str, Any]:
    return request.model_dump()


def _batch_request_to_dict(request: "BatchSearchRequest") -> Dict[str, Any]:
    return request.model_dump()


def _schedule_background_task(coro) -> None:
    task = asyncio.create_task(coro)

    def _on_done(done_task: asyncio.Task) -> None:
        try:
            exc = done_task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            log.error("[CAPTCHA recovery] Background task failed: %s", exc, exc_info=exc)

    task.add_done_callback(_on_done)


def _batch_ctx_from_state(state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not state.get("batch_mode"):
        return None
    return {
        "keywords": state.get("batch_keywords", []),
        "index": state.get("batch_index", 0),
        "batch_request": state.get("batch_request"),
    }


async def _handle_serp_failure(
    request: "SearchRequest",
    error: SerpCaptchaError,
    batch_ctx: Optional[Dict[str, Any]] = None,
    *,
    last_error: Optional[str] = None,
) -> str:
    """
    Hybrid recovery: in-process rotation first, hard restart when exhausted or wedged.
    Returns 'retry' to re-run scrape in the same process.
    """
    reason = last_error or getattr(error, "reason", "sorry")
    force_proxy, _ = get_recovery_overrides()
    state = load_pending_recovery() or {}

    if can_recover_in_process(state) and not getattr(error, "wedged", False):
        try:
            apply_recovery_rotation(
                _search_request_to_dict(request),
                error.browser,
                batch_mode=batch_ctx is not None,
                batch_keywords=(batch_ctx or {}).get("keywords"),
                batch_index=(batch_ctx or {}).get("index", 0),
                current_proxy=force_proxy,
                batch_request=(batch_ctx or {}).get("batch_request"),
                last_error=reason,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return "retry"

    if should_hard_restart(state, wedged=getattr(error, "wedged", False)):
        try:
            prepare_recovery(
                _search_request_to_dict(request),
                error.browser,
                batch_mode=batch_ctx is not None,
                batch_keywords=(batch_ctx or {}).get("keywords"),
                batch_index=(batch_ctx or {}).get("index", 0),
                current_proxy=force_proxy,
                batch_request=(batch_ctx or {}).get("batch_request"),
                last_error=reason,
                for_hard_restart=True,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        restart_backend()

    hits = int(state.get("captcha_hits", 0))
    raise HTTPException(
        status_code=503,
        detail=(
            f"SERP blocked after {hits} recovery attempt(s) ({reason}). "
            "Try SURFOX_HEADFUL=1, update SERP_CAPTCHA_PROXY_POOL, or wait before retrying."
        ),
    )


async def _scrape_google_with_recovery(
    request: "SearchRequest",
    *,
    target_urls: int,
    effective_headless: bool,
    batch_ctx: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Scrape Google with in-process CAPTCHA/block recovery loops."""
    safety_cap = max(3, max_captcha_attempts() + 3)
    all_urls: List[str] = []

    for _ in range(safety_cap):
        force_proxy, serp_browser = get_recovery_overrides()
        try:
            all_urls = await fetch_serp(
                request.keyword,
                k=target_urls,
                headless=effective_headless,
                use_proxy=request.use_proxy,
                device=request.device,
                serp_browser=serp_browser,
                proxy_url=force_proxy,
            )
        except SerpCaptchaError as captcha_exc:
            if await _handle_serp_failure(request, captcha_exc, batch_ctx) == "retry":
                continue
            return []

        if all_urls:
            return all_urls

        empty_error = SerpBlockedError(
            browser=serp_browser,
            message="No organic URLs extracted from SERP",
        )
        if await _handle_serp_failure(
            request, empty_error, batch_ctx, last_error="empty_serp"
        ) == "retry":
            continue
        return []

    raise HTTPException(
        status_code=503,
        detail="SERP scrape exceeded recovery retry limit.",
    )


async def _resume_after_captcha_restart(state: Dict[str, Any]) -> None:
    """Resume in-flight search after hard backend restart."""
    await asyncio.sleep(2)
    try:
        if state.get("batch_mode"):
            await _resume_batch_search(state)
        else:
            req = SearchRequest(**state["request"])
            await _search_core(req)
        clear_recovery_state()
        log.info("[CAPTCHA recovery] Resumed search completed successfully")
    except SerpCaptchaError as captcha_exc:
        req_data = state.get("request") or {}
        req = SearchRequest(**req_data)
        action = await _handle_serp_failure(
            req, captcha_exc, _batch_ctx_from_state(state)
        )
        if action == "retry":
            await _resume_after_captcha_restart(load_pending_recovery() or state)
    except HTTPException as http_exc:
        log.warning(
            "[CAPTCHA recovery] Resume failed: %s",
            getattr(http_exc, "detail", http_exc),
        )
        try:
            update_recovery_state(
                {
                    "last_error": str(getattr(http_exc, "detail", http_exc)),
                }
            )
        except Exception:
            pass
    except Exception:
        log.exception("[CAPTCHA recovery] Failed to resume search after restart")
    finally:
        clear_hard_restart_flag()


async def _resume_batch_search(state: Dict[str, Any]) -> None:
    batch_data = state.get("batch_request") or {}
    keywords = state.get("batch_keywords") or batch_data.get("keywords") or []
    start_idx = int(state.get("batch_index", 0))
    batch_req = BatchSearchRequest(**{**batch_data, "keywords": keywords})

    for i in range(start_idx, len(keywords)):
        kw = keywords[i]
        log.info("[CAPTCHA recovery] Resuming batch keyword %d/%d: %r", i + 1, len(keywords), kw)
        await _search_core(
            SearchRequest(
                keyword=kw,
                k=batch_req.k,
                use_proxy=batch_req.use_proxy,
                headless=batch_req.headless,
                use_browser=batch_req.use_browser,
                device=batch_req.device,
            ),
            batch_ctx={
                "keywords": keywords,
                "index": i,
                "batch_request": _batch_request_to_dict(batch_req),
            },
        )


async def _probe_http_json(name: str, url: str, timeout_seconds: float = 2.5) -> Dict:
    status = {
        "name": name,
        "url": url,
        "ok": False,
        "http_status": None,
        "detail": "",
    }
    try:
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                status["http_status"] = response.status
                text = await response.text()
                status["ok"] = 200 <= response.status < 300
                status["detail"] = text[:500]
    except Exception as exc:
        status["detail"] = str(exc)
    return status


async def check_nlp_service_health() -> Dict:
    probe_tasks = [
        _probe_http_json("triton", f"{NLP_TRITON_URL}/v2/health/ready"),
        _probe_http_json("gliner", f"{NLP_GLINER_URL}/health"),
        _probe_http_json("biencoder", f"{NLP_BIENCODER_URL}/health"),
    ]
    if NLP_USE_RERANKER:
        probe_tasks.append(_probe_http_json("reranker", f"{NLP_CROSSENCODER_URL.rstrip('/')}/health"))
    else:
        probe_tasks.append(
            _probe_http_json("crossencoder", f"{NLP_CROSSENCODER_URL}/model_info")
        )
    checks = await asyncio.gather(*probe_tasks)
    required_names = {"gliner", "biencoder"}
    if NLP_USE_TRITON:
        required_names.add("triton")
    if NLP_USE_RERANKER:
        required_names.add("reranker")
    required = [c for c in checks if c["name"] in required_names]
    return {
        "ok": all(c["ok"] for c in required),
        "checks": checks,
        "rerank_enabled": NLP_USE_RERANKER,
        "hint": (
            "Configure SURF_GLINER_API_URL, SURF_BIENCODER_API_URL, and SURF_CROSSENCODER_API_URL. "
            "For Triton mode, start Triton before the GLiNER/BGE wrappers."
        ),
    }


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Application lifecycle hook."""
    global _recovery_resume_started
    log.info(
        "Backend startup: NLP models | mode=%s | bge_style=%s | rerank_enabled=%s",
        NLP_MODEL_API_MODE,
        BGE_API_STYLE,
        NLP_USE_RERANKER,
    )
    log.info("  GLiNER %s @ %s", NLP_GLINER_MODEL_ID, NLP_GLINER_PREDICT_URL)
    log.info("  BGE %s @ %s", NLP_BGE_MODEL_ID, NLP_BGE_ENCODE_URL)
    log.info("  Reranker %s @ %s", NLP_RERANK_MODEL_ID, NLP_RERANK_URL)
    try:
        health = await check_nlp_service_health()
        if health["ok"]:
            log.info("NLP service health check passed")
        else:
            bad = [
                f"{c['name']}({c['url']})={c['http_status'] or c['detail']}"
                for c in health["checks"]
                if not c["ok"]
            ]
            log.warning("NLP service health check failed: %s", " | ".join(bad))
            log.warning("%s", health["hint"])
    except Exception:
        log.exception("NLP service health check failed unexpectedly")
    try:
        await asyncio.to_thread(init_db)
    except Exception:
        log.exception("PostgreSQL initialization failed")
    pending = load_pending_recovery()
    if pending and is_hard_restart_pending() and not _recovery_resume_started:
        _recovery_resume_started = True
        log.warning(
            "[CAPTCHA recovery] Hard-restart resume scheduled (attempt #%s)",
            pending.get("captcha_hits"),
        )
        _schedule_background_task(_resume_after_captcha_restart(pending))
    elif pending and not is_hard_restart_pending():
        log.info(
            "[CAPTCHA recovery] Pending in-process state found without hard-restart flag; leaving for next search"
        )
    yield


# FastAPI app setup
app = FastAPI(title="Surfox", description="NLP-powered content analysis", lifespan=lifespan)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

RESULTS_DIR = os.getenv('RESULTS_DIR', 'results')
JSON_OUTPUTS_DIR = Path(__file__).resolve().parent / "json outputs"
TARGET_TITLE = "Dog Breeds for Different Lifestyles"
PORT = int(os.getenv("BACKEND_PORT", os.getenv("PORT", "8010")))

URL_PROCESSING_BATCH_SIZE = int(os.getenv("URL_PROCESSING_BATCH_SIZE", "8"))
MAX_PAGE_NLP_TERMS = int(os.getenv("MAX_PAGE_NLP_TERMS", "100"))
MAX_FINAL_NLP_TERMS = int(os.getenv("MAX_FINAL_NLP_TERMS", "300"))
MAX_NUMERICAL_NLPS_PER_TIER = int(os.getenv("MAX_NUMERICAL_NLPS_PER_TIER", "2"))
NLP_PER_ARTICLE_KEEP_RATIO = float(os.getenv("NLP_PER_ARTICLE_KEEP_RATIO", "0.80"))
GLINER_CONTEXT_SIZE = int(os.getenv("GLINER_CONTEXT_SIZE", "600"))
GLINER_STEP_SIZE = int(os.getenv("GLINER_STEP_SIZE", "400"))
GLINER_THRESHOLD = NLP_GLINER_THRESHOLD
SCRAPE_CONCURRENCY = int(os.getenv("SCRAPE_CONCURRENCY", "6"))
NLP_MIN_WORDS = int(os.getenv("NLP_MIN_WORDS", "30"))
NLP_THIN_CONTENT_WORDS = int(os.getenv("NLP_THIN_CONTENT_WORDS", "80"))
NLP_LOW_QUALITY_MAX_NLPS = int(os.getenv("NLP_LOW_QUALITY_MAX_NLPS", "60"))
NLP_MEDIUM_QUALITY_MAX_NLPS = int(os.getenv("NLP_MEDIUM_QUALITY_MAX_NLPS", "80"))
NLP_ERROR_TITLE_MARKERS = [
    x.strip().casefold()
    for x in os.getenv(
        "NLP_ERROR_TITLE_MARKERS",
        "404,403,access denied,captcha,web server is returning an unknown error,error",
    ).split(",")
    if x.strip()
]

FRONTEND_DIR = os.getenv(
    "FRONTEND_DIR",
    str(Path(__file__).resolve().parent.parent / "frontend" / "build"),
)
FRONTEND_INDEX = Path(FRONTEND_DIR) / "index.html"

if Path(FRONTEND_DIR).exists():
    static_dir = Path(FRONTEND_DIR) / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

# NLP extraction uses hosted GLiNER, BGE-M3, and reranker APIs

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class SearchRequest(BaseModel):
    keyword: str
    k: int = 20
    use_proxy: bool = False
    headless: bool = True
    use_browser: bool = False
    device: str = "desktop"

class BatchSearchRequest(BaseModel):
    keywords: List[str]
    k: int = 20
    use_proxy: bool = False
    headless: bool = True
    use_browser: bool = False
    device: str = "desktop"

class MergeRequest(BaseModel):
    selected_urls: List[str]
    session_id: str
    keyword: Optional[str] = None  # used for saving JSON by keyword name

class SelectNlpKeywordsRequest(BaseModel):
    source_keyword: Optional[str] = None
    file_name: Optional[str] = None
    json_output: Dict
    selected_keywords: Optional[Dict[str, List[str]]] = None

class RegisterUserRequest(BaseModel):
    name: str
    email: str
    password: str
    role: str = "content_writer"

class LoginRequest(BaseModel):
    email: str
    password: str

class ForgotPasswordRequest(BaseModel):
    email: str

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

class UpdateUserRoleRequest(BaseModel):
    user_id: int
    role: str

class SaveArticleRequest(BaseModel):
    article_key: str
    session_id: Optional[str] = None
    title: str = "Untitled"
    keyword: Optional[str] = None
    keywords: List[str] = []
    results: List[Dict] = []
    selected_urls: List[str] = []
    content_score: int = 0
    html: str = ""
    text: str = ""
    user_id: Optional[int] = None
    diff_patch: Optional[str] = None

class ArticlePermissionRequest(BaseModel):
    article_key: str
    user_id: int
    can_edit: bool = False
    can_update: bool = False
    assigned_by: Optional[int] = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def scrape_page_content(page, url):
    """Scrape content from a single page."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        html = await page.content()
        return parse_page_html(html, url)
    except Exception:
        log.exception("Failed to scrape %s", url)
        return None

def scrape_page_content_sync(page, url):
    """Sync Playwright fallback scraper."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        html = page.content()
        return parse_page_html(html, url)
    except Exception:
        log.exception("Failed to scrape %s (sync fallback)", url)
        return None

def _scrape_pages_sync(urls: List[str], headless: bool, device: str):
    """Sync fallback for page scraping (Playwright or Scrapling)."""
    if page_backend() == "scrapling":
        scraped_pages = []
        for idx, url in enumerate(urls):
            try:
                page_data = asyncio.run(scrape_url(url))
                if page_data is not None:
                    scraped_pages.append((idx, page_data))
            except Exception:
                log.exception("[sync fallback] Error scraping %s", url)
        return scraped_pages

    fingerprint = get_hardened_fingerprint(device)
    effective_headless = resolve_browser_headless(headless, log)
    scraped_pages = []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=effective_headless,
            args=BASE_CHROMIUM_ARGS,
        )
        try:
            for idx, url in enumerate(urls):
                context = None
                try:
                    context = browser.new_context(
                        user_agent=fingerprint["user_agent"],
                        viewport=fingerprint["viewport"],
                    )
                    context.add_init_script(
                        f"""
                        Object.defineProperty(navigator, 'webdriver', {{get: () => undefined}});
                        Object.defineProperty(navigator, 'hardwareConcurrency', {{get: () => {fingerprint['hardware_concurrency']} }});
                        Object.defineProperty(navigator, 'deviceMemory', {{get: () => {fingerprint['device_memory']} }});
                        """
                    )
                    page = context.new_page()
                    log.info("[sync fallback] Scraping (%d/%d) %s", idx + 1, len(urls), url)
                    page_data = scrape_url_sync(page, url)
                    if page_data is not None:
                        scraped_pages.append((idx, page_data))
                except Exception:
                    log.exception("[sync fallback] Error scraping %s", url)
                finally:
                    try:
                        if context is not None:
                            context.close()
                    except Exception:
                        pass
        finally:
            browser.close()
    return scraped_pages

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Surfox backend is running",
        "nlp": {
            "mode": NLP_MODEL_API_MODE,
            "use_triton": NLP_USE_TRITON,
            "bge_api_style": BGE_API_STYLE,
            "rerank_enabled": NLP_USE_RERANKER,
            "gliner_model": NLP_GLINER_MODEL_ID,
            "gliner_url": NLP_GLINER_PREDICT_URL,
            "bge_model": NLP_BGE_MODEL_ID,
            "biencoder_url": NLP_BGE_ENCODE_URL,
            "rerank_model": NLP_RERANK_MODEL_ID,
            "rerank_url": NLP_RERANK_URL,
        },
    }

@app.get("/nlp-keywords")
async def get_nlp_keywords(
    source_keyword: Optional[str] = None,
    current_user: dict = Depends(get_current_user),
):
    try:
        outputs = await asyncio.to_thread(
            list_keyword_json_outputs, source_keyword, current_user["user_id"]
        )
        return {"total": len(outputs), "items": outputs}
    except Exception as exc:
        log.exception("Could not load NLP keyword JSON outputs from PostgreSQL")
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

@app.post("/nlp-keywords/select")
async def save_selected_nlp_keywords(
    request: SelectNlpKeywordsRequest,
    current_user: dict = Depends(get_current_user),
):
    try:
        source_keyword = (request.source_keyword or request.file_name or "search").strip()
        saved = await asyncio.to_thread(
            upsert_keyword_json_output,
            source_keyword,
            request.file_name or f"{source_keyword}.json",
            request.json_output,
            current_user["user_id"],
        )
        return {"item": saved}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Could not save selected NLP keywords JSON to PostgreSQL")
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

@app.post("/nlp-keywords/import-json")
async def import_saved_json_keywords(current_user: dict = Depends(get_current_admin)):
    try:
        imported = await asyncio.to_thread(import_json_outputs, JSON_OUTPUTS_DIR)
        return {"imported": imported}
    except Exception as exc:
        log.exception("Could not import JSON output keywords into PostgreSQL")
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

@app.delete("/nlp-keywords/{output_id}")
async def delete_nlp_keyword_output(
    output_id: int,
    current_user: dict = Depends(get_current_user),
):
    try:
        deleted = await asyncio.to_thread(
            delete_keyword_json_output, output_id, current_user["user_id"]
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Keyword history not found")
        return {"status": "ok", "message": "Keyword history deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not delete NLP keyword JSON output from PostgreSQL")
        raise HTTPException(status_code=503, detail=f"Database error: {exc}") from exc

@app.post("/auth/register")
async def register_user(request: RegisterUserRequest, background_tasks: BackgroundTasks):
    try:
        user = await asyncio.to_thread(
            create_user,
            request.name,
            request.email,
            request.password,
            request.role,
            None, # None means it will immediately be verified and no verification token is generated
        )
        token = create_access_token(data={"sub": user["email"], "user_id": user["id"]})
        return {**user, "token": token, "message": "Registration successful. You are now logged in."}
    except Exception as exc:
        log.exception("Could not register user")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.get("/auth/verify-email")
async def verify_email(token: str):
    try:
        user = await asyncio.to_thread(verify_email_token, token)
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired verification token.")
        return {**user, "message": "Email verified successfully. You can now log in."}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not verify email")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/auth/login")
async def login_user(request: LoginRequest):
    try:
        user = await asyncio.to_thread(authenticate_user, request.email, request.password)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")
        if not user.get("email_verified", True):
            raise HTTPException(status_code=403, detail="Please verify your email before logging in.")
        token = create_access_token(data={"sub": user["email"], "user_id": user["id"]})
        return {**user, "token": token}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not login user")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/auth/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks):
    try:
        token = secrets.token_urlsafe(32)
        found = await asyncio.to_thread(set_reset_token, request.email, token)
        if found:
            # Fetch user name for the email
            from database import get_article  # reuse conn helper
            background_tasks.add_task(send_reset_password_email, request.email, "User", token)
        # Always return success to avoid email enumeration
        return {"message": "If an account exists with that email, a reset link has been sent."}
    except Exception as exc:
        log.exception("Could not process forgot password")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/auth/reset-password")
async def reset_password(request: ResetPasswordRequest):
    try:
        user = await asyncio.to_thread(reset_password_with_token, request.token, request.new_password)
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token. Please request a new one.")
        return {**user, "message": "Password reset successfully. You can now log in."}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not reset password")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/auth/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    user = await asyncio.to_thread(get_user_by_id, current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/admin/users/role")
async def change_user_role(
    request: UpdateUserRoleRequest,
    current_user: dict = Depends(get_current_admin),
):
    try:
        return await asyncio.to_thread(update_user_role, request.user_id, request.role)
    except Exception as exc:
        log.exception("Could not update user role")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

@app.post("/articles")
async def save_article(request: SaveArticleRequest, current_user: dict = Depends(get_current_user)):
    try:
        payload = request.dict()
        payload["user_id"] = current_user["user_id"]
        return await asyncio.to_thread(upsert_article, payload)
    except Exception as exc:
        log.exception("Could not save article")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/articles")
async def get_all_articles(current_user: dict = Depends(get_current_user)):
    try:
        articles = await asyncio.to_thread(list_articles_for_user, current_user["user_id"])
        return {"items": articles}
    except Exception as exc:
        log.exception("Could not load articles")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/articles/{article_key}")
async def load_article(article_key: str, current_user: dict = Depends(get_current_user)):
    try:
        article = await asyncio.to_thread(
            get_article_for_user, article_key, current_user["user_id"]
        )
        if not article:
            raise HTTPException(status_code=404, detail="Article not found")
        return article
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not load article")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.delete("/articles/{article_key}")
async def remove_article(article_key: str, current_user: dict = Depends(get_current_user)):
    try:
        success = await asyncio.to_thread(delete_article, article_key)
        if not success:
            raise HTTPException(status_code=404, detail="Article not found")
        return {"status": "ok", "message": "Article deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not delete article")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.get("/articles/{article_key}/history")
async def load_article_history(article_key: str, current_user: dict = Depends(get_current_user)):
    try:
        return await asyncio.to_thread(get_article_history, article_key)
    except Exception as exc:
        log.exception("Could not load article history")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/articles/permissions")
async def save_article_permission(request: ArticlePermissionRequest, current_user: dict = Depends(get_current_user)):
    try:
        return await asyncio.to_thread(
            assign_article_permission,
            request.article_key,
            request.user_id,
            request.can_edit,
            request.can_update,
            current_user["user_id"],
        )
    except Exception as exc:
        log.exception("Could not save article permission")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

def is_social_media_url(url: str) -> bool:
    """Check if URL is from social media platforms (YouTube, Facebook, Reddit)"""
    social_domains = ['youtube.com', 'facebook.com', 'reddit.com', 'youtu.be', 'fb.com']
    try:
        domain = urlparse(url).netloc.lower()
        return any(social in domain for social in social_domains)
    except:
        return False

async def _search_core(
    request: SearchRequest,
    user_id: int,
    batch_ctx: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Serialized entry point for the search pipeline (avoids overlapping SERP recovery)."""
    async with _serp_operation_lock:
        return await _search_core_impl(request, user_id, batch_ctx)


async def _search_core_impl(
    request: SearchRequest,
    user_id: int,
    batch_ctx: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Core search pipeline used by both /search and /batch_search.
    Includes auto-saving JSON outputs for the keyword.
    """
    # Start timer
    start_time = time.perf_counter()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_message = ""

    # timing buckets for this query
    timing_steps = {
        "google_search_seconds": 0.0,
        "content_scraping_seconds": 0.0,
        "save_result_files_seconds": 0.0,
        "autosave_json_seconds": 0.0,
        "nlp_total_seconds": 0.0,
        "nlp_preprocess_seconds": 0.0,
        "nlp_gliner_seconds": 0.0,
        "nlp_ranking_seconds": 0.0,
        "nlp_dedup_seconds": 0.0,
        "nlp_clustering_seconds": 0.0,
        "merge_prep_seconds": 0.0,
        "merge_tiering_seconds": 0.0,
    }
    page_scrape_durations: List[float] = []
    page_nlp_durations: List[float] = []
    nlp_sent_count = 0
    nlp_skipped_counts = defaultdict(int)

    # metadata for csv tracking
    target_urls = request.k + 10
    all_urls = []
    urls = []
    scraped_pages = []
    results = []
    status = "success"
    session_dir = Path(RESULTS_DIR) / session_id
    effective_headless = True
    anchor_prep_task = asyncio.create_task(
        asyncio.to_thread(prepare_anchor_bundle, request.keyword)
    )
    merge_output: Optional[Dict[str, Any]] = None
    all_selected_urls: List[str] = []

    try:
        # Create session directory
        session_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Get Google search results (request more to account for social media filtering)
        log.info(f"Searching Google for: {request.keyword}")

        # Headless vs headful browser mode.
        #
        # Default behavior stays headless for server use, but we allow opting into
        # headful mode (useful to manually solve Google CAPTCHA) via env var:
        #   SURFOX_HEADFUL=1
        #
        # This intentionally overrides the request payload to avoid exposing an
        # interactive browser by accident.
        def _env_truthy(name: str) -> bool:
            return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}

        effective_headless = resolve_browser_headless(
            requested_headless=not _env_truthy("SURFOX_HEADFUL"),
            logger=log,
        )

        google_start = time.perf_counter()
        all_urls = await _scrape_google_with_recovery(
            request,
            target_urls=target_urls,
            effective_headless=effective_headless,
            batch_ctx=batch_ctx,
        )
        timing_steps["google_search_seconds"] = time.perf_counter() - google_start

        # SERP ranking map: 1 = top result in returned Google list
        rank_map = {u: i + 1 for i, u in enumerate(all_urls or [])}

        if not all_urls:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No URLs found after SERP recovery attempts. "
                    "Try SURFOX_HEADFUL=1 or configure SERP_CAPTCHA_PROXY_POOL."
                ),
            )

        # IMPORTANT: Keep the exact Google SERP order.
        # Do NOT move social results (YouTube/Facebook/Reddit) to the end.
        # Also de-duplicate URLs while preserving order to avoid double scraping.
        urls = []
        seen_norm = set()
        for u in all_urls[:request.k]:
            nu = normalize_url(u)
            if nu in seen_norm:
                continue
            seen_norm.add(nu)
            urls.append(u)

        log.info("Google search completed in %.2fs", timing_steps["google_search_seconds"])
        log.info("Returning top %d results in SERP order", len(urls))

        # Step 2: Scrape content from ALL SERP URLs first
        scraping_start = time.perf_counter()
        scrape_semaphore = asyncio.Semaphore(max(1, SCRAPE_CONCURRENCY))
        try:
            if page_backend() == "scrapling":

                async def scrape_single_url_scrapling(url: str, idx: int):
                    async with scrape_semaphore:
                        step_started = time.perf_counter()
                        try:
                            log.info("Scraping (%d/%d) %s", idx + 1, len(urls), url)
                            page_data = await scrape_url(url)
                            page_scrape_durations.append(time.perf_counter() - step_started)
                            return page_data
                        except Exception:
                            log.exception("Error scraping %s", url)
                            page_scrape_durations.append(time.perf_counter() - step_started)
                            return None

                scraped_list = await asyncio.gather(
                    *[scrape_single_url_scrapling(url, idx) for idx, url in enumerate(urls)],
                    return_exceptions=True,
                )
            else:
                async with async_playwright() as p:
                    fingerprint = get_hardened_fingerprint(request.device)
                    browser = await p.chromium.launch(
                        headless=effective_headless,
                        args=BASE_CHROMIUM_ARGS,
                    )
                    shared_context = await browser.new_context(
                        user_agent=fingerprint["user_agent"],
                        viewport=fingerprint["viewport"],
                    )
                    await apply_stealth(shared_context, fingerprint)

                    async def scrape_single_url(url: str, idx: int):
                        async with scrape_semaphore:
                            page = None
                            step_started = time.perf_counter()
                            try:
                                page = await shared_context.new_page()
                                log.info("Scraping (%d/%d) %s", idx + 1, len(urls), url)
                                page_data = await scrape_url(url, page=page)
                                page_scrape_durations.append(time.perf_counter() - step_started)
                                return page_data
                            except Exception:
                                log.exception("Error scraping %s", url)
                                page_scrape_durations.append(time.perf_counter() - step_started)
                                return None
                            finally:
                                try:
                                    if page is not None:
                                        await page.close()
                                except Exception:
                                    pass

                    try:
                        scraped_list = await asyncio.gather(
                            *[scrape_single_url(url, idx) for idx, url in enumerate(urls)],
                            return_exceptions=True,
                        )
                    except Exception:
                        log.exception("Error scraping URLs in parallel")
                        scraped_list = []

                    try:
                        await shared_context.close()
                    except Exception:
                        pass
                    await browser.close()

            for idx, page_data in enumerate(scraped_list or []):
                if page_data is None or isinstance(page_data, Exception):
                    continue
                scraped_pages.append((idx, page_data))
        except NotImplementedError:
            log.warning("Async Playwright unavailable for page scraping; using sync fallback.")
            scrape_sync_started = time.perf_counter()
            scraped_pages = await asyncio.to_thread(
                _scrape_pages_sync,
                urls,
                effective_headless,
                request.device,
            )
            page_scrape_durations.append(time.perf_counter() - scrape_sync_started)
        timing_steps["content_scraping_seconds"] = time.perf_counter() - scraping_start

        # Step 3: Process scraped contents — GLiNER (6000) -> ranker (6005, 6010) -> deduplicator
        log.info(
            "[Search] Step 3/3 — NLP pipeline for %d pages (batch_size=%d | max_nlps_per_page=%d)",
            len(scraped_pages),
            URL_PROCESSING_BATCH_SIZE,
            MAX_PAGE_NLP_TERMS,
        )

        async def enrich_single_page(idx: int, page_data: Dict):
            nonlocal nlp_sent_count
            url = page_data.get("url", "")
            started = time.perf_counter()
            try:
                content = page_data.get("content", "") or ""
                word_count = int(page_data.get("word_count", 0) or 0)
                title = (page_data.get("title") or request.keyword or "").strip()
                domain = (page_data.get("domain") or "").strip().lower()

                def _domain_tokens(d: str) -> set[str]:
                    d = (d or "").strip().lower()
                    if not d:
                        return set()
                    d = d.split(":")[0]
                    parts = [p for p in d.split(".") if p and p not in {"www", "m", "amp"}]
                    toks = set(parts)
                    if len(parts) >= 2:
                        toks.add(parts[-2])
                    return toks

                banned_domain_tokens = _domain_tokens(domain)

                def _is_pure_number(txt: str) -> bool:
                    t = (txt or "").strip()
                    return bool(t) and t.isdigit()

                def _is_banned_term(txt: str) -> bool:
                    t = (txt or "").strip()
                    if not t:
                        return True
                    tl = t.casefold()
                    if tl in banned_domain_tokens:
                        return True
                    if _is_pure_number(t):
                        return True
                    return False

                nlp_terms = []
                ranking_method = "biencoder"
                clusters = None
                cluster_scores = None
                entities_list = []
                effective_max_nlps = MAX_PAGE_NLP_TERMS
                page_title_cf = title.casefold()
                has_error_title = any(marker in page_title_cf for marker in NLP_ERROR_TITLE_MARKERS)
                if not content.strip():
                    nlp_skipped_counts["empty_content"] += 1
                elif word_count < NLP_MIN_WORDS:
                    nlp_skipped_counts["too_short"] += 1
                elif has_error_title:
                    nlp_skipped_counts["error_like_title"] += 1
                else:
                    # Adaptive NLP depth by page quality to save GLiNER/BGE cycles.
                    if word_count < NLP_THIN_CONTENT_WORDS:
                        effective_max_nlps = min(MAX_PAGE_NLP_TERMS, NLP_LOW_QUALITY_MAX_NLPS)
                    elif word_count < 250:
                        effective_max_nlps = min(MAX_PAGE_NLP_TERMS, NLP_MEDIUM_QUALITY_MAX_NLPS)
                    nlp_sent_count += 1
                    log.info(
                        "[NLP] Page %d/%d — url=%s | word_count=%d | max_nlps=%d | dedup_threshold=0.85 | gliner_threshold=%.2f | gliner_step=%d",
                        idx + 1,
                        len(scraped_pages),
                        url[:60] + ("..." if len(url) > 60 else ""),
                        word_count,
                        effective_max_nlps,
                        GLINER_THRESHOLD,
                        GLINER_STEP_SIZE,
                    )
                    pipeline_result = await asyncio.to_thread(
                        run_pipeline,
                        content,
                        title or request.keyword,
                        max_nlps=effective_max_nlps,
                        dedup_threshold=0.85,
                        gliner_context_size=GLINER_CONTEXT_SIZE,
                        gliner_step_size=GLINER_STEP_SIZE,
                    )
                    ranking_method = pipeline_result.get("ranking_method", "biencoder")
                    entities_list = pipeline_result.get("entities", [])
                    clusters = pipeline_result.get("clusters")
                    cluster_scores = pipeline_result.get("cluster_scores")
                    pipeline_timing = pipeline_result.get("timing_seconds") or {}
                    timing_steps["nlp_preprocess_seconds"] += float(pipeline_timing.get("preprocess_seconds", 0.0) or 0.0)
                    timing_steps["nlp_gliner_seconds"] += float(pipeline_timing.get("gliner_seconds", 0.0) or 0.0)
                    timing_steps["nlp_ranking_seconds"] += float(pipeline_timing.get("ranking_seconds", 0.0) or 0.0)
                    timing_steps["nlp_dedup_seconds"] += float(pipeline_timing.get("dedup_seconds", 0.0) or 0.0)
                    timing_steps["nlp_clustering_seconds"] += float(pipeline_timing.get("clustering_seconds", 0.0) or 0.0)
                    # Keep backward compatible total accounting but include embedding time in ranking bucket.
                    timing_steps["nlp_ranking_seconds"] += float(pipeline_timing.get("embedding_seconds", 0.0) or 0.0)
                    timing_steps["nlp_total_seconds"] += float(pipeline_timing.get("total_seconds", 0.0) or 0.0)
                    log.info(
                        "[NLP] Page done — url=%s | method=%s | terms=%d",
                        url[:50] + ("..." if len(url) > 50 else ""),
                        ranking_method,
                        len(entities_list),
                    )

                anchor_title = (request.keyword or "").strip()
                nlp_word_exempt: list[str] = []
                if anchor_title:
                    nlp_word_exempt.append(anchor_title)
                page_title = (title or "").strip()
                if page_title and page_title.casefold() != anchor_title.casefold():
                    nlp_word_exempt.append(page_title)
                if anchor_title and not _is_banned_term(anchor_title):
                    nlp_terms.append({
                        "text": anchor_title,
                        "count": 1,
                        "relevance": 1.0,
                        "weightage": 1.0,
                        "source": "gliner",
                        "label": "Phrase",
                    })

                for e in entities_list:
                    score = e.get("crossencoder_score") or e.get("biencoder_score") or 0.0
                    entity_text = e.get("text") or ""
                    if _is_banned_term(entity_text):
                        continue
                    if exceeds_max_nlp_words(entity_text) and not is_exempt_nlp_text(
                        entity_text, nlp_word_exempt
                    ):
                        continue
                    term_entry = {
                        "text": e["text"],
                        "count": e.get("count", 1),
                        "relevance": score,
                        "weightage": score,
                        "source": "gliner",
                        "label": e.get("label") or "Other",
                    }
                    if e.get("embedding_unit") is not None:
                        term_entry["embedding_unit"] = e["embedding_unit"]
                    nlp_terms.append(term_entry)

                seen = set()
                deduped = []
                for t in nlp_terms:
                    key = (t.get("text") or "").strip().casefold()
                    if not key or key in seen:
                        continue
                    seen.add(key)
                    deduped.append(t)
                nlp_terms = deduped

                result = {
                    "rank": rank_map.get(url),
                    "url": url,
                    "domain": domain,
                    "title": page_data.get("title", ""),
                    "description": page_data.get("description", ""),
                    "word_count": word_count,
                    "heading_count": int(page_data.get("heading_count", 0) or 0),
                    "para_count": int(page_data.get("para_count", 0) or 0),
                    "authority": page_data.get("authority", 0),
                    "entities": [],
                    "total_entities": 0,
                    "keyphrases": [],
                    "gpt_terms": [],
                    "nlp_terms": nlp_terms,
                    "total_nlp_terms": len(nlp_terms),
                    "ranking_method": ranking_method,
                    "nlp_clusters": clusters,
                    "nlp_cluster_scores": cluster_scores,
                    "content_preview": content[:500] if content else "",
                }
                page_nlp_durations.append(time.perf_counter() - started)
                return result
            except Exception:
                log.exception("NLP processing error for %s", url)
                page_nlp_durations.append(time.perf_counter() - started)
                return None

        def _batches(items, batch_size: int):
            size = max(1, int(batch_size or 1))
            for i in range(0, len(items), size):
                yield items[i:i + size]

        processed_results = []
        for batch in _batches(scraped_pages, URL_PROCESSING_BATCH_SIZE):
            batch_out = await asyncio.gather(
                *[enrich_single_page(idx, page_data) for idx, page_data in batch],
                return_exceptions=True,
            )
            for item in batch_out:
                if item is None or isinstance(item, Exception):
                    continue
                processed_results.append(item)

        processed_results.sort(key=lambda r: (r.get("rank") is None, r.get("rank", 10**9)))
        results = processed_results
        log.info(
            "[NLP] Dispatch summary — sent=%d | skipped=%d | reasons=%s",
            nlp_sent_count,
            sum(nlp_skipped_counts.values()),
            dict(nlp_skipped_counts),
        )

        save_start = time.perf_counter()
        for idx, result in enumerate(results, 1):
            try:
                file_path = session_dir / f"{idx}.json"
                with open(file_path, "w") as f:
                    json.dump(result, f, indent=2)
            except Exception:
                log.exception("Error saving result to file")
        timing_steps["save_result_files_seconds"] = time.perf_counter() - save_start

        merge_start = time.perf_counter()
        try:
            all_selected_urls = [r.get("url") for r in results if r.get("url")]
            merge_prep_started = time.perf_counter()
            anchor_bundle = await anchor_prep_task
            tiering_prep = build_session_tiering_prep(
                request.keyword,
                results,
                anchor_texts=anchor_bundle.get("anchor_texts"),
                query_unit=anchor_bundle.get("query_unit"),
                query_generation_method=anchor_bundle.get("query_generation_method") or "",
                raw_gliner=anchor_bundle.get("raw_gliner"),
                gliner_variants=anchor_bundle.get("gliner_variants"),
            )
            save_tiering_prep(session_dir, tiering_prep)
            timing_steps["merge_prep_seconds"] = time.perf_counter() - merge_prep_started

            merged_entities, merge_stats, ranking_method = aggregate_entities_from_items(
                results,
                keep_ratio=NLP_PER_ARTICLE_KEEP_RATIO,
            )
            merge_output = build_merge_response(
                merged_entities,
                merge_stats,
                ranking_method,
                request.keyword or session_id,
                tiering_prep=tiering_prep,
                max_numerical_per_tier=MAX_NUMERICAL_NLPS_PER_TIER,
                json_outputs_dir=JSON_OUTPUTS_DIR,
                persist_keyword_json=True,
                upsert_keyword_json=upsert_keyword_json_output,
                user_id=user_id,
            )
            cache_key = selection_cache_key(all_selected_urls)
            save_merge_cache(session_dir, cache_key, merge_output)
            log.info(
                "[Search] Pre-merge complete — Green=%d | Orange=%d | White=%d",
                len(merge_output.get("green_nlps") or []),
                len(merge_output.get("orange_nlps") or []),
                len(merge_output.get("white_nlps") or []),
            )
        except Exception:
            log.exception("[Search] Pre-merge tiering failed")
            merge_output = None
        timing_steps["merge_tiering_seconds"] = time.perf_counter() - merge_start
        timing_steps["autosave_json_seconds"] = timing_steps["merge_tiering_seconds"]

        total_time = time.perf_counter() - start_time
        avg_page_scrape_seconds = (sum(page_scrape_durations) / len(page_scrape_durations)) if page_scrape_durations else 0.0
        avg_page_nlp_seconds = (sum(page_nlp_durations) / len(page_nlp_durations)) if page_nlp_durations else 0.0

        log.info("Scraping completed in %.2fs", timing_steps["content_scraping_seconds"])
        log.info("NLP total completed in %.2fs", timing_steps["nlp_total_seconds"])
        log.info("Total search time: %.2fs", total_time)

        clear_recovery_state()

        timing_payload = {
            "google_search_time_seconds": _round_seconds(timing_steps["google_search_seconds"]),
            "content_scraping_time_seconds": _round_seconds(timing_steps["content_scraping_seconds"]),
            "nlp_total_time_seconds": _round_seconds(timing_steps["nlp_total_seconds"]),
            "nlp_step_times_seconds": {
                "preprocess": _round_seconds(timing_steps["nlp_preprocess_seconds"]),
                "gliner": _round_seconds(timing_steps["nlp_gliner_seconds"]),
                "ranking": _round_seconds(timing_steps["nlp_ranking_seconds"]),
                "deduplication": _round_seconds(timing_steps["nlp_dedup_seconds"]),
                "clustering": _round_seconds(timing_steps["nlp_clustering_seconds"]),
            },
            "save_results_time_seconds": _round_seconds(timing_steps["save_result_files_seconds"]),
            "autosave_json_time_seconds": _round_seconds(timing_steps["autosave_json_seconds"]),
            "merge_prep_time_seconds": _round_seconds(timing_steps["merge_prep_seconds"]),
            "merge_tiering_time_seconds": _round_seconds(timing_steps["merge_tiering_seconds"]),
            "avg_page_scrape_time_seconds": _round_seconds(avg_page_scrape_seconds),
            "avg_page_nlp_time_seconds": _round_seconds(avg_page_nlp_seconds),
            "total_time_seconds": _round_seconds(total_time),
        }

        try:
            await asyncio.to_thread(
                create_search_session,
                user_id,
                session_id,
                request.keyword,
                results,
                timing_payload,
                "completed",
            )
            log.info("[Search] Stored session in PostgreSQL for user_id=%s session_id=%s", user_id, session_id)
            if merge_output is not None:
                try:
                    await asyncio.to_thread(
                        update_search_session_merge,
                        user_id,
                        session_id,
                        all_selected_urls,
                        merge_output,
                    )
                    log.info("[Search] Stored pre-merge output for all %d URLs", len(all_selected_urls))
                except Exception:
                    log.exception("[Search] Could not store pre-merge output in PostgreSQL")
        except Exception:
            log.exception("[Search] Could not store session in PostgreSQL")

        return {
            "session_id": session_id,
            "keyword": request.keyword,
            "total_results": len(results),
            "timing": timing_payload,
            "results": results,
            "merge_output": merge_output,
            "selected_urls": all_selected_urls,
        }
    except Exception as exc:
        status = "failed"
        error_message = str(exc)
        raise
    finally:
        total_time = time.perf_counter() - start_time
        avg_page_scrape_seconds = (sum(page_scrape_durations) / len(page_scrape_durations)) if page_scrape_durations else 0.0
        avg_page_nlp_seconds = (sum(page_nlp_durations) / len(page_nlp_durations)) if page_nlp_durations else 0.0
        csv_row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": session_id,
            "keyword": request.keyword,
            "status": status,
            "error": error_message,
            "requested_k": request.k,
            "google_urls_found": len(all_urls or []),
            "urls_selected_for_scraping": len(urls or []),
            "scrape_success_count": len(scraped_pages or []),
            "scrape_failed_count": max(0, len(urls or []) - len(scraped_pages or [])),
            "total_results_returned": len(results or []),
            "use_proxy": request.use_proxy,
            "use_browser": request.use_browser,
            "headless": effective_headless,
            "device": request.device,
            "total_time_seconds": _round_seconds(total_time),
            "google_search_seconds": _round_seconds(timing_steps["google_search_seconds"]),
            "content_scraping_seconds": _round_seconds(timing_steps["content_scraping_seconds"]),
            "save_result_files_seconds": _round_seconds(timing_steps["save_result_files_seconds"]),
            "autosave_json_seconds": _round_seconds(timing_steps["autosave_json_seconds"]),
            "nlp_total_seconds": _round_seconds(timing_steps["nlp_total_seconds"]),
            "nlp_preprocess_seconds": _round_seconds(timing_steps["nlp_preprocess_seconds"]),
            "nlp_gliner_seconds": _round_seconds(timing_steps["nlp_gliner_seconds"]),
            "nlp_ranking_seconds": _round_seconds(timing_steps["nlp_ranking_seconds"]),
            "nlp_dedup_seconds": _round_seconds(timing_steps["nlp_dedup_seconds"]),
            "nlp_clustering_seconds": _round_seconds(timing_steps["nlp_clustering_seconds"]),
            "avg_page_scrape_seconds": _round_seconds(avg_page_scrape_seconds),
            "avg_page_nlp_seconds": _round_seconds(avg_page_nlp_seconds),
        }
        try:
            append_time_track_row(csv_row)
            log.info("[TimeTrack] Query timing appended to %s", TRACK_CSV_PATH)
        except Exception:
            log.exception("[TimeTrack] Failed writing CSV row")

@app.get("/keyword-nlp")
async def list_keyword_nlp(current_user: dict = Depends(get_current_user)):
    try:
        items = await asyncio.to_thread(list_keyword_nlp_outputs)
        return {"total": len(items), "items": items}
    except Exception as exc:
        log.exception("Could not list keyword NLP outputs")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/keyword-nlp/{slug}")
async def get_keyword_nlp(slug: str, current_user: dict = Depends(get_current_user)):
    try:
        payload = await asyncio.to_thread(load_keyword_nlp, slug)
        return payload
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Could not load keyword NLP output for slug=%s", slug)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/searches")
async def list_user_searches(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
):
    try:
        items = await asyncio.to_thread(
            list_search_sessions, current_user["user_id"], limit, offset
        )
        return {"total": len(items), "items": items}
    except Exception as exc:
        log.exception("Could not list search sessions")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/searches/{session_id}")
async def get_user_search_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        session = await asyncio.to_thread(
            get_search_session, current_user["user_id"], session_id
        )
        if not session:
            raise HTTPException(status_code=404, detail="Search session not found")

        merge_output = session.get("merge_output") or {}
        session_keyword = (session.get("keyword") or "").strip()
        session_dir = Path(RESULTS_DIR) / session_id
        tiering_prep = load_tiering_prep(session_dir)

        updated_merge, tiers_changed = ensure_proportional_tiers_in_merge_output(
            merge_output,
            keyword=session_keyword,
            tiering_prep=tiering_prep,
            max_numerical_per_tier=MAX_NUMERICAL_NLPS_PER_TIER,
        )
        updated_merge, buckets_changed = ensure_word_buckets_in_merge_output(
            updated_merge,
            keyword=session_keyword,
            max_numerical_per_tier=MAX_NUMERICAL_NLPS_PER_TIER,
        )
        instances_changed = False
        if not tiers_changed:
            updated_merge, instances_changed = ensure_keyword_instances_in_merge_output(
                updated_merge,
                keyword=session_keyword,
                tiering_prep=tiering_prep,
            )
        gliner_changed = False
        updated_merge, gliner_changed = ensure_gliner_labels_in_merge_output(
            updated_merge,
            keyword=session_keyword,
        )
        if tiers_changed or buckets_changed or instances_changed or gliner_changed:
            await asyncio.to_thread(
                update_search_session_merge_output,
                current_user["user_id"],
                session_id,
                updated_merge,
            )
            session["merge_output"] = updated_merge
            if tiers_changed:
                log.info("[Backfill] Refreshed proportional tiers for session %s", session_id)
            if buckets_changed:
                log.info("[Backfill] Stored word_buckets for session %s", session_id)
            if instances_changed:
                log.info("[Backfill] Stored keyword_instances for session %s", session_id)
            if gliner_changed:
                log.info("[Backfill] Stored GLiNER labels for session %s", session_id)

        return session
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not load search session")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/searches/{session_id}")
async def delete_user_search_session(
    session_id: str,
    current_user: dict = Depends(get_current_user),
):
    try:
        deleted = await asyncio.to_thread(
            delete_search_session, current_user["user_id"], session_id
        )
        if not deleted:
            raise HTTPException(status_code=404, detail="Search session not found")
        return {"status": "ok", "message": "Search session deleted"}
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Could not delete search session")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search")
async def search_and_process(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    """
    Search Google for results and process entities
    Returns list of URLs and processed data
    - Gets enough URLs to have at least 'k' non-social media URLs
    - Also processes social media URLs if found
    """
    try:
        return await _search_core(request, current_user["user_id"])
    except HTTPException:
        raise
    except Exception as e:
        log.exception("Search error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_search")
async def batch_search_and_process(
    request: BatchSearchRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Run multiple searches sequentially (one-by-one) and auto-save JSON for each keyword.
    """
    keywords = [k.strip() for k in (request.keywords or []) if (k or "").strip()]
    if not keywords:
        raise HTTPException(status_code=400, detail="No keywords provided")

    # Hard cap for safety; user can run multiple batches.
    keywords = keywords[:50]

    started = time.time()
    outputs = []
    batch_ctx_base = {
        "keywords": keywords,
        "batch_request": _batch_request_to_dict(request),
    }
    for i, kw in enumerate(keywords):
        try:
            out = await _search_core(
                SearchRequest(
                    keyword=kw,
                    k=request.k,
                    use_proxy=request.use_proxy,
                    headless=request.headless,
                    use_browser=request.use_browser,
                    device=request.device,
                ),
                current_user["user_id"],
                batch_ctx={**batch_ctx_base, "index": i},
            )
            outputs.append(
                {
                    "keyword": kw,
                    "session_id": out.get("session_id"),
                    "total_results": out.get("total_results", 0),
                    "timing": out.get("timing", {}),
                    "merge_output": out.get("merge_output"),
                    "selected_urls": out.get("selected_urls") or [],
                }
            )
        except Exception as e:
            log.exception("[BatchSearch] keyword=%r failed", kw)
            outputs.append({"keyword": kw, "error": str(e)})

    return {
        "total_keywords": len(keywords),
        "completed": len([o for o in outputs if not o.get("error")]),
        "failed": len([o for o in outputs if o.get("error")]),
        "elapsed_seconds": round(time.time() - started, 2),
        "items": outputs,
    }

def _load_merge_result_items(user_id: int, session_id: str, selected_urls: List[str]) -> List[Dict]:
    """Load per-page result dicts from disk or PostgreSQL fallback."""
    selected_set = set(selected_urls or [])
    session_dir = Path(RESULTS_DIR) / session_id
    items: List[Dict] = []

    if session_dir.exists():
        for result_file in sorted(session_dir.glob("*.json")):
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                url = data.get("url")
                if url in selected_set:
                    items.append(data)
            except Exception:
                log.exception("Error loading merge source file %s", result_file)

    if not items:
        session = get_search_session(user_id, session_id)
        if session:
            for data in session.get("results") or []:
                url = data.get("url")
                if url in selected_set:
                    items.append(data)

    return items


@app.post("/merge")
async def merge_entities(
    request: MergeRequest,
    current_user: dict = Depends(get_current_user),
):
    """
    Merge entities from selected URLs
    """
    try:
        user_id = current_user["user_id"]
        log.info("[Merge] session_id=%s | selected_urls=%d", request.session_id, len(request.selected_urls))

        session_dir = Path(RESULTS_DIR) / request.session_id
        cache_key = selection_cache_key(request.selected_urls)
        tiering_prep = load_tiering_prep(session_dir)
        cached = load_merge_cache(session_dir, cache_key)
        if cached is not None:
            keyword_for_backfill = (request.keyword or request.session_id or "").strip()
            cached, tiers_changed = ensure_proportional_tiers_in_merge_output(
                cached,
                keyword=keyword_for_backfill,
                tiering_prep=tiering_prep,
                max_numerical_per_tier=MAX_NUMERICAL_NLPS_PER_TIER,
            )
            instances_changed = False
            if not tiers_changed:
                cached, instances_changed = ensure_keyword_instances_in_merge_output(
                    cached,
                    keyword=keyword_for_backfill,
                    tiering_prep=tiering_prep,
                )
            gliner_changed = False
            cached, gliner_changed = ensure_gliner_labels_in_merge_output(
                cached,
                keyword=keyword_for_backfill,
            )
            if tiers_changed or instances_changed or gliner_changed:
                save_merge_cache(session_dir, cache_key, cached)
            log.info("[Merge] Cache hit for selection key=%s", cache_key)
            try:
                await asyncio.to_thread(
                    update_search_session_merge,
                    user_id,
                    request.session_id,
                    request.selected_urls,
                    cached,
                )
            except Exception:
                log.exception("[Merge] Could not update search session from cache")
            return cached

        merge_source_items = _load_merge_result_items(
            user_id, request.session_id, request.selected_urls
        )
        merged_entities, stats, ranking_method = aggregate_entities_from_items(
            merge_source_items,
            keep_ratio=NLP_PER_ARTICLE_KEEP_RATIO,
        )
        log.info(
            "[Merge] done — files=%d | unique_entities=%d | returned=%d | ranking_method=%s",
            stats.get("total_files", 0),
            len(merged_entities),
            len(merged_entities),
            ranking_method,
        )

        keyword_for_file = (request.keyword or request.session_id or "merge").strip()
        merge_response = build_merge_response(
            merged_entities,
            stats,
            ranking_method,
            keyword_for_file,
            tiering_prep=tiering_prep,
            max_numerical_per_tier=MAX_NUMERICAL_NLPS_PER_TIER,
            json_outputs_dir=JSON_OUTPUTS_DIR,
            persist_keyword_json=True,
            upsert_keyword_json=upsert_keyword_json_output,
            user_id=user_id,
        )

        save_merge_cache(session_dir, cache_key, merge_response)
        try:
            await asyncio.to_thread(
                update_search_session_merge,
                user_id,
                request.session_id,
                request.selected_urls,
                merge_response,
            )
            log.info("[Merge] Updated search session merge output in PostgreSQL")
        except Exception:
            log.exception("[Merge] Could not update search session in PostgreSQL")

        return merge_response

    except Exception as e:
        log.exception("Merge error")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nlp-health")
async def nlp_health():
    health = await check_nlp_service_health()
    if not health["ok"]:
        raise HTTPException(status_code=503, detail=health)
    return health


@app.get("/", include_in_schema=False)
def serve_frontend_root():
    if FRONTEND_INDEX.exists():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(status_code=404, detail="Frontend build not found")


@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend_paths(full_path: str):
    if not FRONTEND_INDEX.exists():
        raise HTTPException(status_code=404, detail="Frontend build not found")

    requested = Path(FRONTEND_DIR) / full_path
    if requested.exists() and requested.is_file():
        return FileResponse(requested)

    return FileResponse(FRONTEND_INDEX)

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT)