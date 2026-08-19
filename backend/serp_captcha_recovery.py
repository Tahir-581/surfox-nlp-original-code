"""
SERP CAPTCHA recovery: rotate proxy/browser, clear Playwright profile, optional backend restart.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
STATE_PATH = BACKEND_DIR / "serp_recovery_state.json"
HARD_RESTART_FLAG_PATH = BACKEND_DIR / "serp_recovery_hard_restart.flag"
MAIN_SCRIPT = BACKEND_DIR / "main.py"

_DEFAULT_PROXY_POOL: List[str] = [
    "http://108.59.14.208:13080",
    "http://108.59.14.203:13080",
]

SESSION_DIR_CHROMIUM = "google_serp_session_chromium"
SESSION_DIR_FIREFOX = "google_serp_session_firefox"
LEGACY_SESSION_DIR = "google_serp_session"

BrowserKind = Literal["chromium", "firefox"]


class SerpCaptchaError(Exception):
    """Raised when Google SERP returns a CAPTCHA or block page."""

    def __init__(
        self,
        browser: BrowserKind = "chromium",
        message: str = "CAPTCHA hit on SERP",
        *,
        reason: str = "sorry",
        wedged: bool = False,
    ):
        self.browser = browser if browser in ("chromium", "firefox") else "chromium"
        self.reason = reason
        self.wedged = wedged
        super().__init__(message)


class SerpBlockedError(SerpCaptchaError):
    """Soft block: empty SERP or consent wall without /sorry/ URL."""

    def __init__(self, browser: BrowserKind = "chromium", message: str = "SERP blocked or empty"):
        super().__init__(browser, message, reason="empty_serp")


def captcha_proxy_pool() -> List[str]:
    raw = (os.getenv("SERP_CAPTCHA_PROXY_POOL") or "").strip()
    if raw:
        pool = [p.strip() for p in raw.split(",") if p.strip()]
        if pool:
            return pool
    google_proxy = (os.getenv("GOOGLE_PROXY_URL") or "").strip()
    if google_proxy:
        return [google_proxy, *_DEFAULT_PROXY_POOL]
    return list(_DEFAULT_PROXY_POOL)


def default_serp_browser() -> BrowserKind:
    b = (os.getenv("SERP_BROWSER") or "chromium").strip().lower()
    return "firefox" if b == "firefox" else "chromium"


def max_captcha_attempts() -> int:
    raw = os.getenv("SERP_CAPTCHA_MAX_ATTEMPTS", "").strip()
    if not raw:
        return 5
    try:
        return int(raw)
    except ValueError:
        return 5


def in_process_attempt_limit() -> int:
    try:
        return max(0, int(os.getenv("SERP_CAPTCHA_IN_PROCESS_ATTEMPTS", "2")))
    except ValueError:
        return 2


def session_dir_for(browser: BrowserKind) -> str:
    if browser == "firefox":
        return SESSION_DIR_FIREFOX
    return SESSION_DIR_CHROMIUM


def session_path_for(browser: BrowserKind) -> Path:
    return BACKEND_DIR / session_dir_for(browser)


def clear_serp_session(browser: BrowserKind) -> None:
    """Remove Playwright persistent profile for the given browser."""
    for name in (session_dir_for(browser), LEGACY_SESSION_DIR):
        path = BACKEND_DIR / name
        if path.exists():
            try:
                shutil.rmtree(path)
                log.info("[CAPTCHA recovery] Cleared session dir: %s", path)
            except OSError as exc:
                log.warning("[CAPTCHA recovery] Failed to clear %s: %s", path, exc)


_PROXY_COOLDOWN_UNTIL: Dict[str, float] = {}


def proxy_cooldown_seconds() -> int:
    try:
        return max(0, int(os.getenv("SERP_PROXY_COOLDOWN_SEC", "0")))
    except ValueError:
        return 0


def mark_proxy_captcha_hit(proxy_url: str) -> None:
    cooldown = proxy_cooldown_seconds()
    if cooldown > 0 and proxy_url:
        _PROXY_COOLDOWN_UNTIL[proxy_url] = time.time() + cooldown


def _proxy_available(proxy_url: str) -> bool:
    until = _PROXY_COOLDOWN_UNTIL.get(proxy_url)
    return until is None or time.time() >= until


def next_proxy(current: Optional[str]) -> str:
    pool = captcha_proxy_pool()
    if not pool:
        return ""
    if len(pool) == 1:
        return pool[0]
    start_idx = 0
    if current and current in pool:
        start_idx = (pool.index(current) + 1) % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start_idx + offset) % len(pool)]
        if _proxy_available(candidate):
            return candidate
    return pool[start_idx]


def serp_backend() -> str:
    """Active SERP fetch backend: playwright | scrapling | serpapi."""
    return (os.getenv("SERP_BACKEND") or "playwright").strip().lower()


def disable_firefox_rotation() -> bool:
    """When true (default), CAPTCHA recovery stays on chromium instead of toggling to firefox."""
    raw = (os.getenv("SERP_DISABLE_FIREFOX_ROTATION") or "1").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def toggle_browser(current: BrowserKind) -> BrowserKind:
    if disable_firefox_rotation():
        return "chromium"
    return "firefox" if current == "chromium" else "chromium"


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_pending_recovery() -> Optional[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        if state.get("recovery_active"):
            return state
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("[CAPTCHA recovery] Could not read state file: %s", exc)
    return None


def clear_recovery_state() -> None:
    if STATE_PATH.exists():
        try:
            STATE_PATH.unlink()
            log.info("[CAPTCHA recovery] Cleared recovery state")
        except OSError as exc:
            log.warning("[CAPTCHA recovery] Failed to remove state file: %s", exc)
    clear_hard_restart_flag()


def clear_hard_restart_flag() -> None:
    if HARD_RESTART_FLAG_PATH.exists():
        try:
            HARD_RESTART_FLAG_PATH.unlink()
        except OSError as exc:
            log.warning("[CAPTCHA recovery] Failed to remove hard-restart flag: %s", exc)


def mark_hard_restart_pending() -> None:
    try:
        HARD_RESTART_FLAG_PATH.write_text("1", encoding="utf-8")
    except OSError as exc:
        log.warning("[CAPTCHA recovery] Failed to write hard-restart flag: %s", exc)


def is_hard_restart_pending() -> bool:
    return HARD_RESTART_FLAG_PATH.exists()


def get_recovery_overrides() -> Tuple[Optional[str], BrowserKind]:
    """Return (force_proxy_url, serp_browser) when recovery is active."""
    state = load_pending_recovery()
    if not state:
        return None, default_serp_browser()
    proxy = state.get("serp_proxy_url")
    browser = state.get("serp_browser", default_serp_browser())
    if browser not in ("chromium", "firefox"):
        browser = default_serp_browser()
    return proxy, browser


def check_max_attempts_exceeded(captcha_hits: int) -> bool:
    limit = max_captcha_attempts()
    return limit > 0 and captcha_hits > limit


def can_recover_in_process(state: Optional[Dict[str, Any]] = None) -> bool:
    limit = in_process_attempt_limit()
    if limit <= 0:
        return False
    state = state or load_pending_recovery() or {}
    attempts = int(state.get("in_process_attempts", 0))
    return attempts < limit


def should_hard_restart(state: Optional[Dict[str, Any]] = None, *, wedged: bool = False) -> bool:
    if wedged:
        return True
    state = state or load_pending_recovery() or {}
    return not can_recover_in_process(state)


def update_recovery_state(patch: Dict[str, Any]) -> Dict[str, Any]:
    state = load_pending_recovery() or {}
    state.update(patch)
    state["recovery_active"] = True
    _atomic_write_json(STATE_PATH, state)
    return state


def apply_recovery_rotation(
    request_dict: Dict[str, Any],
    failed_browser: BrowserKind,
    *,
    batch_mode: bool = False,
    batch_keywords: Optional[List[str]] = None,
    batch_index: int = 0,
    batch_request: Optional[Dict[str, Any]] = None,
    current_proxy: Optional[str] = None,
    last_error: str = "sorry",
) -> Tuple[str, BrowserKind]:
    """
    In-process rotation: increment counters, rotate proxy/browser, clear failed profile.
    Returns (next_proxy_url, next_browser).
    """
    existing = load_pending_recovery() or {}
    captcha_hits = int(existing.get("captcha_hits", 0)) + 1
    in_process_attempts = int(existing.get("in_process_attempts", 0)) + 1

    if check_max_attempts_exceeded(captcha_hits):
        clear_recovery_state()
        raise RuntimeError(
            f"SERP CAPTCHA recovery exceeded max attempts ({max_captcha_attempts()})"
        )

    pool = captcha_proxy_pool()
    prev_proxy = current_proxy or existing.get("serp_proxy_url") or (pool[0] if pool else "")
    next_proxy_url = next_proxy(prev_proxy)
    backend = serp_backend()
    if backend == "scrapling":
        next_browser = failed_browser if failed_browser in ("chromium", "firefox") else "chromium"
    else:
        next_browser = toggle_browser(failed_browser)
        clear_serp_session(failed_browser)

    state: Dict[str, Any] = {
        "recovery_active": True,
        "captcha_hits": captcha_hits,
        "in_process_attempts": in_process_attempts,
        "serp_proxy_url": next_proxy_url,
        "serp_browser": next_browser,
        "serp_backend": backend,
        "failed_browser": failed_browser,
        "request": request_dict,
        "batch_mode": batch_mode,
        "batch_keywords": batch_keywords or [],
        "batch_index": batch_index,
        "batch_request": batch_request or {},
        "last_error": last_error,
        "hard_restart_pending": False,
    }
    mark_proxy_captcha_hit(prev_proxy)
    _atomic_write_json(STATE_PATH, state)
    log.warning(
        "[CAPTCHA recovery] In-process retry #%d (hit %d) | proxy=%s | browser=%s | backend=%s | reason=%s",
        in_process_attempts,
        captcha_hits,
        next_proxy_url,
        next_browser,
        backend,
        last_error,
    )
    return next_proxy_url, next_browser


def prepare_recovery(
    request_dict: Dict[str, Any],
    failed_browser: BrowserKind,
    *,
    batch_mode: bool = False,
    batch_keywords: Optional[List[str]] = None,
    batch_index: int = 0,
    batch_request: Optional[Dict[str, Any]] = None,
    current_proxy: Optional[str] = None,
    last_error: str = "sorry",
    for_hard_restart: bool = False,
) -> Dict[str, Any]:
    """
    Persist state for hard-restart resume (in-process counter reset).
    """
    existing = load_pending_recovery() or {}
    captcha_hits = int(existing.get("captcha_hits", 0)) + 1

    if check_max_attempts_exceeded(captcha_hits):
        clear_recovery_state()
        raise RuntimeError(
            f"SERP CAPTCHA recovery exceeded max attempts ({max_captcha_attempts()})"
        )

    pool = captcha_proxy_pool()
    prev_proxy = current_proxy or existing.get("serp_proxy_url") or (pool[0] if pool else "")
    next_proxy_url = next_proxy(prev_proxy)
    backend = serp_backend()
    if backend == "scrapling":
        next_browser = failed_browser if failed_browser in ("chromium", "firefox") else "chromium"
    else:
        next_browser = toggle_browser(failed_browser)
        clear_serp_session(failed_browser)

    state: Dict[str, Any] = {
        "recovery_active": True,
        "captcha_hits": captcha_hits,
        "in_process_attempts": 0,
        "serp_proxy_url": next_proxy_url,
        "serp_browser": next_browser,
        "serp_backend": backend,
        "failed_browser": failed_browser,
        "request": request_dict,
        "batch_mode": batch_mode,
        "batch_keywords": batch_keywords or [],
        "batch_index": batch_index,
        "batch_request": batch_request or {},
        "last_error": last_error,
        "hard_restart_pending": for_hard_restart,
    }
    mark_proxy_captcha_hit(prev_proxy)
    _atomic_write_json(STATE_PATH, state)
    log.warning(
        "[CAPTCHA recovery] Prepared hard-restart retry #%d | proxy=%s | browser=%s | backend=%s",
        captcha_hits,
        next_proxy_url,
        next_browser,
        backend,
    )
    return state


def restart_backend() -> None:
    """Terminate backend; supervisor or a new process continues the search."""
    mark_hard_restart_pending()
    managed = os.getenv("SURF_MANAGED_BY_RUN_SERVICES", "").strip() in {
        "1",
        "true",
        "yes",
    }
    if managed:
        log.warning(
            "[CAPTCHA recovery] Exiting for run_services to respawn backend (managed mode)."
        )
        os._exit(0)

    cmd = [sys.executable, str(MAIN_SCRIPT)]
    log.warning("[CAPTCHA recovery] Restarting backend: %s", " ".join(cmd))
    kwargs: Dict[str, Any] = {
        "cwd": str(BACKEND_DIR),
        "env": os.environ.copy(),
    }
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    subprocess.Popen(cmd, **kwargs)
    log.warning("[CAPTCHA recovery] New backend process started; exiting current process.")
    os._exit(0)
