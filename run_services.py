#!/usr/bin/env python3
"""
Run Surfox backend and frontend together.

Usage:
    python run_services.py
    python run_services.py --headful
    python run_services.py --lan --headful
    python run_services.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse, urlunparse

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DEFAULT_FRONTEND_PORT = 3010
DEFAULT_GLINER_URL = "http://localhost:8081"
GLINER_HEALTH_TIMEOUT_SEC = 60
GLINER_HEALTH_POLL_SEC = 2
BACKEND_ENV_FILE = BACKEND_DIR / ".env"
GLINER_SERVER_SCRIPT = BACKEND_DIR / "NLP_Extraction_and_Ranking" / "gliner_server.py"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Surfox backend and frontend together.",
    )
    parser.add_argument(
        "--headful",
        action="store_true",
        help="Run Playwright in headful mode (visible browser; use to solve Google CAPTCHA).",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="LAN mode: build frontend and serve everything on the backend port (best for network access).",
    )
    return parser.parse_args(argv)


def _get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def _is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def _find_available_port(start_port: int) -> int:
    port = start_port
    while not _is_port_available(port):
        port += 1
    return port


HARD_RESTART_FLAG = BACKEND_DIR / "serp_recovery_hard_restart.flag"
RECOVERY_STATE_PATH = BACKEND_DIR / "serp_recovery_state.json"


def _is_captcha_recovery_restart(exit_code: int | None) -> bool:
    if exit_code != 0:
        return False
    if HARD_RESTART_FLAG.exists():
        return True
    if RECOVERY_STATE_PATH.exists():
        try:
            import json

            state = json.loads(RECOVERY_STATE_PATH.read_text(encoding="utf-8"))
            return bool(state.get("recovery_active"))
        except (OSError, json.JSONDecodeError):
            return True
    return False


def _build_frontend(runner: str, runner_cmd: list[str]) -> int:
    print("Building frontend for LAN mode...")
    result = subprocess.run(runner_cmd, cwd=FRONTEND_DIR)
    if result.returncode != 0:
        print("Frontend build failed.")
        return result.returncode
    build_index = FRONTEND_DIR / "build" / "index.html"
    if not build_index.exists():
        print(f"Frontend build output not found: {build_index}")
        return 1
    print("Frontend build complete.")
    return 0


def _terminate_process(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return

    print(f"Stopping {name} (PID: {proc.pid})...")
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        print(f"{name} did not stop in time. Killing...")
        proc.kill()
        proc.wait(timeout=3)


def _normalize_client_url(url: str) -> str:
    """0.0.0.0 is valid for bind addresses, not outbound HTTP (especially on Windows)."""
    parsed = urlparse(url)
    if parsed.hostname in {"0.0.0.0", "::"}:
        port = parsed.port
        netloc = f"localhost:{port}" if port else "localhost"
        return urlunparse(parsed._replace(netloc=netloc)).rstrip("/")
    return url.rstrip("/")


def _gliner_api_url(env: dict[str, str]) -> str:
    raw = env.get("SURF_GLINER_API_URL", DEFAULT_GLINER_URL).strip().rstrip("/") or DEFAULT_GLINER_URL
    return _normalize_client_url(raw)


def _is_local_gliner_url(url: str) -> bool:
    try:
        host = (urlparse(url).hostname or "localhost").lower()
        return host in {"localhost", "127.0.0.1", "::1"}
    except Exception:
        return False


def _fetch_gliner_health(health_url: str) -> dict | None:
    try:
        with urllib.request.urlopen(health_url, timeout=3) as resp:
            if resp.status != 200:
                return None
            body = json.loads(resp.read().decode())
            return body if isinstance(body, dict) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def _wait_for_gliner_health(health_url: str, timeout_sec: float = GLINER_HEALTH_TIMEOUT_SEC) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        body = _fetch_gliner_health(health_url)
        if body and body.get("ready"):
            return True
        time.sleep(GLINER_HEALTH_POLL_SEC)
    return False


def _start_gliner_if_local(env: dict[str, str]) -> subprocess.Popen | None:
    gliner_url = _gliner_api_url(env)
    if not _is_local_gliner_url(gliner_url):
        print(f"GLiNER URL is remote ({gliner_url}); skipping local auto-start.")
        return None

    health_url = f"{gliner_url}/health"
    existing = _fetch_gliner_health(health_url)
    if existing and existing.get("ready"):
        print(f"GLiNER already running at {gliner_url}")
        return None

    print(f"Starting local GLiNER server at {gliner_url}...")
    proc = subprocess.Popen(
        [sys.executable, str(GLINER_SERVER_SCRIPT)],
        cwd=BACKEND_DIR,
        env=env,
    )
    if not _wait_for_gliner_health(health_url):
        _terminate_process(proc, "gliner")
        print("GLiNER server failed to become ready in time.")
        return None

    print(f"GLiNER ready at {gliner_url}")
    return proc


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not BACKEND_DIR.exists():
        print(f"Backend directory not found: {BACKEND_DIR}")
        return 1
    if not FRONTEND_DIR.exists():
        print(f"Frontend directory not found: {FRONTEND_DIR}")
        return 1

    backend_cmd = [sys.executable, "main.py"]
    frontend_runner = None
    frontend_cmd = []
    lan_mode = args.lan

    pnpm_exe = shutil.which("pnpm")
    npm_exe = shutil.which("npm")
    use_pnpm = (FRONTEND_DIR / "pnpm-lock.yaml").exists() and pnpm_exe is not None

    if use_pnpm:
        frontend_runner = pnpm_exe
        frontend_cmd = [frontend_runner, "run", "host"]
        frontend_build_cmd = [frontend_runner, "run", "build"]
    elif npm_exe is not None:
        frontend_runner = npm_exe
        frontend_cmd = [frontend_runner, "run", "host"]
        frontend_build_cmd = [frontend_runner, "run", "build"]
    else:
        print("Neither `pnpm` nor `npm` is available in PATH. Please install Node.js first.")
        return 1

    if lan_mode:
        build_code = _build_frontend(frontend_runner, frontend_build_cmd)
        if build_code != 0:
            return build_code

    if load_dotenv is not None and BACKEND_ENV_FILE.exists():
        load_dotenv(BACKEND_ENV_FILE, override=False)
    backend_env = os.environ.copy()
    backend_env["SURF_MANAGED_BY_RUN_SERVICES"] = "1"
    backend_env["SURFOX_HEADFUL"] = "1" if args.headful else "0"
    if lan_mode:
        backend_env["FRONTEND_DIR"] = str(FRONTEND_DIR / "build")

    pg_host = backend_env.get("POSTGRES_HOST", "localhost")
    pg_port = backend_env.get("POSTGRES_PORT", "5432")
    pg_db = backend_env.get("POSTGRES_DB", "serfox_db")
    pg_user = backend_env.get("POSTGRES_USER", "postgres")

    gliner_proc = _start_gliner_if_local(backend_env)
    if _is_local_gliner_url(_gliner_api_url(backend_env)) and gliner_proc is None:
        existing = _fetch_gliner_health(f"{_gliner_api_url(backend_env)}/health")
        if not (existing and existing.get("ready")):
            print("Local GLiNER is required but not available. Exiting.")
            return 1

    print("Starting backend...")
    if args.headful:
        print("  Browser mode: headful (visible window for CAPTCHA)")
    else:
        print("  Browser mode: headless (pass --headful for visible browser)")
    print(f"  PostgreSQL: {pg_user}@{pg_host}:{pg_port}/{pg_db} (pgAdmin: register this server)")

    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=BACKEND_DIR,
        env=backend_env,
    )

    time.sleep(1)

    backend_port = int(backend_env.get("BACKEND_PORT", backend_env.get("PORT", "8010")))
    lan_ip = _get_lan_ip()
    frontend_proc = None
    frontend_port = None

    if lan_mode:
        print("LAN mode: frontend served by backend (single port).")
    else:
        frontend_port = _find_available_port(DEFAULT_FRONTEND_PORT)
        frontend_env = os.environ.copy()
        frontend_env["PORT"] = str(frontend_port)
        frontend_env["HOST"] = "0.0.0.0"
        frontend_env["DANGEROUSLY_DISABLE_HOST_CHECK"] = "true"
        frontend_env["WDS_SOCKET_HOST"] = lan_ip if lan_ip != "127.0.0.1" else "0.0.0.0"
        if frontend_port != DEFAULT_FRONTEND_PORT:
            print(
                f"Port {DEFAULT_FRONTEND_PORT} is in use. "
                f"Starting frontend on port {frontend_port}."
            )

        print(f"Starting frontend via {Path(frontend_runner).name} run host...")
        frontend_proc = subprocess.Popen(
            frontend_cmd,
            cwd=FRONTEND_DIR,
            env=frontend_env,
        )

    print("\nSurfox services are running:")
    print(f" - GLiNER:   {_gliner_api_url(backend_env)}")
    if lan_mode:
        print(f" - App: http://localhost:{backend_port}")
        if lan_ip != "127.0.0.1":
            print(f"\nLAN access (same network): http://{lan_ip}:{backend_port}")
    else:
        print(f" - Backend:  http://localhost:{backend_port}")
        print(f" - Frontend: http://localhost:{frontend_port}")
        if lan_ip != "127.0.0.1":
            print(f"\nLAN access (same network):")
            print(f" - Frontend: http://{lan_ip}:{frontend_port}  (API proxied through frontend)")
            print(f" - Backend:  http://{lan_ip}:{backend_port}  (direct API/docs)")
    if lan_ip != "127.0.0.1":
        print(
            "\nIf other devices still cannot connect, run scripts/open_lan_access.ps1 "
            "as Administrator (sets Wi-Fi to Private + opens firewall)."
        )
    print("\nPress Ctrl+C to stop.\n")

    def handle_signal(signum, _frame):
        signal_name = signal.Signals(signum).name
        print(f"\nReceived {signal_name}. Shutting down services...")
        if frontend_proc is not None:
            _terminate_process(frontend_proc, "frontend")
        _terminate_process(backend_proc, "backend")
        if gliner_proc is not None:
            _terminate_process(gliner_proc, "gliner")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        while True:
            backend_code = backend_proc.poll()

            if backend_code is not None:
                if _is_captcha_recovery_restart(backend_code):
                    print(
                        "\nCAPTCHA recovery restart detected — respawning backend, keeping frontend..."
                    )
                    if HARD_RESTART_FLAG.exists():
                        try:
                            HARD_RESTART_FLAG.unlink()
                        except OSError:
                            pass
                    time.sleep(1)
                    backend_proc = subprocess.Popen(
                        backend_cmd,
                        cwd=BACKEND_DIR,
                        env=backend_env,
                    )
                    continue
                print(f"\nBackend exited with code {backend_code}. Stopping frontend...")
                if frontend_proc is not None:
                    _terminate_process(frontend_proc, "frontend")
                if gliner_proc is not None:
                    _terminate_process(gliner_proc, "gliner")
                return backend_code

            if frontend_proc is not None:
                frontend_code = frontend_proc.poll()
                if frontend_code is not None:
                    print(f"\nFrontend exited with code {frontend_code}. Stopping backend...")
                    _terminate_process(backend_proc, "backend")
                    if gliner_proc is not None:
                        _terminate_process(gliner_proc, "gliner")
                    return frontend_code

            if gliner_proc is not None:
                gliner_code = gliner_proc.poll()
                if gliner_code is not None:
                    print(f"\nGLiNER exited with code {gliner_code}. Stopping services...")
                    if frontend_proc is not None:
                        _terminate_process(frontend_proc, "frontend")
                    _terminate_process(backend_proc, "backend")
                    return gliner_code

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted. Shutting down services...")
        if frontend_proc is not None:
            _terminate_process(frontend_proc, "frontend")
        _terminate_process(backend_proc, "backend")
        if gliner_proc is not None:
            _terminate_process(gliner_proc, "gliner")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
