#!/usr/bin/env python3
"""
PoC: Compare Playwright vs Scrapling Google SERP CAPTCHA/block rates.

Usage:
  python poc_scrapling_serp.py --backend playwright --attempts 5
  python poc_scrapling_serp.py --backend scrapling_dynamic --attempts 5
  python poc_scrapling_serp.py --backend scrapling_stealthy --keywords keywords.txt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

DEFAULT_KEYWORDS = [
    "best dog breeds 2025",
    "python web scraping tutorial",
    "seo keyword research tools",
    "cloud hosting comparison",
    "machine learning courses online",
    "organic coffee brands",
    "electric vehicle charging stations",
    "home office desk setup",
    "mediterranean diet recipes",
    "budget travel europe 2025",
]


def _load_keywords(path: str | None) -> List[str]:
    if not path:
        return list(DEFAULT_KEYWORDS)
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


async def _run_playwright(keyword: str, proxy_url: str | None) -> Dict[str, Any]:
    from google_search import _scrape_google_results_async
    from serp_captcha_recovery import SerpCaptchaError, SerpBlockedError

    t0 = time.monotonic()
    try:
        urls = await _scrape_google_results_async(
            keyword,
            k=10,
            headless=True,
            use_proxy=bool(proxy_url),
            proxy_url=proxy_url,
        )
        return {
            "keyword": keyword,
            "backend": "playwright",
            "captcha": False,
            "has_root": bool(urls),
            "url_count": len(urls),
            "latency_s": time.monotonic() - t0,
            "error": None,
            "urls_sample": urls[:3],
        }
    except (SerpCaptchaError, SerpBlockedError) as exc:
        return {
            "keyword": keyword,
            "backend": "playwright",
            "captcha": True,
            "has_root": False,
            "url_count": 0,
            "latency_s": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "urls_sample": [],
        }
    except Exception as exc:
        return {
            "keyword": keyword,
            "backend": "playwright",
            "captcha": True,
            "has_root": False,
            "url_count": 0,
            "latency_s": time.monotonic() - t0,
            "error": str(exc),
            "urls_sample": [],
        }


async def _run_scrapling(
    keyword: str,
    proxy_url: str | None,
    fetcher_type: str,
) -> Dict[str, Any]:
    import os

    os.environ["SCRAPLING_SERP_FETCHER"] = fetcher_type
    from serp_backends.scrapling_serp import fetch_serp_scrapling
    from serp_captcha_recovery import SerpCaptchaError, SerpBlockedError

    backend_label = f"scrapling_{fetcher_type}"
    t0 = time.monotonic()
    try:
        urls = await fetch_serp_scrapling(
            keyword,
            k=10,
            headless=True,
            proxy_url=proxy_url,
        )
        return {
            "keyword": keyword,
            "backend": backend_label,
            "captcha": False,
            "has_root": bool(urls),
            "url_count": len(urls),
            "latency_s": time.monotonic() - t0,
            "error": None,
            "urls_sample": urls[:3],
        }
    except (SerpCaptchaError, SerpBlockedError) as exc:
        return {
            "keyword": keyword,
            "backend": backend_label,
            "captcha": True,
            "has_root": False,
            "url_count": 0,
            "latency_s": time.monotonic() - t0,
            "error": f"{type(exc).__name__}: {exc}",
            "urls_sample": [],
        }
    except Exception as exc:
        return {
            "keyword": keyword,
            "backend": backend_label,
            "captcha": True,
            "has_root": False,
            "url_count": 0,
            "latency_s": time.monotonic() - t0,
            "error": str(exc),
            "urls_sample": [],
        }


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(results) or 1
    captcha_rate = sum(1 for r in results if r.get("captcha")) / n
    empty_rate = sum(
        1 for r in results if not r.get("captcha") and r.get("url_count", 0) == 0
    ) / n
    latencies = [r["latency_s"] for r in results if r.get("latency_s") is not None]
    return {
        "attempts": n,
        "captcha_rate": round(captcha_rate, 4),
        "empty_serp_rate": round(empty_rate, 4),
        "p50_latency_s": round(statistics.median(latencies), 3) if latencies else None,
        "p95_latency_s": round(
            sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)], 3
        )
        if latencies
        else None,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="SERP CAPTCHA PoC: Playwright vs Scrapling")
    parser.add_argument(
        "--backend",
        choices=("playwright", "scrapling_dynamic", "scrapling_stealthy"),
        default="playwright",
    )
    parser.add_argument("--keywords", default=None, help="Text file, one keyword per line")
    parser.add_argument("--attempts", type=int, default=0, help="Max runs (0 = all keywords)")
    parser.add_argument("--proxy", default=None, help="Override proxy URL for all attempts")
    parser.add_argument("--out", default=None, help="Write JSONL results to this file")
    args = parser.parse_args()

    keywords = _load_keywords(args.keywords)
    if args.attempts > 0:
        keywords = keywords[: args.attempts]

    from serp_captcha_recovery import captcha_proxy_pool

    pool = captcha_proxy_pool()
    proxy_url = args.proxy or (pool[0] if pool else None)

    results: List[Dict[str, Any]] = []
    for kw in keywords:
        if args.backend == "playwright":
            row = await _run_playwright(kw, proxy_url)
        elif args.backend == "scrapling_stealthy":
            row = await _run_scrapling(kw, proxy_url, "stealthy")
        else:
            row = await _run_scrapling(kw, proxy_url, "dynamic")
        results.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary = _summarize(results)
    print("\n--- summary ---")
    print(json.dumps(summary, indent=2))
    print(
        "\nGo criteria: adopt Scrapling for SERP if CAPTCHA rate drops >=40% vs Playwright "
        "with the same proxy pool."
    )

    if args.out:
        out_path = Path(args.out)
        with out_path.open("w", encoding="utf-8") as f:
            for row in results:
                f.write(json.dumps(row) + "\n")
            f.write(json.dumps({"summary": summary}) + "\n")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
