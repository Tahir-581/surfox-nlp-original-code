"""SERP and page scraping backends (Playwright default, Scrapling optional)."""
from __future__ import annotations

import os
from typing import List, Optional

from serp_captcha_recovery import BrowserKind, serp_backend

from serp_backends.playwright_serp import fetch_serp_playwright


def page_backend() -> str:
    return (os.getenv("PAGE_BACKEND") or "playwright").strip().lower()


async def fetch_serp(
    keyword: str,
    k: int = 20,
    *,
    headless: bool = False,
    use_proxy: bool = False,
    device: str = "desktop",
    serp_browser: Optional[BrowserKind] = None,
    proxy_url: Optional[str] = None,
) -> List[str]:
    backend = serp_backend()
    if backend == "scrapling":
        from serp_backends.scrapling_serp import fetch_serp_scrapling

        return await fetch_serp_scrapling(
            keyword,
            k,
            headless=headless,
            use_proxy=use_proxy,
            device=device,
            proxy_url=proxy_url,
        )
    if backend == "serpapi":
        raise NotImplementedError(
            "SERP_BACKEND=serpapi is not configured. Set SERPAPI_KEY and implement serpapi adapter."
        )
    return await fetch_serp_playwright(
        keyword,
        k,
        headless=headless,
        use_proxy=use_proxy,
        device=device,
        serp_browser=serp_browser,
        proxy_url=proxy_url,
    )
