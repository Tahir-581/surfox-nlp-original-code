"""Playwright SERP backend (existing google_search implementation)."""
from __future__ import annotations

from typing import List, Optional

from serp_captcha_recovery import BrowserKind

from google_search import _scrape_google_results_async


async def fetch_serp_playwright(
    keyword: str,
    k: int = 20,
    *,
    headless: bool = False,
    use_proxy: bool = False,
    device: str = "desktop",
    serp_browser: Optional[BrowserKind] = None,
    proxy_url: Optional[str] = None,
) -> List[str]:
    return await _scrape_google_results_async(
        keyword,
        k=k,
        headless=headless,
        use_proxy=use_proxy,
        device=device,
        serp_browser=serp_browser,
        proxy_url=proxy_url,
    )
