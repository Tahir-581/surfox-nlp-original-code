"""Scrapling DynamicFetcher / StealthyFetcher SERP backend."""
from __future__ import annotations

import asyncio
import logging
import os
import random
from typing import List, Optional
from urllib.parse import quote_plus

from google_search import CHUNK_SIZE, GOOGLE_URL, is_organic_host, normalize_url
from serp_captcha_recovery import SerpBlockedError, SerpCaptchaError, default_serp_browser
from serp_backends.block_detection import (
    classify_scrapling_response,
    extract_organic_hrefs_from_response,
    probe_from_html,
)

log = logging.getLogger(__name__)


def _scrapling_fetcher_type() -> str:
    return (os.getenv("SCRAPLING_SERP_FETCHER") or "dynamic").strip().lower()


def _build_search_url(keyword: str, start: int, device: str) -> str:
    if device == "mobile":
        return (
            f"{GOOGLE_URL}?q={quote_plus(keyword)}"
            f"&client=ms-android-google&sourceid=chrome-mobile&start={start}"
        )
    return f"{GOOGLE_URL}?q={quote_plus(keyword)}&hl=en&gl=us&start={start}"


def _proxy_kwarg(proxy_url: Optional[str]) -> dict:
    if proxy_url:
        return {"proxy": proxy_url}
    return {}


async def _fetch_page(url: str, *, headless: bool, proxy_url: Optional[str]) -> object:
    extra = _proxy_kwarg(proxy_url)
    fetcher_type = _scrapling_fetcher_type()
    try:
        if fetcher_type == "stealthy":
            from scrapling.fetchers import StealthyFetcher

            return await StealthyFetcher.async_fetch(
                url,
                headless=headless,
                **extra,
            )
        from scrapling.fetchers import DynamicFetcher

        return await DynamicFetcher.async_fetch(
            url,
            headless=headless,
            **extra,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Scrapling is not installed. Run: pip install 'scrapling[fetchers]>=0.3.2' && scrapling install"
        ) from exc


async def fetch_serp_scrapling(
    keyword: str,
    k: int = 20,
    *,
    headless: bool = False,
    use_proxy: bool = False,
    device: str = "desktop",
    proxy_url: Optional[str] = None,
) -> List[str]:
    """Fetch Google SERP organic URLs via Scrapling (no persistent Playwright profile)."""
    del use_proxy  # proxy_url is passed explicitly from recovery overrides
    serp_browser = default_serp_browser()
    urls: List[str] = []
    start = 0

    while len(urls) < k:
        search_url = _build_search_url(keyword, start, device)
        log.info("Scrapling SERP fetch: %s", search_url)
        try:
            response = await _fetch_page(search_url, headless=headless, proxy_url=proxy_url)
        except Exception as exc:
            log.error("Scrapling SERP fetch failed: %s", exc)
            raise SerpCaptchaError(
                browser=serp_browser,
                message=f"Scrapling fetch failed: {exc}",
                wedged=True,
            ) from exc

        page_url = str(getattr(response, "url", search_url) or search_url)
        if "/sorry/" in page_url:
            raise SerpCaptchaError(browser=serp_browser, reason="sorry")

        raw_hrefs = extract_organic_hrefs_from_response(response)
        classify_scrapling_response(response, serp_browser, len(raw_hrefs))

        new_found = 0
        for h in raw_hrefs:
            norm = normalize_url(h)
            if is_organic_host(norm) and norm not in urls:
                urls.append(norm)
                new_found += 1

        log.info("Scrapling found %d new results (total %d)", new_found, len(urls))

        if new_found == 0:
            from serp_backends.block_detection import _response_html

            probe = probe_from_html(_response_html(response), page_url)
            if not probe.get("hasSearchRoot"):
                raise SerpBlockedError(browser=serp_browser)

        if new_found == 0 or len(urls) >= k:
            break

        start += CHUNK_SIZE
        await asyncio.sleep(random.uniform(2, 4))

    return urls[:k]
