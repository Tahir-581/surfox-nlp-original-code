"""Shared SERP block / CAPTCHA detection for Playwright and Scrapling."""
from __future__ import annotations

import logging
from typing import Any, List

from serp_captcha_recovery import SerpBlockedError, SerpCaptchaError

log = logging.getLogger(__name__)

BLOCK_PHRASES = (
    "unusual traffic",
    "not a robot",
    "before you continue",
    "verify you are human",
    "captcha",
)


def probe_from_html(html: str, page_url: str = "") -> dict:
    """Build probe dict matching SERP_PAGE_PROBE_JS shape from HTML."""
    lower = (html or "")[:8000].lower()
    title = ""
    if "<title" in lower:
        start = lower.find("<title")
        if start >= 0:
            content_start = lower.find(">", start)
            content_end = lower.find("</title", content_start)
            if content_start >= 0 and content_end > content_start:
                title = lower[content_start + 1 : content_end]

    has_search_root = any(
        marker in lower
        for marker in ('id="search"', "id='search'", 'id="rso"', 'id="main"')
    )
    blocked_text = any(p in lower or p in title for p in BLOCK_PHRASES)
    return {
        "title": title[:200],
        "hasSearchRoot": has_search_root,
        "blockedText": blocked_text,
    }


def classify_probe(
    probe: dict,
    page_url: str,
    serp_browser: str,
    link_count: int,
) -> None:
    if "/sorry/" in (page_url or ""):
        log.warning("CAPTCHA hit. Solve it in the browser window.")
        raise SerpCaptchaError(browser=serp_browser, reason="sorry")

    if probe.get("blockedText"):
        log.warning("SERP block detected (title/body): %s", probe.get("title"))
        raise SerpCaptchaError(
            browser=serp_browser,
            message="SERP block page detected",
            reason="block_text",
        )

    if link_count == 0 and not probe.get("hasSearchRoot"):
        log.warning(
            "Soft SERP block: no organic links and no search root (title=%r)",
            probe.get("title"),
        )
        raise SerpBlockedError(
            browser=serp_browser,
            message="Empty SERP with no search results container",
        )


def classify_scrapling_response(
    response: Any,
    serp_browser: str,
    link_count: int,
) -> None:
    """Classify Scrapling Response after organic URL extraction."""
    page_url = str(getattr(response, "url", "") or "")
    html = _response_html(response)
    probe = probe_from_html(html, page_url)
    classify_probe(probe, page_url, serp_browser, link_count)


def _response_html(response: Any) -> str:
    if hasattr(response, "text"):
        return response.text or ""
    if hasattr(response, "body"):
        body = response.body
        if isinstance(body, bytes):
            return body.decode("utf-8", errors="replace")
        return str(body or "")
    return str(response or "")


def extract_organic_hrefs_from_html(html: str) -> List[str]:
    """Best-effort organic link extraction without browser JS."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    hrefs: List[str] = []
    for h3 in soup.find_all("h3"):
        parent = h3.parent
        if parent and parent.name == "a" and parent.get("href"):
            hrefs.append(parent["href"])
        for a in h3.find_all("a", href=True):
            hrefs.append(a["href"])
    if not hrefs:
        for a in soup.select("div#search a[href], div#rso a[href], div#main a[href]"):
            hrefs.append(a["href"])
    return hrefs


def extract_organic_hrefs_from_response(response: Any) -> List[str]:
    """Extract hrefs from Scrapling Response (css + HTML fallback)."""
    hrefs: List[str] = []
    try:
        if hasattr(response, "css"):
            for node in response.css("h3"):
                for a in node.xpath(".//a[@href]"):
                    href = a.attrib.get("href") if hasattr(a, "attrib") else None
                    if href:
                        hrefs.append(href)
                parent = getattr(node, "parent", None)
                if parent is not None and getattr(parent, "tag", None) == "a":
                    href = parent.attrib.get("href")
                    if href:
                        hrefs.append(href)
    except Exception:
        pass
    if not hrefs:
        hrefs = extract_organic_hrefs_from_html(_response_html(response))
    return hrefs
