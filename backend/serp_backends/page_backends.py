"""Page content scraping: Playwright or Scrapling Fetcher."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from google_search import get_authority

log = logging.getLogger(__name__)

AnchorTriple = Tuple[Optional[str], str, Optional[str]]


def page_backend() -> str:
    return (os.getenv("PAGE_BACKEND") or "playwright").strip().lower()


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _last_word(text: str) -> Optional[str]:
    words = text.split()
    return words[-1] if words else None


def _first_word(text: str) -> Optional[str]:
    words = text.split()
    return words[0] if words else None


def _normalize_neighbor_word(word: Optional[str]) -> Optional[str]:
    if not word:
        return None
    cleaned = re.sub(r"^[^\w]+|[^\w]+$", "", word, flags=re.UNICODE)
    return cleaned or None


def _text_from_sibling(sibling: Any) -> str:
    if isinstance(sibling, NavigableString):
        return str(sibling)
    if isinstance(sibling, Tag):
        return sibling.get_text()
    return ""


def _immediate_prev_word(tag: Tag) -> Optional[str]:
    sibling = tag.previous_sibling
    while sibling is not None:
        text = _text_from_sibling(sibling)
        stripped = text.strip()
        if stripped:
            word = _last_word(stripped)
            if word:
                return word
        sibling = sibling.previous_sibling
    return None


def _immediate_next_word(tag: Tag) -> Optional[str]:
    sibling = tag.next_sibling
    while sibling is not None:
        text = _text_from_sibling(sibling)
        stripped = text.strip()
        if stripped:
            word = _first_word(stripped)
            if word:
                return word
        sibling = sibling.next_sibling
    return None


def _collect_anchor_triples(container: Tag) -> List[AnchorTriple]:
    triples: List[AnchorTriple] = []
    for anchor in container.find_all("a"):
        anchor_text = anchor.get_text(strip=True)
        if not anchor_text:
            continue
        triples.append(
            (
                _normalize_neighbor_word(_immediate_prev_word(anchor)),
                anchor_text,
                _normalize_neighbor_word(_immediate_next_word(anchor)),
            )
        )
    return triples


def _phrase_pattern(*parts: str) -> str:
    segments: List[str] = []
    for part in parts:
        if not part:
            continue
        words = part.split()
        if not words:
            continue
        segments.append(r"\s+".join(re.escape(word) for word in words))
    if not segments:
        return ""
    return r"\b" + r"\s+".join(segments) + r"\b"


def _remove_anchor_triples(text: str, triples: List[AnchorTriple]) -> str:
    result = text or ""
    for prev_word, anchor_text, next_word in triples:
        if not anchor_text:
            continue
        patterns: List[str] = []
        if prev_word and next_word:
            patterns.append(_phrase_pattern(prev_word, anchor_text, next_word))
        if prev_word:
            patterns.append(_phrase_pattern(prev_word, anchor_text))
        if next_word:
            patterns.append(_phrase_pattern(anchor_text, next_word))
        patterns.append(_phrase_pattern(anchor_text))
        patterns.sort(key=len, reverse=True)
        for pattern in patterns:
            if pattern:
                result = re.sub(pattern, " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\s+", " ", result).strip()
    if result and not re.search(r"\w", result, flags=re.UNICODE):
        return ""
    return result


def _decompose_anchors(soup: BeautifulSoup) -> None:
    try:
        for anchor in soup.find_all("a"):
            anchor.decompose()
    except Exception:
        pass


def parse_page_html(html: str, url: str) -> Dict[str, Any]:
    """Parse page HTML into the same dict shape as main.scrape_page_content."""
    soup = BeautifulSoup(html or "", "html.parser")

    title = ""
    title_tag = soup.find("title")
    if title_tag:
        title = title_tag.get_text(strip=True)

    description = ""
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        description = meta_desc.get("content").strip()

    headings: List[str] = []
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        heading_text = heading.get_text(strip=True)
        if heading_text:
            headings.append(heading_text)

    paragraphs: List[str] = []
    for para in soup.find_all("p"):
        para_triples = _collect_anchor_triples(para)
        para_text = re.sub(r"\s+", " ", para.get_text(separator=" ", strip=True))
        if para_triples:
            para_text = _remove_anchor_triples(para_text, para_triples)
        if para_text:
            paragraphs.append(para_text)

    _decompose_anchors(soup)

    # NLP input: title + description + headings + paragraphs.
    content_parts = [part for part in [title, description] + headings + paragraphs if part]
    full_content = " ".join(content_parts)
    domain = urlparse(url).netloc
    authority = get_authority(domain)

    return {
        "url": url,
        "domain": domain,
        "title": title,
        "description": description,
        "headings": headings,
        "paragraphs": paragraphs,
        "word_count": len(full_content.split()),
        "heading_count": len(headings),
        "para_count": len(paragraphs),
        "images_count": 0,
        "authority": authority,
        "content": full_content,
    }


async def scrape_url_scrapling(url: str) -> Optional[Dict[str, Any]]:
    impersonate = os.getenv("SCRAPLING_IMPERSONATE", "chrome124")
    timeout = 30
    try:
        from scrapling.fetchers import AsyncFetcher

        response = await AsyncFetcher.get(
            url,
            impersonate=impersonate,
            timeout=timeout,
        )
        html = getattr(response, "text", None) or ""
        if not html.strip() and _env_truthy("PAGE_SCRAPLING_JS_FALLBACK"):
            from scrapling.fetchers import DynamicFetcher

            response = await DynamicFetcher.async_fetch(url, headless=True, timeout=60)
            html = getattr(response, "text", None) or ""
        if not html.strip():
            return None
        return parse_page_html(html, url)
    except ImportError:
        log.error("Scrapling not installed; cannot use PAGE_BACKEND=scrapling")
        return None
    except Exception:
        log.exception("Scrapling page fetch failed for %s", url)
        return None


async def scrape_url_playwright(page: Any, url: str) -> Optional[Dict[str, Any]]:
    """Use an existing Playwright page (caller manages browser lifecycle)."""
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        html = await page.content()
        return parse_page_html(html, url)
    except Exception:
        log.exception("Failed to scrape %s", url)
        return None


async def scrape_url(
    url: str,
    *,
    page: Any = None,
) -> Optional[Dict[str, Any]]:
    if page_backend() == "scrapling":
        return await scrape_url_scrapling(url)
    if page is None:
        log.error("Playwright page required when PAGE_BACKEND=playwright")
        return None
    return await scrape_url_playwright(page, url)


def scrape_url_sync(page: Any, url: str) -> Optional[Dict[str, Any]]:
    """Sync wrapper for Playwright page scraping."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        html = page.content()
        return parse_page_html(html, url)
    except Exception:
        log.exception("Failed to scrape %s (sync)", url)
        return None
