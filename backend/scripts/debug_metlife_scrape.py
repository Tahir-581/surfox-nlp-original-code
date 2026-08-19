"""One-off debug: show scraped content vs GLiNER input for a URL."""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from serp_backends.page_backends import parse_page_html  # noqa: E402

URL = (
    "https://www.metlifepetinsurance.com/blog/breed-spotlights/"
    "the-best-dog-breeds-for-families-with-kids/"
)


def chunk_starts(text: str, size: int = 600, step: int = 400):
    n = len(text)
    start = 0
    while start < n:
        end = min(start + size, n)
        if end < n and end > start:
            last_space = text.rfind(" ", start, end + 1)
            if last_space > start:
                end = last_space + 1
        yield start, end
        start += step
        if start < n and text[start] not in " \n\t":
            nxt = text.find(" ", start, min(start + 80, n))
            if nxt != -1:
                start = nxt + 1
        if start >= n:
            break


def main() -> None:
    response = httpx.get(
        URL,
        follow_redirects=True,
        timeout=45,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
            )
        },
    )
    response.raise_for_status()
    page = parse_page_html(response.text, URL)
    content = page.get("content") or ""
    chunks = [content[s:e] for s, e in chunk_starts(content, 600, 400)]

    lines: list[str] = []
    lines.append(f"URL: {URL}")
    lines.append("Query context: best dog breeds for families")
    lines.append(f"HTTP status: {response.status_code}")
    lines.append(f"TITLE: {page.get('title') or ''}")
    lines.append(f"DESCRIPTION: {page.get('description') or ''}")
    lines.append(f"WORD COUNT (scraped content for NLP): {page.get('word_count')}")
    lines.append(f"HEADINGS ({page.get('heading_count')}, included in scraped NLP content):")
    for i, heading in enumerate(page.get("headings") or [], 1):
        lines.append(f"  {i}. {heading}")
    lines.append(f"PARAGRAPHS ({page.get('para_count')}):")
    for i, para in enumerate(page.get("paragraphs") or [], 1):
        lines.append(f"--- para {i} ({len(para.split())} words) ---")
        lines.append(para)
    lines.append("=== SCRAPED CONTENT (run_pipeline input) ===")
    lines.append(content)
    lines.append(f"=== GLiNER CHUNKS ({len(chunks)} total) ===")
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"--- chunk {i} ({len(chunk)} chars) ---")
        lines.append(chunk)

    out_path = BACKEND / "metlife_scrape_debug.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Scraped content: {len(content)} chars, {page.get('word_count')} words")
    print(f"GLiNER chunks: {len(chunks)} chunks, {len(content.split())} words")
    print("--- First 2000 chars of scraped content ---")
    print(content[:2000])


if __name__ == "__main__":
    main()
