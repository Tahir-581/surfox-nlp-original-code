"""Tests for page HTML parsing and NLP input filtering."""

from serp_backends.page_backends import (
    _remove_anchor_triples,
    parse_page_html,
)


def test_headings_included_in_content_and_metadata():
    html = """
    <html>
      <head>
        <title>Page Title</title>
        <meta name="description" content="Meta description here.">
      </head>
      <body>
        <h2>Section Title</h2>
        <p>Body paragraph with useful article text.</p>
      </body>
    </html>
    """
    result = parse_page_html(html, "https://example.com/article")

    assert result["headings"] == ["Section Title"]
    assert "Section Title" in result["content"]
    assert "Body paragraph with useful article text." in result["content"]
    assert "Page Title" in result["content"]
    assert "Meta description here." in result["content"]


def test_all_heading_levels_included_in_content():
    html = """
    <html><head><title>T</title></head><body>
    <h1>H1 text</h1><h2>H2 text</h2><h3>H3 text</h3>
    <h4>H4 text</h4><h5>H5 text</h5><h6>H6 text</h6>
    <p>Paragraph body text here.</p>
    </body></html>
    """
    result = parse_page_html(html, "https://example.com/levels")

    for heading in result["headings"]:
        assert heading in result["content"]
    assert "Paragraph body text here." in result["content"]


def test_anchor_triple_removal_from_paragraph():
    html = "<p>See our <a>pricing plans</a> today.</p>"
    result = parse_page_html(html, "https://example.com/pricing")

    assert result["paragraphs"] == ["See ."]
    assert "pricing plans" not in result["content"]
    assert "our" not in result["content"]
    assert "today" not in result["content"]


def test_anchor_link_strips_anchor_phrase_and_neighbors():
    html = "<p>Visit <a>home</a> now.</p>"
    result = parse_page_html(html, "https://example.com/")

    assert result["paragraphs"] == []
    assert "home" not in result["content"]


def test_remove_anchor_triples_phrase_patterns():
    text = "See our pricing plans today."
    scrubbed = _remove_anchor_triples(
        text,
        [("our", "pricing plans", "today")],
    )
    assert scrubbed == "See ."


def test_word_count_includes_headings_in_content():
    html = """
    <html>
      <head><title>T</title></head>
      <body>
        <h1>Heading one two three four five</h1>
        <p>Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.</p>
      </body>
    </html>
    """
    result = parse_page_html(html, "https://example.com/count")

    assert result["heading_count"] == 1
    assert "Heading one" in result["content"]
    assert result["word_count"] == len(result["content"].split())
