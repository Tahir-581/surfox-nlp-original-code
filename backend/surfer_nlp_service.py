"""Load and transform Surfer keyword NLP exports from keyword-nlp-output/."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

_DEFAULT_DIR = Path(__file__).resolve().parent.parent / "keyword-nlp-output"


def keyword_nlp_output_dir() -> Path:
    configured = os.getenv("KEYWORD_NLP_OUTPUT_DIR", "").strip()
    if configured:
        return Path(configured).resolve()
    return _DEFAULT_DIR.resolve()


def _slug_from_filename(filename: str) -> str:
    name = (filename or "").strip()
    if name.lower().endswith(".json"):
        return name[:-5]
    return name


def _resolve_json_path(slug: str) -> Path:
    base = keyword_nlp_output_dir()
    if not base.is_dir():
        raise FileNotFoundError(f"Keyword NLP output directory not found: {base}")

    clean_slug = (slug or "").strip()
    if not clean_slug or "/" in clean_slug or "\\" in clean_slug or ".." in clean_slug:
        raise ValueError("Invalid slug")

    candidate = (base / f"{clean_slug}.json").resolve()
    if not str(candidate).startswith(str(base)):
        raise ValueError("Invalid slug path")
    if not candidate.is_file():
        raise FileNotFoundError(f"Keyword NLP file not found: {clean_slug}")
    return candidate


def _target_midpoint(target_range: Optional[Dict[str, Any]]) -> int:
    if not isinstance(target_range, dict):
        return 0
    lo = int(target_range.get("min") or 0)
    hi = int(target_range.get("max") or 0)
    if lo <= 0 and hi <= 0:
        return 0
    return int(round((lo + hi) / 2))


def term_to_entity(term: Dict[str, Any]) -> Dict[str, Any]:
    target_range = term.get("target_range") if isinstance(term.get("target_range"), dict) else {}
    return {
        "text": (term.get("term") or "").strip(),
        "combined_count": _target_midpoint(target_range),
        "average_weightage": None,
        "sources": ["surfer"],
        "included": bool(term.get("included")),
        "ignored": bool(term.get("ignored")),
        "is_nlp": bool(term.get("is_nlp")),
        "target_range": {
            "min": int(target_range.get("min") or 0),
            "max": int(target_range.get("max") or 0),
        },
        "use_in_heading": bool(term.get("use_in_heading")),
    }


def build_merge_view(raw: Dict[str, Any]) -> Dict[str, Any]:
    nlp_terms = raw.get("nlp_terms") or {}
    terms = nlp_terms.get("terms") if isinstance(nlp_terms, dict) else []
    if not isinstance(terms, list):
        terms = []

    entities = [term_to_entity(t) for t in terms if isinstance(t, dict) and term_to_entity(t)["text"]]
    green_nlps = [e for e in entities if e.get("included")]
    white_nlps = [e for e in entities if not e.get("included")]

    return {
        "entities": entities,
        "green_nlps": green_nlps,
        "orange_nlps": [],
        "white_nlps": white_nlps,
        "total_unique_entities": len(entities),
        "total_entity_occurrences": sum(e.get("combined_count", 0) for e in entities),
        "total_files_processed": 0,
        "ranking_method": "surfer",
        "source": "keyword-nlp-output",
    }


def build_viewer_payload(raw: Dict[str, Any], slug: str) -> Dict[str, Any]:
    merge_view = build_merge_view(raw)
    return {
        "slug": slug,
        "keyword": raw.get("keyword") or slug.replace("_", " "),
        "surfer_id": raw.get("surfer_id"),
        "surfer_link": raw.get("surfer_link"),
        "permalink_hash": raw.get("permalink_hash"),
        "location": raw.get("location"),
        "timestamp": raw.get("timestamp"),
        "status": raw.get("status") or "success",
        "merge_view": merge_view,
    }


def list_keyword_nlp_outputs() -> List[Dict[str, Any]]:
    base = keyword_nlp_output_dir()
    if not base.is_dir():
        return []

    summary_path = base / "_batch_summary.csv"
    if summary_path.is_file():
        items: List[Dict[str, Any]] = []
        with open(summary_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                json_file = (row.get("json_file") or "").strip()
                if not json_file:
                    continue
                slug = _slug_from_filename(json_file)
                json_path = base / json_file
                if not json_path.is_file():
                    continue
                items.append(
                    {
                        "slug": slug,
                        "keyword": (row.get("keyword") or slug.replace("_", " ")).strip(),
                        "status": (row.get("status") or "unknown").strip(),
                        "surfer_link": (row.get("surfer_link") or "").strip() or None,
                        "json_file": json_file,
                    }
                )
        return items

    items = []
    for path in sorted(base.glob("*.json")):
        if path.name.startswith("_"):
            continue
        slug = _slug_from_filename(path.name)
        keyword = slug.replace("_", " ")
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            keyword = raw.get("keyword") or keyword
        except (OSError, json.JSONDecodeError):
            pass
        items.append(
            {
                "slug": slug,
                "keyword": keyword,
                "status": "unknown",
                "surfer_link": None,
                "json_file": path.name,
            }
        )
    return items


def load_keyword_nlp(slug: str) -> Dict[str, Any]:
    path = _resolve_json_path(slug)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Invalid keyword NLP JSON payload")
    return build_viewer_payload(raw, _slug_from_filename(path.name))
