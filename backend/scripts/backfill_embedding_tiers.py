#!/usr/bin/env python3
"""
Backfill embedding-consensus tiers into merge_output for existing sessions.

Usage:
    python scripts/backfill_embedding_tiers.py
    python scripts/backfill_embedding_tiers.py --user-id 27
    python scripts/backfill_embedding_tiers.py --dry-run
    python scripts/backfill_embedding_tiers.py --force
    python scripts/backfill_embedding_tiers.py --nimra-dir ../nimra
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from database import (  # noqa: E402
    init_db,
    list_search_sessions_with_merge,
    update_search_session_merge_output,
    upsert_keyword_json_output,
)
from nlp_embedding_tiers.merge_tiering import (  # noqa: E402
    apply_merge_tiering,
)
from merge_service import merge_output_needs_proportional_tier_refresh  # noqa: E402
from nlp_word_buckets import build_tier_word_buckets  # noqa: E402

log = logging.getLogger(__name__)

NIMRA_SKIP_FILES = {"_manifest.json", "_user.json"}


def _texts(items: list) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        txt = (item.get("text") if isinstance(item, dict) else str(item) or "").strip()
        key = txt.casefold()
        if not txt or key in seen:
            continue
        seen.add(key)
        out.append(txt)
    return out


def _patch_merge_output(merge_output: dict, keyword: str) -> dict:
    entities = merge_output.get("entities") or []
    stats = merge_output.get("average_statistics") or {}
    avg_word_count = float(stats.get("avg_word_count") or 0.0)

    tiering = apply_merge_tiering(
        entities,
        keyword,
        avg_word_count=avg_word_count,
    )
    green = tiering["green_nlps"]
    orange = tiering["orange_nlps"]
    white = tiering["white_nlps"]

    updated = dict(merge_output)
    updated["green_nlps"] = green
    updated["orange_nlps"] = orange
    updated["white_nlps"] = white
    updated["tiering_method"] = tiering["tiering_method"]
    updated["tiering_metadata"] = tiering["tiering_metadata"]
    updated["consensus_metadata"] = tiering["tiering_metadata"]
    updated["word_buckets"] = build_tier_word_buckets(green, white, orange)
    updated["keyword_buckets"] = {
        "Green": _texts(green),
        "Orange": _texts(orange),
        "White": _texts(white),
    }
    return updated


def _safe_keyword_filename(keyword: str) -> str:
    safe_name = re.sub(r"[^\w\s-]", "", keyword or "")
    safe_name = re.sub(r"[-\s]+", "_", safe_name).strip() or "merge"
    return f"{safe_name}.json"


def backfill_database_sessions(
    *,
    user_id: int | None,
    dry_run: bool,
    force: bool,
) -> tuple[int, int]:
    init_db()
    sessions = list_search_sessions_with_merge(user_id=user_id)
    updated = 0
    skipped = 0

    for session in sessions:
        merge_output = session.get("merge_output") or {}
        if not merge_output.get("entities"):
            skipped += 1
            continue
        if not force and not merge_output_needs_proportional_tier_refresh(merge_output):
            skipped += 1
            continue

        keyword = (session.get("keyword") or "").strip()
        session_id = session.get("session_id")
        uid = int(session["user_id"])

        log.info(
            "Backfill embedding tiers session %s (user_id=%s, keyword=%r)",
            session_id,
            uid,
            keyword,
        )
        patched = _patch_merge_output(merge_output, keyword)
        updated += 1

        if dry_run:
            continue

        update_search_session_merge_output(uid, str(session_id), patched)
        payload = patched.get("keyword_buckets") or {}
        if keyword and payload:
            try:
                upsert_keyword_json_output(
                    keyword,
                    _safe_keyword_filename(keyword),
                    payload,
                    uid,
                )
            except Exception:
                log.exception("Could not upsert keyword JSON for session %s", session_id)

    return updated, skipped


def backfill_nimra_files(nimra_dir: Path, *, dry_run: bool, force: bool) -> tuple[int, int]:
    updated = 0
    skipped = 0

    for path in sorted(nimra_dir.glob("*.json")):
        if path.name in NIMRA_SKIP_FILES:
            continue

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Could not read %s", path)
            continue

        merge_output = data.get("merge_output") or {}
        if not merge_output.get("entities"):
            skipped += 1
            continue
        if not force and not merge_output_needs_proportional_tier_refresh(merge_output):
            skipped += 1
            continue

        keyword = (data.get("keyword") or "").strip()
        log.info("Backfill nimra file %s (keyword=%r)", path.name, keyword)
        data["merge_output"] = _patch_merge_output(merge_output, keyword)
        updated += 1

        if dry_run:
            continue

        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill embedding-consensus tiers into merge_output",
    )
    parser.add_argument("--user-id", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--nimra-dir", type=Path, default=None)
    parser.add_argument("--skip-db", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    total_updated = 0
    total_skipped = 0

    if not args.skip_db:
        db_updated, db_skipped = backfill_database_sessions(
            user_id=args.user_id,
            dry_run=args.dry_run,
            force=args.force,
        )
        total_updated += db_updated
        total_skipped += db_skipped
        log.info("Database: updated=%d skipped=%d", db_updated, db_skipped)

    if args.nimra_dir:
        nimra_dir = args.nimra_dir
        if not nimra_dir.is_absolute():
            nimra_dir = (Path.cwd() / nimra_dir).resolve()
        if not nimra_dir.is_dir():
            log.error("Nimra directory not found: %s", nimra_dir)
            return 1
        nimra_updated, nimra_skipped = backfill_nimra_files(
            nimra_dir,
            dry_run=args.dry_run,
            force=args.force,
        )
        total_updated += nimra_updated
        total_skipped += nimra_skipped
        log.info("Nimra files: updated=%d skipped=%d", nimra_updated, nimra_skipped)

    log.info("Done. updated=%d skipped=%d dry_run=%s", total_updated, total_skipped, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
