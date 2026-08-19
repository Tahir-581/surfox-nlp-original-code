#!/usr/bin/env python3
"""
Backfill word_buckets into merge_output for existing search sessions.

Usage:
    python scripts/backfill_word_buckets.py
    python scripts/backfill_word_buckets.py --user-id 27
    python scripts/backfill_word_buckets.py --dry-run
    python scripts/backfill_word_buckets.py --force
    python scripts/backfill_word_buckets.py --nimra-dir ../nimra
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from database import (  # noqa: E402
    init_db,
    list_search_sessions_with_merge,
    update_search_session_merge_output,
)
from nlp_word_buckets import ensure_word_buckets_in_merge_output  # noqa: E402

log = logging.getLogger(__name__)

NIMRA_SKIP_FILES = {"_manifest.json", "_user.json"}


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
        session_id = session.get("session_id")
        merge_output = session.get("merge_output") or {}
        keyword = (session.get("keyword") or "").strip()
        patched, changed = ensure_word_buckets_in_merge_output(
            merge_output,
            keyword=keyword,
            force=force,
        )
        if not changed:
            skipped += 1
            continue

        updated += 1
        log.info(
            "Backfill session %s (user_id=%s, keyword=%r)",
            session_id,
            session.get("user_id"),
            keyword,
        )
        if dry_run:
            continue

        update_search_session_merge_output(
            int(session["user_id"]),
            str(session_id),
            patched,
        )

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
        keyword = (data.get("keyword") or "").strip()
        patched, changed = ensure_word_buckets_in_merge_output(
            merge_output,
            keyword=keyword,
            force=force,
        )
        if not changed:
            skipped += 1
            continue

        updated += 1
        log.info("Backfill nimra file %s (keyword=%r)", path.name, keyword)
        if dry_run:
            continue

        data["merge_output"] = patched
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill word_buckets into merge_output")
    parser.add_argument("--user-id", type=int, default=None, help="Limit DB backfill to one user")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute word_buckets even when already present",
    )
    parser.add_argument(
        "--nimra-dir",
        type=Path,
        default=None,
        help="Also backfill exported nimra session JSON files",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip PostgreSQL sessions (nimra-only mode)",
    )
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
        nimra_dir = args.nimra_dir.resolve()
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
