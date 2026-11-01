#!/usr/bin/env python3
"""
Backfill keyword_instances into merge_output for existing search sessions.

Usage:
    python scripts/backfill_keyword_instances.py
    python scripts/backfill_keyword_instances.py --user-id 27
    python scripts/backfill_keyword_instances.py --dry-run
    python scripts/backfill_keyword_instances.py --force
"""
from __future__ import annotations

import argparse
import logging
import os
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
from merge_service import (  # noqa: E402
    ensure_keyword_instances_in_merge_output,
    load_tiering_prep,
)

log = logging.getLogger(__name__)

_results = os.getenv("RESULTS_DIR", "results")
RESULTS_DIR = Path(_results)
if not RESULTS_DIR.is_absolute():
    RESULTS_DIR = BACKEND_DIR / RESULTS_DIR


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
        session_dir = RESULTS_DIR / str(session_id)
        tiering_prep = load_tiering_prep(session_dir) if session_dir.exists() else None

        patched, changed = ensure_keyword_instances_in_merge_output(
            merge_output,
            keyword=keyword,
            tiering_prep=tiering_prep,
            force=force,
        )
        if not changed:
            skipped += 1
            continue

        updated += 1
        log.info(
            "Backfill session %s (user_id=%s, keyword=%r, instances=%d)",
            session_id,
            session.get("user_id"),
            keyword,
            len(patched.get("keyword_instances") or []),
        )
        if dry_run:
            continue

        update_search_session_merge_output(
            int(session["user_id"]),
            str(session_id),
            patched,
        )

    return updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill keyword_instances into merge_output"
    )
    parser.add_argument("--user-id", type=int, default=None, help="Limit DB backfill to one user")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute keyword_instances even when already present",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    updated, skipped = backfill_database_sessions(
        user_id=args.user_id,
        dry_run=args.dry_run,
        force=args.force,
    )
    log.info("Done. updated=%d skipped=%d dry_run=%s", updated, skipped, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
