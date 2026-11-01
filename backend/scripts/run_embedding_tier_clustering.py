#!/usr/bin/env python3
"""
Run embedding consensus tier clustering on session session JSON or merge entities.

Usage:
    python scripts/run_embedding_tier_clustering.py \\
        --session-file ../session/06_20260610_095911_best_big_dog_breeds.json

    python scripts/run_embedding_tier_clustering.py \\
        --session-dir ../session

    python scripts/run_embedding_tier_clustering.py \\
        --session-file ../session/06_....json \\
        --output-dir ../clustering_runs/run_2026_06_11_001
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BACKEND_DIR.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(BACKEND_DIR / ".env")

from nlp_embedding_tiers.artifacts import new_run_id  # noqa: E402
from nlp_embedding_tiers.config import ClusterConfig  # noqa: E402
from nlp_embedding_tiers.pipeline import load_session_input, run_pipeline  # noqa: E402

log = logging.getLogger(__name__)

SESSION_SKIP = {"_manifest.json", "_user.json"}


def _default_output_dir(keyword: str) -> Path:
    slug = keyword.strip().replace(" ", "_")[:60] or "unknown"
    return REPO_ROOT / "clustering_runs" / new_run_id() / slug


def run_single(session_path: Path, output_dir: Path | None, config: ClusterConfig) -> Path:
    inp = load_session_input(session_path)
    keyword = inp["keyword"]
    out = output_dir or _default_output_dir(keyword)
    out.mkdir(parents=True, exist_ok=True)
    log.info("Running pipeline for keyword=%r → %s", keyword, out)
    run_pipeline(
        keyword=keyword,
        entities=inp["entities"],
        output_dir=out,
        config=config,
        merge_output=inp.get("merge_output"),
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Embedding consensus tier clustering")
    parser.add_argument("--session-file", type=Path, help="Single session session JSON file")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Batch run all session/*.json files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (single-file mode only)",
    )
    parser.add_argument(
        "--clustering-runs-root",
        type=Path,
        default=REPO_ROOT / "clustering_runs",
        help="Root folder for batch outputs",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    config = ClusterConfig.from_env()

    if not args.session_file and not args.session_dir:
        parser.error("Provide --session-file or --session-dir")

    if args.session_file:
        path = args.session_file
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            log.error("File not found: %s", path)
            return 1
        out = args.output_dir
        if out and not out.is_absolute():
            out = (Path.cwd() / out).resolve()
        run_single(path, out, config)
        return 0

    session_dir = args.session_dir
    if not session_dir.is_absolute():
        session_dir = (Path.cwd() / session_dir).resolve()
    if not session_dir.is_dir():
        log.error("Directory not found: %s", session_dir)
        return 1

    files = sorted(
        p for p in session_dir.glob("*.json") if p.name not in SESSION_SKIP
    )
    if not files:
        log.error("No session JSON files in %s", session_dir)
        return 1

    batch_root = args.clustering_runs_root
    if not batch_root.is_absolute():
        batch_root = (Path.cwd() / batch_root).resolve()
    batch_id = new_run_id()
    batch_dir = batch_root / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    errors: list[dict] = []
    for path in files:
        slug = path.stem
        out = batch_dir / slug
        try:
            run_single(path, out, config)
            results.append({"file": str(path), "output_dir": str(out)})
        except Exception as exc:
            log.exception("Failed on %s", path)
            errors.append({"file": str(path), "error": str(exc)})

    summary_path = batch_dir / "batch_summary.json"
    import json

    summary_path.write_text(
        json.dumps(
            {"batch_id": batch_id, "succeeded": results, "failed": errors},
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "Batch done — %d succeeded, %d failed → %s",
        len(results),
        len(errors),
        batch_dir,
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
