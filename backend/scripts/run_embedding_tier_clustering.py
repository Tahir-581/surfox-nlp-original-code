#!/usr/bin/env python3
"""
Run embedding consensus tier clustering on nimra session JSON or merge entities.

Usage:
    python scripts/run_embedding_tier_clustering.py \\
        --nimra-file ../nimra/06_20260610_095911_best_big_dog_breeds.json

    python scripts/run_embedding_tier_clustering.py \\
        --nimra-dir ../nimra

    python scripts/run_embedding_tier_clustering.py \\
        --nimra-file ../nimra/06_....json \\
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
from nlp_embedding_tiers.pipeline import load_nimra_input, run_pipeline  # noqa: E402

log = logging.getLogger(__name__)

NIMRA_SKIP = {"_manifest.json", "_user.json"}


def _default_output_dir(keyword: str) -> Path:
    slug = keyword.strip().replace(" ", "_")[:60] or "unknown"
    return REPO_ROOT / "clustering_runs" / new_run_id() / slug


def run_single(nimra_path: Path, output_dir: Path | None, config: ClusterConfig) -> Path:
    inp = load_nimra_input(nimra_path)
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
    parser.add_argument("--nimra-file", type=Path, help="Single nimra session JSON file")
    parser.add_argument(
        "--nimra-dir",
        type=Path,
        default=None,
        help="Batch run all nimra/*.json files",
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

    if not args.nimra_file and not args.nimra_dir:
        parser.error("Provide --nimra-file or --nimra-dir")

    if args.nimra_file:
        path = args.nimra_file
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

    nimra_dir = args.nimra_dir
    if not nimra_dir.is_absolute():
        nimra_dir = (Path.cwd() / nimra_dir).resolve()
    if not nimra_dir.is_dir():
        log.error("Directory not found: %s", nimra_dir)
        return 1

    files = sorted(
        p for p in nimra_dir.glob("*.json") if p.name not in NIMRA_SKIP
    )
    if not files:
        log.error("No nimra JSON files in %s", nimra_dir)
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
