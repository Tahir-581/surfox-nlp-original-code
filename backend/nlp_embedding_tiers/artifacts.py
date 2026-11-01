"""JSON artifact writers for clustering evaluation runs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunArtifacts:
    """Manages the clustering_runs/<run_id>/ folder layout."""

    def __init__(self, output_dir: Path) -> None:
        self.root = Path(output_dir)
        self.per_query_dir = self.root / "06_per_query_clustering"
        self.root.mkdir(parents=True, exist_ok=True)
        self.per_query_dir.mkdir(parents=True, exist_ok=True)

    def write(self, filename: str, data: Any) -> Path:
        path = self.root / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path

    def write_per_query(self, filename: str, data: Any) -> Path:
        path = self.per_query_dir / filename
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        return path


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y_%m_%d_%H%M%S")
    return f"run_{ts}"
