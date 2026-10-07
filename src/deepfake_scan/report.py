"""JSON report construction and serialization."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from . import __version__


def build_report(
    results,
    *,
    input_dir: Path,
    threshold: float,
    frames_per_video: int,
    model_path: Path | None,
) -> dict:
    """Build the report dictionary for a completed scan."""
    deepfakes = [r for r in results if r.verdict == "deepfake"]
    errors = [r for r in results if r.verdict == "error"]
    return {
        "schema_version": "1.0",
        "tool": {"name": "deepfake-scan", "version": __version__},
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "parameters": {
            "threshold": threshold,
            "frames_per_video": frames_per_video,
            "model": str(model_path) if model_path else None,
        },
        "summary": {
            "total": len(results),
            "deepfakes": len(deepfakes),
            "real": len(results) - len(deepfakes) - len(errors),
            "errors": len(errors),
            "flagged": [r.path for r in deepfakes],
        },
        "results": [asdict(r) for r in results],
    }


def write_report(report: dict, path: Path) -> None:
    """Write the report as JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
