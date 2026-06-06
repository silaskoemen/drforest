"""JSON result persistence helpers for benchmark studies and diagnostics."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np


def default_results_dir(study_name: str) -> Path:
    """Default output directory for a study or diagnostic name."""
    return Path(__file__).resolve().parent / "results" / study_name


def write_json_result(study_name: str, payload: dict[str, Any], results_dir: Path | None = None) -> Path:
    """Write one benchmark payload under ``benchmarks/results/{study_name}``.

    If ``payload["params"]["seed"]`` is present it is included in the filename;
    otherwise the filename remains unique via timestamp and UUID suffix.
    """
    out_dir = default_results_dir(study_name) if results_dir is None else Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    params = payload.get("params")
    seed = params.get("seed", "unknown") if isinstance(params, dict) else "unknown"
    path = out_dir / f"{timestamp}_seed{seed}_{uuid4().hex[:8]}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def _json_default(value: object) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return cast(Any, value).tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"object of type {type(value).__name__} is not JSON serializable")
