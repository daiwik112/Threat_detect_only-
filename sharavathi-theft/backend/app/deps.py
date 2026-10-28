"""
backend/app/deps.py — Settings, project paths, shared app state, error helper.

The backend imports the project packages (grid, simdata, ml, future) from the
sharavathi-theft/ root, so it is runnable as `uvicorn app.main:app` from backend/.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]          # sharavathi-theft/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DISCLAIMER = "Simulated results; suspected anomalies require field verification."


@dataclass
class Settings:
    model_dir: Path = Path(os.getenv("STD_MODEL_DIR", ROOT / "data" / "models"))
    baseline_path: Path = ROOT / "data" / "raw" / "baseline_states.parquet"
    dataset_path: Path = ROOT / "data" / "processed" / "dataset_major_only.parquet"
    splits_path: Path = ROOT / "data" / "processed" / "splits.json"
    dataset_card: Path = ROOT / "data" / "processed" / "dataset_card.md"
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
        ).split(",") if o.strip())


settings = Settings()


@dataclass
class AppState:
    """Loaded once in the FastAPI lifespan."""
    bundle: Any = None                 # ml.infer.ModelBundle
    network: dict | None = None        # cached NetworkResponse
    dataset: Any = None                # lazily loaded DataFrame
    extra: dict = field(default_factory=dict)


state = AppState()


class APIError(Exception):
    """Raised by services; rendered as {error: {code, message}} with the given status."""

    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(message)


def load_flags() -> dict:
    from grid.config import load_feature_flags
    f = load_feature_flags()
    return {"mitigation_enabled": bool(f.mitigation_enabled), "gnn_enabled": bool(f.gnn_enabled),
            "rag_enabled": bool(f.rag_enabled)}


def require_models(*names: str):
    b = state.bundle
    missing = [n for n in names if b is None or getattr(b, n, None) is None]
    if missing:
        raise APIError(503, "models_missing",
                       f"Model artifacts not loaded ({', '.join(missing)}). "
                       "Run `make train` (writes data/models/) and restart the API.")
    return b
