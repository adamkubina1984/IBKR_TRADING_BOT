"""Legacy compatibility shim for model sidecar I/O and metrics.

This module used to contain a second metrics implementation. That duplication led to
drift in label handling and trading metric semantics, so the project now keeps the
canonical implementation in ibkr_trading_bot.utils.metrics.

Only backward-compatible entry points remain here:
- calculate_metrics delegates to the canonical implementation.
- save_model_meta and load_model_meta keep the historical sidecar API.

New code should import from ibkr_trading_bot.utils.metrics or
core.services.model_service directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ibkr_trading_bot.utils.metrics import calculate_metrics as _calculate_metrics


def calculate_metrics(*args, **kwargs) -> dict[str, Any]:
    return _calculate_metrics(*args, **kwargs)


def _meta_path(model_path: str | Path) -> Path:
    p = Path(model_path)
    return p.with_suffix(p.suffix + ".meta.json")


def save_model_meta(model_path: str | Path, meta: dict[str, Any]) -> None:
    mp = _meta_path(model_path)
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_model_meta(model_path: str | Path) -> dict[str, Any]:
    mp = _meta_path(model_path)
    if mp.exists():
        return json.loads(mp.read_text(encoding="utf-8"))
    return {}


__all__ = ["calculate_metrics", "save_model_meta", "load_model_meta"]
