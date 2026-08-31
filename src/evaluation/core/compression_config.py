"""Small helpers for compression configs shared by eval launchers and workers."""

from __future__ import annotations

import re
from typing import Any

DEFAULT_COMPRESSION_ALGORITHM = "compactor"


def normalize_compression_config(comp_cfg: Any) -> dict[str, Any]:
    """Return a plain dict for one compression config entry."""

    if comp_cfg is None or comp_cfg == "none":
        return {}
    if isinstance(comp_cfg, dict):
        return dict(comp_cfg)
    if hasattr(comp_cfg, "items"):
        return dict(comp_cfg.items())
    return {}


def compression_ratio_from_config(comp_cfg: Any) -> float:
    cfg = normalize_compression_config(comp_cfg)
    cr = cfg.get("compression_ratio")
    return float(cr) if cr is not None else 0.0


def compression_algorithm_from_config(comp_cfg: Any) -> str:
    cfg = normalize_compression_config(comp_cfg)
    raw = cfg.get("compression_algorithm", cfg.get("algorithm", DEFAULT_COMPRESSION_ALGORITHM))
    if raw is None or str(raw).strip() == "":
        return DEFAULT_COMPRESSION_ALGORITHM
    return str(raw).strip().lower().replace("-", "_")


def compression_algorithm_label(comp_cfg: Any) -> str:
    algorithm = compression_algorithm_from_config(comp_cfg)
    return re.sub(r"[^a-z0-9_]+", "_", algorithm).strip("_") or DEFAULT_COMPRESSION_ALGORITHM


def compression_suffix_from_config(
    comp_cfg: Any = None,
    *,
    compression_ratio: float | None = None,
) -> str:
    """Return a compact filename suffix for a compression config."""

    cr = compression_ratio_from_config(comp_cfg)
    if compression_ratio is not None and cr <= 0:
        cr = float(compression_ratio)
    if cr <= 0:
        return ""

    algorithm = compression_algorithm_label(comp_cfg)
    if algorithm == DEFAULT_COMPRESSION_ALGORITHM:
        return f"_cr{cr}"
    return f"_{algorithm}_cr{cr}"


def compression_label_from_config(comp_cfg: Any) -> str:
    suffix = compression_suffix_from_config(comp_cfg)
    return suffix[1:] if suffix else "no_compression"
