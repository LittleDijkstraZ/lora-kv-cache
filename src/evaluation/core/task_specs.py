"""Contracts shared by the generative-QA evaluators."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True, kw_only=True)
class GenerativeTaskSpec:
    """Dataset-specific preparation and scoring hooks."""

    benchmark_name: str
    default_eval_filename: str
    prediction_basename: str
    prepare_eval_df: Callable[..., pd.DataFrame]
    answer_scorer_fn: Callable[[str, Any], float]
    metric_sum_fns: dict[str, Callable[[list[dict]], float]] = field(default_factory=dict)
    annotate_predictions_fn: Callable[[list[dict]], None] | None = None
    primary_metric_name: str = "rouge_l"


def resolve_eval_file(
    data_dir: str | Path,
    *,
    default_eval_filename: str,
    eval_filename: str | None = None,
) -> Path:
    """Resolve an evaluation file from a prepared dataset directory."""

    resolved = Path(data_dir) / (eval_filename or default_eval_filename)
    if not resolved.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {resolved}")
    return resolved


__all__ = ["GenerativeTaskSpec", "resolve_eval_file"]
