"""Compatibility helpers for training config constructors."""

from __future__ import annotations

import inspect
from typing import Any, TypeVar

T = TypeVar("T")


def make_training_config(config_cls: type[T], **kwargs: Any) -> T:
    """Instantiate a TrainingArguments-like class with supported kwargs only.

    Transformers/TRL 5.x removed some older knobs such as ``group_by_length``,
    while older TRL releases used ``max_length`` where newer ones expose
    ``max_seq_length``.  Keep the call sites declarative and normalize here.
    """

    params = inspect.signature(config_cls.__init__).parameters
    normalized = dict(kwargs)

    if "max_length" in normalized and "max_length" not in params and "max_seq_length" in params:
        normalized["max_seq_length"] = normalized.pop("max_length")
    elif "max_seq_length" in normalized and "max_seq_length" not in params and "max_length" in params:
        normalized["max_length"] = normalized.pop("max_seq_length")

    supported = {key: value for key, value in normalized.items() if key in params}
    return config_cls(**supported)
