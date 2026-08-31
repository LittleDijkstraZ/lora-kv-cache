"""Helpers for LoRA target-module shorthands."""

from __future__ import annotations

from collections.abc import Iterable


ALL_LINEAR_SHORTHAND = "all-linear"


def normalize_target_modules(value):
    """Return PEFT-compatible target modules, preserving ``all-linear``.

    Hydra and argparse usually hand us a list.  PEFT's all-linear shorthand only
    activates when ``target_modules`` is the string ``"all-linear"``, so a
    single-item list must be collapsed before constructing ``LoraConfig``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if not isinstance(value, Iterable):
        return value
    modules = [str(item) for item in value if str(item)]
    if len(modules) == 1 and modules[0].lower() == ALL_LINEAR_SHORTHAND:
        return ALL_LINEAR_SHORTHAND
    return modules or None


def is_all_linear_target(value) -> bool:
    return normalize_target_modules(value) == ALL_LINEAR_SHORTHAND
