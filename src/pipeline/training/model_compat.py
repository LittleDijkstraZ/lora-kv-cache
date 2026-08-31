"""Model loading and precision helpers shared by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def get_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except (ImportError, OSError) as exc:
        print(f"[model_compat] flash_attn unavailable; falling back to sdpa: {exc}", flush=True)
        return "sdpa"


@dataclass(frozen=True)
class TrainingPrecision:
    """Resolved model dtype plus matching Trainer precision switches."""

    torch_dtype: torch.dtype
    fp16: bool
    bf16: bool
    source: str


_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "torch.float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "full": torch.float32,
    "torch.float32": torch.float32,
}

def _coerce_torch_dtype(value: Any) -> torch.dtype | None:
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return value
    text = str(value).strip().lower()
    if text in {"", "auto", "none", "null"}:
        return None
    if text not in _DTYPE_ALIASES:
        valid = ", ".join(sorted(k for k in _DTYPE_ALIASES if not k.startswith("torch.")))
        raise ValueError(f"Unsupported training torch dtype {value!r}. Expected auto or one of: {valid}")
    return _DTYPE_ALIASES[text]


def _config_dtype(config: Any) -> torch.dtype | None:
    for attr in ("dtype", "torch_dtype"):
        try:
            dtype = getattr(config, attr, None)
        except AttributeError:
            dtype = None
        resolved = _coerce_torch_dtype(dtype)
        if resolved is not None:
            return resolved
    return None


def _cuda_bf16_supported() -> bool:
    checker = getattr(torch.cuda, "is_bf16_supported", None)
    return bool(torch.cuda.is_available() and checker is not None and checker())


def _finalize_precision(dtype: torch.dtype, *, source: str) -> TrainingPrecision:
    if dtype == torch.bfloat16:
        if _cuda_bf16_supported():
            return TrainingPrecision(torch_dtype=torch.bfloat16, fp16=False, bf16=True, source=source)
        if torch.cuda.is_available():
            print(
                "[model_compat] Requested bf16 training but this CUDA device does not "
                "report bf16 support; falling back to fp32.",
                flush=True,
            )
        return TrainingPrecision(torch_dtype=torch.float32, fp16=False, bf16=False, source=f"{source}->fp32")

    if dtype == torch.float16:
        if torch.cuda.is_available():
            return TrainingPrecision(torch_dtype=torch.float16, fp16=True, bf16=False, source=source)
        return TrainingPrecision(torch_dtype=torch.float32, fp16=False, bf16=False, source=f"{source}->fp32")

    return TrainingPrecision(torch_dtype=torch.float32, fp16=False, bf16=False, source=source)


def resolve_training_precision(
    model_name: str,
    *,
    requested: Any = "auto",
    token: str | None = None,
) -> TrainingPrecision:
    """Resolve training dtype from an explicit request or the model config.

    ``requested='auto'`` follows the checkpoint's native dtype when available.
    That keeps bf16-native models off fp16 Trainer/GradScaler paths.
    """

    explicit = _coerce_torch_dtype(requested)
    if explicit is not None:
        return _finalize_precision(explicit, source=f"requested:{explicit}")

    config = None
    try:
        config = AutoConfig.from_pretrained(model_name, token=token)
    except Exception as exc:  # noqa: BLE001 - model loading will surface hard failures later.
        print(f"[model_compat] Could not inspect config dtype for {model_name}: {exc}", flush=True)

    if config is not None:
        dtype = _config_dtype(config)
        if dtype is not None:
            return _finalize_precision(dtype, source=f"config:{type(config).__name__}")

    fallback = torch.float16 if torch.cuda.is_available() else torch.float32
    return _finalize_precision(fallback, source="auto_fallback")


def load_training_model(model_name: str, *, token: str | None = None, torch_dtype: Any = "auto"):
    """Load a trainable causal language model."""
    precision = resolve_training_precision(model_name, requested=torch_dtype, token=token)
    kwargs: dict[str, Any] = {
        "torch_dtype": precision.torch_dtype,
        "attn_implementation": get_attn_implementation(),
        "token": token,
    }
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
