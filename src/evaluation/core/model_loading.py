"""Causal-language-model loading helpers for evaluation."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from src.pipeline.training.model_compat import TrainingPrecision, resolve_training_precision
except ModuleNotFoundError:  # pragma: no cover - script entrypoint uses src/ on PYTHONPATH.
    from pipeline.training.model_compat import TrainingPrecision, resolve_training_precision


def _get_attn_implementation():
    """Return the best available attention implementation."""
    try:
        import flash_attn  # noqa: F401

        return "flash_attention_2"
    except (ImportError, OSError) as e:
        print(f"[model_loading] flash_attn unavailable; falling back to sdpa: {e}")
        return "sdpa"


def resolve_eval_precision(base_model_name, *, requested="auto") -> TrainingPrecision:
    """Resolve the evaluation dtype.

    Evaluation should follow the same native-dtype policy as training. In
    particular, bf16-native checkpoints such as Llama-3.1 and Qwen3 should not
    be silently loaded as fp16.
    """

    return resolve_training_precision(base_model_name, requested=requested)


def get_model_dtype(model, fallback: torch.dtype | None = None) -> torch.dtype | None:
    """Return the dtype of the first model parameter when available."""

    try:
        parameter = next(model.parameters())
    except (AttributeError, StopIteration, TypeError):
        return fallback
    return getattr(parameter, "dtype", fallback)


def move_model_to_eval_device(model, device: str, *, dtype: torch.dtype | None = None):
    """Move an eval model without forcing fp16.

    Real ``torch.nn.Module.to`` supports ``device=`` and ``dtype=`` keywords;
    the fallback keeps small test doubles working without adding a fake dtype
    implementation to every test model.
    """

    if dtype is not None:
        try:
            return model.to(device=device, dtype=dtype)
        except TypeError:
            pass
    return model.to(device)


def apply_model_max_length(model, tokenizer, model_max_length: int | None):
    """Override model/tokenizer context metadata when a run explicitly asks for it."""

    if model_max_length is None:
        return
    try:
        model_max_length = int(model_max_length)
    except (TypeError, ValueError):
        raise ValueError(f"model_max_length must be an integer, got {model_max_length!r}")
    if model_max_length <= 0:
        raise ValueError(f"model_max_length must be positive, got {model_max_length}")

    config = getattr(model, "config", None)
    if config is not None:
        setattr(config, "max_position_embeddings", model_max_length)
    if tokenizer is not None:
        setattr(tokenizer, "model_max_length", model_max_length)
    print(f"[model_loading] model/tokenizer max length override={model_max_length}", flush=True)


def load_model(
    base_model_name,
    device="cuda",
    multi_gpu=False,
    move_to_device=False,
    torch_dtype="auto",
    model_max_length: int | None = None,
):
    """Load base model without adapters (adapters applied later).

    Args:
        base_model_name: HuggingFace model name or path
        device: Device to load on (ignored if multi_gpu=True)
        multi_gpu: If True, use device_map="auto" to distribute across all available GPUs
        move_to_device: If True and not multi_gpu, move model to device using the
                        resolved eval dtype.
        torch_dtype: auto | bfloat16/bf16 | float16/fp16 | float32/fp32.
                     auto follows the checkpoint config when possible.
                        Set to False if you need to apply adapters before moving to device.
        model_max_length: Optional explicit context limit for evaluation metadata.
                          This updates model.config.max_position_embeddings and
                          tokenizer.model_max_length after loading.

    Returns:
        Tuple of (model, tokenizer). Model is in eval mode.
    """
    attn_impl = _get_attn_implementation()
    precision = resolve_eval_precision(base_model_name, requested=torch_dtype)
    print(
        "[model_loading] eval torch_dtype="
        f"{precision.torch_dtype} (source={precision.source})",
        flush=True,
    )
    if multi_gpu:
        # Distribute model layers across all available GPUs
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=precision.torch_dtype,
            attn_implementation=attn_impl,
            device_map="auto",  # Automatically split across GPUs
        )
        print(f"Model distributed across devices: {set(model.hf_device_map.values())}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=precision.torch_dtype,
            attn_implementation=attn_impl,
        )
        if move_to_device:
            model = move_model_to_eval_device(model, device, dtype=precision.torch_dtype)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    apply_model_max_length(model, tokenizer, model_max_length)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


__all__ = [
    "apply_model_max_length",
    "get_model_dtype",
    "load_model",
    "move_model_to_eval_device",
    "resolve_eval_precision",
]
