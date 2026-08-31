import gc
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from src.pipeline.training.lora_targets import is_all_linear_target, normalize_target_modules


def sanitize_model_name(name: str) -> str:
    return name.replace(os.sep, "__")


def resolve_logging_steps(
    total_training_steps: int,
    configured_logging_steps: int | None = None,
) -> int:
    """Choose an explicit cadence or roughly ten log points per run."""

    if configured_logging_steps is not None:
        configured = int(configured_logging_steps)
        if configured < 1:
            raise ValueError("logging_steps must be >= 1")
        return configured
    return max(1, int(total_training_steps) // 10)


# ── Module classification ────────────────────────────────────────────────────

_ATTN_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
_MLP_MODULES = frozenset({"gate_proj", "up_proj", "down_proj", "gate_up_proj"})


def _module_subpath(module: str) -> str:
    if module in _ATTN_MODULES:
        return "self_attn"
    if module in _MLP_MODULES:
        return "mlp"
    raise ValueError(f"Unknown LoRA target module: {module!r}")


def _lora_layer_prefix(config=None) -> str:
    """Return the PEFT parameter prefix for decoder layers."""
    if config is None:
        return "base_model.model.model.layers"
    try:
        config.hidden_size
        config.num_hidden_layers
        return "base_model.model.model.layers"
    except AttributeError:
        if getattr(config, "text_config", None) is not None:
            return "base_model.model.model.language_model.layers"
        return "base_model.model.model.layers"


def _module_lora_path(module: str) -> str:
    """Return the decoder-layer relative path used in PEFT LoRA parameter keys."""
    if "." in module:
        return module
    return f"{_module_subpath(module)}.{module}"


def _lora_key(layer_idx: int, module: str, matrix: str = "lora_A", *, layer_prefix: str | None = None) -> str:
    """Return the safetensors key for a LoRA A or B parameter."""
    prefix = layer_prefix or _lora_layer_prefix()
    return (
        f"{prefix}.{layer_idx}"
        f".{_module_lora_path(module)}.{matrix}.default.weight"
    )


def _text_config(config):
    """Return the decoder/text config that owns LoRA target dimensions."""
    try:
        config.hidden_size
        config.num_hidden_layers
        return config
    except AttributeError:
        nested = getattr(config, "text_config", None)
        if nested is not None:
            try:
                nested.hidden_size
                nested.num_hidden_layers
                return nested
            except AttributeError:
                pass
    raise AttributeError(
        "Model config does not expose hidden_size/num_hidden_layers. "
        "If this is a multimodal wrapper config, expected a text_config with those fields."
    )


def _decoder_layers(model):
    """Return the module list containing transformer decoder layers."""
    candidates = (
        ("model", "layers"),
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "layers"),
        ("language_model", "model", "layers"),
    )
    for path in candidates:
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None:
            return obj
    raise AttributeError("Could not locate decoder layers on loaded model.")


def _get_module_in_features(config) -> Dict[str, int]:
    config = _text_config(config)
    hidden_size = config.hidden_size
    intermediate_size = getattr(config, "intermediate_size", hidden_size * 4)
    head_dim = getattr(config, "head_dim", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    attn_in = (
        num_attention_heads * head_dim
        if head_dim is not None and num_attention_heads is not None
        else hidden_size
    )
    return {
        "q_proj": hidden_size,
        "k_proj": hidden_size,
        "v_proj": hidden_size,
        "o_proj": attn_in,
        "gate_proj": hidden_size,
        "up_proj": hidden_size,
        "down_proj": intermediate_size,
        "gate_up_proj": hidden_size,
    }


def _target_modules_tuple(target_modules) -> tuple[str, ...]:
    modules = normalize_target_modules(target_modules)
    if modules is None:
        return ("q_proj", "k_proj", "v_proj", "o_proj")
    if isinstance(modules, str):
        return (modules,)
    return tuple(str(module) for module in modules)


def _iter_layer_linear_targets(layer, target_modules: tuple[str, ...]):
    all_linear = is_all_linear_target(target_modules)
    for rel_name, module in layer.named_modules():
        if not rel_name or not isinstance(module, torch.nn.Linear):
            continue
        if all_linear or any(rel_name == target or rel_name.endswith(f".{target}") for target in target_modules):
            yield rel_name, module


def _target_pairs_from_loaded_model(model, layer_indices, target_modules: tuple[str, ...]):
    layers = _decoder_layers(model)
    pairs = []
    seen = set()
    for layer_idx in layer_indices:
        for rel_name, module in _iter_layer_linear_targets(layers[layer_idx], target_modules):
            key = (layer_idx, rel_name)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((layer_idx, rel_name, module))
    return pairs


def _is_compatible_shared_a_file(
    existing_state: dict,
    *,
    num_layers: int,
    lora_r: int,
    target_modules: Iterable[str],
    module_in_features: Dict[str, int],
    layer_prefix: str | None = None,
    layers_to_transform=None,
    init_method: str = "random",
    metadata_path: Optional[Path] = None,
) -> bool:
    # Check init_method matches stored metadata
    target_modules = _target_modules_tuple(target_modules)
    if metadata_path is not None and metadata_path.exists():
        try:
            stored = json.loads(metadata_path.read_text())
            if stored.get("init_method") != init_method:
                return False
            if is_all_linear_target(target_modules):
                return (
                    stored.get("target_modules") == list(target_modules)
                    and stored.get("lora_r") == lora_r
                    and bool(existing_state)
                )
        except Exception:
            return False
    elif init_method != "random":
        # No metadata file but non-random init requested → force regen
        return False

    layer_indices = layers_to_transform if layers_to_transform is not None else list(range(num_layers))
    expected_keys = len(layer_indices) * len(target_modules)
    if len(existing_state) != expected_keys:
        return False

    for layer_idx in layer_indices:
        for module in target_modules:
            name = _lora_key(layer_idx, module, "lora_A", layer_prefix=layer_prefix)
            tensor = existing_state.get(name)
            if tensor is None:
                return False
            if tuple(tensor.shape) != (lora_r, module_in_features[module]):
                return False
    return True


# ── PiSSA (SVD-based) initialisation ────────────────────────────────────────

def _svd_init_shared_matrices(
    model_name: str,
    layer_indices,
    target_modules,
    lora_r: int,
    layer_prefix: str | None = None,
) -> Tuple[dict, dict]:
    """Load model weights, run SVD, return (A_state, B_state).

    A_state  : { lora_A key → V_r^T  [r, d_in]  }
    B_state  : { lora_B key → U_r    [d_out, r]  }
    Both returned as float16 tensors.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_token = os.environ.get("HF_AUTH_TOKEN")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SharedA/PiSSA] Loading model weights for SVD ({model_name}) on {device}...", flush=True)
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        device_map=device,
        token=hf_token,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except (AttributeError, ValueError) as exc:
        config = AutoConfig.from_pretrained(model_name, token=hf_token)
        if getattr(config, "text_config", None) is None:
            raise
        print(
            "[SharedA/PiSSA] Falling back to AutoModelForImageTextToText "
            f"for wrapper config ({type(config).__name__}); original error: {exc}",
            flush=True,
        )
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
    model.eval()

    from tqdm import tqdm
    import sys

    A_state: dict = {}
    B_state: dict = {}

    target_modules = _target_modules_tuple(target_modules)
    pairs = _target_pairs_from_loaded_model(model, layer_indices, target_modules)
    if not pairs:
        raise ValueError(f"No linear LoRA target modules matched target_modules={target_modules!r}")

    for layer_idx, module_path, module_obj in tqdm(pairs, desc="[PiSSA] SVD init", unit="matrix", file=sys.stderr):
        W = module_obj.weight.data.float()  # [d_out, d_in], on device
        # Randomized low-rank SVD — only computes top-r singular vectors.
        # torch.svd_lowrank returns (U, S, V) where V is NOT transposed.
        U, _S, V = torch.svd_lowrank(W, q=lora_r, niter=4)
        A_state[_lora_key(layer_idx, module_path, "lora_A", layer_prefix=layer_prefix)] = V.T.contiguous().cpu().to(torch.float16)  # [r, d_in]
        B_state[_lora_key(layer_idx, module_path, "lora_B", layer_prefix=layer_prefix)] = U.contiguous().cpu().to(torch.float16)    # [d_out, r]

    del model
    gc.collect()
    return A_state, B_state


# ── Public API ───────────────────────────────────────────────────────────────

def initialize_shared_A_matrices(
    model_name: str,
    out_dir: str,
    lora_r: int,
    seed: int = 42,
    training_seed: int = 0,
    target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
    layers_to_transform=None,
    init_method: str = "random",
) -> Path:
    """Initialize shared LoRA A matrices once and save to disk.

    Args:
        init_method: ``"random"`` (Gaussian, current behaviour) or
                     ``"pissa"`` (V_r^T from SVD of each weight matrix).
                     When ``"pissa"``, a companion ``shared_B_init.safetensors``
                     is saved with the U_r matrices for optional B initialisation.
    """
    if init_method not in ("random", "pissa"):
        raise ValueError(f"init_method must be 'random' or 'pissa', got {init_method!r}")

    hf_token = os.environ.get("HF_AUTH_TOKEN")
    model_key = sanitize_model_name(model_name)
    shared_A_dir = Path(out_dir) / model_key / "sharedA" / f"seed_{training_seed}"
    shared_A_dir.mkdir(parents=True, exist_ok=True)
    shared_A_path = shared_A_dir / "shared_A.safetensors"
    metadata_path = shared_A_dir / "shared_A_metadata.json"

    print(f"[SharedA] Initializing shared A matrices (init_method={init_method!r}, seed={seed})...", flush=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    config = AutoConfig.from_pretrained(model_name, token=hf_token)
    train_config = _text_config(config)
    layer_prefix = _lora_layer_prefix(config)
    hidden_size = train_config.hidden_size
    num_layers = train_config.num_hidden_layers
    module_in_features = _get_module_in_features(train_config)

    target_modules = _target_modules_tuple(target_modules)
    layer_indices = layers_to_transform if layers_to_transform is not None else list(range(num_layers))

    print(
        f"[SharedA] Model: hidden_size={hidden_size}, num_layers={num_layers}, "
        f"lora_r={lora_r}, target_modules={target_modules}",
        flush=True,
    )

    if shared_A_path.exists():
        existing_state = load_file(str(shared_A_path))
        if _is_compatible_shared_a_file(
            existing_state,
            num_layers=num_layers,
            lora_r=lora_r,
            target_modules=target_modules,
            module_in_features=module_in_features,
            layer_prefix=layer_prefix,
            layers_to_transform=layers_to_transform,
            init_method=init_method,
            metadata_path=metadata_path,
        ):
            print(f"[SharedA] Found compatible shared A matrices at {shared_A_path}", flush=True)
            return shared_A_path
        print(f"[SharedA] Existing file is incompatible, regenerating: {shared_A_path}", flush=True)

    if init_method == "pissa":
        shared_A_state, shared_B_state = _svd_init_shared_matrices(
            model_name=model_name,
            layer_indices=layer_indices,
            target_modules=target_modules,
            lora_r=lora_r,
            layer_prefix=layer_prefix,
        )
        shared_B_path = shared_A_dir / "shared_B_init.safetensors"
        save_file(shared_B_state, str(shared_B_path))
        print(f"[SharedA] Saved {len(shared_B_state)} B_init matrices to {shared_B_path}", flush=True)
    else:
        from tqdm import tqdm
        shared_A_state = {}
        if is_all_linear_target(target_modules):
            from transformers import AutoModelForCausalLM

            model_kwargs = dict(
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                token=hf_token,
            )
            try:
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            except (AttributeError, ValueError):
                from transformers import AutoModelForImageTextToText

                model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
            pairs = _target_pairs_from_loaded_model(model, layer_indices, target_modules)
            if not pairs:
                raise ValueError(f"No linear LoRA target modules matched target_modules={target_modules!r}")
            for layer_idx, module_path, module_obj in tqdm(pairs, desc="[Random] A init", unit="matrix"):
                in_features = module_obj.in_features
                shared_A_state[_lora_key(layer_idx, module_path, "lora_A", layer_prefix=layer_prefix)] = (
                    torch.randn(lora_r, in_features, dtype=torch.float16) * (1.0 / lora_r)
                )
            del model
            gc.collect()
        else:
            pairs = [(layer_idx, module) for layer_idx in layer_indices for module in target_modules]
            for layer_idx, module in tqdm(pairs, desc="[Random] A init", unit="matrix"):
                in_features = module_in_features[module]
                shared_A_state[_lora_key(layer_idx, module, "lora_A", layer_prefix=layer_prefix)] = (
                    torch.randn(lora_r, in_features, dtype=torch.float16) * (1.0 / lora_r)
                )

    save_file(shared_A_state, str(shared_A_path))
    print(f"[SharedA] Saved {len(shared_A_state)} A matrices to {shared_A_path}", flush=True)

    metadata_path.write_text(
        json.dumps(
            {
                "init_method": init_method,
                "lora_r": lora_r,
                "target_modules": list(target_modules),
                "layers_to_transform": list(layers_to_transform) if layers_to_transform is not None else None,
            }
        )
    )
    return shared_A_path


def load_shared_A_matrices(shared_A_path: Path) -> dict:
    print(f"[SharedA] Loading shared A matrices from {shared_A_path}", flush=True)
    return load_file(str(shared_A_path))


def load_shared_B_init_matrices(shared_A_path: Path) -> Optional[dict]:
    """Load B_init matrices (U_r from PiSSA SVD) if they exist, else return None."""
    b_path = Path(shared_A_path).parent / "shared_B_init.safetensors"
    if not b_path.exists():
        return None
    print(f"[SharedA] Loading shared B_init matrices from {b_path}", flush=True)
    return load_file(str(b_path))
