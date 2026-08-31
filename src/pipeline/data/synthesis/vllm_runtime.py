"""vLLM device and tensor-parallel runtime helpers."""

from __future__ import annotations

import os
from typing import Any


def count_visible_cuda_devices() -> int | None:
    """Best-effort count of CUDA devices visible to the current process."""

    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    env_count: int | None = None
    if env_value is not None:
        stripped = env_value.strip()
        if stripped in {"", "-1"}:
            env_count = 0
        else:
            env_count = len([part for part in stripped.split(",") if part.strip()])

    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
        if env_count is not None:
            return env_count
        return 0
    except Exception:
        return env_count


def resolve_vllm_parallelism(
    requested_tensor_parallel_size: int,
    visible_gpu_count: int | None,
    *,
    forced_backend: str | None = None,
) -> dict[str, Any]:
    """Choose a safe local vLLM parallel plan from the requested settings."""

    requested_size = max(1, int(requested_tensor_parallel_size))
    effective_size = requested_size
    if (
        visible_gpu_count is not None
        and visible_gpu_count > 0
        and requested_size > visible_gpu_count
    ):
        effective_size = visible_gpu_count

    backend = forced_backend.strip() if forced_backend else None
    if (
        not backend
        and effective_size > 1
        and visible_gpu_count is not None
        and visible_gpu_count >= effective_size
    ):
        backend = "mp"

    return {
        "requested_tensor_parallel_size": requested_size,
        "tensor_parallel_size": effective_size,
        "visible_gpu_count": visible_gpu_count,
        "distributed_executor_backend": backend,
    }
