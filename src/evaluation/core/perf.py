"""Lightweight evaluation performance and CUDA-memory instrumentation."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


def _gb(num_bytes: int | float | None) -> float | None:
    if num_bytes is None:
        return None
    return float(num_bytes) / (1024.0**3)


class EvalPerfStats:
    """Collect per-stage CUDA peak memory and inference timing.

    CUDA peak counters are reset at each measured stage, so stage memory is the
    absolute process allocation peak during that stage. This is usually the most
    useful number for comparing eval recipes because it includes resident model
    weights plus any temporary KV/cache allocations for that stage.
    """

    def __init__(self, *, enabled: bool | None = None) -> None:
        self.enabled = torch.cuda.is_available() if enabled is None else bool(enabled)
        self.metadata: dict[str, Any] = {}
        self.stage_memory: dict[str, dict[str, Any]] = {}
        self.inference_time_s: dict[str, float] = {}
        self.inference_calls: dict[str, int] = {}
        self.inference: dict[str, Any] = {
            "elapsed_s": None,
            "samples": None,
            "samples_per_s": None,
            "output_tokens": None,
            "output_tokens_per_s": None,
        }
        self.context_prefill_cache: list[dict[str, Any]] = []

    def set_metadata(self, **metadata: Any) -> None:
        self.metadata.update(metadata)

    def _device_indices(self) -> list[int]:
        if not self.enabled:
            return []
        return list(range(torch.cuda.device_count()))

    def _sync(self) -> None:
        if not self.enabled:
            return
        for idx in self._device_indices():
            torch.cuda.synchronize(idx)

    def _reset_peak_memory(self) -> None:
        if not self.enabled:
            return
        for idx in self._device_indices():
            torch.cuda.reset_peak_memory_stats(idx)

    def _memory_snapshot(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "cuda_available": False,
                "device_count": 0,
                "total_peak_allocated_bytes": None,
                "total_peak_reserved_bytes": None,
                "total_peak_allocated_gb": None,
                "total_peak_reserved_gb": None,
                "devices": [],
            }

        devices: list[dict[str, Any]] = []
        total_peak_allocated = 0
        total_peak_reserved = 0
        for idx in self._device_indices():
            current_allocated = int(torch.cuda.memory_allocated(idx))
            current_reserved = int(torch.cuda.memory_reserved(idx))
            peak_allocated = int(torch.cuda.max_memory_allocated(idx))
            peak_reserved = int(torch.cuda.max_memory_reserved(idx))
            total_peak_allocated += peak_allocated
            total_peak_reserved += peak_reserved
            devices.append(
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "current_allocated_bytes": current_allocated,
                    "current_reserved_bytes": current_reserved,
                    "peak_allocated_bytes": peak_allocated,
                    "peak_reserved_bytes": peak_reserved,
                    "current_allocated_gb": _gb(current_allocated),
                    "current_reserved_gb": _gb(current_reserved),
                    "peak_allocated_gb": _gb(peak_allocated),
                    "peak_reserved_gb": _gb(peak_reserved),
                }
            )

        return {
            "cuda_available": True,
            "device_count": len(devices),
            "total_peak_allocated_bytes": total_peak_allocated,
            "total_peak_reserved_bytes": total_peak_reserved,
            "total_peak_allocated_gb": _gb(total_peak_allocated),
            "total_peak_reserved_gb": _gb(total_peak_reserved),
            "devices": devices,
        }

    def _record_memory_snapshot(self, stage_name: str, snapshot: dict[str, Any]) -> None:
        existing = self.stage_memory.get(stage_name)
        if existing is None:
            self.stage_memory[stage_name] = {
                "calls": 1,
                **snapshot,
            }
            return

        existing["calls"] = int(existing.get("calls", 0)) + 1
        existing_peak = existing.get("total_peak_allocated_bytes")
        snapshot_peak = snapshot.get("total_peak_allocated_bytes")
        if existing_peak is None or (
            snapshot_peak is not None and int(snapshot_peak) > int(existing_peak)
        ):
            calls = existing["calls"]
            existing.clear()
            existing.update({"calls": calls, **snapshot})

    @contextmanager
    def memory_stage(self, stage_name: str) -> Iterator[None]:
        self._sync()
        self._reset_peak_memory()
        try:
            yield
        finally:
            self._sync()
            self._record_memory_snapshot(stage_name, self._memory_snapshot())

    @contextmanager
    def inference_stage(self, stage_name: str) -> Iterator[None]:
        self._sync()
        self._reset_peak_memory()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self.inference_time_s[stage_name] = self.inference_time_s.get(stage_name, 0.0) + elapsed
            self.inference_calls[stage_name] = self.inference_calls.get(stage_name, 0) + 1
            self._record_memory_snapshot(stage_name, self._memory_snapshot())

    @contextmanager
    def inference_timer(self, stage_name: str) -> Iterator[None]:
        """Time an inference span without touching CUDA peak-memory counters."""

        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed = time.perf_counter() - start
            self.inference_time_s[stage_name] = self.inference_time_s.get(stage_name, 0.0) + elapsed
            self.inference_calls[stage_name] = self.inference_calls.get(stage_name, 0) + 1

    def record_stage_memory_max(self, stage_name: str, source_stage_names: list[str]) -> None:
        """Record *stage_name* as the max peak-memory snapshot among source stages."""

        candidates = [
            self.stage_memory[name]
            for name in source_stage_names
            if name in self.stage_memory
        ]
        if not candidates:
            return
        best = max(
            candidates,
            key=lambda item: item.get("total_peak_allocated_bytes") or 0,
        )
        self.stage_memory[stage_name] = {
            **best,
            "calls": sum(int(item.get("calls", 0)) for item in candidates),
            "source_stages": list(source_stage_names),
        }

    def set_inference_summary(self, *, samples: int, output_tokens: int, elapsed_s: float) -> None:
        samples = int(samples)
        output_tokens = int(output_tokens)
        elapsed_s = float(elapsed_s)
        self.inference.update(
            {
                "elapsed_s": elapsed_s,
                "samples": samples,
                "samples_per_s": (samples / elapsed_s) if elapsed_s > 0 else None,
                "output_tokens": output_tokens,
                "output_tokens_per_s": (output_tokens / elapsed_s) if elapsed_s > 0 else None,
            }
        )

    def add_context_prefill_cache_record(
        self,
        *,
        batch_size: int,
        logical_context_tokens: int,
        physical_context_tokens: int,
        prefix_keep_tokens: int,
        requested_compression_ratio: float,
        effective_compression_ratio: float,
    ) -> None:
        self.context_prefill_cache.append(
            {
                "batch_size": int(batch_size),
                "logical_context_tokens": int(logical_context_tokens),
                "physical_context_tokens": int(physical_context_tokens),
                "prefix_keep_tokens": int(prefix_keep_tokens),
                "requested_compression_ratio": float(requested_compression_ratio),
                "effective_compression_ratio": float(effective_compression_ratio),
            }
        )

    @staticmethod
    def _summarize_records(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
        values = [int(record[key]) for record in records if record.get(key) is not None]
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        }

    def to_dict(self) -> dict[str, Any]:
        inference_time = dict(sorted(self.inference_time_s.items()))
        inference_calls = dict(sorted(self.inference_calls.items()))
        data = {
            "schema_version": 1,
            "metadata": self.metadata,
            "cuda_available": bool(self.enabled),
            "stage_memory": dict(sorted(self.stage_memory.items())),
            "inference": {
                **self.inference,
                "stage_time_s": inference_time,
                "stage_calls": inference_calls,
            },
        }
        if self.context_prefill_cache:
            data["context_prefill_cache"] = {
                "records": self.context_prefill_cache,
                "logical_context_tokens": self._summarize_records(
                    self.context_prefill_cache,
                    "logical_context_tokens",
                ),
                "physical_context_tokens": self._summarize_records(
                    self.context_prefill_cache,
                    "physical_context_tokens",
                ),
            }
        return data


def save_perf_stats(perf_stats: EvalPerfStats, path: str | Path) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(perf_stats.to_dict(), f, indent=2)
