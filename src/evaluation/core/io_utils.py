"""Result, prediction, and deterministic sharding helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path


def make_result_entry(method: str | None = None, **fields) -> dict:
    entry = {}
    if method is not None:
        entry["method"] = method
    entry.update(fields)
    return entry


def _write_json(payload, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def save_results(results: dict, outfile: str | Path) -> None:
    _write_json(results, outfile)


def save_predictions(predictions: list, path: str | Path) -> None:
    _write_json(predictions, path)


def shard_indices(
    size: int,
    eval_every_n: int = 1,
    shard_id: int = 0,
    num_shards: int = 1,
    shuffle_seed: int = 42,
) -> list[int]:
    if eval_every_n < 1:
        raise ValueError("eval_every_n must be at least 1")
    if num_shards < 1 or not 0 <= shard_id < num_shards:
        raise ValueError("require num_shards >= 1 and 0 <= shard_id < num_shards")
    indices = list(range(0, int(size), int(eval_every_n)))
    if num_shards > 1:
        random.Random(shuffle_seed).shuffle(indices)
        shard_size = (len(indices) + num_shards - 1) // num_shards
        start = shard_id * shard_size
        indices = indices[start : start + shard_size]
    return indices


__all__ = ["make_result_entry", "save_predictions", "save_results", "shard_indices"]
