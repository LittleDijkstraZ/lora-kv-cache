"""Cache helpers for synthetic generative-QA pairs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_CONTENT_KEYS = frozenset(
    {
        "instruct_model",
        "chunk_size",
        "overlap_ratio",
        "n_generations",
        "pairs_per_call",
        "context_window_chunks",
        "temperature",
        "top_p",
        "max_tokens",
        "eval_file",
        "eval_every_n",
    }
)


def find_or_create_cache_dir(
    data_dir: Path,
    instruct_model: str,
    chunk_size: int,
    overlap_ratio: float,
    n_generations: int,
    pairs_per_call: int,
    context_window_chunks: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    eval_path: Path,
    eval_every_n: int,
    synthetic_data_dir: str | None = None,
) -> Path:
    """Return a cache directory matching the requested synthesis settings."""

    if synthetic_data_dir:
        path = Path(synthetic_data_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    model_short = instruct_model.split("/")[-1]
    overlap_str = f"{overlap_ratio}".replace(".", "p")
    temp_str = f"{temperature}".replace(".", "p")
    base_name = (
        f"qa_{chunk_size}_{overlap_str}"
        f"_ngen{n_generations}_ppc{pairs_per_call}"
        f"_ctx{context_window_chunks}_{model_short}_t{temp_str}"
    )
    current_content = {
        "instruct_model": instruct_model,
        "chunk_size": chunk_size,
        "overlap_ratio": overlap_ratio,
        "n_generations": n_generations,
        "pairs_per_call": pairs_per_call,
        "context_window_chunks": context_window_chunks,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "eval_file": eval_path.name,
        "eval_every_n": eval_every_n,
    }

    synthetic_root = data_dir / "synthetic"
    candidates: list[Path] = []
    if synthetic_root.exists():
        for candidate in sorted(synthetic_root.iterdir()):
            if not candidate.is_dir():
                continue
            if candidate.name == base_name or (
                candidate.name.startswith(base_name + "_")
                and candidate.name[len(base_name) + 1 :].isdigit()
            ):
                candidates.append(candidate)

    for candidate in candidates:
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue
        with open(metadata_path) as handle:
            stored = json.load(handle)
        stored_content = {key: stored.get(key) for key in _CONTENT_KEYS}
        if stored_content == current_content:
            log.info("Reusing matching synthetic cache at %s", candidate)
            return candidate
        mismatches = sorted(
            key
            for key in _CONTENT_KEYS
            if stored_content.get(key) != current_content.get(key)
        )
        log.warning(
            "Skipping cache %s because metadata differs for: %s",
            candidate,
            ", ".join(mismatches),
        )

    cache_dir = synthetic_root / base_name
    suffix = 1
    while cache_dir.exists():
        cache_dir = synthetic_root / f"{base_name}_{suffix}"
        suffix += 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info("Created synthetic cache directory: %s", cache_dir)
    return cache_dir
