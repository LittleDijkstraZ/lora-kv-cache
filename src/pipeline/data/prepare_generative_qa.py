#!/usr/bin/env python3
"""Prepare generative QA training data from an eval JSONL file.

This is the generic generative-QA prepare path used by the generative-QA
handler. Input records are loaded through the shared canonical record layer,
clustered through the shared cluster-strategy layer, and materialized to the
standard per-cluster training layout plus ``id_to_cluster.json``.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline.data.qa_materialization import (
    assign_clusters_to_records,
    load_canonical_generative_records,
    materialize_generative_training_data,
)


def prepare_from_eval_file(
    eval_path: Path,
    eval_every_n: int,
    outdir: Path,
    *,
    chunk_size: int,
    overlap_ratio: float,
    data_cfg,
) -> dict[str, int]:
    """Build training data directly from a generative eval JSONL file."""
    chunk_size = int(chunk_size)
    overlap_ratio = float(overlap_ratio)
    eval_every_n = int(eval_every_n)
    if eval_every_n < 1:
        raise ValueError("eval_every_n must be at least 1")

    eval_path = Path(eval_path)
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_path}")

    all_records = load_canonical_generative_records(eval_path)
    training_records = all_records[::eval_every_n]
    if not training_records:
        raise ValueError(f"No training examples selected from {eval_path}")

    print(
        f"[prepare_generative_qa] train_from_eval_file: using {len(training_records)} / "
        f"{len(all_records)} examples (every {eval_every_n}-th) from {eval_path.name}"
    )
    record_to_cluster, _ = assign_clusters_to_records(
        training_records,
        data_cfg,
        default_strategy="metadata_field",
        dedupe_by_context=True,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    result = materialize_generative_training_data(
        training_records,
        record_to_cluster,
        outdir,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
    )
    cluster_indices = sorted(set(record_to_cluster.values()))
    print(f"[prepare_generative_qa] Using {len(cluster_indices)} cluster(s): {cluster_indices}")
    total_chunks = sum(len(v) for v in result.cluster_map.values())
    print(
        f"[prepare_generative_qa] Saved {total_chunks} training chunks across "
        f"{len(cluster_indices)} cluster(s) -> {outdir}"
    )
    return result.doc_to_cluster
