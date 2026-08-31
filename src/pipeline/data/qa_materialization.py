"""Canonicalize, cluster, and materialize generative-QA documents."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jsonlines

from src.pipeline.data.cluster_strategy import (
    assign_clusters_by_metadata_field,
    cfg_get,
    metadata_value_for_record,
    resolve_cluster_strategy,
)
from src.pipeline.data.contracts import (
    GENERATIVE_TASK_FAMILY,
    CanonicalGenerativeQARecord,
    CanonicalQARecord,
)
from src.pipeline.data.data_utils import shatter_document


@dataclass
class MaterializationResult:
    """Summary of the document and cluster sidecars written to disk."""

    doc_to_cluster: dict[str, int]
    cluster_map: dict[str, list[Any]]


def flatten_context(context) -> str:
    """Normalize supported context shapes to a plain string."""

    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentences = context.get("sentences") or []
        parts: list[str] = []
        for title, document_sentences in zip(titles, sentences):
            parts.append(str(title))
            parts.append(" ".join(str(sentence) for sentence in document_sentences))
            parts.append("")
        return "\n".join(parts).strip()
    return str(context)


def _merge_metadata(example: dict, reserved_keys: set[str]) -> dict[str, Any]:
    metadata = dict(example.get("metadata") or {})
    for key, value in example.items():
        if key not in reserved_keys and key not in metadata:
            metadata[key] = value
    return metadata


def load_canonical_generative_records(
    eval_path: Path,
) -> list[CanonicalGenerativeQARecord]:
    """Load a generative-QA JSONL file into canonical records."""

    records: list[CanonicalGenerativeQARecord] = []
    reserved = {
        "_id",
        "id",
        "context",
        "question",
        "answer",
        "answers",
        "cluster_idx",
        "metadata",
    }
    with jsonlines.open(eval_path) as reader:
        for example in reader:
            record_id = str(example.get("_id", example.get("id")))
            if record_id in {"None", ""}:
                raise ValueError(f"Missing _id/id in generative eval example: {example}")

            answers = example.get("answers")
            if answers is None and "answer" in example:
                answers = [example["answer"]]
            if isinstance(answers, str):
                answers = [answers]

            records.append(
                CanonicalGenerativeQARecord(
                    record_id=record_id,
                    task_family=GENERATIVE_TASK_FAMILY,
                    question=str(example["question"]),
                    context=flatten_context(example.get("context", "")),
                    answers=list(answers or []),
                    cluster_idx=example.get("cluster_idx"),
                    metadata=_merge_metadata(example, reserved),
                )
            )
    return records


def assign_clusters_to_records(
    records: list[CanonicalQARecord],
    data_cfg,
    *,
    default_strategy: str = "metadata_field",
    dedupe_by_context: bool = False,
) -> tuple[dict[str, int], dict[str, int] | None]:
    """Assign records to document clusters using metadata or supplied ids."""

    if not records:
        return {}, None

    spec = resolve_cluster_strategy(data_cfg, default_strategy=default_strategy)
    expected_num_clusters = cfg_get(data_cfg, "num_clusters", None)

    def dedupe_key(record: CanonicalQARecord):
        return (
            metadata_value_for_record(record, str(spec.metadata_field)),
            record.context,
        )

    if dedupe_by_context:
        representative_by_key: dict[tuple[Any, str], CanonicalQARecord] = {}
        for record in records:
            representative_by_key.setdefault(dedupe_key(record), record)
        assignment_records = list(representative_by_key.values())
        representative_id_by_key = {
            key: record.record_id for key, record in representative_by_key.items()
        }
    else:
        assignment_records = list(records)
        representative_id_by_key = {}

    record_to_cluster, value_to_cluster = assign_clusters_by_metadata_field(
        assignment_records,
        metadata_field=str(spec.metadata_field),
        expected_num_clusters=expected_num_clusters,
    )

    if dedupe_by_context:
        record_to_cluster = {
            record.record_id: record_to_cluster[
                representative_id_by_key[dedupe_key(record)]
            ]
            for record in records
        }
    return record_to_cluster, value_to_cluster


def _write_cluster_chunks(
    records: list[CanonicalQARecord],
    record_to_cluster: dict[str, int],
    outdir: Path,
    *,
    chunk_size: int,
    overlap_ratio: float,
    filename_builder: Callable[[CanonicalQARecord, int], str],
    cluster_map_entry_builder: Callable[[CanonicalQARecord, int, str], Any],
) -> MaterializationResult:
    cluster_map: dict[str, list[Any]] = defaultdict(list)
    doc_to_cluster: dict[str, int] = {}
    records_by_cluster: dict[int, list[CanonicalQARecord]] = defaultdict(list)
    for record in records:
        cluster_idx = int(record_to_cluster[record.record_id])
        records_by_cluster[cluster_idx].append(record)
        doc_to_cluster[record.record_id] = cluster_idx

    for cluster_idx, cluster_records in sorted(records_by_cluster.items()):
        cluster_dir = outdir / str(cluster_idx)
        cluster_dir.mkdir(parents=True, exist_ok=True)

        all_chunks: list[str] = []
        seen_contexts: set[str] = set()
        for record in cluster_records:
            if record.context in seen_contexts:
                continue
            seen_contexts.add(record.context)

            chunks = shatter_document(record.context, chunk_size, overlap_ratio)
            for chunk_idx, chunk in enumerate(chunks):
                filename = filename_builder(record, chunk_idx)
                (cluster_dir / filename).write_text(chunk)
                all_chunks.append(chunk)
                cluster_map[str(cluster_idx)].append(
                    cluster_map_entry_builder(record, chunk_idx, filename)
                )

        with open(cluster_dir / f"{cluster_idx}.json", "w") as handle:
            json.dump(all_chunks, handle, indent=2)

    with open(outdir / "cluster_map.json", "w") as handle:
        json.dump(cluster_map, handle, indent=2)
    with open(outdir / "doc_to_cluster.json", "w") as handle:
        json.dump(doc_to_cluster, handle, indent=2)
    return MaterializationResult(
        doc_to_cluster=doc_to_cluster,
        cluster_map=dict(cluster_map),
    )


def materialize_generative_training_data(
    records: list[CanonicalGenerativeQARecord],
    record_to_cluster: dict[str, int],
    outdir: Path,
    *,
    chunk_size: int,
    overlap_ratio: float,
) -> MaterializationResult:
    """Write chunk directories and document/cluster mapping sidecars."""

    def filename_builder(record: CanonicalGenerativeQARecord, chunk_idx: int) -> str:
        stub = hashlib.md5(record.record_id.encode()).hexdigest()[:12]
        return f"{stub}_chunk{chunk_idx}.txt"

    def cluster_map_entry_builder(
        record: CanonicalGenerativeQARecord,
        chunk_idx: int,
        filename: str,
    ) -> str:
        del record, chunk_idx
        return filename.removesuffix(".txt")

    result = _write_cluster_chunks(
        records,
        record_to_cluster,
        outdir,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
        filename_builder=filename_builder,
        cluster_map_entry_builder=cluster_map_entry_builder,
    )
    with open(outdir / "id_to_cluster.json", "w") as handle:
        json.dump(result.doc_to_cluster, handle, indent=2)
    return result
