#!/usr/bin/env python3
"""Record-loading and shattering helpers for synthetic generation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import jsonlines

from src.pipeline.data.data_utils import shatter_document_with_metadata
from src.pipeline.task_modes import QA_SHORT_TASK

log = logging.getLogger(__name__)


def _flatten_context(ctx) -> str:
    if isinstance(ctx, str):
        return ctx
    if isinstance(ctx, dict):
        titles = ctx.get("title") or []
        sentences = ctx.get("sentences") or []
        parts = []
        for title, sents in zip(titles, sentences):
            parts.append(title)
            parts.append(" ".join(sents))
            parts.append("")
        return "\n".join(parts).strip()
    return str(ctx)


def _get_id(ex: dict) -> str:
    return str(ex["_id"] if "_id" in ex else ex["id"])


def _get_document_dedupe_key(ex: dict) -> tuple[str, str]:
    """Return a stable key for one source document in an eval-style record."""

    metadata = ex.get("metadata")
    for key in ("document_id", "patient_id"):
        if isinstance(metadata, dict) and metadata.get(key) is not None:
            return (key, str(metadata[key]))
        if ex.get(key) is not None:
            return (key, str(ex[key]))

    ctx = ex.get("context", "")
    if ctx:
        return ("context", str(ctx))
    return ("id", _get_id(ex))


def load_training_examples(eval_path: Path, eval_every_n: int) -> list[dict]:
    """Load, normalize, and deduplicate eval examples by context."""
    with jsonlines.open(eval_path) as reader:
        all_examples = list(reader)
    training_examples = all_examples[::eval_every_n]
    log.info(
        "Synthesize: %d / %d examples (every %d-th) from %s",
        len(training_examples),
        len(all_examples),
        eval_every_n,
        eval_path.name,
    )

    for ex in training_examples:
        ex["context"] = _flatten_context(ex.get("context", ""))

    seen_documents: set[tuple[str, str]] = set()
    unique_examples: list[dict] = []
    for ex in training_examples:
        key = _get_document_dedupe_key(ex)
        if key not in seen_documents:
            seen_documents.add(key)
            unique_examples.append(ex)
    log.info(
        "Deduplicated %d examples -> %d unique documents",
        len(training_examples),
        len(unique_examples),
    )
    return unique_examples


def load_doc_to_cluster_map(cluster_dir: Path) -> dict[str, int]:
    """Load the document-to-cluster mapping emitted by the prepare step."""
    doc_to_cluster_path = cluster_dir / "doc_to_cluster.json"
    if not doc_to_cluster_path.exists():
        raise FileNotFoundError(
            f"doc_to_cluster.json not found at {doc_to_cluster_path}. "
            "Run the 'prepare' step first."
        )
    with open(doc_to_cluster_path) as fh:
        return json.load(fh)


def build_chunk_records(
    training_examples: list[dict],
    doc_to_cluster: dict[str, int],
    *,
    chunk_size: int,
    overlap_ratio: float,
) -> tuple[list[dict], dict[str, list[str]]]:
    """Re-shatter documents into chunk records and a context lookup."""
    all_chunk_records: list[dict] = []
    doc_chunks: dict[str, list[str]] = {}

    for ex in training_examples:
        doc_id = _get_id(ex)
        context = ex.get("context", "")
        if not context:
            continue

        shattered = shatter_document_with_metadata(
            context,
            chunk_size,
            overlap_ratio,
        )
        doc_chunks[doc_id] = [section.text for section in shattered.sections]
        cluster_idx = doc_to_cluster.get(doc_id, 0)

        for chunk_idx, section in enumerate(shattered.sections):
            all_chunk_records.append(
                {
                    "doc_id": doc_id,
                    "chunk_idx": chunk_idx,
                    "chunk_text": section.text,
                    "full_context": context,
                    "normalized_context": shattered.normalized_text,
                    "start_char": section.start_char,
                    "end_char": section.end_char,
                    "cluster_idx": cluster_idx,
                }
            )

    return all_chunk_records, doc_chunks


def remap_cached_cluster_indices(
    synthetic_records: list[dict],
    doc_to_cluster: dict[str, int],
) -> None:
    """Recompute cluster indices for cached records using the current cluster map."""
    for rec in synthetic_records:
        rec["cluster_idx"] = doc_to_cluster.get(rec["doc_id"], 0)


def scatter_synthetic_records(
    cluster_dir: Path,
    synthetic_records: list[dict],
) -> int:
    """Scatter generated synthetic records into per-cluster jsonl files."""
    cluster_records: dict[int, list[dict]] = {}
    for rec in synthetic_records:
        cluster_id = int(rec.get("cluster_idx", 0))
        cluster_records.setdefault(cluster_id, []).append(rec)

    n_written = 0
    for cluster_id, records in cluster_records.items():
        target_dir = cluster_dir / str(cluster_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        qa_jsonl = target_dir / "qa_pairs.jsonl"
        qa_metadata_json = target_dir / "qa_pairs.metadata.json"
        written_records = records
        with jsonlines.open(qa_jsonl, mode="w") as writer:
            writer.write_all(written_records)

        qa_metadata = {
            "cluster_id": cluster_id,
            "n_records": len(written_records),
            "task": QA_SHORT_TASK,
        }
        with open(qa_metadata_json, "w") as fh:
            json.dump(qa_metadata, fh, indent=2, sort_keys=True)

        log.info(
            "Cluster %d: wrote %d synthetic records -> %s",
            cluster_id,
            len(written_records),
            qa_jsonl,
        )
        n_written += len(written_records)

    log.info("Synthesis complete. Total synthetic records written: %d", n_written)
    return n_written
