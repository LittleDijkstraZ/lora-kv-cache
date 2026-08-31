"""Dataset construction for the paper's four training formats."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from src.pipeline.data.data_utils import load_cluster_chunks
from src.pipeline.task_modes import normalize_task_mode, normalize_task_modes
from src.pipeline.training.input_formatter import (
    CHUNK_NEXT_PROMPT_FORMAT,
    CTX_QA_FORMAT,
    QA_ONLY_FORMAT,
    RAW_FORMAT,
    format_next_chunks_for_sft,
    format_record_for_sft,
    format_text_for_ntp,
)
from src.pipeline.training.mixed_label_utils import (
    build_continuation_labels,
    tokenize_continuation_example,
)


SOURCE_KINDS = {"synthetic_records", "raw_chunks", "next_chunks"}


def load_mixed_sources_config(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as handle:
        value = json.load(handle)
    if not isinstance(value, list) or not value:
        raise ValueError("mixed_sources_file must contain a non-empty JSON list")
    return value


def _load_qa_records(cluster_dir: Path) -> list[dict]:
    path = cluster_dir / "qa_pairs.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic QA records not found: {path}")
    records: list[dict] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                record["task"] = normalize_task_mode(record.get("task", "qa_short"))
                records.append(record)
    return records


def _validate_source(source: dict[str, Any], index: int) -> dict[str, Any]:
    value = dict(source)
    value["name"] = str(value.get("name") or f"source_{index}")
    value["kind"] = str(value.get("kind", "synthetic_records")).strip().lower()
    value["input_format"] = str(value.get("input_format", "")).strip().lower()
    value["train_answer_only"] = bool(value.get("train_answer_only", False))
    value["append_eos"] = bool(value.get("append_eos", False))
    value["sample_weight"] = float(value.get("sample_weight", 1.0))
    value["context_window_chunks"] = int(value.get("context_window_chunks", 0) or 0)
    value["n_next"] = int(value.get("n_next", 1) or 1)
    value["cluster_root"] = value.get("cluster_root")
    if value["kind"] not in SOURCE_KINDS:
        raise ValueError(f"Unsupported mixed source kind: {value['kind']!r}")
    if value["sample_weight"] < 0 or value["context_window_chunks"] < 0:
        raise ValueError("sample_weight and context_window_chunks must be non-negative")
    if value["kind"] == "synthetic_records":
        if value["input_format"] not in {QA_ONLY_FORMAT, CTX_QA_FORMAT}:
            raise ValueError("synthetic_records require qa_only or ctx_qa")
        if value["context_window_chunks"] and value["input_format"] != CTX_QA_FORMAT:
            raise ValueError("context_window_chunks is supported only with ctx_qa")
    elif value["kind"] == "raw_chunks":
        if value["input_format"] != RAW_FORMAT:
            raise ValueError("raw_chunks require input_format=raw")
        if value["train_answer_only"] or value["context_window_chunks"]:
            raise ValueError("raw_chunks do not support answer-only or context-window options")
    else:
        if value["input_format"] not in {RAW_FORMAT, CHUNK_NEXT_PROMPT_FORMAT}:
            raise ValueError("next_chunks require raw or chunk_next_prompt")
        if value["train_answer_only"] or value["context_window_chunks"]:
            raise ValueError("next_chunks do not support answer-only or context-window options")
        if value["n_next"] < 1:
            raise ValueError("next_chunks require n_next >= 1")
    return value


def _source_cluster_dir(default: Path, cluster_id: int, source: dict) -> Path:
    root = source.get("cluster_root")
    return Path(root) / str(cluster_id) if root else default


def _chunk_lookup(records: list[dict]) -> dict[str, dict[int, str]]:
    result: dict[str, dict[int, str]] = {}
    for record in records:
        try:
            chunk_index = int(record["chunk_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        document_id = str(record.get("doc_id", ""))
        text = str(record.get("chunk_text") or record.get("context", "")).strip()
        if document_id and text:
            result.setdefault(document_id, {})[chunk_index] = text
    return result


def _with_context_window(record: dict, lookup: dict, radius: int) -> dict:
    if radius <= 0:
        return record
    try:
        center = int(record["chunk_idx"])
    except (KeyError, TypeError, ValueError):
        return record
    chunks = lookup.get(str(record.get("doc_id", "")), {})
    texts = [
        chunks[index]
        for index in range(center - radius, center + radius + 1)
        if index in chunks
    ]
    if not texts:
        return record
    value = dict(record)
    value["chunk_text"] = "\n\n".join(texts)
    value["context"] = value["chunk_text"]
    return value


def _check_length(
    *,
    cluster_id: int,
    source_name: str,
    length: int,
    max_seq_length: int,
    disable_truncation: bool,
    fail_on_truncation: bool,
) -> None:
    if length > max_seq_length and (disable_truncation or fail_on_truncation):
        raise RuntimeError(
            f"[Cluster {cluster_id}] {source_name!r} has {length} tokens, "
            f"exceeding max_seq_length={max_seq_length}."
        )


def _example(input_ids: list[int], labels: list[int]) -> dict[str, list[int]]:
    return {
        "input_ids": list(input_ids),
        "attention_mask": [1] * len(input_ids),
        "labels": list(labels),
    }


def _synthetic_examples(
    *,
    cluster_id: int,
    records: list[dict],
    source: dict,
    tokenizer,
    max_seq_length: int,
    disable_truncation: bool,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    task_filter = normalize_task_modes(source.get("task")) if source.get("task") else []
    selected = [
        record
        for record in records
        if not task_filter or record.get("task", "qa_short") in task_filter
    ]
    lookup = _chunk_lookup(records) if source["context_window_chunks"] else {}
    examples: list[dict[str, list[int]]] = []
    for record in selected:
        record = _with_context_window(record, lookup, source["context_window_chunks"])
        full_text, prefix = format_record_for_sft(
            record,
            tokenizer,
            source["input_format"],
            task_mode=record.get("task"),
            append_eos=source["append_eos"],
        )
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        _check_length(
            cluster_id=cluster_id,
            source_name=source["name"],
            length=len(all_ids),
            max_seq_length=max_seq_length,
            disable_truncation=disable_truncation,
            fail_on_truncation=fail_on_truncation,
        )
        if source["train_answer_only"]:
            tokenized = tokenize_continuation_example(
                tokenizer, full_text, prefix, max_seq_length=max_seq_length
            )
            if tokenized is None:
                continue
            input_ids, mask = tokenized
            labels = build_continuation_labels(input_ids, mask)
        else:
            input_ids = all_ids[:max_seq_length]
            labels = list(input_ids)
        if input_ids:
            examples.append(_example(input_ids, labels))
    return examples, {
        "name": source["name"],
        "kind": source["kind"],
        "input_format": source["input_format"],
        "sample_weight": source["sample_weight"],
        "raw_record_count": len(selected),
        "final_example_count": len(examples),
    }


def _raw_examples(
    *,
    cluster_id: int,
    chunks: list[str],
    source: dict,
    tokenizer,
    max_seq_length: int,
    disable_truncation: bool,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    examples: list[dict[str, list[int]]] = []
    for chunk in chunks:
        text = format_text_for_ntp(chunk, tokenizer, RAW_FORMAT)
        all_ids = tokenizer.encode(text, add_special_tokens=True)
        _check_length(
            cluster_id=cluster_id,
            source_name=source["name"],
            length=len(all_ids),
            max_seq_length=max_seq_length,
            disable_truncation=disable_truncation,
            fail_on_truncation=fail_on_truncation,
        )
        input_ids = all_ids[:max_seq_length]
        if input_ids:
            examples.append(_example(input_ids, input_ids))
    return examples, {
        "name": source["name"],
        "kind": source["kind"],
        "input_format": source["input_format"],
        "sample_weight": source["sample_weight"],
        "raw_record_count": len(chunks),
        "final_example_count": len(examples),
    }


def _next_chunk_examples(
    *,
    cluster_id: int,
    chunks: list[str],
    source: dict,
    tokenizer,
    max_seq_length: int,
    disable_truncation: bool,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    n_next = source["n_next"]
    examples: list[dict[str, list[int]]] = []
    for index in range(max(0, len(chunks) - n_next)):
        target = "\n\n".join(chunks[index + 1 : index + 1 + n_next])
        full_text, prefix = format_next_chunks_for_sft(
            current_text=chunks[index],
            next_text=target,
            tokenizer=tokenizer,
            input_format=source["input_format"],
            append_eos=source["append_eos"],
        )
        all_ids = tokenizer.encode(full_text, add_special_tokens=False)
        _check_length(
            cluster_id=cluster_id,
            source_name=source["name"],
            length=len(all_ids),
            max_seq_length=max_seq_length,
            disable_truncation=disable_truncation,
            fail_on_truncation=fail_on_truncation,
        )
        tokenized = tokenize_continuation_example(
            tokenizer, full_text, prefix, max_seq_length=max_seq_length
        )
        if tokenized is None:
            continue
        input_ids, mask = tokenized
        examples.append(_example(input_ids, build_continuation_labels(input_ids, mask)))
    return examples, {
        "name": source["name"],
        "kind": source["kind"],
        "input_format": source["input_format"],
        "n_next": n_next,
        "sample_weight": source["sample_weight"],
        "raw_record_count": len(chunks),
        "final_example_count": len(examples),
    }


def _target_sizes(manifests: list[dict[str, Any]]) -> list[int]:
    total = sum(int(item["final_example_count"]) for item in manifests)
    weights = [float(item["sample_weight"]) for item in manifests]
    if total <= 0 or sum(weights) <= 0:
        raise RuntimeError("Mixed training produced no weighted examples")
    exact = [total * weight / sum(weights) for weight in weights]
    sizes = [int(value) for value in exact]
    for index in sorted(
        range(len(exact)), key=lambda i: exact[i] - sizes[i], reverse=True
    )[: total - sum(sizes)]:
        sizes[index] += 1
    return sizes


def _sample(pool: list[dict], count: int, rng: random.Random) -> list[dict]:
    if not pool:
        raise RuntimeError("A mixed training source produced zero examples")
    if count <= len(pool):
        indices = rng.sample(range(len(pool)), count)
    else:
        indices = [rng.randrange(len(pool)) for _ in range(count)]
    return [pool[index] for index in indices]


def build_mixed_training_examples(
    *,
    cluster_dir: Path,
    cluster_id: int,
    tokenizer,
    mixed_sources: list[dict[str, Any]],
    max_seq_length: int,
    seed: int,
    disable_truncation: bool,
    fail_on_truncation: bool,
) -> tuple[list[dict[str, list[int]]], dict[str, Any]]:
    sources = [_validate_source(source, index) for index, source in enumerate(mixed_sources)]
    pools: list[list[dict[str, list[int]]]] = []
    manifests: list[dict[str, Any]] = []
    for source in sources:
        source_dir = _source_cluster_dir(cluster_dir, cluster_id, source)
        if source["kind"] == "synthetic_records":
            pool, manifest = _synthetic_examples(
                cluster_id=cluster_id,
                records=_load_qa_records(source_dir),
                source=source,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                disable_truncation=disable_truncation,
                fail_on_truncation=fail_on_truncation,
            )
        elif source["kind"] == "raw_chunks":
            pool, manifest = _raw_examples(
                cluster_id=cluster_id,
                chunks=load_cluster_chunks(source_dir),
                source=source,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                disable_truncation=disable_truncation,
                fail_on_truncation=fail_on_truncation,
            )
        else:
            pool, manifest = _next_chunk_examples(
                cluster_id=cluster_id,
                chunks=load_cluster_chunks(source_dir),
                source=source,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                disable_truncation=disable_truncation,
                fail_on_truncation=fail_on_truncation,
            )
        pools.append(pool)
        manifests.append(manifest)

    rng = random.Random(seed)
    examples: list[dict[str, list[int]]] = []
    for pool, manifest, count in zip(pools, manifests, _target_sizes(manifests)):
        examples.extend(_sample(pool, count, rng))
        manifest["mixed_target_examples"] = count
    rng.shuffle(examples)
    return examples, {
        "cluster_id": cluster_id,
        "mixed_epoch_size": len(examples),
        "sources": manifests,
    }
