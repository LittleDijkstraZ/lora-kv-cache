#!/usr/bin/env python3
"""Generate fixed-budget synthetic QA pairs with vLLM."""

from __future__ import annotations

import gc
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Any

import jsonlines

from src.pipeline.data.synthesis.cache import find_or_create_cache_dir
from src.pipeline.data.synthesis.prompting import build_prompt, generate_batch, parse_output
from src.pipeline.data.synthesis.records import (
    build_chunk_records,
    load_doc_to_cluster_map,
    load_training_examples,
    remap_cached_cluster_indices,
    scatter_synthetic_records,
)
from src.pipeline.data.synthesis.vllm_runtime import (
    count_visible_cuda_devices,
    resolve_vllm_parallelism,
)
from src.pipeline.task_modes import QA_SHORT_TASK

log = logging.getLogger(__name__)


def _is_retryable_vllm_startup_error(exc: BaseException) -> bool:
    message = str(exc)
    if (
        "Error in memory profiling" in message
        and "Initial free memory" in message
        and "current free memory" in message
        and "release GPU memory while vLLM is profiling" in message
    ):
        return True
    return (
        "Engine core initialization failed" in message
        and "Failed core proc" in message
    )


def _init_vllm_with_memory_race_retries(
    llm_class: Any,
    llm_kwargs: dict[str, Any],
) -> Any:
    retries = int(os.environ.get("LORA_KV_CACHE_VLLM_INIT_RETRIES", "3"))
    delay_seconds = float(
        os.environ.get("LORA_KV_CACHE_VLLM_INIT_RETRY_SECONDS", "20")
    )
    for attempt in range(retries + 1):
        try:
            return llm_class(**llm_kwargs)
        except RuntimeError as exc:
            if not _is_retryable_vllm_startup_error(exc) or attempt >= retries:
                raise
            wait_seconds = delay_seconds * (attempt + 1)
            log.warning(
                "vLLM startup hit a transient memory-profiling race; retrying "
                "in %.0f seconds (%d/%d).",
                wait_seconds,
                attempt + 1,
                retries,
            )
            time.sleep(wait_seconds)
    raise RuntimeError("unreachable")


def _request_key(record: dict[str, Any]) -> tuple[str, int]:
    return str(record["doc_id"]), int(record["chunk_idx"])


def _build_output_record(
    record: dict[str, Any],
    pair: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task": QA_SHORT_TASK,
        "doc_id": record["doc_id"],
        "chunk_idx": record["chunk_idx"],
        "chunk_text": record["chunk_text"],
        "full_context": record.get("full_context", ""),
        "cluster_idx": record["cluster_idx"],
        "requested_pairs": int(record["requested_pairs"]),
        "question": pair["question"],
        "answer": pair["answer"],
    }


def _rebuild_synthetic_records_from_raw_outputs(
    raw_outputs_path: Path,
    chunk_records: list[dict[str, Any]],
    *,
    dedupe_successful_requests: bool,
) -> list[dict[str, Any]]:
    """Reparse cached raw outputs after JSON-parser improvements."""

    if not raw_outputs_path.exists():
        return []

    request_map = {_request_key(record): record for record in chunk_records}
    records: list[dict[str, Any]] = []
    satisfied_requests: set[tuple[str, int]] = set()
    with raw_outputs_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            raw_row = json.loads(line)
            key = str(raw_row["doc_id"]), int(raw_row["chunk_idx"])
            if dedupe_successful_requests and key in satisfied_requests:
                continue
            base_record = request_map.get(key)
            if base_record is None:
                continue
            requested_pairs = max(int(raw_row.get("requested_pairs", 0)), 0)
            if requested_pairs == 0:
                continue
            pairs = parse_output(str(raw_row.get("raw_output", "")))[:requested_pairs]
            if not pairs:
                continue
            if dedupe_successful_requests:
                satisfied_requests.add(key)
            record = {**base_record, "requested_pairs": requested_pairs}
            records.extend(_build_output_record(record, pair) for pair in pairs)
    return records


def _surrounding_context(
    record: dict[str, Any],
    document_chunks: dict[str, list[str]],
    radius: int,
) -> str | None:
    if radius <= 0:
        return None
    chunks = document_chunks.get(str(record["doc_id"]), [])
    center = int(record["chunk_idx"])
    start = max(0, center - radius)
    stop = min(len(chunks), center + radius + 1)
    neighbours = chunks[start:center] + chunks[center + 1 : stop]
    return "\n\n".join(neighbours) if neighbours else None


def _consume_outputs(
    request_records: list[dict[str, Any]],
    raw_outputs: list[str],
    *,
    raw_writer,
    synthetic_records: list[dict[str, Any]],
    requests_with_output: set[tuple[str, int]],
    retry_number: int | None = None,
) -> tuple[int, int, int]:
    """Parse one generation batch and append raw and structured records."""

    if len(request_records) != len(raw_outputs):
        raise RuntimeError(
            f"vLLM returned {len(raw_outputs)} outputs for "
            f"{len(request_records)} requests"
        )
    failed_calls = 0
    over_generated_calls = 0
    truncated_pairs = 0
    for record, raw_output in zip(request_records, raw_outputs):
        requested_pairs = int(record["requested_pairs"])
        raw_row = {
            "doc_id": record["doc_id"],
            "chunk_idx": record["chunk_idx"],
            "requested_pairs": requested_pairs,
            "raw_output": raw_output,
        }
        if retry_number is not None:
            raw_row["retry"] = retry_number
        raw_writer.write(raw_row)

        parsed_pairs = parse_output(raw_output)
        if len(parsed_pairs) > requested_pairs:
            over_generated_calls += 1
            truncated_pairs += len(parsed_pairs) - requested_pairs
        pairs = parsed_pairs[:requested_pairs]
        if not pairs:
            failed_calls += 1
            continue

        requests_with_output.add(_request_key(record))
        synthetic_records.extend(
            _build_output_record(record, pair) for pair in pairs
        )
    return failed_calls, over_generated_calls, truncated_pairs


def _clear_cuda_cache() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        gc.collect()


def synthesize_for_cluster_dir(
    instruct_model: str,
    eval_path: Path,
    eval_every_n: int,
    cluster_dir: Path,
    chunk_size: int,
    overlap_ratio: float,
    n_generations: int,
    *,
    synthetic_data_dir: str | None = None,
    data_dir: Path | None = None,
    overwrite: bool = False,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 256,
    max_model_len: int | None = None,
    dtype: str = "auto",
    enforce_eager: bool = False,
    attention_backend: str = "FLASH_ATTN",
    temperature: float = 0.7,
    top_p: float = 1.0,
    max_tokens: int = 512,
    pairs_per_call: int = 1,
    max_retries: int = 0,
    context_window_chunks: int = 0,
) -> None:
    """Generate fixed-count QA pairs for every document chunk using vLLM."""

    if data_dir is None and synthetic_data_dir is None:
        raise ValueError("Either data_dir or synthetic_data_dir must be provided.")
    n_generations = int(n_generations)
    pairs_per_call = int(pairs_per_call)
    max_retries = int(max_retries)
    context_window_chunks = int(context_window_chunks)
    if n_generations < 1:
        raise ValueError("n_generations must be at least 1")
    if pairs_per_call < 1:
        raise ValueError("pairs_per_call must be at least 1")
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if context_window_chunks < 0:
        raise ValueError("context_window_chunks must be non-negative")

    eval_path = Path(eval_path)
    cache_dir = find_or_create_cache_dir(
        data_dir=data_dir or Path("."),
        instruct_model=instruct_model,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
        n_generations=n_generations,
        pairs_per_call=pairs_per_call,
        context_window_chunks=context_window_chunks,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        eval_path=eval_path,
        eval_every_n=eval_every_n,
        synthetic_data_dir=synthetic_data_dir,
    )
    cache_path = cache_dir / "synthetic_pairs.jsonl"
    metadata_path = cache_dir / "metadata.json"
    raw_outputs_path = cache_dir / "raw_outputs.jsonl"

    training_examples = load_training_examples(eval_path, eval_every_n)
    doc_to_cluster = load_doc_to_cluster_map(cluster_dir)
    chunk_records, document_chunks = build_chunk_records(
        training_examples,
        doc_to_cluster,
        chunk_size=chunk_size,
        overlap_ratio=overlap_ratio,
    )
    if not chunk_records:
        raise RuntimeError(f"No document chunks were created from {eval_path.name}")
    for record in chunk_records:
        record["requested_pairs"] = pairs_per_call
    log.info("Synthesizing over %d document chunks", len(chunk_records))

    if cache_path.exists() and not overwrite:
        log.info("Reusing synthetic cache at %s", cache_path)
        with jsonlines.open(cache_path) as reader:
            synthetic_records = list(reader)
        for record in synthetic_records:
            record["task"] = QA_SHORT_TASK
        remap_cached_cluster_indices(synthetic_records, doc_to_cluster)

        reparsed_records = _rebuild_synthetic_records_from_raw_outputs(
            raw_outputs_path,
            chunk_records,
            dedupe_successful_requests=(n_generations == 1),
        )
        if len(reparsed_records) > len(synthetic_records):
            remap_cached_cluster_indices(reparsed_records, doc_to_cluster)
            log.warning(
                "The current JSON parser recovered %d cached records instead of %d; "
                "updating %s.",
                len(reparsed_records),
                len(synthetic_records),
                cache_path,
            )
            synthetic_records = reparsed_records
            with jsonlines.open(cache_path, mode="w") as writer:
                writer.write_all(synthetic_records)
            metadata = {}
            if metadata_path.exists():
                with open(metadata_path) as handle:
                    metadata = json.load(handle)
            metadata.update(
                {
                    "n_records": len(synthetic_records),
                    "reparsed_from_raw_outputs": True,
                }
            )
            with open(metadata_path, "w") as handle:
                json.dump(metadata, handle, indent=2)
    else:
        visible_gpu_count = count_visible_cuda_devices()
        if multiprocessing.get_start_method(allow_none=True) != "spawn":
            multiprocessing.set_start_method("spawn", force=True)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

        from vllm import LLM

        parallel_plan = resolve_vllm_parallelism(
            tensor_parallel_size,
            visible_gpu_count,
            forced_backend=os.environ.get("LORA_KV_CACHE_VLLM_DISTRIBUTED_BACKEND"),
        )
        effective_tensor_parallel_size = int(parallel_plan["tensor_parallel_size"])
        distributed_backend = parallel_plan["distributed_executor_backend"]
        if effective_tensor_parallel_size != int(
            parallel_plan["requested_tensor_parallel_size"]
        ):
            log.warning(
                "Requested tensor_parallel_size=%d but only %s CUDA device(s) are "
                "visible; using %d.",
                int(parallel_plan["requested_tensor_parallel_size"]),
                visible_gpu_count,
                effective_tensor_parallel_size,
            )

        llm_kwargs: dict[str, Any] = {
            "model": instruct_model,
            "tensor_parallel_size": effective_tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
        }
        if attention_backend:
            llm_kwargs["attention_config"] = {"backend": attention_backend}
        if distributed_backend:
            llm_kwargs["distributed_executor_backend"] = distributed_backend
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        log.info(
            "Starting vLLM model=%s tensor_parallel_size=%d visible_gpus=%s",
            instruct_model,
            effective_tensor_parallel_size,
            visible_gpu_count,
        )
        llm = None
        try:
            llm = _init_vllm_with_memory_race_retries(LLM, llm_kwargs)
            tokenizer = llm.get_tokenizer()
            prompts = [
                build_prompt(
                    tokenizer,
                    record["chunk_text"],
                    pairs_per_call,
                    surrounding_context=_surrounding_context(
                        record,
                        document_chunks,
                        context_window_chunks,
                    ),
                )
                for record in chunk_records
                for _ in range(n_generations)
            ]
            expanded_records = [
                record
                for record in chunk_records
                for _ in range(n_generations)
            ]
            log.info(
                "Generating %d prompts (%d chunks x %d generations)",
                len(prompts),
                len(chunk_records),
                n_generations,
            )

            synthetic_records: list[dict[str, Any]] = []
            requests_with_output: set[tuple[str, int]] = set()
            failed_calls = 0
            over_generated_calls = 0
            truncated_pairs = 0
            generation_calls = 0

            raw_outputs = generate_batch(
                llm,
                prompts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            generation_calls += len(raw_outputs)
            with jsonlines.open(raw_outputs_path, mode="w") as raw_writer:
                failed, over_generated, truncated = _consume_outputs(
                    expanded_records,
                    raw_outputs,
                    raw_writer=raw_writer,
                    synthetic_records=synthetic_records,
                    requests_with_output=requests_with_output,
                )
                failed_calls += failed
                over_generated_calls += over_generated
                truncated_pairs += truncated

            request_map = {_request_key(record): record for record in chunk_records}
            for retry_index in range(max_retries):
                failed_records = [
                    request_map[key]
                    for key in request_map
                    if key not in requests_with_output
                ]
                if not failed_records:
                    break
                log.info(
                    "Retry %d/%d for %d chunks with no parseable QA output",
                    retry_index + 1,
                    max_retries,
                    len(failed_records),
                )
                retry_prompts = [
                    build_prompt(
                        tokenizer,
                        record["chunk_text"],
                        pairs_per_call,
                        surrounding_context=_surrounding_context(
                            record,
                            document_chunks,
                            context_window_chunks,
                        ),
                    )
                    for record in failed_records
                    for _ in range(n_generations)
                ]
                retry_records = [
                    record
                    for record in failed_records
                    for _ in range(n_generations)
                ]
                retry_outputs = generate_batch(
                    llm,
                    retry_prompts,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                )
                generation_calls += len(retry_outputs)
                with jsonlines.open(raw_outputs_path, mode="a") as raw_writer:
                    failed, over_generated, truncated = _consume_outputs(
                        retry_records,
                        retry_outputs,
                        raw_writer=raw_writer,
                        synthetic_records=synthetic_records,
                        requests_with_output=requests_with_output,
                        retry_number=retry_index + 1,
                    )
                    failed_calls += failed
                    over_generated_calls += over_generated
                    truncated_pairs += truncated
        finally:
            if llm is not None:
                del llm
            _clear_cuda_cache()

        if not synthetic_records:
            raise RuntimeError(
                f"vLLM produced no parseable QA records with model {instruct_model!r}"
            )

        request_failures = len(chunk_records) - len(requests_with_output)
        if over_generated_calls:
            log.warning(
                "Truncated %d extra pairs across %d over-generated calls",
                truncated_pairs,
                over_generated_calls,
            )
        log.info(
            "Generated %d QA records; %d/%d chunks had no parseable output",
            len(synthetic_records),
            request_failures,
            len(chunk_records),
        )

        with jsonlines.open(cache_path, mode="w") as writer:
            writer.write_all(synthetic_records)
        metadata = {
            "task": QA_SHORT_TASK,
            "instruct_model": instruct_model,
            "chunk_size": chunk_size,
            "overlap_ratio": overlap_ratio,
            "n_generations": n_generations,
            "pairs_per_call": pairs_per_call,
            "context_window_chunks": context_window_chunks,
            "tensor_parallel_size": tensor_parallel_size,
            "effective_tensor_parallel_size": effective_tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": max_num_seqs,
            "max_model_len": max_model_len,
            "dtype": dtype,
            "enforce_eager": enforce_eager,
            "attention_backend": attention_backend,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "eval_file": eval_path.name,
            "eval_every_n": eval_every_n,
            "n_chunks": len(chunk_records),
            "n_records": len(synthetic_records),
            "n_generation_calls": generation_calls,
            "n_failed_calls": failed_calls,
            "n_requests_all_failed": request_failures,
            "max_retries": max_retries,
        }
        with open(metadata_path, "w") as handle:
            json.dump(metadata, handle, indent=2)
        log.info("Cached synthetic QA data at %s", cache_path)

    if not synthetic_records:
        raise RuntimeError("Synthetic QA cache is empty")
    scatter_synthetic_records(cluster_dir, synthetic_records)
