"""Runtime for evaluation with document-specific adapters."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable

import torch

from .eval_runtime import (
    build_seed_outfile,
    _filter_or_raise_overlength_generative_df,
    list_seed_adapter_dirs,
    resolve_press_and_compression_ratio,
    resolve_cluster_dir_from_pipeline_config,
)
from .compression_config import compression_suffix_from_config
from .io_utils import make_result_entry
from .model_loading import get_model_dtype, move_model_to_eval_device
from .perf import EvalPerfStats, save_perf_stats
from .predictions import BASE_SCORE_ADAPTER_PREFILL_MODE
from .task_specs import GenerativeTaskSpec, resolve_eval_file

log = logging.getLogger(__name__)

FORCED_ADAPTER_METHOD = "forced_adapter"
FORCED_ADAPTER_BASEPREFILL_METHOD = "forced_adapter_baseprefill"
FORCED_ADAPTER_BASESCORE_ADAPTERPREFILL_METHOD = "forced_adapter_basescore_adapterprefill"
DOCUMENT_ADAPTER_METHODS = {
    FORCED_ADAPTER_METHOD,
    FORCED_ADAPTER_BASEPREFILL_METHOD,
    FORCED_ADAPTER_BASESCORE_ADAPTERPREFILL_METHOD,
}
OUTPUT_STEMS = {
    FORCED_ADAPTER_METHOD: "forced_adapter_analysis",
    FORCED_ADAPTER_BASEPREFILL_METHOD: "forced_adapter_baseprefill_analysis",
    FORCED_ADAPTER_BASESCORE_ADAPTERPREFILL_METHOD: (
        "forced_adapter_basescore_adapterprefill_analysis"
    ),
}


def _cluster_dir_lookup(seed_adapter_dirs: list[Path]) -> dict[int, Path]:
    lookup: dict[int, Path] = {}
    for slot_idx, train_dir in enumerate(seed_adapter_dirs):
        parts = train_dir.name.split("_")
        if len(parts) >= 3 and parts[0] == "cluster" and parts[-1] == "train":
            # Prefer the explicit cluster id encoded in names like
            # cluster_12_train; fall back to slot order for older layouts.
            lookup[int(parts[1])] = train_dir
        else:
            lookup[slot_idx] = train_dir
    return lookup


def evaluate_generative_task_document_adapter(
    *,
    spec: GenerativeTaskSpec,
    args,
    load_model_fn: Callable[..., Any],
    batch_get_predictions_fn: Callable[..., Any],
    get_max_length_fn: Callable[..., int],
    set_peft_adapter_scaling_fn: Callable[..., None],
    save_results_fn: Callable[[dict, str | Path], None],
    save_predictions_fn: Callable[[list, str | Path], None],
    find_latest_checkpoint_fn: Callable[[str | Path], Path | None],
) -> None:
    """Run document-adapter evaluation for a generative benchmark."""

    # Import PEFT lazily so base-model evaluation does not require it at import time.
    from peft import PeftModel

    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.out)
    eval_file = resolve_eval_file(
        args.data_dir,
        default_eval_filename=spec.default_eval_filename,
        eval_filename=getattr(args, "eval_file", None),
    )

    eval_every_n = getattr(args, "eval_every_n", 1)
    multi_gpu = getattr(args, "multi_gpu", False)
    shard_id = getattr(args, "shard_id", 0)
    num_shards = getattr(args, "num_shards", 1)
    max_new_tokens = getattr(args, "max_new_tokens", 64)
    max_length_arg = getattr(args, "max_length", None)
    disable_truncation = getattr(args, "disable_truncation", False)
    skip_overlength = getattr(args, "skip_overlength", False)
    batch_size = getattr(args, "batch_size", 4)
    adapter_scaling = getattr(args, "adapter_scaling", None)
    model_max_length = getattr(args, "model_max_length", None)
    include_context = getattr(args, "include_context", True)
    task = getattr(args, "task", "qa")
    compressed_generation_mode = getattr(args, "compressed_generation_mode", "context_prefill")
    method_name = getattr(args, "method", FORCED_ADAPTER_METHOD)
    if method_name not in DOCUMENT_ADAPTER_METHODS:
        raise ValueError(f"Unsupported document-adapter method: {method_name}")
    if method_name == FORCED_ADAPTER_BASEPREFILL_METHOD:
        adapter_prefill_mode = "base"
    elif method_name == FORCED_ADAPTER_BASESCORE_ADAPTERPREFILL_METHOD:
        adapter_prefill_mode = BASE_SCORE_ADAPTER_PREFILL_MODE
    else:
        adapter_prefill_mode = "adapter"

    press, compression_ratio = resolve_press_and_compression_ratio(
        args,
        benchmark_name=spec.benchmark_name,
    )
    compression_kwargs = getattr(args, "compression_kwargs", None)
    compression_suffix = compression_suffix_from_config(
        compression_kwargs,
        compression_ratio=compression_ratio,
    )

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        perf_stats = EvalPerfStats()
        perf_stats.set_metadata(
            benchmark=spec.benchmark_name,
            method=method_name,
            model_name=args.model_name,
            adapter_dir=str(adapter_dir),
            seed=int(seed),
            batch_size=int(batch_size),
            max_new_tokens=int(max_new_tokens),
            compressed_generation_mode=compressed_generation_mode,
            include_context=bool(include_context),
            task=task,
            compression_ratio=float(compression_ratio),
            compression_scope=(
                getattr(press, "compression_scope", None) if press is not None else None
            ),
        )
        out_stem = OUTPUT_STEMS[method_name] + compression_suffix
        outfile = build_seed_outfile(
            output_dir,
            seed=seed,
            out_stem=out_stem,
            shard_id=shard_id,
            num_shards=num_shards,
        )

        seed_adapter_dirs = list_seed_adapter_dirs(adapter_dir, seed)
        if not seed_adapter_dirs:
            raise FileNotFoundError(f"No adapters found in {adapter_dir / f'seed_{seed}'}")

        # Load the base model once per seed.  Each cluster adapter is then
        # wrapped onto this model for only the examples assigned to that cluster.
        with perf_stats.memory_stage("model_load"):
            model, tokenizer = load_model_fn(
                args.model_name,
                device=args.device,
                multi_gpu=multi_gpu,
                move_to_device=False,
                torch_dtype=getattr(args, "torch_dtype", "auto"),
                model_max_length=model_max_length,
            )
        base_dtype = get_model_dtype(model)
        max_length = None if disable_truncation else get_max_length_fn(tokenizer, max_length_arg)
        cluster_dir = resolve_cluster_dir_from_pipeline_config(adapter_dir)
        # The task-specific dataframe builder also attaches cluster_idx, either
        # from the eval file itself or from the training sidecar when available.
        df = spec.prepare_eval_df(
            tokenizer,
            eval_file,
            cluster_dir=cluster_dir,
            include_context=include_context,
            task=task,
        )
        if max_length is None:
            df = _filter_or_raise_overlength_generative_df(
                df,
                model=model,
                max_new_tokens=max_new_tokens,
                skip_overlength=skip_overlength,
            )
        if "cluster_idx" not in df.columns or not df["cluster_idx"].notna().any():
            raise ValueError(
                f"{spec.benchmark_name} document-adapter evaluation requires cluster_idx assignments."
            )

        from .io_utils import shard_indices

        # Shard before grouping so each worker evaluates a disjoint subset, then
        # group the local shard by document cluster id.
        df_eval = df.iloc[shard_indices(len(df), eval_every_n, shard_id, num_shards)].copy()
        cluster_id_to_dir = _cluster_dir_lookup(seed_adapter_dirs)
        cluster_keys = sorted(cluster_id_to_dir)
        total_metric_sums = {metric_name: 0.0 for metric_name in spec.metric_sum_fns}
        total_samples = 0
        all_preds: list[dict[str, Any]] = []
        pred_file = outfile.parent / f"{spec.prediction_basename}{compression_suffix}.json"

        for cluster_idx in cluster_keys:
            cluster_df = df_eval[df_eval["cluster_idx"] == cluster_idx].copy()
            if len(cluster_df) == 0:
                continue
            cluster_train_dir = cluster_id_to_dir[cluster_idx]

            adapter_path = find_latest_checkpoint_fn(cluster_train_dir)
            if adapter_path is None:
                log.warning(
                    "Skipping cluster %s: no checkpoint in %s",
                    cluster_idx,
                    cluster_train_dir,
                )
                continue

            # PeftModel.from_pretrained returns a PEFT wrapper around the base
            # model.  We delete it after this cluster so the next adapter can be
            # loaded without keeping all adapters resident at once.
            cluster_model = PeftModel.from_pretrained(model, adapter_path)
            if adapter_scaling is not None:
                set_peft_adapter_scaling_fn(cluster_model, adapter_scaling)
            if not multi_gpu:
                cluster_model = move_model_to_eval_device(
                    cluster_model,
                    args.device,
                    dtype=base_dtype,
                )
            cluster_model.eval()

            with perf_stats.inference_timer("inference_total"):
                _, count, preds = batch_get_predictions_fn(
                    cluster_model,
                    tokenizer,
                    cluster_df,
                    device=args.device,
                    batch_size=batch_size,
                    max_length=max_length,
                    collect_predictions=True,
                    max_new_tokens=max_new_tokens,
                    answer_scorer_fn=spec.answer_scorer_fn,
                    press=press,
                    compressed_generation_mode=compressed_generation_mode,
                    adapter_prefill_mode=adapter_prefill_mode,
                    perf_stats=perf_stats,
                )
            if spec.annotate_predictions_fn is not None:
                spec.annotate_predictions_fn(preds)
            metric_sums = {
                metric_name: float(metric_sum_fn(preds))
                for metric_name, metric_sum_fn in spec.metric_sum_fns.items()
            }

            all_preds.extend(preds)
            for metric_name, metric_sum in metric_sums.items():
                total_metric_sums[metric_name] += metric_sum
            total_samples += int(count)

            del cluster_model
            torch.cuda.empty_cache()
            gc.collect()

        metric_scores = {
            metric_name: (metric_sum / total_samples if total_samples > 0 else 0.0)
            for metric_name, metric_sum in total_metric_sums.items()
        }

        if all_preds:
            # Cluster evaluation runs examples out of original order; sort so
            # prediction files remain easy to compare against base runs.
            all_preds.sort(key=lambda row: row.get("index", 0))
        if all_preds:
            save_predictions_fn(all_preds, pred_file)

        metric_payload = {}
        for metric_name, metric_score in metric_scores.items():
            metric_payload[metric_name] = metric_score
            metric_payload[f"{metric_name}_sum"] = total_metric_sums[metric_name]

        results = {
            spec.benchmark_name: make_result_entry(
                total=int(total_samples),
                method=method_name,
                benchmark=spec.benchmark_name,
                primary_metric=spec.primary_metric_name,
                shard_id=shard_id if num_shards > 1 else None,
                num_shards=num_shards if num_shards > 1 else None,
                **metric_payload,
            )
        }
        save_results_fn(results, outfile)
        perf_stats.record_stage_memory_max(
            "inference_total",
            [
                "context_prefill_tokenize_to_device",
                "context_prefill_compress",
                "suffix_tokenize_to_device",
                "suffix_prefill",
                "decode",
                "full_prompt_tokenize_to_device",
                "full_prompt_generate",
            ],
        )
        output_tokens = sum(
            int(pred.get("confidence_token_count") or 0) for pred in all_preds
        )
        perf_stats.set_inference_summary(
            samples=int(total_samples),
            output_tokens=output_tokens,
            elapsed_s=float(perf_stats.inference_time_s.get("inference_total", 0.0)),
        )
        save_perf_stats(
            perf_stats,
            outfile.parent / f"perf_metrics{compression_suffix}.json",
        )
