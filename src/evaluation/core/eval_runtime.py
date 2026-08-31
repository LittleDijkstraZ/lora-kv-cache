"""Shared runtime for the paper's generative-QA evaluation."""

from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import torch

from .compression_config import (
    compression_algorithm_from_config,
    compression_suffix_from_config,
)
from .io_utils import make_result_entry, save_predictions, save_results
from .model_loading import get_model_dtype, move_model_to_eval_device
from .perf import EvalPerfStats, save_perf_stats
from .task_specs import GenerativeTaskSpec, resolve_eval_file

log = logging.getLogger(__name__)


def _model_context_limit(model) -> int | None:
    config = getattr(model, "config", None)
    value = getattr(config, "max_position_embeddings", None)
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 0 < value < 10**7 else None


def _filter_or_raise_overlength_generative_df(
    df,
    *,
    model,
    max_new_tokens: int,
    skip_overlength: bool,
):
    """Reject or explicitly skip prompts that exceed the model context."""

    if "num_tokens" not in df.columns:
        return df
    limit = _model_context_limit(model)
    if limit is None:
        return df

    required = df["num_tokens"].astype(int) + int(max_new_tokens or 0)
    over_mask = required > limit
    count = int(over_mask.sum())
    if count == 0:
        return df

    message = (
        f"{count}/{len(df)} examples require more than {limit} tokens "
        f"including max_new_tokens={max_new_tokens}."
    )
    if not skip_overlength:
        raise RuntimeError(
            message
            + " Reduce max_new_tokens or pass --skip_overlength to drop them explicitly."
        )
    filtered = df.loc[~over_mask].reset_index(drop=True)
    if filtered.empty:
        raise RuntimeError(message + " Skipping them would remove the entire evaluation set.")
    log.warning("%s Skipping those examples.", message)
    return filtered


def list_seed_adapter_dirs(adapter_dir: str | Path, seed: int) -> list[Path]:
    """Return document-adapter training directories for one seed."""

    seed_dir = Path(adapter_dir) / f"seed_{seed}"
    return sorted(path for path in seed_dir.glob("cluster_*_train") if path.is_dir())


def resolve_cluster_dir_from_pipeline_config(
    adapter_dir: str | Path | None,
) -> Path | None:
    """Recover the prepared document-cluster directory from a saved run config."""

    if adapter_dir is None:
        return None
    config_path = Path(adapter_dir).parent.parent / "pipeline_config.yaml"
    if not config_path.is_file():
        return None

    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(config_path)
        data_dir = OmegaConf.select(cfg, "data.data_dir")
        cluster_folder = OmegaConf.select(cfg, "data.cluster_docs_folder")
        configured_adapter_dir = OmegaConf.select(cfg, "pipeline.adapter_dir")
        if not data_dir or not cluster_folder:
            return None
        relative = Path(data_dir) / str(cluster_folder)
        if relative.is_absolute():
            return relative

        candidates: list[Path] = []
        if configured_adapter_dir:
            root = config_path.parent
            for _ in Path(str(configured_adapter_dir)).parts:
                root = root.parent
            candidates.append(root / relative)
        candidates.extend(parent / relative for parent in config_path.parents)
        for candidate in candidates:
            if (candidate / "id_to_cluster.json").is_file():
                return candidate
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not resolve cluster data from %s: %s", config_path, exc)
    return None


def build_press_from_compression_kwargs(compression_kwargs: dict[str, Any] | None):
    """Construct the selected Compactor or Expected Attention wrapper."""

    config = compression_kwargs or {}
    ratio = config.get("compression_ratio")
    if ratio is None or float(ratio) <= 0:
        return None
    try:
        from src.core.kvpress_wrapper import IndexedScorerPress
    except ModuleNotFoundError:  # pragma: no cover - direct script path
        from core.kvpress_wrapper import IndexedScorerPress

    return IndexedScorerPress(
        compression_ratio=float(ratio),
        chunk_size=int(config.get("chunk_size", 256)),
        sink_size_start=int(config.get("sink_size_start", 32)),
        sink_size_end=int(config.get("sink_size_end", 32)),
        compression_scope=str(config.get("compression_scope", "context_only")),
        context_min_keep_tokens=int(config.get("context_min_keep_tokens", 0)),
        compression_algorithm=compression_algorithm_from_config(config),
        n_future_positions=int(config.get("n_future_positions", 512)),
        n_sink=int(config.get("n_sink", 4)),
        use_covariance=bool(config.get("use_covariance", True)),
        use_vnorm=bool(config.get("use_vnorm", True)),
        epsilon=float(config.get("epsilon", 0.0)),
    )


def resolve_press_and_compression_ratio(
    args,
    *,
    benchmark_name: str,
) -> tuple[Any | None, float]:
    config = getattr(args, "compression_kwargs", None) or {}
    requested = float(config.get("compression_ratio") or 0.0)
    try:
        press = build_press_from_compression_kwargs(config)
    except ImportError as exc:
        if requested > 0:
            raise RuntimeError(
                f"{benchmark_name}: compression was requested but KVPress is unavailable"
            ) from exc
        return None, 0.0
    return press, requested if press is not None else 0.0


def build_seed_outfile(
    output_dir: str | Path,
    *,
    seed: int,
    out_stem: str,
    shard_id: int,
    num_shards: int,
) -> Path:
    path = Path(output_dir) / f"seed_{seed}" / f"{out_stem}.json"
    if num_shards > 1:
        path = (
            path.parent
            / "shards"
            / f"{path.stem}.shard_{shard_id}_of_{num_shards}.json"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _prediction_path(
    output_dir: str | Path,
    *,
    seed: int,
    basename: str,
    compression_suffix: str,
) -> Path:
    return Path(output_dir) / f"seed_{seed}" / f"{basename}{compression_suffix}.json"


def evaluate_generative_task(
    *,
    spec: GenerativeTaskSpec,
    load_model_fn: Callable[..., Any],
    batch_get_predictions_fn: Callable[..., Any],
    get_max_length_fn: Callable[..., int],
    model_name: str,
    method: str,
    seed: int,
    eval_file: str | Path,
    device: str = "cuda",
    batch_size: int = 4,
    max_length: int | None = None,
    disable_truncation: bool = False,
    max_new_tokens: int = 64,
    skip_overlength: bool = False,
    eval_every_n: int = 1,
    shard_id: int = 0,
    num_shards: int = 1,
    return_counts: bool = False,
    multi_gpu: bool = False,
    press=None,
    save_predictions_file: str | Path | None = None,
    include_context: bool = True,
    task: str = "qa_short",
    compressed_generation_mode: str = "context_prefill",
    save_perf_file: str | Path | None = None,
    torch_dtype: str = "auto",
    model_max_length: int | None = None,
):
    """Evaluate the base model on one prepared generative-QA dataset."""

    if method != "base":
        raise ValueError("The shared base evaluator only accepts method='base'.")
    perf = EvalPerfStats() if save_perf_file is not None else None
    if perf is not None:
        perf.set_metadata(
            benchmark=spec.benchmark_name,
            method=method,
            model_name=model_name,
            seed=seed,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            compressed_generation_mode=compressed_generation_mode,
            include_context=include_context,
            compression_ratio=float(getattr(press, "compression_ratio", 0.0)),
        )

    with perf.memory_stage("model_load") if perf is not None else nullcontext():
        model, tokenizer = load_model_fn(
            model_name,
            device=device,
            multi_gpu=multi_gpu,
            move_to_device=False,
            torch_dtype=torch_dtype,
            model_max_length=model_max_length,
        )
        if not multi_gpu:
            model = move_model_to_eval_device(
                model,
                device,
                dtype=get_model_dtype(model),
            )

    with perf.memory_stage("data_prepare") if perf is not None else nullcontext():
        effective_max_length = (
            None if disable_truncation else get_max_length_fn(tokenizer, max_length)
        )
        df = spec.prepare_eval_df(
            tokenizer,
            eval_file,
            include_context=include_context,
            task=task,
        )
        if effective_max_length is None:
            df = _filter_or_raise_overlength_generative_df(
                df,
                model=model,
                max_new_tokens=max_new_tokens,
                skip_overlength=skip_overlength,
            )

    with perf.inference_timer("inference_total") if perf is not None else nullcontext():
        _, total, predictions = batch_get_predictions_fn(
            model,
            tokenizer,
            df,
            device=device,
            batch_size=batch_size,
            press=press,
            max_length=effective_max_length,
            eval_every_n=eval_every_n,
            shard_id=shard_id,
            num_shards=num_shards,
            collect_predictions=True,
            max_new_tokens=max_new_tokens,
            answer_scorer_fn=spec.answer_scorer_fn,
            compressed_generation_mode=compressed_generation_mode,
            perf_stats=perf,
        )

    if spec.annotate_predictions_fn is not None:
        spec.annotate_predictions_fn(predictions)
    metric_sums = {
        name: float(metric_fn(predictions))
        for name, metric_fn in spec.metric_sum_fns.items()
    }
    metric_scores = {
        name: value / total if total else 0.0 for name, value in metric_sums.items()
    }
    if save_predictions_file is not None:
        save_predictions(predictions, save_predictions_file)
    if perf is not None:
        output_tokens = sum(
            int(row.get("confidence_token_count") or 0) for row in predictions
        )
        perf.set_inference_summary(
            samples=int(total),
            output_tokens=output_tokens,
            elapsed_s=float(perf.inference_time_s.get("inference_total", 0.0)),
        )
        save_perf_stats(perf, save_perf_file)

    if return_counts:
        return metric_scores, metric_sums, int(total)
    return metric_scores[spec.primary_metric_name]


def evaluate_generative_task_all_seeds(
    *,
    spec: GenerativeTaskSpec,
    args,
    evaluate_fn: Callable[..., Any],
) -> None:
    """Run the base evaluation for every requested seed."""

    output_dir = Path(args.out)
    eval_file = resolve_eval_file(
        args.data_dir,
        default_eval_filename=spec.default_eval_filename,
        eval_filename=getattr(args, "eval_file", None),
    )
    press, compression_ratio = resolve_press_and_compression_ratio(
        args,
        benchmark_name=spec.benchmark_name,
    )
    compression_config = getattr(args, "compression_kwargs", None)
    compression_suffix = compression_suffix_from_config(
        compression_config,
        compression_ratio=compression_ratio,
    )
    shard_id = getattr(args, "shard_id", 0)
    num_shards = getattr(args, "num_shards", 1)
    prediction_root = getattr(args, "save_predictions_dir", None) or output_dir

    for seed in range(args.start_seed, args.start_seed + args.num_seeds):
        outfile = build_seed_outfile(
            output_dir,
            seed=seed,
            out_stem=f"base_analysis{compression_suffix}",
            shard_id=shard_id,
            num_shards=num_shards,
        )
        prediction_file = _prediction_path(
            prediction_root,
            seed=seed,
            basename=spec.prediction_basename,
            compression_suffix=compression_suffix,
        )
        perf_file = outfile.parent / f"perf_metrics{compression_suffix}.json"
        metric_scores, metric_sums, total = evaluate_fn(
            model_name=args.model_name,
            method="base",
            seed=seed,
            eval_file=eval_file,
            device=args.device,
            batch_size=getattr(args, "batch_size", 4),
            max_length=getattr(args, "max_length", None),
            disable_truncation=getattr(args, "disable_truncation", False),
            max_new_tokens=getattr(args, "max_new_tokens", 64),
            skip_overlength=getattr(args, "skip_overlength", False),
            eval_every_n=getattr(args, "eval_every_n", 1),
            shard_id=shard_id,
            num_shards=num_shards,
            return_counts=True,
            multi_gpu=getattr(args, "multi_gpu", False),
            press=press,
            save_predictions_file=prediction_file,
            include_context=getattr(args, "include_context", True),
            task=getattr(args, "task", "qa_short"),
            compressed_generation_mode=getattr(
                args,
                "compressed_generation_mode",
                "context_prefill",
            ),
            save_perf_file=perf_file,
            torch_dtype=getattr(args, "torch_dtype", "auto"),
            model_max_length=getattr(args, "model_max_length", None),
        )

        metric_fields = {}
        for name, score in metric_scores.items():
            metric_fields[name] = score
            metric_fields[f"{name}_sum"] = metric_sums[name]
        save_results(
            {
                spec.benchmark_name: make_result_entry(
                    total=total,
                    method="base",
                    benchmark=spec.benchmark_name,
                    primary_metric=spec.primary_metric_name,
                    shard_id=shard_id if num_shards > 1 else None,
                    num_shards=num_shards if num_shards > 1 else None,
                    **metric_fields,
                )
            },
            outfile,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "_filter_or_raise_overlength_generative_df",
    "build_seed_outfile",
    "evaluate_generative_task",
    "evaluate_generative_task_all_seeds",
    "list_seed_adapter_dirs",
    "resolve_cluster_dir_from_pipeline_config",
    "resolve_press_and_compression_ratio",
]
