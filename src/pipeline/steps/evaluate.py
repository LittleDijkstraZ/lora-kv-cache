"""Step: evaluate — run evaluation with the instruct model and trained adapters."""

import json
import logging
import subprocess
import sys
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from src.pipeline._utils import (
    PROJECT_ROOT,
    sanitize_model_name,
    resolve_eval_methods,
    resolve_results_dir,
)

log = logging.getLogger(__name__)


def run(cfg: DictConfig, handler) -> None:
    """Evaluate with the instruct model and trained adapters."""
    dataset = cfg.pipeline.get("dataset", "narrativeqa")
    benchmark = cfg.evaluate.get("benchmark", None) or dataset
    eval_methods = resolve_eval_methods(cfg)
    log.info(
        "[evaluate] Evaluating dataset %s as benchmark %s with methods: %s",
        dataset,
        benchmark,
        eval_methods,
    )

    train_model_subdir = sanitize_model_name(cfg.model.base_model)
    full_adapter_dir = Path(cfg.pipeline.adapter_dir) / train_model_subdir

    resolved_results_dir = resolve_results_dir(
        cfg,
        default_name=Path(cfg.pipeline.adapter_dir).name,
        project_root=PROJECT_ROOT,
    )
    with open_dict(cfg):
        cfg.evaluate.results_dir = str(resolved_results_dir)

    cmd = [
        sys.executable,
        "-m",
        "src.evaluation.cli",
        "--adapter_dir", str(full_adapter_dir),
        "--model_name", cfg.model.instruct_model,
        "--benchmark", benchmark,
        "--batch_size", str(cfg.evaluate.batch_size),
        "--torch_dtype", str(cfg.evaluate.get("torch_dtype", "auto")),
        "--eval_every_n", str(cfg.evaluate.eval_every_n),
        "--results_dir", str(resolved_results_dir),
    ]
    no_compression_batch_size = cfg.evaluate.get("no_compression_batch_size", None)
    if no_compression_batch_size is not None:
        cmd.extend(["--no_compression_batch_size", str(no_compression_batch_size)])

    if cfg.evaluate.get("parallel_gpus", True):
        cmd.append("--parallel_gpus")
    if cfg.evaluate.get("multi_gpu", False):
        cmd.append("--multi_gpu")
    if eval_methods:
        cmd.extend(["--methods", *eval_methods])
    if cfg.evaluate.skip_existing:
        cmd.append("--skip_existing")

    eval_file = cfg.evaluate.get("eval_file", None)
    if eval_file:
        cmd.extend(["--eval_file", str(eval_file)])

    evaluate_task = cfg.evaluate.get("task", None)
    if evaluate_task:
        cmd.extend(["--task", str(evaluate_task)])

    max_length = cfg.evaluate.get("max_length", None)
    if max_length is not None:
        cmd.extend(["--max_length", str(max_length)])

    model_max_length = cfg.evaluate.get("model_max_length", None)
    if model_max_length is not None:
        cmd.extend(["--model_max_length", str(model_max_length)])

    max_new_tokens = cfg.evaluate.get("max_new_tokens", None)
    if max_new_tokens is not None:
        cmd.extend(["--max_new_tokens", str(max_new_tokens)])

    if cfg.evaluate.get("disable_truncation", False):
        cmd.append("--disable_truncation")

    if cfg.evaluate.get("skip_overlength", False):
        cmd.append("--skip_overlength")

    compressed_generation_mode = cfg.evaluate.get("compressed_generation_mode", "context_prefill")
    # Forward the effective value so standalone evaluation receives the same setting.
    cmd.extend(["--compressed_generation_mode", str(compressed_generation_mode)])

    adapter_scaling = cfg.evaluate.get("adapter_scaling", 2.0)
    if adapter_scaling != 2.0:
        cmd.extend(["--adapter_scaling", str(adapter_scaling)])

    compression_configs = list(cfg.evaluate.get("compression_configs") or [])
    if compression_configs:
        # OmegaConf parses YAML `none` as Python None; the evaluation CLI expects the string "none".
        normalized = [
            "none" if (c is None or c == "none") else dict(c)
            for c in compression_configs
        ]
        cmd.extend(["--compression_configs", json.dumps(normalized)])

    context_modes = list(cfg.evaluate.get("context_modes") or [])
    if context_modes:
        cmd.extend(["--context_modes"] + context_modes)

    log.info("Evaluation command: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("Evaluation failed with return code %d", result.returncode)
        raise RuntimeError(f"Evaluation failed with return code {result.returncode}")
    else:
        log.info("Evaluation completed successfully")

    if resolved_results_dir.exists():
        OmegaConf.save(cfg, resolved_results_dir / "pipeline_config.yaml")
        log.info("Saved pipeline_config.yaml -> %s", resolved_results_dir)
    else:
        log.warning(
            "Results dir missing after successful evaluation; skipping pipeline_config snapshot: %s",
            resolved_results_dir,
        )
