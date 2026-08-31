"""Subprocess orchestration for the paper's evaluation matrix."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.evaluation.core.compression_config import (
    compression_label_from_config,
    compression_ratio_from_config,
    compression_suffix_from_config,
    normalize_compression_config,
)


log = logging.getLogger(__name__)

SUPPORTED_BENCHMARKS = {"longhealth", "narrativeqa"}
SUPPORTED_METHODS = (
    "base",
    "forced_adapter",
    "forced_adapter_baseprefill",
    "forced_adapter_basescore_adapterprefill",
)
OUTPUT_STEMS = {
    "base": "base_analysis",
    "forced_adapter": "forced_adapter_analysis",
    "forced_adapter_baseprefill": "forced_adapter_baseprefill_analysis",
    "forced_adapter_basescore_adapterprefill": (
        "forced_adapter_basescore_adapterprefill_analysis"
    ),
}


@dataclass
class EvalRunConfig:
    benchmark: str
    model_name: str
    adapter_path: str
    data_dir: str
    results_dir: str
    seed: int = 0
    num_seeds: int = 1
    batch_size: int = 1
    no_compression_batch_size: int | None = None
    torch_dtype: str = "auto"
    methods: list[str] = field(default_factory=lambda: list(SUPPORTED_METHODS))
    compression_configs: list[Any] = field(default_factory=lambda: ["none"])
    eval_every_n: int = 1
    skip_existing: bool = False
    multi_gpu: bool = False
    parallel_gpus: bool = False
    context_modes: list[str] = field(default_factory=lambda: ["with_context"])
    extra_eval_args: list[str] = field(default_factory=list)
    eval_script: str = ""
    project_root: str = ""


@dataclass(frozen=True)
class EvalTask:
    method: str
    compression_config: Any
    context_mode: str


def _project_root(cfg: EvalRunConfig) -> Path:
    return Path(cfg.project_root) if cfg.project_root else Path(__file__).resolve().parents[2]


def _eval_script(cfg: EvalRunConfig) -> Path:
    if cfg.eval_script:
        return Path(cfg.eval_script)
    return _project_root(cfg) / "src" / "evaluation" / "eval_method_all_compression.py"


def _output_subdir(task: EvalTask) -> str:
    if task.context_mode == "no_context":
        return f"{task.method}_no_context"
    return task.method


def _output_filename(task: EvalTask) -> str:
    suffix = compression_suffix_from_config(task.compression_config)
    return f"{OUTPUT_STEMS[task.method]}{suffix}.json"


def _should_run(task: EvalTask) -> bool:
    ratio = compression_ratio_from_config(task.compression_config)
    if task.context_mode == "no_context" and ratio > 0:
        return False
    if task.method == "forced_adapter_baseprefill" and task.context_mode == "no_context":
        return False
    if task.method == "forced_adapter_basescore_adapterprefill":
        return task.context_mode == "with_context" and ratio > 0
    return True


def _tasks(cfg: EvalRunConfig) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    seen_no_context: set[str] = set()
    for comp_cfg in cfg.compression_configs:
        for method in cfg.methods:
            for context_mode in cfg.context_modes:
                task = EvalTask(method, comp_cfg, context_mode)
                if not _should_run(task):
                    continue
                if context_mode == "no_context":
                    if method in seen_no_context:
                        continue
                    seen_no_context.add(method)
                tasks.append(task)
    return tasks


def _task_complete(cfg: EvalRunConfig, results_dir: Path, task: EvalTask) -> bool:
    filename = _output_filename(task)
    subdir = _output_subdir(task)
    return all(
        (results_dir / subdir / f"seed_{seed}" / filename).is_file()
        for seed in range(cfg.seed, cfg.seed + cfg.num_seeds)
    )


def _batch_size(cfg: EvalRunConfig, task: EvalTask) -> int:
    if (
        cfg.no_compression_batch_size is not None
        and task.context_mode == "with_context"
        and compression_ratio_from_config(task.compression_config) == 0
    ):
        return int(cfg.no_compression_batch_size)
    return int(cfg.batch_size)


def _command(cfg: EvalRunConfig, task: EvalTask, results_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(_eval_script(cfg)),
        "--method",
        task.method,
        "--model_name",
        cfg.model_name,
        "--adapter_dir",
        cfg.adapter_path,
        "--data_dir",
        cfg.data_dir,
        "--start_seed",
        str(cfg.seed),
        "--num_seeds",
        str(cfg.num_seeds),
        "--out",
        str(results_dir / _output_subdir(task)),
        "--batch_size",
        str(_batch_size(cfg, task)),
        "--torch_dtype",
        cfg.torch_dtype,
        "--benchmark",
        cfg.benchmark,
    ]
    ratio = compression_ratio_from_config(task.compression_config)
    if ratio > 0:
        cmd.extend(
            [
                "--compression_config_json",
                json.dumps(normalize_compression_config(task.compression_config), sort_keys=True),
            ]
        )
    if cfg.eval_every_n > 1:
        cmd.extend(["--eval_every_n", str(cfg.eval_every_n)])
    if cfg.multi_gpu:
        cmd.append("--multi_gpu")
    if task.context_mode == "no_context":
        cmd.append("--no_context")
    cmd.extend(cfg.extra_eval_args)
    return cmd


def _gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0
    return [str(index) for index in range(count)] or ["0"]


def _run_task(
    cfg: EvalRunConfig,
    task: EvalTask,
    results_dir: Path,
    gpu_id: str | None,
) -> None:
    command = _command(cfg, task, results_dir)
    env = os.environ.copy()
    root = _project_root(cfg)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    label = (
        f"{task.method}/{task.context_mode}/"
        f"{compression_label_from_config(task.compression_config)}"
    )
    log.info("Starting %s", label)
    subprocess.run(command, cwd=root, env=env, check=True)
    log.info("Completed %s", label)


def _validate(cfg: EvalRunConfig) -> None:
    if cfg.benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {cfg.benchmark}")
    unknown = sorted(set(cfg.methods) - set(SUPPORTED_METHODS))
    if unknown:
        raise ValueError(f"Unsupported evaluation methods: {unknown}")
    if not cfg.compression_configs:
        raise ValueError("compression_configs must not be empty")
    if not cfg.context_modes or not set(cfg.context_modes) <= {"with_context", "no_context"}:
        raise ValueError("context_modes must contain with_context and/or no_context")
    adapter_methods = set(cfg.methods) - {"base"}
    if adapter_methods and not Path(cfg.adapter_path).is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {cfg.adapter_path}")


def run_evaluation(cfg: EvalRunConfig) -> None:
    _validate(cfg)
    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(cfg)
    config_payload["compression_configs"] = [
        "none" if value is None or value == "none" else dict(value)
        for value in cfg.compression_configs
    ]
    (results_dir / "eval_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n"
    )

    tasks = [
        task
        for task in _tasks(cfg)
        if not (cfg.skip_existing and _task_complete(cfg, results_dir, task))
    ]
    if not tasks:
        log.info("All requested evaluation tasks already exist.")
        return

    if not cfg.parallel_gpus or len(tasks) == 1:
        for task in tasks:
            _run_task(cfg, task, results_dir, None)
        return

    gpu_queue: queue.Queue[str] = queue.Queue()
    gpu_ids = _gpu_ids()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    def run_on_available_gpu(task: EvalTask) -> None:
        gpu_id = gpu_queue.get()
        try:
            _run_task(cfg, task, results_dir, gpu_id)
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(run_on_available_gpu, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
