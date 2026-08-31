"""Step: finetune — train LoRA adapters in parallel across GPUs."""

import json
import logging
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from src.pipeline._utils import PROJECT_ROOT

log = logging.getLogger(__name__)

_MIXED_TRAINING_SCRIPT = PROJECT_ROOT / "src" / "pipeline" / "training" / "finetune_mixed.py"


def _normalize_lora_target_modules(value):
    """Preserve PEFT shorthands like ``all-linear`` while normalizing lists."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    modules = [str(item) for item in list(value) if str(item)]
    if len(modules) == 1 and modules[0].lower() == "all-linear":
        return "all-linear"
    return modules or None


def _extend_target_modules_arg(cmd: list[str], value) -> None:
    modules = _normalize_lora_target_modules(value)
    if not modules:
        return
    if isinstance(modules, str):
        cmd.extend(["--target_modules", modules])
    else:
        cmd.extend(["--target_modules"] + modules)


def _get_visible_gpu_ids(num_gpus: int) -> list[str]:
    """Return the caller's visible GPU identifiers in launch order."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        gpu_ids = [gpu_id.strip() for gpu_id in visible.split(",") if gpu_id.strip()]
        if gpu_ids:
            return gpu_ids
    return [str(gpu_id) for gpu_id in range(num_gpus)]


def _is_nullish(value) -> bool:
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def run(cfg: DictConfig, handler) -> None:
    """Fine-tune LoRA adapters on document chunks in parallel across GPUs."""
    objective = str(cfg.training.get("objective", "sft")).strip().lower()
    if objective != "sft":
        raise ValueError(f"Unknown training.objective={objective!r}; expected 'sft'.")

    import torch
    from src.pipeline.training.shared_a import (
        initialize_shared_A_matrices,
        sanitize_model_name as training_sanitize_model_name,
    )

    log.info("[finetune] Fine-tuning document adapters...")

    base_model = cfg.model.base_model
    adapter_dir = PROJECT_ROOT / cfg.pipeline.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cluster_dir = handler.get_cluster_dir(cfg)
    model_key = training_sanitize_model_name(base_model)
    max_seq_length = cfg.model.max_seq_length
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    visible_gpu_ids = _get_visible_gpu_ids(num_gpus)
    train_python = sys.executable
    mixed_sources_file: Path | None = None

    mixed_sources = cfg.training.get("mixed_sources", None)
    if not mixed_sources:
        raise ValueError("training.mixed_sources must be set")
    mixed_sources_payload = OmegaConf.to_container(mixed_sources, resolve=True)
    mixed_sources_file = adapter_dir / "mixed_sources.json"
    mixed_sources_file.write_text(json.dumps(mixed_sources_payload, indent=2))

    log.info("Detected %d GPUs for parallel training", num_gpus)
    log.info("Visible GPU IDs: %s", visible_gpu_ids)

    for seed in range(cfg.pipeline.seed, cfg.pipeline.seed + cfg.pipeline.num_seeds):
        _run_seed(
            cfg=cfg,
            seed=seed,
            base_model=base_model,
            adapter_dir=adapter_dir,
            cluster_dir=cluster_dir,
            model_key=model_key,
            max_seq_length=max_seq_length,
            num_gpus=num_gpus,
            visible_gpu_ids=visible_gpu_ids,
            train_python=train_python,
            initialize_shared_A_matrices=initialize_shared_A_matrices,
            mixed_sources_file=mixed_sources_file,
        )


def _run_seed(
    cfg,
    seed: int,
    base_model: str,
    adapter_dir: Path,
    cluster_dir: Path,
    model_key: str,
    max_seq_length: int,
    num_gpus: int,
    visible_gpu_ids: list[str],
    train_python: str,
    initialize_shared_A_matrices,
    mixed_sources_file: Path | None,
) -> None:
    shared_a_path = (
        adapter_dir / model_key / "sharedA" / f"seed_{seed}" / "shared_A.safetensors"
    )
    metadata_path = shared_a_path.parent / "shared_A_metadata.json"
    if (not shared_a_path.exists()) or metadata_path.exists():
        log.info("Preparing shared A matrices for seed %d", seed)
        layers = cfg.lora.get("layers_to_transform", None)
        initialize_shared_A_matrices(
            model_name=base_model,
            out_dir=str(adapter_dir),
            lora_r=cfg.lora.r,
            seed=42,
            training_seed=seed,
            layers_to_transform=list(layers) if layers else None,
            target_modules=_normalize_lora_target_modules(cfg.lora.get("target_modules")),
            init_method=str(cfg.lora.get("lora_init", "random")),
        )

    cluster_dirs = sorted(
        d for d in cluster_dir.iterdir() if d.is_dir() and d.name.isdigit()
    )

    def is_cluster_completed(cluster_id: int) -> bool:
        train_dir = (
            adapter_dir / model_key / "sharedA" / f"seed_{seed}" / f"cluster_{cluster_id}_train"
        )
        return bool(
            train_dir.exists()
            and list(train_dir.glob("checkpoint-*/adapter_model.safetensors"))
        )

    pending = [int(d.name) for d in cluster_dirs if not is_cluster_completed(int(d.name))]
    for cdir in cluster_dirs:
        cid = int(cdir.name)
        if cid not in pending:
            log.info("Cluster %d already completed, skipping", cid)

    if not pending:
        log.info("All clusters already completed for seed %d", seed)
        return

    log.info("Training %d clusters in parallel across %d GPUs (seed %d)", len(pending), num_gpus, seed)

    logs_dir = adapter_dir / model_key / "sharedA" / f"seed_{seed}" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    gpu_queue: queue.Queue = queue.Queue()
    for gpu_id in visible_gpu_ids:
        gpu_queue.put(gpu_id)

    def run_cluster(cluster_id: int) -> tuple[int, bool, str]:
        gpu_id = gpu_queue.get()
        try:
            return _run_cluster_subprocess(
                cfg=cfg,
                cluster_id=cluster_id,
                gpu_id=gpu_id,
                base_model=base_model,
                cluster_dir=cluster_dir,
                adapter_dir=adapter_dir,
                seed=seed,
                max_seq_length=max_seq_length,
                logs_dir=logs_dir,
                train_python=train_python,
                mixed_sources_file=mixed_sources_file,
            )
        finally:
            gpu_queue.put(gpu_id)

    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        futures = {executor.submit(run_cluster, cid): cid for cid in pending}
        failed_clusters: list[int] = []
        for future in as_completed(futures):
            cid = futures[future]
            try:
                _, success, _ = future.result()
                if not success:
                    log.error("Cluster %d training failed", cid)
                    failed_clusters.append(cid)
            except Exception as exc:
                log.error("Cluster %d raised exception: %s", cid, exc)
                failed_clusters.append(cid)

    if failed_clusters:
        failed = ", ".join(str(cid) for cid in sorted(failed_clusters))
        raise RuntimeError(f"Training failed for cluster(s): {failed}")

def _run_cluster_subprocess(
    cfg,
    cluster_id: int,
    gpu_id: str,
    base_model: str,
    cluster_dir: Path,
    adapter_dir: Path,
    seed: int,
    max_seq_length: int,
    logs_dir: Path,
    train_python: str,
    mixed_sources_file: Path | None = None,
) -> tuple[int, bool, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    report_to = list(cfg.training.get("report_to") or [])
    if report_to:
        env.pop("WANDB_DISABLED", None)
        env.pop("WANDB_MODE", None)
    else:
        env["WANDB_DISABLED"] = "true"
        env["WANDB_MODE"] = "disabled"

    log_file = logs_dir / f"cluster_{cluster_id}_gpu{gpu_id}.log"
    cmd = [
        train_python,
        str(_MIXED_TRAINING_SCRIPT),
        "--model_name", base_model,
        "--cluster_dir", str(cluster_dir),
        "--cluster_id", str(cluster_id),
        "--out_dir", str(adapter_dir),
        "--seed", str(seed),
        "--max_seq_length", str(max_seq_length),
        "--per_device_train_batch_size", str(cfg.training.per_device_train_batch_size),
        "--gradient_accumulation_steps", str(cfg.training.gradient_accumulation_steps),
        "--learning_rate", str(cfg.training.learning_rate),
        "--weight_decay", str(cfg.training.weight_decay),
        "--torch_dtype", str(cfg.training.get("torch_dtype", "auto")),
        "--num_train_epochs", str(cfg.training.num_train_epochs),
        "--save_epochs", str(cfg.training.save_epochs),
        "--lora_r", str(cfg.lora.r),
        "--lora_alpha", str(cfg.lora.alpha),
        "--lora_dropout", str(cfg.lora.dropout),
    ]
    if not _is_nullish(cfg.training.get("max_steps", None)):
        cmd.extend(["--max_steps", str(cfg.training.max_steps)])
    if mixed_sources_file is None:
        raise ValueError("No mixed_sources_file was prepared")
    cmd.extend(["--mixed_sources_file", str(mixed_sources_file)])
    logging_steps = cfg.training.get("logging_steps", None)
    if logging_steps is not None:
        cmd.extend(["--logging_steps", str(logging_steps)])
    early_stop_metric = cfg.training.get("early_stop_train_metric", None)
    early_stop_threshold = cfg.training.get("early_stop_train_threshold", None)
    if not _is_nullish(early_stop_metric) or not _is_nullish(early_stop_threshold):
        if _is_nullish(early_stop_metric) or _is_nullish(early_stop_threshold):
            raise ValueError(
                "Both training.early_stop_train_metric and "
                "training.early_stop_train_threshold must be set together."
            )
        cmd.extend(
            [
                "--early_stop_train_metric",
                str(early_stop_metric),
                "--early_stop_train_threshold",
                str(early_stop_threshold),
                "--early_stop_train_mode",
                str(cfg.training.get("early_stop_train_mode", "max")),
                "--early_stop_train_min_steps",
                str(cfg.training.get("early_stop_train_min_steps", 0)),
                "--early_stop_train_patience",
                str(cfg.training.get("early_stop_train_patience", 1)),
            ]
        )

    if cfg.training.get("disable_truncation", False):
        cmd.append("--disable_truncation")
    if cfg.training.get("fail_on_truncation", False):
        cmd.append("--fail_on_truncation")
    if report_to:
        cmd.extend(["--report_to"] + report_to)
    layers = cfg.lora.get("layers_to_transform", None)
    if layers:
        cmd.extend(["--lora_layers"] + [str(l) for l in layers])
    _extend_target_modules_arg(cmd, cfg.lora.get("target_modules"))
    cmd.append("--use_dora" if cfg.lora.get("use_dora", True) else "--no-use_dora")
    if cfg.lora.get("train_A", False):
        cmd.append("--train_lora_A")
    lora_init = str(cfg.lora.get("lora_init", "random"))
    cmd.extend(["--lora_init", lora_init])
    lora_b_init = str(cfg.lora.get("lora_B_init", "zero"))
    cmd.extend(["--lora_B_init", lora_b_init])
    resume_ckpt = cfg.pipeline.get("resume_from_checkpoint", None)
    if resume_ckpt:
        cmd.extend(["--resume_from_checkpoint", str(resume_ckpt)])

    log.info("Starting cluster %d on GPU %s (log: %s)", cluster_id, gpu_id, log_file)
    with open(log_file, "w") as fh:
        fh.write(f"=== Cluster {cluster_id} on GPU {gpu_id} ===\n")
        fh.write(f"Command: {' '.join(cmd)}\n{'=' * 60}\n\n")
        result = subprocess.run(
            cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, text=True, cwd=str(PROJECT_ROOT)
        )

    if result.returncode != 0:
        log.error("Cluster %d failed (see %s)", cluster_id, log_file)
        return (cluster_id, False, str(log_file))
    log.info("Cluster %d completed on GPU %s", cluster_id, gpu_id)
    return (cluster_id, True, str(log_file))
