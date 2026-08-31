#!/usr/bin/env python3
"""Fine-tune LoRA adapters on mixed tokenized training sources."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)

from src.pipeline.training.shared_a import (
    initialize_shared_A_matrices,
    load_shared_A_matrices,
    load_shared_B_init_matrices,
    resolve_logging_steps,
    sanitize_model_name,
)
from src.pipeline.training.mixed_collator import MixedDataCollator
from src.pipeline.training.mixed_data import (
    build_mixed_training_examples,
    load_mixed_sources_config,
)
from src.pipeline.training.lora_targets import normalize_target_modules
from src.pipeline.training.model_compat import load_training_model, resolve_training_precision
from src.pipeline.training.training_args_compat import make_training_config


def _extract_logits(outputs) -> torch.Tensor | None:
    if isinstance(outputs, dict):
        return outputs.get("logits")
    logits = getattr(outputs, "logits", None)
    if logits is not None:
        return logits
    if isinstance(outputs, (tuple, list)) and len(outputs) > 1:
        return outputs[1]
    return None


@torch.no_grad()
def _compute_causal_lm_token_metrics(
    logits: torch.Tensor | None,
    labels: torch.Tensor | None,
    *,
    ignore_index: int = -100,
    entropy_chunk_size: int = 64,
) -> dict[str, float]:
    """Compute next-token metrics over supervised causal-LM labels."""

    if logits is None or labels is None:
        return {"correct": 0.0, "total": 0.0, "entropy_sum": 0.0}
    if logits.ndim != 3 or labels.ndim != 2:
        return {"correct": 0.0, "total": 0.0, "entropy_sum": 0.0}
    if logits.size(1) < 2 or labels.size(1) < 2:
        return {"correct": 0.0, "total": 0.0, "entropy_sum": 0.0}

    shift_logits = logits[:, :-1, :].detach()
    shift_labels = labels[:, 1:].detach()
    valid_mask = shift_labels.ne(ignore_index)

    total = int(valid_mask.sum().item())
    if total == 0:
        return {"correct": 0.0, "total": 0.0, "entropy_sum": 0.0}

    correct = 0.0
    entropy_sum = 0.0
    chunk_size = max(1, int(entropy_chunk_size))

    for row_logits, row_labels, row_mask in zip(shift_logits, shift_labels, valid_mask):
        valid_positions = row_mask.nonzero(as_tuple=False).flatten()
        if valid_positions.numel() == 0:
            continue

        for position_chunk in valid_positions.split(chunk_size):
            chunk_logits = row_logits.index_select(0, position_chunk)
            chunk_labels = row_labels.index_select(0, position_chunk)

            correct += float(chunk_logits.argmax(dim=-1).eq(chunk_labels).sum().item())

            log_probs = torch.nn.functional.log_softmax(chunk_logits.float(), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            entropy_sum += float(entropy.sum().item())

    return {"correct": correct, "total": float(total), "entropy_sum": entropy_sum}


class MixedMetricsTrainer(Trainer):
    """Trainer that records supervised-token accuracy and entropy."""

    def __init__(self, *args, metric_ignore_index: int = -100, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_ignore_index = metric_ignore_index
        self._reset_token_metric_buffer()

    def _reset_token_metric_buffer(self) -> None:
        self._token_metric_correct = 0.0
        self._token_metric_total = 0.0
        self._token_metric_entropy_sum = 0.0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ):
        labels = inputs.get("labels")
        loss, outputs = super().compute_loss(
            model,
            inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        if model.training and labels is not None:
            metrics = _compute_causal_lm_token_metrics(
                _extract_logits(outputs),
                labels,
                ignore_index=self.metric_ignore_index,
            )
            if metrics["total"] > 0:
                self._token_metric_correct += metrics["correct"]
                self._token_metric_total += metrics["total"]
                self._token_metric_entropy_sum += metrics["entropy_sum"]

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        if "loss" in logs and self._token_metric_total > 0:
            logs = dict(logs)
            logs["mean_token_accuracy"] = self._token_metric_correct / self._token_metric_total
            logs["entropy"] = self._token_metric_entropy_sum / self._token_metric_total
            self._reset_token_metric_buffer()
        super().log(logs, start_time=start_time)


class TrainMetricEarlyStoppingCallback(TrainerCallback):
    """Stop mixed training once a training log metric clears a threshold."""

    def __init__(
        self,
        *,
        metric: str,
        threshold: float,
        mode: str = "max",
        min_steps: int = 0,
        patience: int = 1,
    ):
        metric = str(metric).strip()
        if not metric:
            raise ValueError("early stop metric must be a non-empty string.")
        mode = str(mode).strip().lower()
        if mode not in {"max", "min"}:
            raise ValueError("early stop mode must be either 'max' or 'min'.")
        min_steps = int(min_steps)
        patience = int(patience)
        if min_steps < 0:
            raise ValueError("early stop min_steps must be >= 0.")
        if patience < 1:
            raise ValueError("early stop patience must be >= 1.")

        self.metric = metric
        self.threshold = float(threshold)
        self.mode = mode
        self.min_steps = min_steps
        self.patience = patience
        self._consecutive_hits = 0

    def _passes_threshold(self, value: float) -> bool:
        if self.mode == "max":
            return value >= self.threshold
        return value <= self.threshold

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ) -> TrainerControl:
        del args, kwargs
        if logs is None or int(state.global_step) < self.min_steps:
            return control
        if self.metric not in logs:
            return control

        try:
            value = float(logs[self.metric])
        except (TypeError, ValueError):
            return control

        if self._passes_threshold(value):
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        if self._consecutive_hits >= self.patience:
            print(
                "[EarlyStop] stopping at step "
                f"{state.global_step}: {self.metric}={value:.6g} "
                f"{'>=' if self.mode == 'max' else '<='} {self.threshold:.6g} "
                f"for {self._consecutive_hits}/{self.patience} log events",
                flush=True,
            )
            control.should_training_stop = True
        return control


def _ensure_final_checkpoint_for_global_step(trainer: Trainer, run_out_dir: Path) -> None:
    """Persist a final PEFT checkpoint when early stopping beats save_steps."""

    global_step = int(getattr(trainer.state, "global_step", 0) or 0)
    if global_step < 1:
        return

    checkpoint_dir = run_out_dir / f"checkpoint-{global_step}"
    adapter_path = checkpoint_dir / "adapter_model.safetensors"
    if adapter_path.exists():
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"[EarlyStop] Saving final checkpoint -> {checkpoint_dir}", flush=True)
    trainer.save_model(str(checkpoint_dir))


def run_once(
    model_name: str,
    cluster_dir: Path,
    cluster_id: int,
    out_dir: str,
    max_seq_length: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    weight_decay: float,
    num_train_epochs: int,
    save_epochs: int,
    max_steps: int | None,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    seed: int,
    mixed_sources: list[dict],
    shared_A_state: dict | None = None,
    shared_B_init_state: dict | None = None,
    resume_from_checkpoint: str | None = None,
    layers_to_transform: list[int] | None = None,
    logging_steps: int | None = None,
    report_to: list[str] | None = None,
    lora_B_init: str = "zero",
    target_modules: list[str] | None = None,
    use_dora: bool = False,
    train_lora_A: bool = False,
    disable_truncation: bool = False,
    fail_on_truncation: bool = False,
    torch_dtype: str = "auto",
    early_stop_train_metric: str | None = None,
    early_stop_train_threshold: float | None = None,
    early_stop_train_mode: str = "max",
    early_stop_train_min_steps: int = 0,
    early_stop_train_patience: int = 1,
) -> None:
    """Train one adapter on one cluster using a mixed tokenized dataset."""

    hf_token = os.environ.get("HF_AUTH_TOKEN")
    precision = resolve_training_precision(model_name, requested=torch_dtype, token=hf_token)
    print(
        f"[Cluster {cluster_id}] Training precision: dtype={precision.torch_dtype}, "
        f"fp16={precision.fp16}, bf16={precision.bf16}, source={precision.source}",
        flush=True,
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mixed_examples, mixed_manifest = build_mixed_training_examples(
        cluster_dir=cluster_dir,
        cluster_id=cluster_id,
        tokenizer=tokenizer,
        mixed_sources=mixed_sources,
        max_seq_length=max_seq_length,
        seed=seed,
        disable_truncation=disable_truncation,
        fail_on_truncation=fail_on_truncation,
    )
    if not mixed_examples:
        print(f"[Cluster {cluster_id}] No mixed training examples found, skipping")
        return

    print(f"[Cluster {cluster_id}] Mixed epoch size: {mixed_manifest['mixed_epoch_size']}")
    for source_manifest in mixed_manifest["sources"]:
        print(
            f"[Cluster {cluster_id}] Source {source_manifest['name']}: "
            f"{source_manifest['final_example_count']} examples -> "
            f"{source_manifest['mixed_target_examples']} mixed samples"
        )

    model = load_training_model(model_name, torch_dtype=precision.torch_dtype, token=hf_token)
    model.config.use_cache = False

    dataset = Dataset.from_list(mixed_examples)
    dataset = dataset.shuffle(seed=seed)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=normalize_target_modules(target_modules)
        or ["gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers_to_transform if layers_to_transform else None,
        lora_dropout=lora_dropout,
        bias="none",
        use_dora=use_dora,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if shared_A_state is not None:
        print(f"[Cluster {cluster_id}] Applying shared A matrices...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_A" in name and name in shared_A_state:
                    param.copy_(shared_A_state[name].to(param.device))

    if lora_B_init == "pissa" and shared_B_init_state is not None:
        print(f"[Cluster {cluster_id}] Applying PiSSA B_init (U_r) matrices...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "lora_B" in name and name in shared_B_init_state:
                    param.copy_(shared_B_init_state[name].to(param.device))

    if not train_lora_A:
        for name, param in model.named_parameters():
            if "lora_A" in name:
                param.requires_grad = False

    model_key = sanitize_model_name(model_name)
    run_out_dir = Path(out_dir) / model_key / "sharedA" / f"seed_{seed}" / f"cluster_{cluster_id}_train"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    (run_out_dir / "mixed_source_manifest.json").write_text(json.dumps(mixed_manifest, indent=2))

    num_samples = len(dataset)
    steps_per_epoch = max(
        1,
        (num_samples + (per_device_train_batch_size * gradient_accumulation_steps) - 1)
        // (per_device_train_batch_size * gradient_accumulation_steps),
    )
    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError("--max_steps must be >= 1 when set.")
    total_training_steps = (
        max_steps if max_steps is not None else max(1, steps_per_epoch * int(num_train_epochs))
    )
    save_steps = total_training_steps if max_steps is not None else steps_per_epoch * save_epochs
    total_checkpoints = 1 if max_steps is not None else max(1, num_train_epochs // save_epochs)
    effective_logging_steps = resolve_logging_steps(
        total_training_steps=total_training_steps,
        configured_logging_steps=logging_steps,
    )

    print(f"[Cluster {cluster_id}] Mixed dataset size: {num_samples}")
    print(f"[Cluster {cluster_id}] Steps per epoch: {steps_per_epoch}, save_steps={save_steps}")
    print(
        f"[Cluster {cluster_id}] Total training steps: {total_training_steps}, "
        f"logging every {effective_logging_steps} steps"
    )

    training_args = make_training_config(
        TrainingArguments,
        output_dir=str(run_out_dir),
        do_train=True,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps if max_steps is not None else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=seed,
        fp16=precision.fp16,
        bf16=precision.bf16,
        logging_strategy="steps",
        logging_steps=effective_logging_steps,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=total_checkpoints,
        group_by_length=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        torch_empty_cache_steps=100,
        report_to=list(report_to or []),
    )

    callbacks: list[TrainerCallback] = []
    if early_stop_train_metric is not None or early_stop_train_threshold is not None:
        if early_stop_train_metric is None or early_stop_train_threshold is None:
            raise ValueError(
                "early_stop_train_metric and early_stop_train_threshold must be set together."
            )
        early_stop_callback = TrainMetricEarlyStoppingCallback(
            metric=early_stop_train_metric,
            threshold=float(early_stop_train_threshold),
            mode=early_stop_train_mode,
            min_steps=early_stop_train_min_steps,
            patience=early_stop_train_patience,
        )
        callbacks.append(early_stop_callback)
        print(
            "[EarlyStop] enabled: "
            f"metric={early_stop_callback.metric}, "
            f"threshold={early_stop_callback.threshold}, "
            f"mode={early_stop_callback.mode}, "
            f"min_steps={early_stop_callback.min_steps}, "
            f"patience={early_stop_callback.patience}",
            flush=True,
        )

    trainer = MixedMetricsTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=MixedDataCollator(pad_token_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if callbacks:
        _ensure_final_checkpoint_for_global_step(trainer, run_out_dir)

    del model
    del trainer
    del dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune on mixed tokenized datasets")
    p.add_argument("--model_name", required=True, help="Base model checkpoint")
    p.add_argument(
        "--cluster_dir",
        type=str,
        required=True,
        help="Directory containing cluster folders (0/, 1/, ...)",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cluster_id", type=int, default=None, help="Train specific cluster only")
    p.add_argument("--mixed_sources_file", type=str, required=True)

    p.add_argument("--max_seq_length", type=int, default=32768)
    p.add_argument("--num_train_epochs", type=int, default=4)
    p.add_argument("--save_epochs", type=int, default=4, help="Save checkpoint every N epochs")
    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional fixed optimizer-step budget. Overrides epoch-derived length when set.",
    )
    p.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Trainer logging cadence in optimizer steps. Default None picks an adaptive value.",
    )
    p.add_argument(
        "--early_stop_train_metric",
        type=str,
        default=None,
        help="Optional training log metric for train-only early stopping.",
    )
    p.add_argument(
        "--early_stop_train_threshold",
        type=float,
        default=None,
        help="Metric threshold for train-only early stopping.",
    )
    p.add_argument(
        "--early_stop_train_mode",
        type=str,
        choices=["max", "min"],
        default="max",
        help="Use max for metric >= threshold, min for metric <= threshold.",
    )
    p.add_argument(
        "--early_stop_train_min_steps",
        type=int,
        default=0,
        help="Do not early-stop before this optimizer step.",
    )
    p.add_argument(
        "--early_stop_train_patience",
        type=int,
        default=1,
        help="Number of consecutive matching log events before stopping.",
    )
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.001)
    p.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        help="Training dtype override: auto, bfloat16/bf16, float16/fp16, or float32/fp32.",
    )

    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--lora_layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to apply LoRA to (default: all layers)",
    )
    p.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from",
    )
    p.add_argument(
        "--disable_truncation",
        action="store_true",
        default=False,
        help=(
            "Do not allow mixed examples to exceed --max_seq_length. "
            "Any overlength example raises instead of being truncated."
        ),
    )
    p.add_argument(
        "--fail_on_truncation",
        action="store_true",
        default=False,
        help=(
            "Raise immediately if a mixed example would exceed --max_seq_length. "
            "In-range examples are left unchanged."
        ),
    )
    p.add_argument(
        "--report_to",
        type=str,
        nargs="+",
        default=[],
        help="Optional trainer integrations to report to.",
    )
    p.add_argument(
        "--lora_init",
        type=str,
        default="pissa",
        choices=["random", "pissa"],
        help="How to initialise the shared A matrix.",
    )
    p.add_argument(
        "--lora_B_init",
        type=str,
        default="zero",
        choices=["zero", "pissa"],
        help="How to initialise B matrices.",
    )
    p.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["gate_proj", "up_proj", "down_proj"],
        help="LoRA target module names.",
    )
    p.add_argument(
        "--use_dora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable DoRA (Weight-Decomposed Low-Rank Adaptation).",
    )
    p.add_argument(
        "--train_lora_A",
        action="store_true",
        default=False,
        help="Allow LoRA A matrices to update during training instead of freezing them.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    cluster_base = Path(args.cluster_dir)
    if not cluster_base.exists():
        raise FileNotFoundError(f"Cluster directory not found: {cluster_base}")

    cluster_dirs = sorted([d for d in cluster_base.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Found {len(cluster_dirs)} cluster directories")

    mixed_sources = load_mixed_sources_config(args.mixed_sources_file)

    model_key = sanitize_model_name(args.model_name)
    shared_A_path = Path(args.out_dir) / model_key / "sharedA" / f"seed_{args.seed}" / "shared_A.safetensors"
    target_modules = normalize_target_modules(args.target_modules) or [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    print("Preparing shared A matrices...")
    initialize_shared_A_matrices(
        model_name=args.model_name,
        out_dir=args.out_dir,
        lora_r=args.lora_r,
        seed=42,
        training_seed=args.seed,
        target_modules=target_modules,
        layers_to_transform=args.lora_layers,
        init_method=args.lora_init,
    )

    shared_A_state = load_shared_A_matrices(shared_A_path)
    print(f"Loaded {len(shared_A_state)} shared A matrices")
    shared_B_init_state = load_shared_B_init_matrices(shared_A_path)
    if shared_B_init_state is not None:
        print(f"Loaded {len(shared_B_init_state)} shared B_init matrices")

    for cluster_dir in cluster_dirs:
        cluster_id = int(cluster_dir.name)

        if args.cluster_id is not None and cluster_id != args.cluster_id:
            continue

        print(f"\n{'=' * 60}")
        print(f"Training cluster {cluster_id}: {cluster_dir}")
        print(f"{'=' * 60}")

        run_once(
            model_name=args.model_name,
            cluster_dir=cluster_dir,
            cluster_id=cluster_id,
            out_dir=args.out_dir,
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_train_epochs=args.num_train_epochs,
            save_epochs=args.save_epochs,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            seed=args.seed,
            mixed_sources=mixed_sources,
            shared_A_state=shared_A_state,
            shared_B_init_state=shared_B_init_state,
            resume_from_checkpoint=args.resume_from_checkpoint,
            layers_to_transform=args.lora_layers,
            report_to=args.report_to,
            lora_B_init=args.lora_B_init,
            target_modules=target_modules,
            use_dora=args.use_dora,
            train_lora_A=bool(args.train_lora_A),
            disable_truncation=args.disable_truncation,
            fail_on_truncation=args.fail_on_truncation,
            torch_dtype=args.torch_dtype,
            early_stop_train_metric=args.early_stop_train_metric,
            early_stop_train_threshold=args.early_stop_train_threshold,
            early_stop_train_mode=args.early_stop_train_mode,
            early_stop_train_min_steps=args.early_stop_train_min_steps,
            early_stop_train_patience=args.early_stop_train_patience,
        )


if __name__ == "__main__":
    main()
