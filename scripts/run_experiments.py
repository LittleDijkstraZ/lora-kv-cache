#!/usr/bin/env python3
"""Plan or launch the experiments reported in the paper."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass


MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
}
DATASETS = ("narrativeqa", "longhealth")
GROUPS = ("main", "training-formats", "compression-methods", "target-modules")


@dataclass(frozen=True)
class Job:
    group: str
    dataset: str
    model: str
    name: str
    config: str
    step: str
    adapter_dir: str
    results_dir: str
    overrides: tuple[str, ...] = ()

    def command(self) -> list[str]:
        model_id = MODELS[self.model]
        return [
            sys.executable,
            "-m",
            "src.pipeline.cli",
            "--config-name",
            self.config,
            f"step={self.step}",
            f"model.base_model={model_id}",
            f"model.instruct_model={model_id}",
            f"pipeline.adapter_dir={self.adapter_dir}",
            f"evaluate.results_dir={self.results_dir}",
            *self.overrides,
        ]


def main_job(dataset: str, model: str) -> Job:
    return Job(
        group="main",
        dataset=dataset,
        model=model,
        name="qa",
        config=dataset,
        step="all",
        adapter_dir=f"adapters/{dataset}/{model}/qa",
        results_dir=f"results/main/{dataset}/{model}/qa",
    )


def target_module_jobs(dataset: str, model: str) -> list[Job]:
    variants = {
        "attention": "[q_proj,k_proj,v_proj,o_proj]",
        "all-linear": "[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]",
    }
    return [
        Job(
            group="target-modules",
            dataset=dataset,
            model=model,
            name=name,
            config=dataset,
            step="all",
            adapter_dir=f"adapters/{dataset}/{model}/{name}",
            results_dir=f"results/target-modules/{dataset}/{model}/{name}",
            overrides=(
                f"lora.target_modules={modules}",
                "evaluate.methods=[base,forced_adapter]",
            ),
        )
        for name, modules in variants.items()
    ]


def training_format_jobs(dataset: str, model: str) -> list[Job]:
    context_batch = (
        ("training.per_device_train_batch_size=2", "training.gradient_accumulation_steps=2")
        if dataset == "narrativeqa"
        else ("training.per_device_train_batch_size=4", "training.gradient_accumulation_steps=1")
    )
    common = ("evaluate.methods=[forced_adapter]",)
    return [
        Job(
            group="training-formats",
            dataset=dataset,
            model=model,
            name="context-qa",
            config=dataset,
            step="all",
            adapter_dir=f"adapters/{dataset}/{model}/context-qa",
            results_dir=f"results/training-formats/{dataset}/{model}/context-qa",
            overrides=common
            + (
                "training.input_format=ctx_qa",
                "training.mixed_sources.0.name=context_qa",
                "training.mixed_sources.0.input_format=ctx_qa",
                "training.mixed_sources.0.context_window_chunks=1",
                *context_batch,
            ),
        ),
        Job(
            group="training-formats",
            dataset=dataset,
            model=model,
            name="chunk-continuation",
            config=dataset,
            step="all",
            adapter_dir=f"adapters/{dataset}/{model}/chunk-continuation",
            results_dir=f"results/training-formats/{dataset}/{model}/chunk-continuation",
            overrides=common
            + (
                "pipeline.skip_steps=[synthesize]",
                "data.input_mode=ntp",
                "training.input_format=chunk_next_prompt",
                "training.task=null",
                "training.train_answer_only=false",
                "training.mixed_sources=[{name:next_chunk,kind:next_chunks,input_format:chunk_next_prompt,n_next:1,sample_weight:1.0}]",
            ),
        ),
        Job(
            group="training-formats",
            dataset=dataset,
            model=model,
            name="raw-context",
            config=dataset,
            step="all",
            adapter_dir=f"adapters/{dataset}/{model}/raw-context",
            results_dir=f"results/training-formats/{dataset}/{model}/raw-context",
            overrides=common
            + (
                "pipeline.skip_steps=[synthesize]",
                "data.input_mode=ntp",
                "data.chunk_size=32768",
                "training.input_format=raw",
                "training.task=null",
                "training.train_answer_only=false",
                "training.max_steps=1000",
                "training.num_train_epochs=1",
                "training.save_epochs=1",
                "training.logging_steps=10",
                "training.early_stop_train_metric=mean_token_accuracy",
                "training.early_stop_train_threshold=0.95",
                "training.early_stop_train_mode=max",
                "training.early_stop_train_min_steps=50",
                "training.early_stop_train_patience=2",
                "training.per_device_train_batch_size=1",
                "training.gradient_accumulation_steps=1",
                "training.mixed_sources=[{name:raw_context,kind:raw_chunks,input_format:raw,sample_weight:1.0}]",
            ),
        ),
    ]


def compression_method_job(dataset: str, model: str) -> Job:
    return Job(
        group="compression-methods",
        dataset=dataset,
        model=model,
        name="expected-attention",
        config=f"expected_attention_{dataset}",
        step="evaluate",
        adapter_dir=f"adapters/{dataset}/{model}/qa",
        results_dir=f"results/compression-methods/{dataset}/{model}/expected-attention",
    )


def build_jobs(datasets: list[str], models: list[str]) -> list[Job]:
    jobs: list[Job] = []
    for dataset in datasets:
        for model in models:
            jobs.append(main_job(dataset, model))
            jobs.extend(training_format_jobs(dataset, model))
            jobs.append(compression_method_job(dataset, model))
            jobs.extend(target_module_jobs(dataset, model))
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", *GROUPS), default="all")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=tuple(MODELS), default=list(MODELS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Execute instead of printing commands.")
    mode.add_argument(
        "--validate",
        action="store_true",
        help="Compose every selected Hydra config without starting a job.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    jobs = build_jobs(args.datasets, args.models)
    if args.group != "all":
        selected_groups = {args.group}
        if args.group != "main":
            selected_groups.add("main")
        jobs = [job for job in jobs if job.group in selected_groups]

    for job in jobs:
        command = job.command()
        if args.validate:
            validation_command = command[:5] + ["--cfg", "job"] + command[5:]
            subprocess.run(
                validation_command,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            continue
        print(f"[{job.group}] [{job.dataset}] [{job.model}] [{job.name}]")
        print(shlex.join(command))
        if args.run:
            subprocess.run(command, check=True)
    if args.validate:
        print(f"Validated {len(jobs)} experiment configs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
