#!/usr/bin/env python3
"""Run the paper's generative-QA evaluation matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_runner import EvalRunConfig, run_evaluation


BENCHMARKS = ("longhealth", "narrativeqa")
METHODS = (
    "base",
    "forced_adapter",
    "forced_adapter_baseprefill",
    "forced_adapter_basescore_adapterprefill",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--data_dir")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--no_compression_batch_size", type=int)
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--compression_configs", default='["none"]')
    parser.add_argument("--eval_every_n", type=int, default=1)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--parallel_gpus", action="store_true")
    parser.add_argument(
        "--context_modes",
        nargs="+",
        choices=("with_context", "no_context"),
        default=["with_context"],
    )
    parser.add_argument("--eval_file")
    parser.add_argument("--task", choices=("qa", "qa_short"), default="qa_short")
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--model_max_length", type=int)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--disable_truncation", action="store_true")
    parser.add_argument("--skip_overlength", action="store_true")
    parser.add_argument("--adapter_scaling", type=float, default=2.0)
    parser.add_argument(
        "--compressed_generation_mode",
        choices=("context_prefill",),
        default="context_prefill",
    )
    return parser


def _compression_configs(parser: argparse.ArgumentParser, raw: str) -> list:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        parser.error(f"invalid --compression_configs JSON: {exc}")
    if not isinstance(value, list) or not value:
        parser.error("--compression_configs must be a non-empty JSON list")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    data_dir = args.data_dir or str(Path("data") / args.benchmark)
    extra_eval_args: list[str] = [
        "--task",
        args.task,
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--compressed_generation_mode",
        args.compressed_generation_mode,
        "--adapter_scaling",
        str(args.adapter_scaling),
    ]
    if args.eval_file:
        extra_eval_args.extend(["--eval_file", args.eval_file])
    if args.max_length is not None:
        extra_eval_args.extend(["--max_length", str(args.max_length)])
    if args.model_max_length is not None:
        extra_eval_args.extend(["--model_max_length", str(args.model_max_length)])
    if args.disable_truncation:
        extra_eval_args.append("--disable_truncation")
    if args.skip_overlength:
        extra_eval_args.append("--skip_overlength")

    cfg = EvalRunConfig(
        benchmark=args.benchmark,
        model_name=args.model_name,
        adapter_path=args.adapter_dir,
        data_dir=data_dir,
        results_dir=args.results_dir,
        seed=args.seed,
        num_seeds=args.num_seeds,
        batch_size=args.batch_size,
        no_compression_batch_size=args.no_compression_batch_size,
        torch_dtype=args.torch_dtype,
        methods=args.methods,
        compression_configs=_compression_configs(parser, args.compression_configs),
        eval_every_n=args.eval_every_n,
        skip_existing=args.skip_existing,
        multi_gpu=args.multi_gpu,
        parallel_gpus=args.parallel_gpus,
        context_modes=args.context_modes,
        extra_eval_args=extra_eval_args,
    )
    run_evaluation(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
