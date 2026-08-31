#!/usr/bin/env python3
"""Evaluate one model/method/compression setting on generative QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METHODS = (
    "base",
    "forced_adapter",
    "forced_adapter_baseprefill",
    "forced_adapter_basescore_adapterprefill",
)
BENCHMARKS = ("longhealth", "narrativeqa")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--adapter_dir")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--num_seeds", type=int, default=1)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--adapter_scaling", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch_dtype", default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--compression_config_json")
    parser.add_argument("--compression_ratio", type=float)
    parser.add_argument("--chunk_size", type=int, default=256)
    parser.add_argument("--sink_size_start", type=int, default=32)
    parser.add_argument("--sink_size_end", type=int, default=32)
    parser.add_argument(
        "--compression_scope",
        choices=("context_only",),
        default="context_only",
    )
    parser.add_argument("--context_min_keep_tokens", type=int, default=0)
    parser.add_argument(
        "--compressed_generation_mode",
        choices=("context_prefill",),
        default="context_prefill",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_length", type=int)
    parser.add_argument("--model_max_length", type=int)
    parser.add_argument("--disable_truncation", action="store_true")
    parser.add_argument("--skip_overlength", action="store_true")
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--eval_every_n", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--save_predictions", dest="save_predictions_dir")
    parser.add_argument("--eval_file")
    parser.add_argument("--task", choices=("qa", "qa_short"), default="qa_short")
    parser.add_argument("--no_context", action="store_true")
    return parser


def _compression_kwargs(args: argparse.Namespace) -> dict:
    if args.compression_config_json:
        try:
            value = json.loads(args.compression_config_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid --compression_config_json: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("--compression_config_json must contain a JSON object")
        return value
    return {
        "compression_ratio": args.compression_ratio,
        "chunk_size": args.chunk_size,
        "sink_size_start": args.sink_size_start,
        "sink_size_end": args.sink_size_end,
        "compression_scope": args.compression_scope,
        "context_min_keep_tokens": args.context_min_keep_tokens,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        parser.error("require num_shards >= 1 and 0 <= shard_id < num_shards")
    try:
        args.compression_kwargs = _compression_kwargs(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.include_context = not args.no_context
    if args.method == "base":
        from evaluation.methods.evaluate_qa import evaluate_qa_all_seeds

        evaluate_qa_all_seeds(args)
    else:
        if not args.adapter_dir:
            parser.error("--adapter_dir is required for document-adapter methods")
        from evaluation.methods.evaluate_document_adapter import evaluate_document_adapter

        evaluate_document_adapter(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
