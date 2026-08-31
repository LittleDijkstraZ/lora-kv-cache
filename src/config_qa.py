"""Structured Hydra configuration for the QA pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


if not OmegaConf.has_resolver("qa_default_adapter_dir"):
    OmegaConf.register_new_resolver(
        "qa_default_adapter_dir",
        lambda adapter_dir, config_name: adapter_dir or f"adapters/{config_name}",
    )


@dataclass
class ModelConfig:
    base_model: str = "Qwen/Qwen3-4B"
    instruct_model: str = "Qwen/Qwen3-4B"
    max_seq_length: int = 32768


@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    share_A: bool = True
    train_A: bool = False
    target_modules: list[str] = field(
        default_factory=lambda: ["gate_proj", "up_proj", "down_proj"]
    )
    layers_to_transform: list[int] | None = None
    lora_init: str = "pissa"
    lora_B_init: str = "zero"
    use_dora: bool = True


@dataclass
class TrainingSourceConfig:
    name: str = "qa"
    kind: str = "synthetic_records"
    input_format: str = "qa_only"
    task: Any = None
    train_answer_only: bool = True
    append_eos: bool = False
    sample_weight: float = 1.0
    context_window_chunks: int = 0
    n_next: int | None = None
    synthetic_data_dir: str | None = None
    cluster_root: str | None = None


@dataclass
class TrainingConfig:
    objective: str = "sft"
    learning_rate: float = 1e-4
    num_train_epochs: int = 4
    max_steps: int | None = None
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.001
    save_epochs: int = 4
    torch_dtype: str = "auto"
    logging_steps: int | None = None
    early_stop_train_metric: str | None = None
    early_stop_train_threshold: float | None = None
    early_stop_train_mode: str = "max"
    early_stop_train_min_steps: int = 0
    early_stop_train_patience: int = 1
    input_format: str = "qa_only"
    task: Any = "qa_short"
    append_eos: bool = False
    train_answer_only: bool = True
    disable_truncation: bool = True
    fail_on_truncation: bool = True
    mixed_sources: list[TrainingSourceConfig] | None = None
    report_to: list[str] = field(default_factory=list)


@dataclass
class DataConfig:
    chunk_size: int = 64
    overlap_ratio: float = 0.0
    num_clusters: int = 20
    data_dir: str = "data/narrativeqa"
    cluster_strategy: str | None = "metadata_field"
    cluster_metadata_field: str | None = "document_id"
    train_from_eval_file: bool = True
    cluster_docs_folder: str | None = None
    input_mode: str = "synthetic_qa"
    synthetic_data_dir: str | None = None


@dataclass
class PipelineConfig:
    seed: int = 0
    num_seeds: int = 1
    adapter_dir: str | None = None
    dataset: str = "narrativeqa"
    skip_steps: list[str] = field(default_factory=list)
    resume_from_checkpoint: str | None = None


@dataclass
class EvaluationConfig:
    batch_size: int = 1
    no_compression_batch_size: int | None = None
    eval_every_n: int = 1
    benchmark: str | None = None
    methods: list[str] | None = None
    context_modes: list[str] = field(default_factory=lambda: ["with_context"])
    adapter_scaling: float = 2.0
    torch_dtype: str = "auto"
    parallel_gpus: bool = True
    multi_gpu: bool = False
    skip_existing: bool = False
    results_dir: str | None = None
    eval_file: str = "narrativeqa.jsonl"
    task: str = "qa_short"
    max_length: int | None = None
    model_max_length: int | None = None
    max_new_tokens: int = 128
    disable_truncation: bool = True
    skip_overlength: bool = True
    compressed_generation_mode: str = "context_prefill"
    compression_configs: list[Any] = field(default_factory=lambda: ["none"])


@dataclass
class SynthesisConfig:
    instruct_model: str | None = None
    n_generations: int = 1
    pairs_per_call: int = 4
    context_window_chunks: int = 2
    temperature: float = 0.4
    top_p: float = 0.9
    max_tokens: int = 2048
    tensor_parallel_size: int = 4
    gpu_memory_utilization: float = 0.95
    max_num_seqs: int = 32
    max_model_len: int | None = 32768
    dtype: str = "auto"
    enforce_eager: bool = False
    attention_backend: str = "FLASH_ATTN"
    overwrite: bool = False
    max_retries: int = 2


@dataclass
class QAPipelineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    evaluate: EvaluationConfig = field(default_factory=EvaluationConfig)
    synthesize: SynthesisConfig = field(default_factory=SynthesisConfig)
    step: str = "all"
