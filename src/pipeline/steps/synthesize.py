"""Step: synthesize - generate synthetic task pairs."""

import logging

from omegaconf import DictConfig

from src.pipeline._utils import PROJECT_ROOT

log = logging.getLogger(__name__)


def run(cfg: DictConfig, handler) -> None:
    """Generate synthetic QA pairs. No-op unless input_mode=synthetic_qa."""
    input_mode = cfg.data.get("input_mode", "ntp")
    if input_mode != "synthetic_qa":
        log.info(
            "[synthesize] input_mode=%s - skipping (only runs for synthetic_qa).",
            input_mode,
        )
        return

    log.info("[synthesize] Synthesizing task pairs...")

    from src.pipeline.data.synthesis.synthesize import synthesize_for_cluster_dir

    data_dir = PROJECT_ROOT / cfg.data.data_dir
    cluster_dir = handler.get_cluster_dir(cfg)

    eval_file = cfg.evaluate.get("eval_file", None)
    if not eval_file:
        raise ValueError("evaluate.eval_file must be set for the synthesize step.")

    synthetic_data_dir = cfg.data.get("synthetic_data_dir", None)
    synth_cfg = cfg.get("synthesize", {})
    synth_model = synth_cfg.get("instruct_model", None) or cfg.model.instruct_model

    synthesize_for_cluster_dir(
        instruct_model=synth_model,
        eval_path=data_dir / eval_file,
        eval_every_n=cfg.evaluate.get("eval_every_n", 1),
        cluster_dir=cluster_dir,
        chunk_size=cfg.data.chunk_size,
        overlap_ratio=cfg.data.overlap_ratio,
        synthetic_data_dir=str(synthetic_data_dir) if synthetic_data_dir else None,
        data_dir=data_dir,
        overwrite=bool(synth_cfg.get("overwrite", False)),
        temperature=float(synth_cfg.get("temperature", 0.7)),
        top_p=float(synth_cfg.get("top_p", 1.0)),
        max_tokens=int(synth_cfg.get("max_tokens", 512)),
        pairs_per_call=int(synth_cfg.get("pairs_per_call", 1)),
        context_window_chunks=int(synth_cfg.get("context_window_chunks", 0)),
        n_generations=int(synth_cfg.get("n_generations", 3)),
        tensor_parallel_size=int(synth_cfg.get("tensor_parallel_size", 1)),
        gpu_memory_utilization=float(synth_cfg.get("gpu_memory_utilization", 0.9)),
        max_model_len=synth_cfg.get("max_model_len", None),
        dtype=str(synth_cfg.get("dtype", "auto")),
        enforce_eager=bool(synth_cfg.get("enforce_eager", False)),
        attention_backend=str(synth_cfg.get("attention_backend", "FLASH_ATTN")),
        max_num_seqs=int(synth_cfg.get("max_num_seqs", 256)),
        max_retries=int(synth_cfg.get("max_retries", 0)),
    )
    log.info("[synthesize] Synthesis complete.")
