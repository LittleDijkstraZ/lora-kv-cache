#!/usr/bin/env python3
"""Hydra entrypoint for data preparation, synthesis, training, and evaluation."""

import gc
import logging
import sys
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_qa import QAPipelineConfig
from src.pipeline._utils import resolve_adapter_dir
from src.pipeline.dataset_handlers import get_handler
from src.pipeline.steps import evaluate, finetune, synthesize

log = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="qa_config_schema", node=QAPipelineConfig)

STEP_ORDER = ["prepare", "synthesize", "finetune", "evaluate"]
ADAPTER_CONFIG_STEPS = {"prepare", "synthesize", "finetune"}


def _build_steps(handler) -> dict:
    return {
        "prepare": lambda cfg: handler.prepare(cfg),
        "synthesize": lambda cfg: synthesize.run(cfg, handler),
        "finetune": lambda cfg: finetune.run(cfg, handler),
        "evaluate": lambda cfg: evaluate.run(cfg, handler),
    }


def run_pipeline(cfg: DictConfig) -> None:
    handler = get_handler(cfg)
    cluster_folder_name = handler.get_cluster_folder_name(cfg)

    with open_dict(cfg):
        cfg.data.cluster_docs_folder = cluster_folder_name
        cfg.pipeline.adapter_dir = resolve_adapter_dir(cfg)

    log.info("Dataset        : %s", cfg.pipeline.get("dataset", "narrativeqa"))
    log.info("Task family    : %s", getattr(handler, "task_family", "unknown"))
    log.info("Adapter dir    : %s", cfg.pipeline.adapter_dir)
    log.info("Cluster folder : %s", cluster_folder_name)

    steps = _build_steps(handler)
    requested_step = getattr(cfg, "step", "all")
    skip_steps = list(cfg.pipeline.get("skip_steps", []) or [])

    if requested_step == "all":
        steps_to_run = list(STEP_ORDER)
    elif requested_step in steps:
        steps_to_run = [requested_step]
    else:
        log.error("Unknown step '%s'. Valid: %s, all", requested_step, ", ".join(steps))
        return

    if skip_steps:
        unknown = [s for s in skip_steps if s not in steps]
        if unknown:
            log.warning("Unknown skip_steps entries (ignored): %s", unknown)
        steps_to_run = [s for s in steps_to_run if s not in skip_steps]
        log.info("Skipping steps: %s", skip_steps)

    has_adapter_config_step = any(step_name in ADAPTER_CONFIG_STEPS for step_name in steps_to_run)
    adapter_dir = PROJECT_ROOT / cfg.pipeline.adapter_dir
    saved_cfg_path = adapter_dir / "pipeline_config.yaml"
    if has_adapter_config_step:
        adapter_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, saved_cfg_path)
        log.info("Config saved to: %s", saved_cfg_path)
    else:
        log.info(
            "Evaluate-only run detected; leaving adapter metadata unchanged at: %s",
            saved_cfg_path,
        )

    log.info("Full config:\n%s", OmegaConf.to_yaml(cfg))

    for step_name in steps_to_run:
        log.info("=" * 60)
        log.info("Running step: %s", step_name)
        log.info("=" * 60)
        steps[step_name](cfg)
        log.info("Step '%s' completed", step_name)

        try:
            import torch
            if torch.cuda.is_available():
                gc.collect()
                torch.cuda.empty_cache()
                log.info("GPU cache cleared after step '%s'", step_name)
        except ImportError:
            pass

    log.info("Pipeline complete.")


@hydra.main(version_base=None, config_path="../../conf", config_name="narrativeqa")
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
