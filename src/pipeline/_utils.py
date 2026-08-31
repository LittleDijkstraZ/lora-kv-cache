"""Shared pipeline utilities (paths, naming, config helpers)."""

import os
import sys
from pathlib import Path

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_EVAL_METHODS = ["base", "forced_adapter"]


def sanitize_model_name(name: str) -> str:
    return name.replace(os.sep, "__")


def resolve_eval_methods(cfg: DictConfig) -> list[str]:
    methods = cfg.evaluate.get("methods", None)
    if methods:
        return [str(m) for m in methods]
    return list(DEFAULT_EVAL_METHODS)


def _get_active_config_name(cfg: DictConfig) -> str | None:
    if HydraConfig.initialized():
        config_name = HydraConfig.get().job.config_name
        if config_name:
            return str(config_name)

    hydra_cfg = cfg.get("hydra", None)
    if hydra_cfg is None:
        return None

    job_cfg = hydra_cfg.get("job", None)
    if job_cfg is None:
        return None

    config_name = job_cfg.get("config_name", None)
    if not config_name:
        return None
    return str(config_name)


def _get_adapter_base_dir(cfg: DictConfig) -> Path:
    explicit = cfg.pipeline.get("adapter_dir", None)
    if explicit:
        return Path(explicit)

    config_name = _get_active_config_name(cfg)
    if not config_name:
        raise ValueError(
            "pipeline.adapter_dir is null but Hydra job.config_name is unavailable"
        )
    return Path("adapters") / config_name


def resolve_adapter_dir(cfg: DictConfig) -> str:
    """Resolve a deterministic adapter directory."""
    return str(_get_adapter_base_dir(cfg))


def resolve_results_dir(
    cfg: DictConfig,
    *,
    default_name: str | None = None,
    project_root: Path | None = None,
) -> Path:
    """Resolve a deterministic results directory for the active pipeline."""
    explicit = cfg.evaluate.get("results_dir", None)
    if explicit:
        return Path(explicit)

    run_name = default_name or Path(cfg.pipeline.adapter_dir).name
    base = Path("results") / str(run_name)
    root = project_root or PROJECT_ROOT
    return root / base
