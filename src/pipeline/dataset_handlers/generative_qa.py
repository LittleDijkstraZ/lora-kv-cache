"""Generative QA dataset handler."""

import logging
from pathlib import Path

from omegaconf import DictConfig

from src.pipeline._utils import PROJECT_ROOT
from src.pipeline.data.contracts import GENERATIVE_TASK_FAMILY

log = logging.getLogger(__name__)


class GenerativeQAHandler:
    """Handler for the generic generative-QA-from-eval-file path."""

    task_family = GENERATIVE_TASK_FAMILY

    def get_cluster_folder_name(self, cfg: DictConfig) -> str:
        overlap_str = f"{cfg.data.overlap_ratio}".replace(".", "p")
        return (
            f"prepared/documents_{int(cfg.data.num_clusters)}/"
            f"chunks_{int(cfg.data.chunk_size)}_overlap_{overlap_str}"
        )

    def get_cluster_dir(self, cfg: DictConfig) -> Path:
        return PROJECT_ROOT / cfg.data.data_dir / self.get_cluster_folder_name(cfg)

    def prepare(self, cfg: DictConfig) -> None:
        log.info("[prepare] Preparing generative QA data via the eval-file path...")
        data_dir = PROJECT_ROOT / cfg.data.data_dir
        cluster_dir = self.get_cluster_dir(cfg)

        if cluster_dir.exists() and (
            (cluster_dir / "cluster_map.json").exists()
            or (cluster_dir / "doc_to_cluster.json").exists()
        ):
            log.info("Cluster data already exists at %s, skipping.", cluster_dir)
            return

        if cluster_dir.exists() and (data_dir / cfg.evaluate.eval_file).exists():
            existing_clusters = [
                d for d in cluster_dir.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if len(existing_clusters) >= cfg.data.num_clusters:
                log.info("Generative QA data already exists at %s, skipping.", cluster_dir)
                return

        if not cfg.data.get("train_from_eval_file", False):
            raise RuntimeError(
                "train_from_eval_file must be True for the generative QA eval-file path. "
                "Set data.train_from_eval_file=true in the config."
            )

        from src.pipeline.data.prepare_generative_qa import prepare_from_eval_file

        eval_path = data_dir / cfg.evaluate.eval_file
        eval_every_n = cfg.evaluate.get("eval_every_n", 1)
        log.info(
            "train_from_eval_file=True: building from %s (every %d-th example)",
            eval_path,
            eval_every_n,
        )
        cluster_dir.mkdir(parents=True, exist_ok=True)
        prepare_from_eval_file(
            eval_path,
            eval_every_n,
            cluster_dir,
            chunk_size=cfg.data.chunk_size,
            overlap_ratio=cfg.data.overlap_ratio,
            data_cfg=cfg.data,
        )
        log.info("Generative QA training data prepared at %s", cluster_dir)
