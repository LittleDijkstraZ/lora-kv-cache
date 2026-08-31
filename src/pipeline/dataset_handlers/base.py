"""DatasetHandler protocol — the contract every dataset must implement."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from omegaconf import DictConfig


@runtime_checkable
class DatasetHandler(Protocol):
    """Encapsulates everything that differs between datasets.

    To add a new dataset, implement this protocol under
    ``src/pipeline/dataset_handlers`` and register the handler in that package.
    """

    task_family: str

    def get_cluster_folder_name(self, cfg: DictConfig) -> str:
        """Return the cluster folder name derived from configuration."""
        ...

    def get_cluster_dir(self, cfg: DictConfig) -> Path:
        """Return the absolute path to the cluster directory for this run."""
        ...

    def prepare(self, cfg: DictConfig) -> None:
        """Chunk and cluster the prepared documents for this dataset."""
        ...
