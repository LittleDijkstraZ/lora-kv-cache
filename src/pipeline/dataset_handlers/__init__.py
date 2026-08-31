"""Dataset handlers for the supported LongHealth and NarrativeQA workflows."""

from src.pipeline.dataset_handlers.base import DatasetHandler
from src.pipeline.dataset_handlers.generative_qa import GenerativeQAHandler

REGISTRY: dict[str, type] = {
    "longhealth": GenerativeQAHandler,
    "narrativeqa": GenerativeQAHandler,
}


def get_handler(cfg) -> DatasetHandler:
    """Return the DatasetHandler instance for the configured dataset."""
    dataset = cfg.pipeline.get("dataset", "narrativeqa")
    if dataset not in REGISTRY:
        valid = ", ".join(sorted(REGISTRY))
        raise ValueError(f"Unknown pipeline.dataset={dataset!r}; valid: {valid}")
    return REGISTRY[dataset]()
