"""Document-adapter evaluation wrapper."""

from __future__ import annotations

from ..core.model_loading import load_model
from ..core.predictions import batch_get_predictions
from ..core.io_utils import save_predictions, save_results
from ..core.document_adapter_runtime import evaluate_generative_task_document_adapter
from ..core.shared_utils import (
    get_max_length,
    find_latest_checkpoint,
    set_peft_adapter_scaling,
)

from .evaluate_qa import get_generative_task_spec


def evaluate_document_adapter(args) -> None:
    """Evaluate a generative-QA task with its document-specific adapter."""

    spec = get_generative_task_spec(getattr(args, "benchmark", "narrativeqa"))
    evaluate_generative_task_document_adapter(
        spec=spec,
        args=args,
        load_model_fn=load_model,
        batch_get_predictions_fn=batch_get_predictions,
        get_max_length_fn=get_max_length,
        set_peft_adapter_scaling_fn=set_peft_adapter_scaling,
        save_results_fn=save_results,
        save_predictions_fn=save_predictions,
        find_latest_checkpoint_fn=find_latest_checkpoint,
    )


__all__ = ["evaluate_document_adapter"]
