"""Generative-QA preparation and scoring for LongHealth and NarrativeQA."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..core.eval_runtime import (
    evaluate_generative_task,
    evaluate_generative_task_all_seeds,
)
from ..core.model_loading import load_model
from ..core.qa_metrics import (
    exact_match_score,
    extract_boxed_answer,
    rouge_l_score,
)
from ..core.predictions import batch_get_predictions
from ..core.shared_utils import get_max_length, setup_generation_prompt_templates
from ..core.task_specs import GenerativeTaskSpec


def _flatten_context(context) -> str:
    if isinstance(context, str):
        return context
    if isinstance(context, dict):
        titles = context.get("title") or []
        sentence_groups = context.get("sentences") or []
        sections: list[str] = []
        for title, sentences in zip(titles, sentence_groups):
            sections.extend([str(title), " ".join(sentences), ""])
        return "\n".join(sections).strip()
    return str(context)


def prepare_qa_eval_df(
    tokenizer,
    eval_file: str | Path,
    cluster_dir: Path | None = None,
    include_context: bool = True,
    task: str = "qa_short",
) -> pd.DataFrame:
    """Load prepared JSONL records and build the exact generation prompts."""

    eval_file = Path(eval_file)
    if not eval_file.is_file():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    rows = [json.loads(line) for line in eval_file.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Evaluation file is empty: {eval_file}")

    template_start, prompt_template, prompt_end, template_end = (
        setup_generation_prompt_templates(tokenizer, task=task)
    )
    context_header = "Here is a context: \n\n"
    if not include_context:
        if template_start.endswith(context_header):
            template_start = template_start[: -len(context_header)]
        prompt_template = prompt_template.lstrip("\n")

    def build_prompt(row: dict) -> tuple[str, str, str, int, int]:
        context = _flatten_context(row.get("context", "")) if include_context else ""
        context_start = len(template_start)
        context_end = context_start + len(context)
        context_prefill = template_start + context
        question_suffix = (
            prompt_template.format(query=row["question"]) + prompt_end + template_end
        )
        return (
            context_prefill + question_suffix,
            context_prefill,
            question_suffix,
            context_start,
            context_end,
        )

    frame = pd.DataFrame(rows)
    if "_id" not in frame.columns and "id" in frame.columns:
        frame["_id"] = frame["id"]
    if "_id" not in frame.columns:
        raise ValueError(f"Evaluation records in {eval_file} require an '_id' field")
    if "answers" not in frame.columns:
        raise ValueError(f"Evaluation records in {eval_file} require an 'answers' field")

    parts = frame.apply(build_prompt, axis=1)
    frame["text"] = parts.apply(lambda value: value[0])
    frame["context_prefill_text"] = parts.apply(lambda value: value[1])
    frame["question_suffix_text"] = parts.apply(lambda value: value[2])
    frame["context_char_start"] = parts.apply(lambda value: value[3])
    frame["context_char_end"] = parts.apply(lambda value: value[4])
    frame["label"] = frame["answers"]

    sidecar = Path(cluster_dir) / "id_to_cluster.json" if cluster_dir else None
    if sidecar is not None and sidecar.is_file():
        assignments = json.loads(sidecar.read_text())
        frame["cluster_idx"] = frame["_id"].astype(str).map(assignments)

    frame["num_tokens"] = frame["text"].apply(
        lambda text: len(
            tokenizer(
                text,
                add_special_tokens=True,
                return_attention_mask=False,
            )["input_ids"]
        )
    )
    return frame.sort_values("num_tokens", ascending=False).reset_index(drop=True)


def _gold_answers(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _exact_match(raw_prediction: str, gold_answers) -> float:
    return exact_match_score(
        extract_boxed_answer(raw_prediction),
        _gold_answers(gold_answers),
    )


def _annotate_predictions(predictions: list[dict]) -> None:
    for prediction in predictions:
        answer = extract_boxed_answer(str(prediction.get("predicted", "")))
        golds = _gold_answers(prediction.get("label"))
        prediction["extracted_answer"] = answer
        prediction["em"] = exact_match_score(answer, golds)
        prediction["rouge_l"] = rouge_l_score(answer, golds)


def _metric_sum(predictions: list[dict], name: str) -> float:
    return sum(float(prediction.get(name, 0.0)) for prediction in predictions)


def _make_spec(benchmark: str, filename: str) -> GenerativeTaskSpec:
    return GenerativeTaskSpec(
        benchmark_name=benchmark,
        default_eval_filename=filename,
        prediction_basename=f"{benchmark}_predictions",
        prepare_eval_df=prepare_qa_eval_df,
        answer_scorer_fn=_exact_match,
        metric_sum_fns={
            "rouge_l": lambda predictions: _metric_sum(predictions, "rouge_l"),
            "em": lambda predictions: _metric_sum(predictions, "em"),
        },
        annotate_predictions_fn=_annotate_predictions,
    )


TASK_SPECS = {
    "longhealth": _make_spec("longhealth", "longhealth.jsonl"),
    "narrativeqa": _make_spec("narrativeqa", "narrativeqa.jsonl"),
}


def get_generative_task_spec(benchmark: str) -> GenerativeTaskSpec:
    try:
        return TASK_SPECS[str(benchmark)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}; choose from {sorted(TASK_SPECS)}"
        ) from exc


def evaluate_qa(
    *,
    model_name: str,
    method: str,
    seed: int,
    eval_file: str | Path,
    benchmark: str,
    **kwargs,
):
    """Evaluate the base model on one of the two paper datasets."""

    return evaluate_generative_task(
        spec=get_generative_task_spec(benchmark),
        load_model_fn=load_model,
        batch_get_predictions_fn=batch_get_predictions,
        get_max_length_fn=get_max_length,
        model_name=model_name,
        method=method,
        seed=seed,
        eval_file=eval_file,
        **kwargs,
    )


def evaluate_qa_all_seeds(args) -> None:
    spec = get_generative_task_spec(args.benchmark)
    evaluate_generative_task_all_seeds(
        spec=spec,
        args=args,
        evaluate_fn=lambda **kwargs: evaluate_qa(
            benchmark=spec.benchmark_name,
            **kwargs,
        ),
    )


__all__ = [
    "TASK_SPECS",
    "evaluate_qa",
    "evaluate_qa_all_seeds",
    "get_generative_task_spec",
    "prepare_qa_eval_df",
]
