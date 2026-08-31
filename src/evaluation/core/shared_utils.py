"""Small utilities shared by the generative-QA evaluators."""

from __future__ import annotations

import re
from pathlib import Path

try:
    from src.pipeline.task_modes import get_qa_boxed_instruction, validate_eval_task_mode
except ModuleNotFoundError:  # pragma: no cover - direct script path
    from pipeline.task_modes import get_qa_boxed_instruction, validate_eval_task_mode


def get_max_length(tokenizer, max_length: int | None = None) -> int:
    if max_length is not None:
        return int(max_length)
    model_limit = int(getattr(tokenizer, "model_max_length", 131072))
    if model_limit > 10**6:
        model_limit = 131072
    return int(model_limit * 0.9)


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints: list[tuple[int, Path]] = []
    for path in checkpoint_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    return max(checkpoints, default=(0, None), key=lambda value: value[0])[1]


def set_peft_adapter_scaling(
    peft_model,
    scaling: float,
    adapter_name: str = "default",
) -> None:
    from peft.tuners.lora import LoraLayer

    patched = 0
    for module in peft_model.modules():
        if isinstance(module, LoraLayer) and adapter_name in module.scaling:
            module.scaling[adapter_name] = float(scaling)
            patched += 1
    if patched == 0:
        raise RuntimeError(f"No LoRA layers found for adapter {adapter_name!r}")


def setup_generation_prompt_templates(
    tokenizer,
    *,
    no_thinking: bool = True,
    task: str = "qa_short",
):
    """Build the context, question, and boxed-answer prompt fragments."""

    task = validate_eval_task_mode(task)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            "{{ message['content'] }}<|eot_id|>"
            "{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
        )

    placeholder = "__CONTEXT_AND_QUESTION__"
    kwargs = {
        "conversation": [{"role": "user", "content": placeholder}],
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        rendered = tokenizer.apply_chat_template(
            **kwargs,
            enable_thinking=not no_thinking,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(**kwargs)
    if rendered.count(placeholder) != 1:
        raise ValueError("Tokenizer chat template did not preserve the prompt placeholder")
    template_start, template_end = rendered.split(placeholder)
    return (
        template_start + "Here is a context: \n\n",
        "\nQuestion: {query} ",
        get_qa_boxed_instruction(task),
        template_end + "The answer is \\boxed{",
    )


__all__ = [
    "find_latest_checkpoint",
    "get_max_length",
    "set_peft_adapter_scaling",
    "setup_generation_prompt_templates",
]
