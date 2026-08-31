"""Generative-QA prediction utilities used by the paper experiments."""

import copy
import math
from contextlib import contextmanager, nullcontext

import torch
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from .io_utils import shard_indices


BASE_SCORE_ADAPTER_PREFILL_MODE = "base_score_adapter_prefill"
ADAPTER_PREFILL_MODES = {"adapter", "base", BASE_SCORE_ADAPTER_PREFILL_MODE}


def _prediction_row_id(value):
    """Return a JSON-friendly stable row id for saved prediction records."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def _raise_if_untruncated_prompt_exceeds_model(model, *, prompt_length: int, max_new_tokens: int) -> None:
    """Fail loudly when no tokenizer truncation is active and the model context is known."""

    config = getattr(model, "config", None)
    limit = getattr(config, "max_position_embeddings", None)
    if limit is None:
        return
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        return
    if limit <= 0 or limit >= 10**7:
        return
    required = int(prompt_length) + int(max_new_tokens or 0)
    if required > limit:
        raise RuntimeError(
            "Evaluation prompt exceeds model context with truncation disabled: "
            f"prompt_length={prompt_length}, max_new_tokens={max_new_tokens}, "
            f"required={required}, model.config.max_position_embeddings={limit}."
        )


def _debug_print_compactorpress_start(*, press, length: int) -> None:
    print(f"\n[KV compression] Starting generation...")
    print(f"[KV compression] Input sequence length: {length} tokens")
    expected_compressed = int(length * (1 - press.compression_ratio))
    print(
        f"[KV compression] Expected KV cache size after compression: ~{expected_compressed} tokens "
        f"(keeping {100*(1-press.compression_ratio):.0f}%)"
    )


def _debug_print_compactorpress_end(*, outputs, press_context) -> None:
    print(f"[KV compression] Generation complete.")

    kv_cache = getattr(outputs, "past_key_values", None)
    if kv_cache is not None:
        if isinstance(kv_cache, tuple):
            # Older transformers: tuple[layer] -> (k, v, ...)
            key_tensor = kv_cache[0][0]
            key_shape = key_tensor.shape
            compressed_len = key_shape[2]
            print(f"[KV compression] KV cache seq_length: {compressed_len} tokens")
            print(f"[KV compression] KV cache key shape (layer 0): {key_shape}")
        else:
            # Newer transformers (Cache API): `past_key_values` is a `Cache` (e.g. `DynamicCache`),
            # where per-layer tensors live under `kv_cache.layers[layer_idx].keys/values`.
            from transformers.cache_utils import Cache

            if isinstance(kv_cache, Cache):
                compressed_len = kv_cache.get_seq_length()
                key_tensor = None
                key_layer_idx = None
                for layer_idx, layer_cache in enumerate(kv_cache.layers):
                    candidate = getattr(layer_cache, "keys", None)
                    if candidate is not None:
                        key_tensor = candidate
                        key_layer_idx = layer_idx
                        break
                print(f"[KV compression] KV cache seq_length (via get_seq_length): {compressed_len} tokens")
                if key_tensor is None:
                    print("[KV compression] KV cache has no layer with `.keys`; skipping key-shape debug.")
                else:
                    key_shape = key_tensor.shape
                    print(f"[KV compression] KV cache key shape (layer {key_layer_idx}): {key_shape}")
                    if key_tensor.ndim == 4:
                        print(
                            f"[KV compression]   -> [batch={key_shape[0]}, num_heads={key_shape[1]}, "
                            f"seq_len={key_shape[2]}, head_dim={key_shape[3]}]"
                        )
            else:
                # Unexpected cache type: keep debug print resilient.
                print(f"[KV compression] KV cache type: {type(kv_cache)}")

    if hasattr(press_context, "compression_stats"):
        print(f"[KV compression] Stats: {press_context.compression_stats}")


def _trim_generated_token_ids(
    sequence: torch.Tensor,
    *,
    prompt_padded_len: int,
    pad_token_id: int | None,
    terminator_ids: set[int],
) -> list[int]:
    """Return generated token ids excluding padding and explicit terminators."""
    gen = sequence[prompt_padded_len:]
    out: list[int] = []
    for tok in gen.tolist():
        if pad_token_id is not None and tok == pad_token_id:
            break
        if tok in terminator_ids:
            break
        out.append(int(tok))
    return out


def _compute_generation_confidence_from_scores(
    *,
    sequences: torch.Tensor,
    scores: tuple[torch.Tensor, ...] | list[torch.Tensor],
    prompt_padded_len: int,
    pad_token_id: int | None,
    terminator_ids: set[int],
) -> list[dict[str, float | int | None]]:
    """Compute per-sample generation confidence from ``generate(..., output_scores=True)``.

    Confidence is exp(mean log-prob) over generated (non-pad, non-terminator) tokens.
    """
    batch_size = sequences.shape[0]
    token_ids_per_sample = [
        _trim_generated_token_ids(
            sequences[j],
            prompt_padded_len=prompt_padded_len,
            pad_token_id=pad_token_id,
            terminator_ids=terminator_ids,
        )
        for j in range(batch_size)
    ]

    logprob_sums = [0.0 for _ in range(batch_size)]
    token_counts = [0 for _ in range(batch_size)]

    for step_idx, step_scores in enumerate(scores):
        # step_scores: [batch, vocab]
        step_log_probs = torch.log_softmax(step_scores.float(), dim=-1)
        for j in range(batch_size):
            toks = token_ids_per_sample[j]
            if step_idx >= len(toks):
                continue
            tok = toks[step_idx]
            logprob_sums[j] += float(step_log_probs[j, tok].item())
            token_counts[j] += 1

    conf: list[dict[str, float | int | None]] = []
    for j in range(batch_size):
        n = token_counts[j]
        if n <= 0:
            conf.append(
                {
                    "confidence": None,
                    "confidence_mean_logprob": None,
                    "confidence_sum_logprob": None,
                    "confidence_token_count": 0,
                }
            )
            continue
        sum_lp = logprob_sums[j]
        mean_lp = sum_lp / n
        conf.append(
            {
                "confidence": float(math.exp(mean_lp)),
                "confidence_mean_logprob": float(mean_lp),
                "confidence_sum_logprob": float(sum_lp),
                "confidence_token_count": int(n),
            }
        )
    return conf


def _press_uses_context_only_compression(press) -> bool:
    return bool(press is not None and getattr(press, "compression_scope", "context_only") == "context_only")


def _document_id_for_row(df, row_idx: int) -> str:
    row = df.iloc[row_idx]
    document_id = None
    if "metadata" in df.columns:
        metadata = row.get("metadata") if hasattr(row, "get") else None
        if isinstance(metadata, dict):
            document_id = (
                metadata.get("document_id")
                or metadata.get("article_id")
                or metadata.get("doc_id")
                or metadata.get("paper_id")
                or metadata.get("patient_id")
            )
    if document_id is None:
        for key in ("document_id", "article_id", "doc_id", "paper_id", "patient_id"):
            if key in df.columns:
                document_id = row.get(key) if hasattr(row, "get") else None
                if document_id is not None:
                    break
    if (
        document_id is None
        or document_id != document_id  # NaN
        or str(document_id).strip() == ""
    ):
        raise RuntimeError(
            "compression_scope='context_only' requires a document id for "
            "context-aware batching. Expected metadata.document_id, "
            "metadata.article_id, metadata.patient_id, or a document-id column."
        )
    return str(document_id)


def _sort_key_for_context_batching(df, row_idx: int) -> int:
    if "num_tokens" in df.columns:
        try:
            return int(df.iloc[row_idx]["num_tokens"])
        except (TypeError, ValueError):
            pass
    return len(str(df.iloc[row_idx]["text"]))


def _build_context_only_batches(df, indices: list[int], batch_size: int) -> list[list[int]]:
    """Group context-only compression batches by document and then by prompt length."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")

    group_order: list[str] = []
    grouped: dict[str, list[int]] = {}
    for row_idx in indices:
        document_id = _document_id_for_row(df, row_idx)
        if document_id not in grouped:
            grouped[document_id] = []
            group_order.append(document_id)
        grouped[document_id].append(row_idx)

    batches: list[list[int]] = []
    for document_id in group_order:
        group_indices = sorted(
            grouped[document_id],
            key=lambda idx: _sort_key_for_context_batching(df, idx),
            reverse=True,
        )
        for start in range(0, len(group_indices), batch_size):
            batches.append(group_indices[start : start + batch_size])
    return batches


def _build_context_prefill_batches(df, indices: list[int], batch_size: int) -> list[list[int]]:
    """Group suffix batches by exact context prefill, preserving document boundaries."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive; got {batch_size}")
    required = {"context_prefill_text", "question_suffix_text"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' requires dataframe "
            f"columns {missing}; generative QA prep should populate split prompt fields."
        )

    group_order: list[tuple[str, str]] = []
    grouped: dict[tuple[str, str], list[int]] = {}
    for row_idx in indices:
        document_id = _document_id_for_row(df, row_idx)
        context_text = str(df.iloc[row_idx]["context_prefill_text"])
        key = (document_id, context_text)
        if key not in grouped:
            grouped[key] = []
            group_order.append(key)
        grouped[key].append(row_idx)

    def suffix_sort_key(row_idx: int) -> int:
        return len(str(df.iloc[row_idx]["question_suffix_text"]))

    batches: list[list[int]] = []
    for key in group_order:
        group_indices = sorted(grouped[key], key=suffix_sort_key, reverse=True)
        for start in range(0, len(group_indices), batch_size):
            batches.append(group_indices[start : start + batch_size])
    return batches


def _context_prefix_token_len_from_offsets(
    *,
    tokenizer,
    context_prefill_text: str,
    context_char_start: int,
) -> int:
    encoded = tokenizer(
        context_prefill_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' requires a fast "
            "tokenizer with return_offsets_mapping support."
        )
    return int(sum(1 for _start, end in offsets if int(end) <= int(context_char_start)))


def _effective_context_only_ratio(
    *,
    requested_compression_ratio: float,
    total_prefill_tokens: int,
    prefix_keep_tokens: int,
) -> float:
    """Convert context-only CR into a full-prompt CR with a protected prefix sink."""
    if total_prefill_tokens <= 0:
        return 0.0
    context_tokens = max(0, int(total_prefill_tokens) - int(prefix_keep_tokens))
    context_keep = int(math.floor(context_tokens * (1.0 - float(requested_compression_ratio)) + 1e-6))
    desired_keep = min(int(total_prefill_tokens), int(prefix_keep_tokens) + context_keep)
    if desired_keep >= total_prefill_tokens:
        return 0.0
    # The press uses int(k_len * (1 - cr)); nudge the keep fraction upward so
    # floating point roundoff does not accidentally lose one token.
    keep_fraction = min(1.0, (float(desired_keep) + 1e-3) / float(total_prefill_tokens))
    return max(0.0, min(0.999999, 1.0 - keep_fraction))


@contextmanager
def _temporary_press_full_prompt_span(
    press,
    *,
    compression_ratio: float,
    sink_size_start: int,
    sink_size_end: int = 0,
):
    old_ratio = getattr(press, "compression_ratio", None)
    old_scope = getattr(press, "compression_scope", None)
    old_sink_start = getattr(press, "sink_size_start", None)
    old_sink_end = getattr(press, "sink_size_end", None)
    if hasattr(press, "clear_context_compressible_mask"):
        press.clear_context_compressible_mask()
    press.compression_ratio = float(compression_ratio)
    press.compression_scope = "full_prompt"
    press.sink_size_start = int(sink_size_start)
    press.sink_size_end = int(sink_size_end)
    try:
        yield press
    finally:
        if old_ratio is not None:
            press.compression_ratio = old_ratio
        if old_scope is not None:
            press.compression_scope = old_scope
        if old_sink_start is not None:
            press.sink_size_start = old_sink_start
        if old_sink_end is not None:
            press.sink_size_end = old_sink_end
        if hasattr(press, "clear_context_compressible_mask"):
            press.clear_context_compressible_mask()


def _context_char_spans_for_rows(df, row_indices: list[int]) -> list[tuple[int, int]] | None:
    if "context_char_start" not in df.columns or "context_char_end" not in df.columns:
        return None
    starts = list(df["context_char_start"])
    ends = list(df["context_char_end"])
    spans: list[tuple[int, int]] = []
    for row_idx in row_indices:
        spans.append((int(starts[row_idx]), int(ends[row_idx])))
    return spans


def _build_context_compressible_mask(
    *,
    offset_mapping,
    context_spans: list[tuple[int, int]],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Return a boolean mask for context tokens in a padded/truncated batch."""
    if offset_mapping is None:
        raise RuntimeError(
            "compression_scope='context_only' requires tokenizer offset mappings. "
            "Use a fast tokenizer that supports return_offsets_mapping=True."
        )
    offsets = offset_mapping
    if not torch.is_tensor(offsets):
        offsets = torch.as_tensor(offsets, dtype=torch.long)
    offsets = offsets.to(device=attention_mask.device)
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError(f"offset_mapping must have shape [batch, seq, 2]; got {tuple(offsets.shape)}")
    if offsets.shape[:2] != attention_mask.shape:
        raise ValueError(
            "offset_mapping and attention_mask shape mismatch: "
            f"offsets={tuple(offsets.shape)} attention_mask={tuple(attention_mask.shape)}"
        )
    if len(context_spans) != offsets.shape[0]:
        raise ValueError(
            f"Expected {offsets.shape[0]} context spans for the batch; got {len(context_spans)}"
        )

    starts = offsets[..., 0]
    ends = offsets[..., 1]
    context_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    valid = attention_mask.bool()
    for batch_idx, (ctx_start, ctx_end) in enumerate(context_spans):
        if ctx_end <= ctx_start:
            continue
        overlaps_context = (starts[batch_idx] < ctx_end) & (ends[batch_idx] > ctx_start)
        context_mask[batch_idx] = overlaps_context & valid[batch_idx]
    return context_mask


def _press_target_for_model(model):
    return (
        model.base_model.model
        if hasattr(model, "base_model") and hasattr(model.base_model, "model")
        else model
    )


def _adapter_prefill_context(model, adapter_prefill_mode: str):
    """Return the context used while building the compressed context cache.

    The default keeps adapters enabled. ``base`` temporarily disables PEFT adapters for the
    context prefill/compression forward pass; suffix prefill and decode still
    run through the adapter.
    """

    if adapter_prefill_mode == "adapter":
        return nullcontext()
    if adapter_prefill_mode != "base":
        raise ValueError(
            "adapter_prefill_mode must be 'adapter' or 'base' in this context; "
            f"got {adapter_prefill_mode!r}"
        )
    disable_adapter = getattr(model, "disable_adapter", None)
    if not callable(disable_adapter):
        raise RuntimeError(
            "adapter_prefill_mode='base' requires a PEFT model with "
            "disable_adapter() support."
        )
    return disable_adapter()


def _reset_press_indices(press) -> None:
    reset_indices = getattr(press, "reset_indices", None)
    if callable(reset_indices):
        reset_indices()


def _snapshot_press_kept_indices(press) -> dict[int, torch.Tensor]:
    kept_indices = getattr(press, "kept_indices", None)
    if not kept_indices:
        return {}
    snapshot: dict[int, torch.Tensor] = {}
    for layer_idx, indices in kept_indices.items():
        if not torch.is_tensor(indices):
            indices = torch.as_tensor(indices, dtype=torch.long)
        snapshot[int(layer_idx)] = indices.detach().cpu().clone().to(dtype=torch.long)
    return snapshot


def _fixed_kept_indices_context(press, kept_indices: dict[int, torch.Tensor]):
    use_fixed_kept_indices = getattr(press, "use_fixed_kept_indices", None)
    if not callable(use_fixed_kept_indices):
        raise RuntimeError(
            "adapter_prefill_mode='base_score_adapter_prefill' requires a press "
            "with use_fixed_kept_indices() replay support."
        )
    return use_fixed_kept_indices(kept_indices)


def _run_context_prefill_forward(
    *,
    model,
    context_inputs,
    context_cache,
    press,
    effective_ratio: float,
    prefix_keep_tokens: int,
    adapter_prefill_mode: str,
    fixed_kept_indices: dict[int, torch.Tensor] | None = None,
) -> None:
    if press is None:
        if fixed_kept_indices is not None:
            raise RuntimeError("Fixed compression keep indices require a compression press.")
        press_scope = nullcontext()
        fixed_context = nullcontext()
        hook_context = nullcontext()
    else:
        _reset_press_indices(press)
        press_scope = _temporary_press_full_prompt_span(
            press,
            compression_ratio=effective_ratio,
            sink_size_start=prefix_keep_tokens,
            sink_size_end=0,
        )
        fixed_context = (
            _fixed_kept_indices_context(press, fixed_kept_indices)
            if fixed_kept_indices is not None
            else nullcontext()
        )
        hook_context = press(_press_target_for_model(model))

    with press_scope:
        with fixed_context:
            with _adapter_prefill_context(model, adapter_prefill_mode):
                with hook_context:
                    _forward_causal_lm(
                        model,
                        input_ids=context_inputs["input_ids"],
                        attention_mask=context_inputs.get("attention_mask"),
                        past_key_values=context_cache,
                    )


def _forward_causal_lm(model, **kwargs):
    try:
        return model(**kwargs, use_cache=True, logits_to_keep=1)
    except TypeError:
        try:
            return model(**kwargs, use_cache=True, num_logits_to_keep=1)
        except TypeError:
            return model(**kwargs, use_cache=True)


def _repeat_linear_attention_states_for_batch(layer, repeats: int) -> bool:
    """Repeat linear-attention state tensors that HF cache layers may not handle."""
    repeated = False
    max_batch_size = None
    for attr in ("conv_states", "recurrent_states"):
        state = getattr(layer, attr, None)
        if not isinstance(state, torch.Tensor):
            continue
        setattr(layer, attr, state.repeat_interleave(repeats, dim=0))
        max_batch_size = int(getattr(layer, attr).shape[0])
        repeated = True

    if repeated and hasattr(layer, "max_batch_size") and max_batch_size is not None:
        layer.max_batch_size = max_batch_size
    return repeated


def _repeat_cache_layer_for_batch(layer, repeats: int) -> None:
    linear_batch_sizes_before = {
        attr: int(state.shape[0])
        for attr in ("conv_states", "recurrent_states")
        if isinstance((state := getattr(layer, attr, None)), torch.Tensor)
    }
    repeat_fn = getattr(layer, "batch_repeat_interleave", None)
    if callable(repeat_fn):
        repeat_fn(repeats)
    elif not linear_batch_sizes_before:
        raise AttributeError(
            f"{type(layer).__name__} does not support batch_repeat_interleave "
            "and has no linear-attention state tensors to repeat."
        )

    # Some hybrid/linear-attention cache layers expose conv/recurrent states but
    # either do not implement batch_repeat_interleave at all, or inherit the
    # full-attention implementation that repeats only keys/values. Repeat those
    # state tensors here if their batch dimension was left unchanged.
    needs_linear_repeat = False
    for attr, before_batch_size in linear_batch_sizes_before.items():
        state = getattr(layer, attr, None)
        if isinstance(state, torch.Tensor) and int(state.shape[0]) == before_batch_size:
            needs_linear_repeat = True
            break
    if needs_linear_repeat:
        _repeat_linear_attention_states_for_batch(layer, repeats)


def _repeat_cache_for_batch(cache, batch_size: int):
    repeated = copy.deepcopy(cache)
    if batch_size == 1:
        return repeated

    repeats = int(batch_size)
    layers = getattr(repeated, "layers", None)
    if layers is not None:
        for layer in layers:
            _repeat_cache_layer_for_batch(layer, repeats)
    else:
        repeated.batch_repeat_interleave(repeats)
    return repeated


def _build_suffix_position_ids(
    *,
    suffix_attention_mask: torch.Tensor,
    logical_context_length: int,
) -> torch.Tensor:
    suffix_positions = suffix_attention_mask.long().cumsum(dim=-1) - 1
    suffix_positions = suffix_positions.masked_fill(suffix_attention_mask == 0, 0)
    suffix_positions = suffix_positions + int(logical_context_length)
    return suffix_positions.masked_fill(suffix_attention_mask == 0, 0)


def _generated_token_ids_without_terminators(
    token_ids: torch.Tensor,
    *,
    pad_token_id: int | None,
    terminator_ids: set[int],
) -> list[int]:
    out: list[int] = []
    for tok in token_ids.tolist():
        tok = int(tok)
        if pad_token_id is not None and tok == pad_token_id:
            break
        if tok in terminator_ids:
            break
        out.append(tok)
    return out


def _compute_confidence_from_generated_scores(
    *,
    generated_ids: torch.Tensor,
    scores: list[torch.Tensor],
    pad_token_id: int | None,
    terminator_ids: set[int],
) -> list[dict[str, float | int | None]]:
    batch_size = generated_ids.shape[0]
    logprob_sums = [0.0 for _ in range(batch_size)]
    token_counts = [0 for _ in range(batch_size)]
    for step_idx, step_scores in enumerate(scores):
        step_log_probs = torch.log_softmax(step_scores.float(), dim=-1)
        for batch_idx in range(batch_size):
            tok = int(generated_ids[batch_idx, step_idx].item())
            if pad_token_id is not None and tok == pad_token_id:
                continue
            if tok in terminator_ids:
                continue
            logprob_sums[batch_idx] += float(step_log_probs[batch_idx, tok].item())
            token_counts[batch_idx] += 1

    conf: list[dict[str, float | int | None]] = []
    for batch_idx in range(batch_size):
        n = token_counts[batch_idx]
        if n <= 0:
            conf.append(
                {
                    "confidence": None,
                    "confidence_mean_logprob": None,
                    "confidence_sum_logprob": None,
                    "confidence_token_count": 0,
                }
            )
            continue
        sum_lp = logprob_sums[batch_idx]
        mean_lp = sum_lp / n
        conf.append(
            {
                "confidence": float(math.exp(mean_lp)),
                "confidence_mean_logprob": float(mean_lp),
                "confidence_sum_logprob": float(sum_lp),
                "confidence_token_count": int(n),
            }
        )
    return conf


def _context_prefill_generate_batch(
    *,
    model,
    tokenizer,
    df,
    batch_row_indices: list[int],
    batch_answers: list,
    device,
    press,
    max_new_tokens: int,
    terminator_ids: set[int],
    answer_scorer_fn,
    collect_predictions: bool,
    row_ids: list,
    debug_print: bool,
    example_offset: int,
    adapter_prefill_mode: str = "adapter",
    perf_stats=None,
) -> tuple[float, list[dict] | None]:
    first_row = batch_row_indices[0]
    context_prefill_text = str(df.iloc[first_row]["context_prefill_text"])
    if any(str(df.iloc[row_idx]["context_prefill_text"]) != context_prefill_text for row_idx in batch_row_indices):
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' requires every suffix "
            "batch to share one exact context_prefill_text."
        )

    if "context_char_start" not in df.columns:
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' requires context_char_start metadata."
        )
    context_char_start = int(df.iloc[first_row]["context_char_start"])

    with (
        perf_stats.memory_stage("context_prefill_tokenize_to_device")
        if perf_stats is not None
        else nullcontext()
    ):
        context_inputs = tokenizer(
            context_prefill_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
    context_length = int(context_inputs["input_ids"].shape[1])
    prefix_keep_tokens = _context_prefix_token_len_from_offsets(
        tokenizer=tokenizer,
        context_prefill_text=context_prefill_text,
        context_char_start=context_char_start,
    )
    requested_ratio = float(getattr(press, "compression_ratio", 0.0))
    effective_ratio = _effective_context_only_ratio(
        requested_compression_ratio=requested_ratio,
        total_prefill_tokens=context_length,
        prefix_keep_tokens=prefix_keep_tokens,
    )

    context_cache = DynamicCache(config=getattr(model, "config", None))
    with torch.no_grad():
        if adapter_prefill_mode == BASE_SCORE_ADAPTER_PREFILL_MODE:
            if press is None or effective_ratio <= 0:
                raise RuntimeError(
                    "adapter_prefill_mode='base_score_adapter_prefill' "
                    "requires a compression press and compression_ratio > 0."
                )
            score_cache = DynamicCache(config=getattr(model, "config", None))
            with (
                perf_stats.inference_stage("context_prefill_base_score_compress")
                if perf_stats is not None
                else nullcontext()
            ):
                _run_context_prefill_forward(
                    model=model,
                    context_inputs=context_inputs,
                    context_cache=score_cache,
                    press=press,
                    effective_ratio=effective_ratio,
                    prefix_keep_tokens=prefix_keep_tokens,
                    adapter_prefill_mode="base",
                )
            kept_indices = _snapshot_press_kept_indices(press)
            if not kept_indices:
                raise RuntimeError(
                    "base_score_adapter_prefill captured no compression keep "
                    "indices from the base scoring pass."
                )
            del score_cache
            with (
                perf_stats.inference_stage("context_prefill_adapter_prefill_replay")
                if perf_stats is not None
                else nullcontext()
            ):
                _run_context_prefill_forward(
                    model=model,
                    context_inputs=context_inputs,
                    context_cache=context_cache,
                    press=press,
                    effective_ratio=effective_ratio,
                    prefix_keep_tokens=prefix_keep_tokens,
                    adapter_prefill_mode="adapter",
                    fixed_kept_indices=kept_indices,
                )
        else:
            with (
                perf_stats.inference_stage("context_prefill_compress")
                if perf_stats is not None
                else nullcontext()
            ):
                _run_context_prefill_forward(
                    model=model,
                    context_inputs=context_inputs,
                    context_cache=context_cache,
                    press=press,
                    effective_ratio=effective_ratio,
                    prefix_keep_tokens=prefix_keep_tokens,
                    adapter_prefill_mode=adapter_prefill_mode,
                )

    physical_context_length = int(context_cache.get_seq_length())
    suffix_texts = [str(df.iloc[row_idx]["question_suffix_text"]) for row_idx in batch_row_indices]
    with (
        perf_stats.memory_stage("suffix_tokenize_to_device")
        if perf_stats is not None
        else nullcontext()
    ):
        suffix_inputs = tokenizer(
            suffix_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        ).to(device)
    suffix_attention_mask = suffix_inputs["attention_mask"]
    batch_size = int(suffix_inputs["input_ids"].shape[0])
    if perf_stats is not None:
        perf_stats.add_context_prefill_cache_record(
            batch_size=batch_size,
            logical_context_tokens=context_length,
            physical_context_tokens=physical_context_length,
            prefix_keep_tokens=prefix_keep_tokens,
            requested_compression_ratio=requested_ratio,
            effective_compression_ratio=effective_ratio,
        )
    position_ids = _build_suffix_position_ids(
        suffix_attention_mask=suffix_attention_mask,
        logical_context_length=context_length,
    )
    physical_attention_mask = torch.cat(
        [
            torch.ones(
                (batch_size, physical_context_length),
                dtype=suffix_attention_mask.dtype,
                device=suffix_attention_mask.device,
            ),
            suffix_attention_mask,
        ],
        dim=1,
    )

    batch_cache = _repeat_cache_for_batch(context_cache, batch_size)
    with torch.no_grad():
        with (
            perf_stats.inference_stage("suffix_prefill")
            if perf_stats is not None
            else nullcontext()
        ):
            outputs = _forward_causal_lm(
                model,
                input_ids=suffix_inputs["input_ids"],
                attention_mask=physical_attention_mask,
                position_ids=position_ids,
                past_key_values=batch_cache,
            )
        next_scores = outputs.logits[:, -1, :]
        next_token = next_scores.argmax(dim=-1)
        score_steps = [next_scores.detach()]
        generated_steps = [next_token.detach().clone()]
        finished = torch.zeros((batch_size,), dtype=torch.bool, device=next_token.device)
        if terminator_ids:
            terminator_tensor = torch.tensor(sorted(terminator_ids), device=next_token.device)
            finished |= (next_token[:, None] == terminator_tensor[None, :]).any(dim=1)

        running_attention_mask = torch.cat(
            [
                physical_attention_mask,
                torch.ones((batch_size, 1), dtype=physical_attention_mask.dtype, device=physical_attention_mask.device),
            ],
            dim=1,
        )
        next_position_ids = (position_ids[:, -1:] + 1).contiguous()
        pad_or_eos = tokenizer.pad_token_id
        if pad_or_eos is None:
            pad_or_eos = tokenizer.eos_token_id

        with (
            perf_stats.inference_stage("decode")
            if perf_stats is not None
            else nullcontext()
        ):
            for _step in range(max(0, int(max_new_tokens) - 1)):
                input_token = next_token.masked_fill(finished, int(pad_or_eos))
                outputs = _forward_causal_lm(
                    model,
                    input_ids=input_token[:, None],
                    attention_mask=running_attention_mask,
                    position_ids=next_position_ids,
                    past_key_values=batch_cache,
                )
                next_scores = outputs.logits[:, -1, :]
                next_token = next_scores.argmax(dim=-1)
                next_token = next_token.masked_fill(finished, int(pad_or_eos))
                generated_steps.append(next_token.detach().clone())
                score_steps.append(next_scores.detach())
                if terminator_ids:
                    finished |= (next_token[:, None] == terminator_tensor[None, :]).any(dim=1)
                if bool(finished.all().item()):
                    break
                running_attention_mask = torch.cat(
                    [
                        running_attention_mask,
                        torch.ones(
                            (batch_size, 1),
                            dtype=running_attention_mask.dtype,
                            device=running_attention_mask.device,
                        ),
                    ],
                    dim=1,
                )
                next_position_ids = next_position_ids + 1

    generated_ids = torch.stack(generated_steps, dim=1)
    confidence_batch = _compute_confidence_from_generated_scores(
        generated_ids=generated_ids,
        scores=score_steps,
        pad_token_id=tokenizer.pad_token_id,
        terminator_ids=terminator_ids,
    )

    correct = 0.0
    predictions = [] if collect_predictions else None
    for j, answer in enumerate(batch_answers):
        token_ids = _generated_token_ids_without_terminators(
            generated_ids[j],
            pad_token_id=tokenizer.pad_token_id,
            terminator_ids=terminator_ids,
        )
        raw_decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        score = float(answer_scorer_fn(raw_decoded, answer))
        is_correct = score
        correct += score
        display_pred = raw_decoded.strip()
        if debug_print:
            print(
                f"  Example {example_offset + j}: Generated='{display_pred}', "
                f"Expected='{answer}', Correct={is_correct}"
            )
        if collect_predictions:
            conf_fields = confidence_batch[j]
            record = {
                "_row_position": int(batch_row_indices[j]),
                "index": _prediction_row_id(row_ids[batch_row_indices[j]]),
                "predicted": display_pred,
                "label": answer,
                "correct": is_correct,
                "confidence": conf_fields["confidence"],
                "confidence_mean_logprob": conf_fields["confidence_mean_logprob"],
                "confidence_sum_logprob": conf_fields["confidence_sum_logprob"],
                "confidence_token_count": conf_fields["confidence_token_count"],
            }
            predictions.append(record)

    return correct, predictions


def _batch_get_predictions_context_prefill_generate(
    model,
    tokenizer,
    df,
    device="cuda",
    batch_size=16,
    press=None,
    debug_print=False,
    max_length=None,
    eval_every_n=1,
    shard_id: int = 0,
    num_shards: int = 1,
    collect_predictions: bool = False,
    max_new_tokens: int = 10,
    answer_scorer_fn=None,
    adapter_prefill_mode: str = "adapter",
    perf_stats=None,
):
    if press is None and adapter_prefill_mode == "adapter":
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' without KV compression "
            "is only used for non-adapter prefill diagnostics."
        )
    if press is not None and getattr(press, "compression_scope", "context_only") != "context_only":
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' expects the requested "
            "compression config to use compression_scope='context_only'."
        )
    if max_length is not None:
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' currently requires "
            "--disable_truncation / max_length=None."
        )
    if num_shards > 1:
        raise RuntimeError(
            "compressed_generation_mode='context_prefill' does not currently "
            "support sharded evaluation; run with num_shards=1."
        )

    terminators = [
        t
        for t in [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        if t is not None
    ]
    terminator_ids = set(int(t) for t in terminators)

    all_answers = list(df["label"])
    row_ids = list(df.index)
    indices = shard_indices(len(df), eval_every_n, shard_id, num_shards)
    eval_batches = _build_context_prefill_batches(df, indices, batch_size)

    if eval_every_n > 1:
        print(f"Evaluating every {eval_every_n} samples: {len(indices)}/{len(df)} samples")

    print(f"\n{'='*60}")
    if press is None:
        print("Context-prefill ENABLED without KV compression:")
        print("  compression_ratio: 0.0")
        print("  chunk_size: n/a")
    else:
        print("KV compression ENABLED:")
        print(f"  compression_ratio: {press.compression_ratio}")
        print(f"  chunk_size: {press.chunk_size}")
    print("  compressed_generation_mode: context_prefill")
    print(f"  adapter_prefill_mode: {adapter_prefill_mode}")
    if press is None:
        print("  context compression: disabled; full context cache is retained")
    else:
        print("  context compression: press over template+context with prefix sink")
    print(f"{'='*60}\n")

    correct = 0.0
    predictions: list = [] if collect_predictions else None  # type: ignore[assignment]
    example_idx = 0
    printed_first_example = False
    for batch_idx, batch_row_indices in enumerate(tqdm(eval_batches, desc="Evaluating", unit="batch")):
        batch_answers = [all_answers[i] for i in batch_row_indices]
        if debug_print and not printed_first_example:
            prompt = str(df.iloc[batch_row_indices[0]]["text"])
            print(f"\n{'='*80}")
            print("FIRST EXAMPLE OF DATASET:")
            print(f"  Total chars: {len(prompt)}")
            print(f"  Total tokens: {len(tokenizer.encode(prompt))}")
            print(f"{'='*80}")
            print("--- FULL INPUT ---")
            print(prompt)
            print("--- END OF INPUT ---")
            print(f"  Expected answer: {batch_answers[0]}")
            print(f"{'='*80}")
            printed_first_example = True

        batch_correct, batch_predictions = _context_prefill_generate_batch(
            model=model,
            tokenizer=tokenizer,
            df=df,
            batch_row_indices=batch_row_indices,
            batch_answers=batch_answers,
            device=device,
            press=press,
            max_new_tokens=max_new_tokens,
            terminator_ids=terminator_ids,
            answer_scorer_fn=answer_scorer_fn,
            collect_predictions=collect_predictions,
            row_ids=row_ids,
            debug_print=debug_print,
            example_offset=example_idx,
            adapter_prefill_mode=adapter_prefill_mode,
            perf_stats=perf_stats,
        )
        correct += batch_correct
        if collect_predictions and batch_predictions:
            predictions.extend(batch_predictions)
        example_idx += len(batch_row_indices)

        if debug_print and example_idx >= 3 * batch_size:
            print(f"\n[Debug] Stopping debug output after {example_idx} examples...")
            debug_print = False

    if collect_predictions:
        predictions.sort(key=lambda item: item.pop("_row_position"))
        return correct, len(indices), predictions
    return correct, len(indices)


def batch_get_predictions(
    model,
    tokenizer,
    df,
    device="cuda",
    batch_size=16,
    press=None,
    debug_print=False,
    max_length=None,
    eval_every_n=1,
    shard_id: int = 0,
    num_shards: int = 1,
    collect_predictions: bool = False,
    max_new_tokens: int = 10,
    answer_scorer_fn=None,
    compressed_generation_mode: str = "context_prefill",
    adapter_prefill_mode: str = "adapter",
    perf_stats=None,
):
    """Evaluate model predictions with optional KV cache compression.

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for the model.
        df: DataFrame with formatted generative-QA prompts in ``"text"`` and
            gold-answer objects in ``"label"``.
        device: Device to run on.
        batch_size: Batch size for evaluation.
        press: Optional compression press for the KV cache.
        debug_print: Whether to print debug info for each example.
        max_length: Optional max sequence length for truncation.
        eval_every_n: Evaluate every N samples (default: 1, i.e., all samples).
        shard_id: Which shard to evaluate (0-indexed).
        num_shards: Total number of shards.
        collect_predictions: If True, return a list of per-sample prediction dicts
            alongside (correct_sum, total).  Each dict has keys: index, predicted,
            label, correct.
        max_new_tokens: Maximum tokens to generate per example (default: 10).
        answer_scorer_fn: Callable ``(decoded_text, label) -> float`` used by
            the generative-QA benchmark. The decoded text is passed through
            before stripping, and labels are taken directly from ``df["label"]``.
        adapter_prefill_mode: ``"adapter"`` keeps PEFT adapters active during
            context prefill/compression.  ``"base"`` disables adapters only for
            that prefill and re-enables them for suffix/decode.
            ``"base_score_adapter_prefill"`` first scores/compresses with base
            adapters disabled, then replays the same kept indices while building
            the actual context KV cache with adapters enabled.
    """
    if adapter_prefill_mode not in ADAPTER_PREFILL_MODES:
        raise ValueError(
            "adapter_prefill_mode must be one of "
            f"{sorted(ADAPTER_PREFILL_MODES)}; "
            f"got {adapter_prefill_mode!r}"
        )
    if compressed_generation_mode not in {"full_prompt", "context_prefill"}:
        raise ValueError(
            "compressed_generation_mode must be 'full_prompt' or 'context_prefill'; "
            f"got {compressed_generation_mode!r}"
        )
    if answer_scorer_fn is None:
        raise ValueError("Generative-QA evaluation requires answer_scorer_fn.")
    if adapter_prefill_mode != "adapter" and compressed_generation_mode != "context_prefill":
        raise RuntimeError(
            "non-adapter prefill modes only have defined semantics for "
            "compressed_generation_mode='context_prefill'."
        )
    if compressed_generation_mode == "context_prefill" and (
        press is not None or adapter_prefill_mode != "adapter"
    ):
        return _batch_get_predictions_context_prefill_generate(
            model,
            tokenizer,
            df,
            device=device,
            batch_size=batch_size,
            press=press,
            debug_print=debug_print,
            max_length=max_length,
            eval_every_n=eval_every_n,
            shard_id=shard_id,
            num_shards=num_shards,
            collect_predictions=collect_predictions,
            max_new_tokens=max_new_tokens,
            answer_scorer_fn=answer_scorer_fn,
            adapter_prefill_mode=adapter_prefill_mode,
            perf_stats=perf_stats,
        )

    # Sequence Terminators
    # Build terminator list, filtering out None for models (e.g. Qwen) that
    # don't have the Llama-specific <|eot_id|> token in their vocabulary.
    terminators = [
        t for t in [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        if t is not None
    ]
    terminator_ids = set(int(t) for t in terminators)

    all_prompts = list(df["text"])
    row_ids = list(df.index)
    compression_scope = getattr(press, "compression_scope", "context_only") if press is not None else None
    if press is not None and compression_scope not in {"full_prompt", "context_only"}:
        raise RuntimeError(
            "KV compression in generation supports compression_scope='full_prompt' "
            "or compression_scope='context_only'."
        )
    all_answers = list(df["label"])

    # Subsample + shard (shuffle-balanced for even GPU utilisation).
    indices = shard_indices(len(all_prompts), eval_every_n, shard_id, num_shards)
    if num_shards > 1:
        print(
            f"Sharding enabled: shard {shard_id}/{num_shards} -> {len(indices)} samples "
            f"(after eval_every_n={eval_every_n})"
        )

    need_context_mask = _press_uses_context_only_compression(press)
    if need_context_mask and num_shards > 1:
        raise RuntimeError(
            "compression_scope='context_only' does not currently support "
            "sharded evaluation; run with num_shards=1."
        )
    eval_batches = (
        _build_context_only_batches(df, indices, batch_size)
        if need_context_mask
        else [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    )

    if eval_every_n > 1:
        print(f"Evaluating every {eval_every_n} samples: {len(indices)}/{len(all_prompts)} samples")

    correct = 0
    predictions: list = [] if collect_predictions else None  # type: ignore[assignment]

    # Print compression info once at the start
    if press is not None:
        print(f"\n{'='*60}")
        print(f"KV compression ENABLED:")
        print(f"  compression_ratio: {press.compression_ratio}")
        print(f"  chunk_size: {press.chunk_size}")
        print(f"  sink_size_start: {press.sink_size_start}")
        print(f"  sink_size_end: {press.sink_size_end}")
        print(f"  compression_scope: {getattr(press, 'compression_scope', 'context_only')}")
        if getattr(press, "compression_scope", "context_only") == "context_only":
            print(f"  context_min_keep_tokens: {getattr(press, 'context_min_keep_tokens', 0)}")
        print(f"{'='*60}\n")

    if max_length is not None:
        print(f"Truncation enabled: max_length={max_length}")

    example_idx = 0
    printed_first_example = False
    for batch_idx, batch_row_indices in enumerate(tqdm(eval_batches, desc="Evaluating", unit="batch")):
        batch_prompts = [all_prompts[i] for i in batch_row_indices]
        batch_answers = [all_answers[i] for i in batch_row_indices]

        # Tokenize with optional truncation
        with (
            perf_stats.memory_stage("full_prompt_tokenize_to_device")
            if perf_stats is not None
            else nullcontext()
        ):
            if max_length is not None:
                old_truncation_side = getattr(tokenizer, "truncation_side", None)
                if old_truncation_side is not None:
                    # Long-context QA prompts put the question at the tail; keep it under caps.
                    tokenizer.truncation_side = "left"
                try:
                    tokenize_kwargs = {
                        "return_tensors": "pt",
                        "padding": True,
                        "truncation": True,
                        "max_length": max_length,
                    }
                    if need_context_mask:
                        tokenize_kwargs["return_offsets_mapping"] = True
                    tokenized = tokenizer(batch_prompts, **tokenize_kwargs)
                finally:
                    if old_truncation_side is not None:
                        tokenizer.truncation_side = old_truncation_side
            else:
                tokenize_kwargs = {
                    "return_tensors": "pt",
                    "padding": True,
                }
                if need_context_mask:
                    tokenize_kwargs["return_offsets_mapping"] = True
                tokenized = tokenizer(batch_prompts, **tokenize_kwargs)
            offset_mapping = tokenized.pop("offset_mapping", None) if need_context_mask else None
            inputs = tokenized.to(device)
        length = inputs["input_ids"].shape[1]
        if max_length is None:
            _raise_if_untruncated_prompt_exceeds_model(
                model,
                prompt_length=length,
                max_new_tokens=max_new_tokens,
            )

        if need_context_mask:
            context_spans = _context_char_spans_for_rows(df, batch_row_indices)
            if context_spans is None:
                raise RuntimeError(
                    "compression_scope='context_only' requires dataframe columns "
                    "'context_char_start' and 'context_char_end'."
                )
            context_mask = _build_context_compressible_mask(
                offset_mapping=offset_mapping,
                context_spans=context_spans,
                attention_mask=inputs["attention_mask"],
            )
            press.set_context_compressible_mask(
                context_mask,
                attention_mask=inputs["attention_mask"],
            )

        # Debug print: show ONLY the first example of the dataset (full content)
        if debug_print and not printed_first_example:
            prompt = batch_prompts[0]
            print(f"\n{'='*80}")
            print(f"FIRST EXAMPLE OF DATASET:")
            print(f"  Total chars: {len(prompt)}")
            print(f"  Total tokens: {len(tokenizer.encode(prompt))}")
            print(f"{'='*80}")
            print(f"--- FULL INPUT ---")
            print(prompt)
            print(f"--- END OF INPUT ---")
            print(f"  Expected answer: {batch_answers[0]}")
            print(f"{'='*80}")
            printed_first_example = True

        with torch.no_grad():
            should_return_dict = bool((debug_print and batch_idx == 0) or collect_predictions)
            confidence_batch: list[dict[str, float | int | None]] | None = None

            # Apply the KV cache compression press if provided.
            if press is not None:
                # kvpress navigates model.model.layers; for PEFT-wrapped models
                # (PeftModelForCausalLM) we must pass the underlying CausalLM so
                # hooks are registered on the correct layer objects.
                _press_target = (
                    model.base_model.model
                    if hasattr(model, "base_model") and hasattr(model.base_model, "model")
                    else model
                )
                with press(_press_target) as press_context:
                    # Print KV cache info before generation
                    if debug_print and batch_idx == 0:
                        _debug_print_compactorpress_start(press=press, length=length)

                    with (
                        perf_stats.inference_stage("full_prompt_generate")
                        if perf_stats is not None
                        else nullcontext()
                    ):
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=False,
                            temperature=None,
                            top_p=None,
                            eos_token_id=terminators,
                            pad_token_id=tokenizer.pad_token_id,
                            return_dict_in_generate=should_return_dict,
                            output_scores=collect_predictions,
                        )

                    # Try to print compression stats after first batch
                    if debug_print and batch_idx == 0:
                        _debug_print_compactorpress_end(outputs=outputs, press_context=press_context)

                    has_sequences = should_return_dict and hasattr(outputs, "sequences")
                    sequences = outputs.sequences if has_sequences else outputs

                    if collect_predictions and has_sequences and hasattr(outputs, "scores"):
                        confidence_batch = _compute_generation_confidence_from_scores(
                            sequences=sequences,
                            scores=outputs.scores,
                            prompt_padded_len=length,
                            pad_token_id=tokenizer.pad_token_id,
                            terminator_ids=terminator_ids,
                        )
            else:
                with (
                    perf_stats.inference_stage("full_prompt_generate")
                    if perf_stats is not None
                    else nullcontext()
                ):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        eos_token_id=terminators,
                        pad_token_id=tokenizer.pad_token_id,
                        return_dict_in_generate=should_return_dict,
                        output_scores=collect_predictions,
                    )

                has_sequences = should_return_dict and hasattr(outputs, "sequences")
                sequences = outputs.sequences if has_sequences else outputs

                if collect_predictions and has_sequences and hasattr(outputs, "scores"):
                    confidence_batch = _compute_generation_confidence_from_scores(
                        sequences=sequences,
                        scores=outputs.scores,
                        prompt_padded_len=length,
                        pad_token_id=tokenizer.pad_token_id,
                        terminator_ids=terminator_ids,
                    )

            if confidence_batch is None:
                confidence_batch = [
                    {
                        "confidence": None,
                        "confidence_mean_logprob": None,
                        "confidence_sum_logprob": None,
                        "confidence_token_count": 0,
                    }
                    for _ in range(len(batch_answers))
                ]

            for j, (output, answer) in enumerate(zip(sequences, batch_answers)):
                raw_decoded = tokenizer.decode(output[length:], skip_special_tokens=True)
                score = float(answer_scorer_fn(raw_decoded, answer))
                is_correct = score
                correct += score
                display_pred = raw_decoded.strip()
                if debug_print:
                    print(f"  Example {example_idx + j}: Generated='{display_pred}', Expected='{answer}', Correct={is_correct}")
                if collect_predictions:
                    conf_fields = confidence_batch[j]
                    predictions.append({
                        "_row_position": int(batch_row_indices[j]),
                        "index": _prediction_row_id(row_ids[batch_row_indices[j]]),
                        "predicted": display_pred,
                        "label": answer,
                        "correct": is_correct,
                        "confidence": conf_fields["confidence"],
                        "confidence_mean_logprob": conf_fields["confidence_mean_logprob"],
                        "confidence_sum_logprob": conf_fields["confidence_sum_logprob"],
                        "confidence_token_count": conf_fields["confidence_token_count"],
                    })

        example_idx += len(batch_prompts)

        # Only print first few batches to avoid too much output
        if debug_print and example_idx >= 3 * batch_size:
            print(f"\n[Debug] Stopping debug output after {example_idx} examples...")
            debug_print = False

    if collect_predictions:
        predictions.sort(key=lambda item: item.pop("_row_position"))
        return correct, len(indices), predictions
    return correct, len(indices)


__all__ = ["batch_get_predictions"]
