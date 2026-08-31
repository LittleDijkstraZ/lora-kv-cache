# SPDX-FileCopyrightText: Copyright 2026 Chunsheng Zuo and contributors
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright Vivek Chari
# SPDX-License-Identifier: Apache-2.0
#
# Portions adapted from NVIDIA/kvpress 0.5.3 and modified by LoRA-KV-Cache contributors
# in 2026. See THIRD_PARTY_NOTICES.md for attribution and licensing details.

"""Context-aware wrappers for Compactor and Expected Attention compression."""

import logging
from contextlib import contextmanager
from typing import Generator

import torch
import torch.nn.functional as F
from kvpress import CompactorPress
from kvpress.utils import extract_keys_and_values

try:
    from kvpress.presses.expected_attention_press import ExpectedAttentionPress
except ImportError:  # pragma: no cover - depends on kvpress version
    ExpectedAttentionPress = None

log = logging.getLogger(__name__)


class CompactorScorer:
    """Score tokens using the local Compactor scoring implementation."""

    def __init__(self, owner: "IndexedScorerPress") -> None:
        self.owner = owner

    def score(self, **kwargs) -> torch.Tensor:
        return self.owner._score_compactor(**kwargs)


class ExpectedAttentionScorer:
    """Score tokens with kvpress ExpectedAttention while reusing local indexing."""

    def __init__(
        self,
        owner: "IndexedScorerPress",
        *,
        n_future_positions: int = 512,
        n_sink: int = 4,
        use_covariance: bool = True,
        use_vnorm: bool = True,
        epsilon: float = 0.0,
    ) -> None:
        if ExpectedAttentionPress is None:
            raise ImportError(
                "kvpress ExpectedAttentionPress is unavailable in this environment."
            )
        self.owner = owner
        self._expected_attention_press = ExpectedAttentionPress(
            compression_ratio=owner.compression_ratio,
            n_future_positions=int(n_future_positions),
            n_sink=int(n_sink),
            use_covariance=bool(use_covariance),
            use_vnorm=bool(use_vnorm),
            epsilon=float(epsilon),
        )

    def score(self, **kwargs) -> torch.Tensor:
        press = self._expected_attention_press
        press.compression_ratio = self.owner.compression_ratio
        old_n_sink = press.n_sink
        # Upstream ExpectedAttention requires at least one non-sink token.  The
        # Context-only compression may pass short document spans, so shrink the
        # inner sink count for that score call instead of failing the eval.
        keys = kwargs.get("keys")
        if isinstance(keys, torch.Tensor):
            press.n_sink = min(int(old_n_sink), max(0, int(keys.shape[2]) - 1))
        try:
            return press.score(**kwargs)
        finally:
            press.n_sink = old_n_sink


class IndexedScorerPress(CompactorPress):
    """
    KV press wrapper that captures kept token indices.

    The outer class owns context-only masking, top-k selection, fixed-index
    replay, and KV gather. A scorer object supplies Compactor or Expected
    Attention token scores.

    After compression, `self.kept_indices` contains a dict mapping layer index
    to the token indices that were kept for that layer.
    """

    def __init__(self, *args, **kwargs):
        compression_algorithm = str(
            kwargs.pop("compression_algorithm", kwargs.pop("algorithm", "compactor"))
        ).strip().lower().replace("-", "_")
        expected_attention_kwargs = {
            "n_future_positions": kwargs.pop("n_future_positions", 512),
            "n_sink": kwargs.pop("n_sink", 4),
            "use_covariance": kwargs.pop("use_covariance", True),
            "use_vnorm": kwargs.pop("use_vnorm", True),
            "epsilon": kwargs.pop("epsilon", 0.0),
        }
        self.compression_scope = str(kwargs.pop("compression_scope", "full_prompt"))
        if self.compression_scope not in {"full_prompt", "context_only"}:
            raise ValueError(
                "compression_scope must be 'full_prompt' or 'context_only'; "
                f"got {self.compression_scope!r}"
            )
        self.context_min_keep_tokens = int(
            kwargs.pop("context_min_keep_tokens", kwargs.pop("min_context_keep_tokens", 0))
        )
        if compression_algorithm not in {"compactor", "expected_attention"}:
            raise ValueError(
                "compression_algorithm must be 'compactor' or 'expected_attention'; "
                f"got {compression_algorithm!r}"
            )
        self.compression_algorithm = compression_algorithm
        self.deterministic_leverage = bool(kwargs.pop("deterministic_leverage", True))
        self.deterministic_leverage_seed = int(kwargs.pop("deterministic_leverage_seed", 1729))
        super().__init__(*args, **kwargs)
        if self.compression_algorithm == "expected_attention":
            self._scorer = ExpectedAttentionScorer(self, **expected_attention_kwargs)
        else:
            self._scorer = CompactorScorer(self)
        self.kept_indices = {}  # layer_idx -> indices tensor
        self._layer_counter = 0
        self._fixed_kept_indices: dict[int, torch.Tensor] | None = None
        self.leverage_fallback_count = 0
        # In context-only mode, evaluation code sets these once per tokenized
        # batch.  The context mask marks tokens that may be pruned; the valid
        # mask excludes padding so padding is never forced into the cache.
        self._context_compressible_mask: torch.Tensor | None = None
        self._valid_token_mask: torch.Tensor | None = None

    @contextmanager
    def _deterministic_leverage_rng(self, *, module, device: torch.device) -> Generator:
        """Make Compactor's random Gaussian sketch reproducible per layer.

        kvpress leverage scoring draws a fresh random sketch with ``torch.randn``.
        Without isolating that RNG, repeated evals and different suffix batch
        sizes can choose different compressed KV tokens for the same context.
        """
        if not self.deterministic_leverage:
            yield
            return

        if device.type == "cuda":
            device_index = device.index
            if device_index is None and torch.cuda.is_available():
                device_index = torch.cuda.current_device()
            devices = [int(device_index)] if device_index is not None else []
        else:
            devices = []

        layer_idx = int(getattr(module, "layer_idx", self._layer_counter))
        seed = int(self.deterministic_leverage_seed) + layer_idx
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            yield

    @contextmanager
    def __call__(self, model) -> Generator:
        """Apply kvpress hooks to full-attention decoder layers."""
        self.post_init_from_model(model)
        hooks = []
        skipped = 0
        try:
            language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
            for layer in language_model.layers:
                self_attn = getattr(layer, "self_attn", None)
                if self_attn is None:
                    skipped += 1
                    continue
                if getattr(self_attn, "is_sliding", False):
                    continue
                if hasattr(language_model, "rotary_emb"):
                    self_attn.rotary_emb = language_model.rotary_emb
                # kvpress works by installing forward hooks on attention
                # modules.  The hook rewrites each layer's KV cache after the
                # prompt prefill pass, before decode tokens are generated.
                hooks.append(self_attn.register_forward_hook(self.forward_hook, with_kwargs=True))

            if not hooks:
                raise RuntimeError(f"No self-attention layers found for kvpress hooks in {type(model)}")
            if skipped:
                log.info(
                    "kvpress skipped %d layer(s) without self_attn; "
                    "compression will apply to %d full-attention layer(s).",
                    skipped,
                    len(hooks),
                )
            yield self
        finally:
            for forward_hook in hooks:
                forward_hook.remove()

    def reset_indices(self):
        """Clear kept indices before a new forward pass."""
        self.kept_indices = {}
        self._layer_counter = 0

    @contextmanager
    def use_fixed_kept_indices(self, kept_indices: dict[int, torch.Tensor]) -> Generator:
        """Replay a previously captured keep-index plan instead of rescoring.

        This is used by the base-score/adapter-prefill diagnostic: the base
        model chooses which context KV positions survive compression, then the
        adapter reruns prefill while gathering exactly those positions.
        """
        normalized = {
            int(layer_idx): indices.detach().cpu().to(dtype=torch.long)
            for layer_idx, indices in kept_indices.items()
        }
        old_fixed_kept_indices = self._fixed_kept_indices
        self._fixed_kept_indices = normalized
        try:
            yield self
        finally:
            self._fixed_kept_indices = old_fixed_kept_indices

    def forward_hook(self, module, input, kwargs: dict, output):
        """Compress the KV cache during prompt prefill."""
        hidden_states = kwargs["hidden_states"]
        cache = kwargs.get("past_key_values") or kwargs.get("past_key_value")
        if cache is None:
            return output

        q_len = hidden_states.shape[1]
        cache_position = kwargs.get("cache_position")
        # Compress only the prompt prefill.  During decode q_len is 1, and the
        # cache already contains the compressed prompt plus previous decode KV.
        if cache_position is not None:
            if cache_position[-1] > q_len:
                return output
        elif q_len <= 1:
            return output

        cache_layer = cache.layers[module.layer_idx]
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        keys, values = self.compress(module, hidden_states, keys, values, output[1], kwargs)

        cache_layer.keys = keys
        cache_layer.values = values

        return output

    def set_context_compressible_mask(
        self,
        context_mask: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """Set the per-batch token mask used by ``compression_scope=context_only``.

        ``context_mask`` is ``True`` exactly for context tokens that may be
        pruned. Valid non-context tokens are forced to stay in the KV cache.
        """
        if context_mask.ndim != 2:
            raise ValueError(f"context_mask must have shape [batch, seq]; got {tuple(context_mask.shape)}")
        self._context_compressible_mask = context_mask.detach().bool()
        self._valid_token_mask = attention_mask.detach().bool() if attention_mask is not None else None

    def clear_context_compressible_mask(self) -> None:
        """Clear any context-only mask from the previous batch."""
        self._context_compressible_mask = None
        self._valid_token_mask = None

    def _get_context_masks(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._context_compressible_mask is None:
            raise RuntimeError(
                "compression_scope='context_only' requires a context mask. "
                "The evaluation dataframe must provide context span metadata."
            )
        context_mask = self._context_compressible_mask.to(device=device)
        if context_mask.shape != (batch_size, seq_len):
            raise ValueError(
                "context mask shape does not match current KV sequence: "
                f"mask={tuple(context_mask.shape)} expected={(batch_size, seq_len)}"
            )
        if self._valid_token_mask is None:
            valid_mask = torch.ones_like(context_mask, dtype=torch.bool, device=device)
        else:
            valid_mask = self._valid_token_mask.to(device=device)
            if valid_mask.shape != (batch_size, seq_len):
                raise ValueError(
                    "valid-token mask shape does not match current KV sequence: "
                    f"mask={tuple(valid_mask.shape)} expected={(batch_size, seq_len)}"
                )
        return context_mask & valid_mask, valid_mask

    def _score_compactor(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        l_scores = None
        try:
            # Compactor blends leverage scores with non-causal attention scores.
            # The leverage solve can be numerically fragile on some layers, so
            # fall back to attention-only scores when it fails in known ways.
            with self._deterministic_leverage_rng(module=module, device=hidden_states.device):
                l_scores = self._leverage_press.score(
                    module=module,
                    hidden_states=hidden_states,
                    keys=keys,
                    values=values,
                    attentions=attentions,
                    kwargs=kwargs,
                )
            if not torch.isfinite(l_scores).all():
                raise RuntimeError("Leverage scores contain non-finite values.")
        except RuntimeError as exc:
            if "Cholesky failed" not in str(exc) and "non-finite" not in str(exc):
                raise
            self.leverage_fallback_count += 1
            if self.leverage_fallback_count <= 3:
                log.warning(
                    "Compactor leverage scoring failed (%s); falling back to "
                    "non-causal attention scores for this layer.",
                    exc,
                )

        attn_scores = self._non_causal_press.score(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )

        if l_scores is None:
            return attn_scores
        assert attn_scores.shape == l_scores.shape, "CompactorPress only supports prefill at the moment"
        blending = self.blending if self.blending is not None else self.compression_ratio
        blending = 0.35 if blending is None else blending
        return blending * l_scores + attn_scores

    def _score_inner(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        return self._scorer.score(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )

    def _score_context_only(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[-2]
        context_mask, valid_mask = self._get_context_masks(
            batch_size=batch_size,
            seq_len=seq_len,
            device=hidden_states.device,
        )
        # In context-only mode, the contract is to prune only the document
        # context. The sink_size_* knobs apply only to full-prompt pruning;
        # questions, instructions, and chat markup survive
        # because every non-context token is forced to +inf below.
        scores = keys.new_full((keys.shape[0], keys.shape[1], keys.shape[2]), float("-inf"))
        cos, sin = kwargs["position_embeddings"]

        for batch_idx in range(batch_size):
            context_positions = context_mask[batch_idx].nonzero(as_tuple=False).flatten()
            if context_positions.numel() == 0:
                continue
            # The eval prompt builds context as one contiguous span.  Score that
            # slice only, leaving all other prompt tokens protected.
            start = int(context_positions[0].item())
            end = int(context_positions[-1].item()) + 1
            sliced_kwargs = {
                "position_embeddings": (
                    cos[batch_idx : batch_idx + 1, start:end, :],
                    sin[batch_idx : batch_idx + 1, start:end, :],
                )
            }
            sample_scores = self._score_inner(
                module=module,
                hidden_states=hidden_states[batch_idx : batch_idx + 1, start:end, :],
                keys=keys[batch_idx : batch_idx + 1, :, start:end, :],
                values=values[batch_idx : batch_idx + 1, :, start:end, :],
                attentions=attentions,
                kwargs=sliced_kwargs,
            )
            scores[batch_idx : batch_idx + 1, :, start:end] = sample_scores

        forced_keep_mask = valid_mask & ~context_mask
        return scores.masked_fill(forced_keep_mask[:, None, :], float("inf"))

    def score(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> torch.Tensor:
        """Score KV positions and protect non-compressible tokens."""
        n_queries = hidden_states.shape[-2]
        assert keys.shape[-2] == n_queries, "CompactorPress only supports prefill at the moment"

        if self.compression_scope == "context_only":
            return self._score_context_only(module, hidden_states, keys, values, attentions, kwargs)

        # Legacy full-prompt mode still protects fixed sink tokens at both
        # ends, but everything between those sinks is compressible.
        left_keep = min(self.sink_size_start, n_queries)
        right_keep = min(self.sink_size_end, max(0, n_queries - left_keep))
        start_idx, end_idx = left_keep, (None if right_keep == 0 else -right_keep)

        hs = hidden_states[:, start_idx:end_idx]
        inner_keys = keys[..., start_idx:end_idx, :]
        inner_values = values[..., start_idx:end_idx, :]
        cos, sin = kwargs["position_embeddings"]
        sliced_kwargs = {
            "position_embeddings": (
                cos[..., start_idx:end_idx, :],
                sin[..., start_idx:end_idx, :],
            )
        }

        scores = self._score_inner(module, hs, inner_keys, inner_values, attentions, sliced_kwargs)

        return F.pad(scores, (left_keep, right_keep), value=scores.detach().max())

    def _n_kept(self, k_len: int, device: torch.device) -> int:
        if self.compression_scope != "context_only":
            return int(k_len * (1 - self.compression_ratio))

        context_mask, valid_mask = self._get_context_masks(
            batch_size=int(self._context_compressible_mask.shape[0]),
            seq_len=k_len,
            device=device,
        )
        context_counts = context_mask.sum(dim=1)
        forced_keep_counts = (valid_mask & ~context_mask).sum(dim=1)
        context_keep_counts = torch.floor(
            context_counts.float() * float(1 - self.compression_ratio)
        ).to(torch.long)
        if self.context_min_keep_tokens > 0:
            min_keep = torch.full_like(context_keep_counts, int(self.context_min_keep_tokens))
            context_keep_counts = torch.where(
                context_counts > 0,
                torch.maximum(context_keep_counts, torch.minimum(context_counts, min_keep)),
                context_keep_counts,
            )
        context_keep_counts = torch.minimum(context_keep_counts, context_counts)
        target_counts = forced_keep_counts + context_keep_counts
        if target_counts.numel() == 0:
            return 0
        # torch.topk uses one k for the whole batch.  Use the largest per-row
        # target so every row has enough slots for all forced-kept tokens; rows
        # with shorter contexts may retain a little extra context.
        return min(k_len, int(target_counts.max().item()))

    def compress(
        self,
        module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override compress to capture kept indices.
        """
        if self.compression_ratio == 0:
            return keys, values

        k_len = keys.shape[2]
        current_layer = int(self._layer_counter)
        if self._fixed_kept_indices is not None:
            fixed_indices = self._fixed_kept_indices.get(current_layer)
            if fixed_indices is None:
                raise RuntimeError(
                    "Fixed Compactor keep indices are missing for layer "
                    f"{current_layer}; available layers={sorted(self._fixed_kept_indices)}"
                )
            indices = fixed_indices.to(device=keys.device, dtype=torch.long)
            if indices.ndim != 3:
                raise ValueError(
                    "Fixed Compactor keep indices must have shape "
                    f"[batch, num_heads, n_kept]; got {tuple(indices.shape)}"
                )
            if indices.shape[:2] != keys.shape[:2]:
                raise ValueError(
                    "Fixed Compactor keep indices batch/head dimensions do not "
                    f"match KV cache: indices={tuple(indices.shape)} keys={tuple(keys.shape)}"
                )
            if indices.numel() > 0 and (
                int(indices.min().item()) < 0 or int(indices.max().item()) >= k_len
            ):
                raise ValueError(
                    "Fixed Compactor keep indices are out of range for KV cache "
                    f"length {k_len}."
                )
        else:
            scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

            # Get indices of KV pairs with the highest scores (to keep)
            n_kept = self._n_kept(k_len, keys.device)

            topk_result = scores.topk(n_kept, dim=-1)
            indices = topk_result.indices  # [batch, num_heads, n_kept]

        self.kept_indices[current_layer] = indices.detach().cpu()
        self._layer_counter += 1

        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)
        keys = keys.gather(2, indices_expanded).contiguous()
        values = values.gather(2, indices_expanded).contiguous()

        return keys, values
