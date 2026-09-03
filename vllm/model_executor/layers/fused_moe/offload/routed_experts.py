# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload-mode routed experts: host banks + global LRU slot-cache GEMM.

``OffloadRoutedExperts`` is the ``RoutedExperts`` used when the ``expert_cache``
offload backend is active. It swaps the quant method for
``OffloadUnquantizedFusedMoEMethod``, which (a) never moves the expert weights to
the GPU in ``process_weights_after_loading`` -- they stay as pinned host banks --
and (b) runs the forward as: ``ensure`` the routed experts into the global slot
cache (rewriting ``topk_ids`` to slot ids), stream the misses host -> device, then
execute a standard fused MoE over the slot cache (``cache_size`` local experts).

The weight diversion itself happens earlier, in
:class:`~vllm.model_executor.offloader.expert_cache.ExpertCacheOffloader.wrap_modules`,
so a MoE model larger than VRAM never holds all experts on the GPU at once.

Prefill handling (phase 1): the slot-cache ``ensure`` path is correct for any
batch (decode or prefill), so it is the default. When a batch is detected as
prefill/mixed (query_len > 1 for at least one sequence) the simpler synchronous
``materialize_layer`` path is used instead, which stages the whole expert layer at
once rather than running the per-token LRU ensure loop. Detection is best-effort
(see :func:`_batch_is_prefill`); when it cannot tell, the always-correct ensure
path is used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.offload.slot_cache import (
    get_global_slot_cache,
)
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.runner.shared_experts import (
        SharedExperts,
    )

logger = init_logger(__name__)


class OffloadUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized (bf16) MoE method for the expert_cache offload backend.

    Expert weights stay in pinned host memory; the GEMM runs over the global GPU
    slot cache using the Triton fused-MoE experts backend (which consumes plain
    ``[E, N, K]`` weights and an arbitrary local expert count, so it operates on
    the slot cache directly).
    """

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Build the slot-space GEMM kernel; do NOT move weights to the GPU.

        The base implementation shuffles weights into a GPU kernel format, which
        would materialize the full expert tensors in VRAM -- exactly what offload
        avoids. The expert weights remain pinned host banks (registered by the
        offloader); here we only build the modular kernel used to run the slot
        cache.
        """
        # Deliberately NOT calling super(): the weights must stay on the host.
        import vllm.model_executor.layers.fused_moe.modular_kernel as mk
        from vllm.model_executor.layers.fused_moe.all2all_utils import (
            maybe_make_prepare_finalize,
        )
        from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        prepare_finalize = maybe_make_prepare_finalize(
            moe=self.moe,
            quant_config=self.moe_quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        )
        assert prepare_finalize is not None
        experts = TritonExperts(moe_config=self.moe, quant_config=self.moe_quant_config)
        self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        cache = get_global_slot_cache()
        layer_id = getattr(layer, "_offload_layer_id", None)
        if layer_id is None:
            raise RuntimeError(
                "expert_cache offload: layer was not registered with the expert "
                "slot cache (was the offloader's wrap_modules run?)"
            )

        # The slot ids are int32; match the routing ids before the in-place rewrite.
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        if _batch_is_prefill():
            # Simple synchronous prefill path: materialize the whole expert layer
            # into the first num_experts slots (position == expert id) and compute.
            # Routing ids pass through unmapped.
            cache.materialize_layer(layer_id)
            cache.copy_missing()
            w13, w2 = cache.bank_views(cache.num_experts)
            num_local = cache.num_experts
        else:
            # Decode: LRU-ensure the routed experts, streaming only the misses.
            # ``ensure_experts`` rewrites ``topk_ids`` in place to slot ids.
            cache.ensure_experts(layer_id, topk_ids)
            cache.copy_missing()
            w13, w2 = cache.bank_views()
            num_local = cache.cache_size

        assert self.moe_kernel is not None
        # Shared experts (if any) are executed by the runner; the offload path only
        # produces the routed output.
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=w13,
            w2=w2,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            activation=layer.activation,
            global_num_experts=num_local,
            expert_map=None,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=None,
            shared_experts_input=None,
        )


class OffloadRoutedExperts(RoutedExperts):
    """RoutedExperts variant for the expert_cache offload backend."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config,
        moe_config: FusedMoEConfig,
    ):
        # Offload only supports the unquantized (bf16) path. If a quant config
        # resolved to a real MoE method, this subclass should not have been
        # selected; fail loudly rather than silently computing wrong results.
        if quant_config is not None:
            method = quant_config.get_quant_method(self, prefix)
            if method is not None:
                raise NotImplementedError(
                    "expert_cache offload supports unquantized (bf16) MoE experts "
                    f"only; got a quantized MoE method for {prefix!r}. Disable "
                    "expert_cache offload or use an unquantized model."
                )
        return OffloadUnquantizedFusedMoEMethod(moe_config)


def _batch_is_prefill() -> bool:
    """Best-effort detection that the current batch has a prefill/mixed sequence.

    Returns True when the forward-context batch descriptor reports strictly more
    tokens than requests (query_len > 1 somewhere). When the descriptor is absent
    or does not carry a request count (the common eager/piecewise case), returns
    False and the always-correct ensure path is used.
    """
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        batch_descriptor = getattr(ctx, "batch_descriptor", None)
        if batch_descriptor is not None:
            num_tokens = getattr(batch_descriptor, "num_tokens", None)
            num_reqs = getattr(batch_descriptor, "num_reqs", None)
            if num_tokens is not None and num_reqs is not None:
                return num_tokens > num_reqs
    except AssertionError:
        # No forward context set (e.g. standalone test invocation).
        return False
    return False
