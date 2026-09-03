# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload-mode routed experts: host banks + global LRU slot-cache GEMM.

``OffloadRoutedExperts`` is the ``RoutedExperts`` used when the ``expert_cache``
offload backend is active. It swaps the quant method for an offload-aware variant:

* unquantized (bf16) -> :class:`OffloadUnquantizedFusedMoEMethod`
* block-quantized FP8 (``weight_block_size``) -> :class:`OffloadFp8MoEMethod`
* any other quantization -> loud ``NotImplementedError`` (no silent fallback)

Each offload method (a) never moves the expert weights to the GPU in
``process_weights_after_loading`` -- they stay as pinned host banks -- and (b) runs
the forward as: ``ensure`` the routed experts into the global slot cache (rewriting
``topk_ids`` to slot ids), stream the misses host -> device, then execute a standard
fused MoE over the slot cache (``cache_size`` local experts) using vLLM's existing
Triton experts backend.

For FP8 the per-block weight scales are small but are indexed by expert/slot id in
the GEMM, so they are banked alongside the weights and moved with them; the slot
GEMM reads the per-slot scale cache (see :class:`OffloadFp8MoEMethod`).

The weight diversion itself happens earlier, in
:class:`~vllm.model_executor.offloader.expert_cache.ExpertCacheOffloader.wrap_modules`,
so a MoE model larger than VRAM never holds all experts on the GPU at once.

Prefill handling: the slot-cache ``ensure`` path is correct for any batch (decode or
prefill), so it is the default. When a batch is detected as prefill/mixed
(query_len > 1 for at least one sequence) the simpler synchronous
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
    ExpertSlotCache,
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


def _offload_forward(
    method,
    layer: RoutedExperts,
    x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """Shared ensure/copy/slot-GEMM forward for every offload quant method.

    ``method`` must provide ``moe_kernel`` (built lazily) and is otherwise agnostic
    to the element format: the slot weight caches are addressed by name and the
    per-format scales are already baked into ``method.moe_kernel``'s quant config.
    """
    cache = get_global_slot_cache()
    layer_id = getattr(layer, "_offload_layer_id", None)
    if layer_id is None:
        raise RuntimeError(
            "expert_cache offload: layer was not registered with the expert "
            "slot cache (was the offloader's wrap_modules run?)"
        )

    # Lazily build the slot-space GEMM kernel once the slot cache exists. The
    # build is host-side (no device sync) and happens during the eager warmup that
    # precedes any CUDA-graph capture.
    if method.moe_kernel is None:
        method.build_offload_kernel(layer, cache)

    # The slot ids are int32; match the routing ids before the in-place rewrite.
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)

    # Double-buffered prefill streams the whole expert layer into a borrowed
    # buffer and computes the GEMM over a buffer view offset by buffer_id * E.
    # That is only correct when the GEMM has no per-row weight-scale bank read by
    # expert id: such scales live in the fixed slot-scale cache at expert ids
    # [0:E) and cannot follow the buffer_id * E weight offset (buffer 1 would
    # read buffer 0's scales). So overlap is restricted to scale-free formats
    # (bf16); formats with scale banks (fp8_block, ...) prefill via materialize.
    has_scale_banks = any("scale" in name for name in cache.bank_schema)

    overlap_prefill = False
    if _batch_is_prefill():
        if cache.prefill_overlap and not has_scale_banks:
            # Double-buffered streaming prefill: the first MoE layer opens the
            # batch; every layer prefetches itself and its successor into
            # alternating borrowed buffers, waits its buffer, computes over it
            # (position == expert id, routing ids unmapped), then releases it.
            overlap_prefill = True
            if layer_id == 0:
                cache.begin_prefill()
            cache.prefetch_prefill_layer(layer_id)
            cache.prefetch_prefill_layer(layer_id + 1)
            views = cache.wait_prefill_layer(layer_id)
            w13 = views[cache.bank_schema.index("w13")]
            w2 = views[cache.bank_schema.index("w2")]
            num_local = cache.num_experts
        else:
            # Synchronous materialize of the whole expert layer into the first
            # num_experts slots (position == expert id), routing ids unmapped.
            # Used when overlap is disabled, when the cache is too small to lend
            # two buffers (degraded mode), or when the format banks per-row
            # scales (see has_scale_banks above).
            cache.materialize_layer(layer_id)
            cache.copy_missing()
            w13 = cache.bank_caches["w13"][: cache.num_experts]
            w2 = cache.bank_caches["w2"][: cache.num_experts]
            num_local = cache.num_experts
    else:
        # Decode: LRU-ensure the routed experts, streaming only the misses.
        # ``ensure_experts`` rewrites ``topk_ids`` in place to slot ids.
        cache.ensure_experts(layer_id, topk_ids)
        cache.copy_missing()
        w13 = cache.bank_caches["w13"]
        w2 = cache.bank_caches["w2"]
        num_local = cache.cache_size

    assert method.moe_kernel is not None
    # Shared experts (if any) are executed by the runner; the offload path only
    # produces the routed output.
    out = method.moe_kernel.apply(
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
    if overlap_prefill:
        cache.release_prefill_layer(layer_id)
    return out


class OffloadUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized (bf16) MoE method for the expert_cache offload backend.

    Expert weights stay in pinned host memory; the GEMM runs over the global GPU
    slot cache using the Triton fused-MoE experts backend (which consumes plain
    ``[E, N, K]`` weights and an arbitrary local expert count, so it operates on
    the slot cache directly).
    """

    offload_quant_format = "bf16"

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        """Do NOT move the expert weights to the GPU.

        The base implementation shuffles weights into a GPU kernel format, which
        would materialize the full expert tensors in VRAM -- exactly what offload
        avoids. The expert weights remain pinned host banks (registered by the
        offloader); the slot-space GEMM kernel is built lazily by
        :meth:`build_offload_kernel` once the slot cache exists.
        """
        # Deliberately NOT calling super(): the weights must stay on the host.

    def build_offload_kernel(
        self, layer: RoutedExperts, cache: ExpertSlotCache
    ) -> None:
        """Build the slot-space Triton GEMM kernel (bf16 has no scales)."""
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
        return _offload_forward(self, layer, x, topk_weights, topk_ids)


def _make_offload_fp8_method(base_method):
    """Build an offload FP8 MoE method from an already-resolved ``Fp8MoEMethod``.

    Defined as a factory (rather than a top-level class) so the ``Fp8MoEMethod``
    import stays lazy; it reuses the resolved backend selection instead of
    re-running it.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    class OffloadFp8MoEMethod(Fp8MoEMethod):
        """Block-quantized FP8 MoE method for the expert_cache offload backend.

        Expert weights (float8_e4m3fn) and their per-block float32 scales stay in
        pinned host banks. The slot-space GEMM uses vLLM's Triton fp8 experts
        backend; because the kernel indexes the weight scale by expert/slot id, the
        scales are banked alongside the weights and served from a per-slot scale
        cache.
        """

        offload_quant_format = "fp8_block"

        def __init__(self, base: Fp8MoEMethod):
            # Reuse the resolved Fp8MoEMethod state (quant config, block shape and
            # backend already selected); do not re-run selection.
            self.__dict__.update(base.__dict__)

        def process_weights_after_loading(self, layer: RoutedExperts) -> None:
            """Do NOT move the expert weights/scales to the GPU (see bf16)."""
            # Deliberately NOT calling the base implementation: the weights and
            # block scales must stay on the host; the kernel is built lazily.

        def build_offload_kernel(
            self, layer: RoutedExperts, cache: ExpertSlotCache
        ) -> None:
            """Build the slot-space Triton fp8 GEMM kernel over the slot caches.

            The per-slot scale caches are the GEMM's weight scales; they are filled
            by the fused miss copy whenever an expert (weight rows + scale rows)
            lands in a slot, so indexing them by slot id matches the rewritten
            ``topk_ids``.
            """
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from vllm.model_executor.layers.fused_moe.all2all_utils import (
                maybe_make_prepare_finalize,
            )
            from vllm.model_executor.layers.fused_moe.config import (
                fp8_w8a8_moe_quant_config,
            )
            from vllm.model_executor.layers.fused_moe.experts.triton_moe import (
                TritonExperts,
            )

            moe_quant_config = fp8_w8a8_moe_quant_config(
                w1_scale=cache.bank_caches["w13_scale"],
                w2_scale=cache.bank_caches["w2_scale"],
                a1_scale=None,
                a2_scale=None,
                block_shape=self.moe_block_shape,
            )
            self.moe_quant_config = moe_quant_config

            prepare_finalize = maybe_make_prepare_finalize(
                moe=self.moe,
                quant_config=moe_quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            )
            assert prepare_finalize is not None
            experts = TritonExperts(moe_config=self.moe, quant_config=moe_quant_config)
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
            return _offload_forward(self, layer, x, topk_weights, topk_ids)

    return OffloadFp8MoEMethod(base_method)


class OffloadRoutedExperts(RoutedExperts):
    """RoutedExperts variant for the expert_cache offload backend."""

    def _get_quant_method(
        self,
        prefix: str,
        quant_config,
        moe_config: FusedMoEConfig,
    ):
        # Unquantized (bf16) and block-quantized FP8 are supported; every other
        # quantization fails loudly rather than silently computing wrong results.
        if quant_config is None:
            return OffloadUnquantizedFusedMoEMethod(moe_config)

        method = quant_config.get_quant_method(self, prefix)
        if method is None:
            # The model has a quant config but this MoE layer is not quantized.
            return OffloadUnquantizedFusedMoEMethod(moe_config)

        from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        if isinstance(method, Fp8MoEMethod):
            if method.block_quant:
                return _make_offload_fp8_method(method)
            raise NotImplementedError(
                "expert_cache offload supports block-quantized FP8 "
                "(weight_block_size) MoE experts; per-tensor FP8 needs a weight "
                f"requantization that is not implemented for host banks ({prefix!r}). "
                "Use a block-quantized checkpoint or disable expert_cache offload."
            )

        raise NotImplementedError(
            "expert_cache offload supports unquantized (bf16) and block-quantized "
            f"FP8 MoE experts only; got {type(method).__name__} for {prefix!r}. "
            "Disable expert_cache offload. (NVFP4/MXFP4/AWQ are deferred.)"
        )


def _batch_is_prefill() -> bool:
    """Best-effort detection that the current batch has a prefill/mixed sequence.

    Returns True when the forward-context batch descriptor reports strictly more
    tokens than requests (query_len > 1 somewhere), selecting the prefill path
    (double-buffered streaming when overlap is enabled, else synchronous
    materialize). When the descriptor is absent or does not carry a request
    count, returns False and the always-correct decode ensure path is used.

    The request count is populated on the eager (CUDAGraphMode.NONE) batch
    descriptor; it stays None on the piecewise descriptor, which doubles as a
    CUDA-graph cache key and must not specialize on num_reqs, so piecewise
    prefill conservatively uses the decode path. This gate therefore controls
    when the prefill overlap buffers are exercised.
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
