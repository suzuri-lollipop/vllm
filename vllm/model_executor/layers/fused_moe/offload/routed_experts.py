# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offload-mode routed experts: host banks + global LRU slot-cache GEMM.

``OffloadRoutedExperts`` is the ``RoutedExperts`` used when the ``expert_cache``
offload backend is active. It swaps the quant method for an offload-aware variant:

* unquantized (bf16) -> :class:`OffloadUnquantizedFusedMoEMethod`
* block-quantized FP8 (``weight_block_size``) -> :class:`OffloadFp8MoEMethod`
* NVFP4 on the VLLM_CUTLASS backend -> the class built by
  :func:`_make_offload_nvfp4_method`
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
from vllm.platforms import current_platform

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

    # Expert parallelism: the router emits GLOBAL expert ids, but this rank
    # banks only its own shard of the experts (create_weights is handed
    # num_local_experts), so translate to local ids up front. Doing it here
    # rather than passing expert_map down is what keeps the ids in slot space
    # for the kernel, exactly as in the non-EP case; each rank then computes a
    # partial sum and MoERunner's final all-reduce (ep_size > 1) combines them.
    #
    # ``expert_map`` yields -1 for an expert another rank owns. A raw -1 must
    # NOT reach the GEMM: this path always calls the kernel with
    # ``expert_map=None``, so nothing downstream applies the "-1 means skip"
    # convention -- the prefill branch materializes the layer so that
    # position == expert id, and the decode branch's LRU leaves ids it did not
    # rewrite untouched. Feeding -1 in as an index is what produced fluent but
    # meaningless output on real hardware, since under ep_size=2 roughly half
    # of all routings are non-local.
    #
    # Instead, neutralize those routings by zeroing their router weight and
    # clamping the id to a valid expert. The clamped expert is still computed
    # but contributes exactly zero (the weight multiplies its output, or its
    # input under apply_router_weight_on_input), and the owning rank
    # contributes that expert with its proper weight, so the all-reduced sum
    # is exact. This is also kernel-agnostic, which matters because the slot
    # GEMM is chosen per quant format.
    expert_map = layer.expert_map
    if expert_map is not None:
        local_ids = expert_map[topk_ids.long()]
        non_local = local_ids < 0
        topk_ids = local_ids.masked_fill(non_local, 0).to(torch.int32)
        topk_weights = topk_weights.masked_fill(non_local, 0)

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


class _CpuCreateWeightsMixin:
    """Allocates this layer's expert params directly on the host, never the GPU.

    Mixed in ahead of the base quant method (MRO-first) so ``create_weights``
    runs under a CPU device context before falling through to the base
    implementation's ``torch.empty(...)`` calls.

    Without this, a many-expert MoE layer (this project's Qwen4Exp/qwen4_exp
    checkpoints run 512 experts/layer over 48 layers) can transiently exceed
    a GPU memory ceiling on this project's WSL2/WDDM host well below the
    physical VRAM total -- confirmed on real hardware via ``nvidia-smi``: the
    CUDA "out of memory" error fired during
    :meth:`~vllm.model_executor.offloader.expert_cache.ExpertCacheOffloader._to_host_bank`
    with only a few GiB actually resident, and neither
    ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments`` nor an explicit
    ``torch.cuda.empty_cache()`` after every layer's diversion changed the
    failure point -- pointing at a WSL2 paravirtualized-GPU (``/dev/dxg``)
    per-process commit limit rather than allocator fragmentation or genuine
    VRAM exhaustion. Constructing directly on the host sidesteps the GPU for
    this step entirely; :class:`~vllm.model_executor.offloader.expert_cache.ExpertCacheOffloader`'s
    diversion/pinning bookkeeping is unaffected since it already treats the
    incoming tensor as arbitrary source data (``_to_host_bank``'s ``.to("cpu")``
    becomes a no-op).
    """

    def create_weights(self, layer: RoutedExperts, *args, **kwargs) -> None:
        with torch.device("cpu"):
            super().create_weights(layer, *args, **kwargs)


class OffloadUnquantizedFusedMoEMethod(_CpuCreateWeightsMixin, UnquantizedFusedMoEMethod):
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


def _make_offload_nvfp4_method(base_method):
    """Build an offload NVFP4 MoE method from an already-resolved
    ``ModelOptNvFp4FusedMoE`` (restricted to the VLLM_CUTLASS backend; see the
    class docstring).

    Defined as a factory (rather than a top-level class) so the ModelOpt import
    stays lazy, mirroring :func:`_make_offload_fp8_method`.
    """
    from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

    class OffloadModelOptNvfp4MoEMethod(_CpuCreateWeightsMixin, ModelOptNvFp4FusedMoE):
        """NVFP4 MoE method for the expert_cache offload backend.

        Two NVFP4 kernel backends are supported, each with its own bank
        schema (see slot_cache.BANK_SCHEMAS["nvfp4"] / ["nvfp4_emulation"]):

        * VLLM_CUTLASS: post-conversion per-expert tensors (weights, block
          scales, and two per-expert scalar scale rows) keep a leading
          ``num_experts`` dimension with no cross-expert mixing
          (``swizzle_blockscale`` only reshapes the block/K dimensions), so a
          whole expert row can be banked and slot-swapped independently.
          Needs a transient GPU round trip to convert (below).
        * EMULATION: weights/block-scales are never repacked at all (the
          Triton kernel dequantizes on the fly), so only the two per-expert
          weight_scale_2 rows need banking; the activation scale collapses to
          a single global scalar (not per-expert) and is never banked.

        Other NVFP4 backends (FlashInfer TRT-LLM/CuteDSL, Marlin, Humming)
        repack weights into batched/grouped-GEMM buffers or bake a layer-wide
        shared scale factor across ALL experts at conversion time, which is
        not safe to reason about per-expert -- they are rejected rather than
        silently mishandled.

        The class name must contain "ModelOpt":
        ``RoutedExperts.weight_loader`` (vllm/model_executor/layers/fused_moe/routed_experts.py)
        branches on ``self.quant_method.__class__.__name__`` containing that
        substring to pick the ModelOpt-specific weight_scale_2/input_scale
        loading path; naming this class e.g. "OffloadNvfp4MoEMethod" makes
        every one of those checks silently miss and fall through to the
        generic (wrong) loader, confirmed on real hardware as a late
        ``ValueError: quant method must be one of [...]`` deep in weight
        loading -- long after this class's own code has already run
        correctly, which is what made it look unrelated to the class name.
        """

        _CUTLASS_BANKED_PARAMS = (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_2",
            "w13_input_scale",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_2",
            "w2_input_scale",
        )
        _EMULATION_BANKED_PARAMS = (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_2",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_2",
        )

        def __init__(self, base: ModelOptNvFp4FusedMoE):
            # Reuse the resolved ModelOptNvFp4FusedMoE state (backend already
            # selected); do not re-run selection.
            self.__dict__.update(base.__dict__)
            if self.nvfp4_backend not in (
                NvFp4MoeBackend.VLLM_CUTLASS,
                NvFp4MoeBackend.EMULATION,
            ):
                raise NotImplementedError(
                    "expert_cache offload supports NVFP4 MoE experts only "
                    "with the VLLM_CUTLASS or EMULATION backend (per-expert "
                    f"rows stay independently sliceable); got "
                    f"{self.nvfp4_backend.value!r}. Pass --moe-backend cutlass "
                    "(or 'emulation') to select one explicitly, or disable "
                    "expert_cache offload."
                )
            if self.use_a16:
                raise NotImplementedError(
                    "expert_cache offload does not support W4A16 NVFP4 "
                    "checkpoints; disable expert_cache offload."
                )
            self.offload_quant_format = (
                "nvfp4"
                if self.nvfp4_backend == NvFp4MoeBackend.VLLM_CUTLASS
                else "nvfp4_emulation"
            )

        def process_weights_after_loading(self, layer: RoutedExperts) -> None:
            if self.nvfp4_backend == NvFp4MoeBackend.EMULATION:
                self._process_weights_emulation(layer)
            else:
                self._process_weights_cutlass(layer)

        def _process_weights_cutlass(self, layer: RoutedExperts) -> None:
            """Convert on a transient GPU copy; keep only the result on the host.

            The banked params were diverted to pinned host memory at
            construction time (before the checkpoint was loaded), so they
            hold real loaded values here but live on the host. They are
            copied to the GPU, converted via the ordinary (unmodified)
            ModelOpt/CUTLASS path -- which also fuses the activation scale
            into ``w{13,2}_weight_scale_2`` in place
            (``CutlassExpertsFp4.process_weights_after_loading``) -- and then
            copied back into fresh pinned host tensors. The throwaway kernel
            ``super()`` builds (bound to the transient GPU tensors) is
            discarded; a real one is built lazily in
            :meth:`build_offload_kernel` once the slot cache exists.
            """
            device = torch.device(
                current_platform.device_type, torch.cuda.current_device()
            )
            # Keep the Parameter objects that exist *now*: the base
            # implementation below calls replace_parameter(), which installs
            # brand new Parameter objects on the layer. ExpertCacheOffloader
            # banked these original ones (it holds the Parameter itself, not a
            # snapshot of .data), so the harvest at the end has to land on them
            # too -- see the comment there.
            originals = {
                name: getattr(layer, name) for name in self._CUTLASS_BANKED_PARAMS
            }
            for param in originals.values():
                param.data = param.data.to(device=device, non_blocking=True)

            super().process_weights_after_loading(layer)

            # `a{13,2}_gscale` bank rows must hold the RECIPROCAL of the
            # activation scale (see slot_cache.py): the slot cache is a
            # shared, mutable buffer, so the reciprocal has to be baked in
            # once per expert now, not recomputed from whatever expert
            # happens to occupy a slot at kernel-build time.
            layer.w13_input_scale.data = 1.0 / layer.w13_input_scale.data
            layer.w2_input_scale.data = 1.0 / layer.w2_input_scale.data

            from vllm.model_executor.offloader.base import get_offloader

            # Match the offloader's configured pin_memory (--moe-cache-pin-memory),
            # not a bare platform check: on this project's WSL2/WDDM host, pinned
            # allocation has its own ceiling well below both VRAM and host RAM
            # (confirmed on real hardware), so a user who disabled it for
            # ExpertCacheOffloader needs it disabled here too.
            pin = get_offloader().pin_memory
            for name in self._CUTLASS_BANKED_PARAMS:
                param = getattr(layer, name)
                host = torch.empty_like(param.data, device="cpu", pin_memory=pin)
                host.copy_(param.data)
                param.data = host
                # Point the banked Parameter at the converted host tensor as
                # well. replace_parameter() left it detached from the layer
                # while it still owned the GPU copy made above, so without
                # this every offloaded layer leaks a full expert layer of
                # VRAM -- confirmed on real hardware as a load-time OOM that
                # filled the GPU at the same point no matter how many layers
                # --moe-gpu-resident-layers kept resident -- and post_init()
                # would wire the slot cache to the stale, unconverted weights.
                original = originals[name]
                if original is not param:
                    original.data = host
            # Force a lazy rebuild against the slot cache; the kernel/config
            # built by super() above referenced the transient GPU tensors.
            self.moe_kernel = None
            self.moe_quant_config = None

            # The conversion above (super().process_weights_after_loading)
            # materializes every banked tensor for this layer's 512 experts on
            # the GPU at once (plus CUTLASS's own padding/reorder scratch);
            # release it back to the allocator now rather than letting it
            # accumulate into the next layer's conversion -- confirmed on real
            # hardware as a genuine (not fragmentation-only) transient peak.
            torch.cuda.empty_cache()

        def _process_weights_emulation(self, layer: RoutedExperts) -> None:
            """Collapse two scalars on the host; nothing else needs converting.

            EMULATION repacks nothing -- weights/block-scales are banked
            exactly as loaded -- so the only real work
            ``ModelOptNvFp4FusedMoE.process_weights_after_loading`` does for
            this backend is: collapse ``w13_weight_scale_2`` from its
            checkpoint ``[E, 2]`` shard shape to ``[E]``, and collapse each of
            ``w{13,2}_input_scale`` to a single global scalar
            (``1 / max(input_scale)``). Both are ordinary CPU tensor ops
            (unlike CUTLASS's ``swizzle_blockscale``, which calls ``.cuda()``
            unconditionally), so this never touches the GPU.

            Deliberately NOT calling ``super()``: its full path builds a
            throwaway kernel/experts object to run this same conversion,
            which for ``Nvfp4QuantizationEmulationTritonExperts`` captures
            ``self.w1_scale_val``/``self.w2_scale_val`` as instance attributes
            referencing this layer's (GPU-resident, had it been staged there)
            banks -- confirmed on real hardware as VRAM accumulating across
            the 48-layer conversion loop until OOM, because those references
            outlive `self.moe_kernel = None` in ways not fully tracked down.
            Skipping super() entirely (mirroring bf16/fp8_block) sidesteps
            the whole class of leak by construction: nothing is ever staged
            on the GPU here for anything to hold onto.
            """
            if self.moe.is_act_and_mul and not torch.allclose(
                layer.w13_weight_scale_2.data[:, 0], layer.w13_weight_scale_2.data[:, 1]
            ):
                logger.warning_once(
                    "w1_weight_scale_2 must match w3_weight_scale_2. "
                    "Accuracy may be affected."
                )
            layer.w13_weight_scale_2.data = layer.w13_weight_scale_2.data[
                :, 0
            ].contiguous()
            layer.w13_input_scale.data = (
                1.0 / layer.w13_input_scale.data.max().to(torch.float32)
            ).reshape(())
            layer.w2_input_scale.data = (
                1.0 / layer.w2_input_scale.data.max().to(torch.float32)
            ).reshape(())
            self.moe_kernel = None
            self.moe_quant_config = None

        def build_offload_kernel(
            self, layer: RoutedExperts, cache: ExpertSlotCache
        ) -> None:
            if self.nvfp4_backend == NvFp4MoeBackend.EMULATION:
                self._build_offload_kernel_emulation(layer, cache)
            else:
                self._build_offload_kernel_cutlass(layer, cache)

        def _build_offload_kernel_cutlass(
            self, layer: RoutedExperts, cache: ExpertSlotCache
        ) -> None:
            """Build the slot-space CUTLASS FP4 GEMM kernel over the slot caches."""
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from vllm.model_executor.layers.fused_moe.all2all_utils import (
                maybe_make_prepare_finalize,
            )
            from vllm.model_executor.layers.fused_moe.config import (
                nvfp4_moe_quant_config,
            )
            from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
                CutlassExpertsFp4,
            )

            # Bypass make_nvfp4_moe_quant_config's own `1.0 / a_scale`: the
            # a{13,2}_gscale banks already hold the reciprocal (see
            # _process_weights_cutlass), and everything below is a direct
            # reference into the slot cache's live GPU buffers so later slot
            # swaps are picked up automatically, exactly like
            # OffloadFp8MoEMethod's w{13,2}_scale banks.
            moe_quant_config = nvfp4_moe_quant_config(
                g1_alphas=cache.bank_caches["w13_gscale2"],
                g2_alphas=cache.bank_caches["w2_gscale2"],
                a1_gscale=cache.bank_caches["a13_gscale"],
                a2_gscale=cache.bank_caches["a2_gscale"],
                w1_scale=cache.bank_caches["w13_qscale"],
                w2_scale=cache.bank_caches["w2_qscale"],
                is_scale_swizzled=True,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )
            self.moe_quant_config = moe_quant_config

            prepare_finalize = maybe_make_prepare_finalize(
                moe=self.moe,
                quant_config=moe_quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            )
            assert prepare_finalize is not None
            experts = CutlassExpertsFp4(moe_config=self.moe, quant_config=moe_quant_config)
            self.moe_kernel = mk.FusedMoEKernel(prepare_finalize, experts)

        def _build_offload_kernel_emulation(
            self, layer: RoutedExperts, cache: ExpertSlotCache
        ) -> None:
            """Build the slot-space Triton dequant+GEMM kernel over the slot caches."""
            import vllm.model_executor.layers.fused_moe.modular_kernel as mk
            from vllm.model_executor.layers.fused_moe.all2all_utils import (
                maybe_make_prepare_finalize,
            )
            from vllm.model_executor.layers.fused_moe.config import (
                nvfp4_moe_quant_config,
            )
            from vllm.model_executor.layers.fused_moe.experts.nvfp4_emulation_moe import (
                Nvfp4QuantizationEmulationTritonExperts,
            )

            # w{13,2}_input_scale are two host-resident scalars (collapsed by
            # _process_weights_emulation, never staged on the GPU there to
            # avoid the leak that method's docstring explains); this lazy,
            # one-time, one-layer move is too small to matter.
            layer.w13_input_scale.data = layer.w13_input_scale.data.to(cache.device)
            layer.w2_input_scale.data = layer.w2_input_scale.data.to(cache.device)
            moe_quant_config = nvfp4_moe_quant_config(
                g1_alphas=cache.bank_caches["w13_gscale2"],
                g2_alphas=cache.bank_caches["w2_gscale2"],
                a1_gscale=layer.w13_input_scale,
                a2_gscale=layer.w2_input_scale,
                w1_scale=cache.bank_caches["w13_qscale"],
                w2_scale=cache.bank_caches["w2_qscale"],
                is_scale_swizzled=False,
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            )
            self.moe_quant_config = moe_quant_config

            prepare_finalize = maybe_make_prepare_finalize(
                moe=self.moe,
                quant_config=moe_quant_config,
                allow_new_interface=True,
                use_monolithic=False,
            )
            assert prepare_finalize is not None
            experts = Nvfp4QuantizationEmulationTritonExperts(
                moe_config=self.moe, quant_config=moe_quant_config
            )
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

    return OffloadModelOptNvfp4MoEMethod(base_method)


def _make_offload_fp8_method(base_method):
    """Build an offload FP8 MoE method from an already-resolved ``Fp8MoEMethod``.

    Defined as a factory (rather than a top-level class) so the ``Fp8MoEMethod``
    import stays lazy; it reuses the resolved backend selection instead of
    re-running it.
    """
    from vllm.model_executor.layers.quantization.fp8 import Fp8MoEMethod

    class OffloadFp8MoEMethod(_CpuCreateWeightsMixin, Fp8MoEMethod):
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

        from vllm.model_executor.layers.quantization.modelopt import (
            ModelOptNvFp4FusedMoE,
        )

        if isinstance(method, ModelOptNvFp4FusedMoE):
            # _make_offload_nvfp4_method itself rejects any backend other than
            # VLLM_CUTLASS (and W4A16 checkpoints) with a NotImplementedError.
            return _make_offload_nvfp4_method(method)

        raise NotImplementedError(
            "expert_cache offload supports unquantized (bf16), block-quantized "
            "FP8, and NVFP4 (VLLM_CUTLASS backend) MoE experts only; got "
            f"{type(method).__name__} for {prefix!r}. Disable expert_cache "
            "offload. (Compressed-tensors NVFP4, MXFP4, AWQ are deferred.)"
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
