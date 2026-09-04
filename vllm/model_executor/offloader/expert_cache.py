# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-granularity MoE offloading backend ("expert_cache").

Dives the routed expert weights (w13/w2 of every MoE layer) into pinned host
memory at model-construction time and rebuilds them on the GPU as a single global
LRU slot cache (see :mod:`vllm.model_executor.layers.fused_moe.offload`). This is
the offloader half; the forward / GEMM half lives on
:class:`~vllm.model_executor.layers.fused_moe.offload.routed_experts.OffloadRoutedExperts`.

The diversion happens in :meth:`wrap_modules` (during ``make_layers``) so a MoE
model larger than VRAM never holds all expert weights on the GPU at once -- each
decoder layer's experts are moved to pinned host right after that layer is
constructed, mirroring how the UVA/prefetch offloaders bound their GPU footprint.
The global slot cache is finalized in :meth:`post_init`, after every layer has
registered its host banks.
"""

from __future__ import annotations

from collections.abc import Generator

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.offload.slot_cache import (
    BANK_SCHEMAS,
    BANK_TO_PARAM,
    ExpertSlotCache,
    set_global_slot_cache,
)
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

# Attribute marker set on diverted expert params so the generic weight loader /
# process_weights_after_loading machinery (device_loading_context) leaves them on
# the host instead of bouncing them through the GPU.
EXPERT_OFFLOADED_ATTR = "_vllm_is_expert_offloaded"


class ExpertCacheOffloader(BaseOffloader):
    """Offloader that keeps routed MoE experts in pinned host memory and serves
    them through a global GPU LRU slot cache.

    Args:
        cache_size: number of expert slots in the global GPU slot cache.
        pin_memory: page-lock the host banks (required for fast async H2D).
        prefill_overlap: enable double-buffered prefill expert streaming
            (degrades to synchronous materialize when cache_size < 2*experts).
    """

    def __init__(
        self,
        cache_size: int,
        pin_memory: bool = True,
        prefill_overlap: bool = True,
    ):
        self.cache_size = cache_size
        self.pin_memory = pin_memory and is_pin_memory_available()
        self.prefill_overlap = prefill_overlap
        # Bank layout, detected from the first diverted layer's quant method
        # ("bf16" or "fp8_block") and required to be uniform across layers.
        self.quant_format: str | None = None

        # Collected during wrap_modules, one entry per MoE layer in model order.
        # Each entry maps bank name -> pinned host tensor [num_experts, ...].
        self._layer_banks: list[dict[str, torch.Tensor]] = []
        self._num_experts: int | None = None
        self.cache: ExpertSlotCache | None = None

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
    ) -> list[nn.Module]:
        """Divert each constructed module's routed expert weights to pinned host."""
        modules = []
        for module in modules_generator:
            self._divert_module(module)
            modules.append(module)
        return modules

    def _divert_module(self, module: nn.Module) -> None:
        # Lazy import to avoid a cycle (offload.routed_experts imports layers).
        from vllm.model_executor.layers.fused_moe.offload.routed_experts import (
            OffloadRoutedExperts,
        )

        for sub in module.modules():
            if isinstance(sub, OffloadRoutedExperts):
                self._divert_routed_experts(sub)

    @staticmethod
    def _layer_quant_format(layer) -> str:
        """Bank layout for this layer, from its offload quant method."""
        qm = getattr(layer, "quant_method", None)
        fmt = getattr(qm, "offload_quant_format", None)
        return fmt or "bf16"

    def _divert_routed_experts(self, layer) -> None:
        """Move this layer's expert weights (and, for fp8, block scales) to pinned
        host and register them."""
        if getattr(layer, "_offload_diverted", False):
            return

        fmt = self._layer_quant_format(layer)
        assert fmt in BANK_SCHEMAS, f"unknown offload quant_format {fmt!r}"
        if self.quant_format is None:
            self.quant_format = fmt
        elif self.quant_format != fmt:
            raise ValueError(
                "expert_cache offload requires a uniform expert format across "
                f"MoE layers; got {fmt!r} vs {self.quant_format!r}"
            )

        banks: dict[str, nn.Parameter] = {}
        for bank_name in BANK_SCHEMAS[fmt]:
            param_name = BANK_TO_PARAM[bank_name]
            param = getattr(layer, param_name, None)
            if param is None:
                raise RuntimeError(
                    f"expert_cache offload: layer {layer.layer_name!r} has no "
                    f"{param_name!r} (bank {bank_name!r}, format {fmt!r})"
                )
            host_tensor = self._to_host_bank(param.data)
            # Keep the parameter object pointing at the pinned host bank so weight
            # loading fills it directly; mark it so downstream machinery leaves it
            # on the host.
            param.data = host_tensor
            setattr(param, EXPERT_OFFLOADED_ATTR, True)
            # Store the Parameter object, not this snapshot of `.data`: some
            # formats (NVFP4) reassign `param.data` again in
            # process_weights_after_loading (host -> transient GPU -> a new
            # converted host tensor), which runs AFTER this diversion but
            # BEFORE post_init() builds the slot cache. Keeping only the
            # Parameter lets post_init() read whatever `.data` currently is,
            # instead of silently wiring the slot cache to this stale,
            # unconverted snapshot (confirmed on real hardware: the CUTLASS
            # kernel would have been fed never-swizzled/never-padded weights).
            banks[bank_name] = param

        # Validate a uniform expert count across all offloaded layers.
        num_experts = int(next(iter(banks.values())).shape[0])
        if self._num_experts is None:
            self._num_experts = num_experts
        elif num_experts != self._num_experts:
            raise ValueError(
                "expert_cache offload requires a uniform expert count across "
                f"MoE layers; got {num_experts} vs {self._num_experts}"
            )

        layer._offload_diverted = True
        layer._offload_layer_id = len(self._layer_banks)
        self._layer_banks.append(banks)

    def _to_host_bank(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a weight tensor to pageable host memory.

        Page-locking is deferred to :meth:`_pin_host_banks`; see the reason
        there.
        """
        return tensor.detach().to("cpu").contiguous()

    def _pin_host_banks(self) -> None:
        """Page-lock the host banks, one bank at a time, after construction.

        Pinned host memory is mapped into the GPU's address space -- that
        mapping is exactly what lets the fused miss-copy kernel dereference a
        host bank directly -- and on WDDM/WSL2 those mappings are charged
        against the same per-GPU budget as device memory. Page-locking during
        diversion therefore makes the banks compete with the model's own
        construction allocations at their *peak*, and on a 24 GiB card the two
        together overflow it: observed on real hardware as a driver-level
        "CUDA error: out of memory" while building a later layer's attention
        quant scales, with tens of GiB of host RAM still free. The settled
        footprint does fit (7.1 GiB device + 16 GiB of banks per rank), so
        pinning here -- after weight loading and process_weights_after_loading
        have released their transients -- pins against that rather than the
        peak.

        Each pageable bank is released as soon as its pinned copy exists, so
        host RAM never holds two full sets.
        """
        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            torch.cuda.empty_cache()
        for banks in self._layer_banks:
            for param in banks.values():
                host = param.data
                if host.is_pinned():
                    continue
                # device="cpu" is explicit because post_init() may still run
                # under the construction device context, whose torch_function
                # mode fills in device=cuda for factory calls that omit it.
                pinned = torch.empty_like(host, device="cpu", pin_memory=True)
                pinned.copy_(host)
                param.data = pinned

    def post_init(self) -> None:
        """Finalize the global slot cache once all layers have registered."""
        if not self._layer_banks:
            logger.warning(
                "expert_cache offload enabled but no offloaded MoE layers were "
                "found; the slot cache will not be created."
            )
            return
        if self._num_experts is None:
            return
        assert self.quant_format is not None  # set on first diversion

        if self.pin_memory:
            self._pin_host_banks()

        num_layers = len(self._layer_banks)
        # Determine the target device (prefer CUDA if available).
        from vllm.platforms import current_platform

        device = (
            torch.device(current_platform.device_type)
            if current_platform.is_cuda_alike()
            else torch.device("cpu")
        )

        cache = ExpertSlotCache(
            num_layers=num_layers,
            num_experts=self._num_experts,
            cache_size=self.cache_size,
            device=device,
            quant_format=self.quant_format,
            pin_memory=self.pin_memory,
            prefill_overlap=self.prefill_overlap,
        )
        # sources[name] -> list of per-layer host tensors. Read `.data` now
        # (not at diversion time): formats whose process_weights_after_loading
        # reassigns it (see _divert_routed_experts) have already run by the
        # time post_init() is called (model_loader.load_model() completes
        # fully -- construct, load weights, process_weights_after_loading --
        # before the runner calls get_offloader().post_init()).
        sources: dict[str, list[torch.Tensor]] = {
            name: [banks[name].data for banks in self._layer_banks]
            for name in BANK_SCHEMAS[self.quant_format]
        }
        cache.set_bank_sources(sources)
        set_global_slot_cache(cache)
        self.cache = cache

        total = cache.slot_cache_bytes()
        logger.info(
            "expert_cache offload: format %s, %d MoE layers x %d experts, %d GPU "
            "slots (%s slot cache), host banks %s, miss copy %s",
            self.quant_format,
            num_layers,
            self._num_experts,
            self.cache_size,
            format_gib(total),
            "pinned" if self.pin_memory else "pageable",
            cache.miss_copy_description,
        )

    # ------------------------------------------------------------------
    # Stream lifecycle / capture hooks (called by the model runner and the
    # CUDA-graph capture paths via the generic offloader interface)
    # ------------------------------------------------------------------

    def sync_prev_onload(self) -> None:
        """Join the miss-copy stream before CUDA-graph capture/replay.

        Ensures any host->device miss copies queued on the copy stream by a
        previous (eager) forward are complete before the graph is captured or
        replayed, so captured/recorded work never races in-flight copies.
        """
        cache = self.cache
        if cache is not None and cache.copy_stream is not None:
            torch.cuda.current_stream().wait_stream(cache.copy_stream)

    def reset_offload_state(self) -> None:
        """Cold-start the expert slot cache.

        Called around CUDA-graph capture: the dummy forwards run during capture
        mutate the LRU bookkeeping, so the cache is reset before capture begins
        and again once capture completes, letting real inference start from a
        clean (cold) residency state. Mirrors FreeToken's
        ``GraphRunner._reset_moe_offload_cache``.
        """
        cache = self.cache
        if cache is not None:
            cache.reset()
