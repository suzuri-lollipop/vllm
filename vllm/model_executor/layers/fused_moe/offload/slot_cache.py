# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global LRU expert slot cache for expert-granularity MoE offloading.

This is the vLLM port of FreeToken's ``OffloadMoeCache`` (the bf16 / GPU-decode
core). Expert weights live in pinned host memory ("banks"); a single global GPU
slot cache holds the experts currently resident on the device, addressed by slot
``0 .. cache_size - 1`` and shared across ALL MoE layers. Per-step bookkeeping is
done by :mod:`vllm.model_executor.layers.fused_moe.offload.lru`.

Bank schema registry
--------------------
``BANK_SCHEMAS`` mirrors FreeToken's ``_BANK_SCHEMAS``: it is the single place a
quant format's bank layout is declared. Only dense ``bf16`` (banks ``w13`` and
``w2``) is implemented in phase 1; the other formats are listed (commented) to
show where quantized banks plug in later. The cache machinery itself is
layout-agnostic and just moves expert rows.

CUDA-graph status
-----------------
All per-step bookkeeping (ensure / materialize / reset) is fixed-shape and
device-side. The miss copy is device-driven too: ``copy_missing`` launches the
fused multi-bank gather (:mod:`...offload.fused_copy`), which reads the miss
count ``num_indices`` on the device -- no host round trip -- so the whole
decode movement path (ensure -> copy -> slot GEMM) is capturable into FULL CUDA
graphs. If the fused plan cannot be built (unaligned banks, unpinned sources,
no UVA aliasing), ``copy_missing`` falls back to the legacy host-count path,
which stays correct for eager and piecewise execution but is NOT capturable.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.offload.fused_copy import (
    FusedCopyPlan,
    build_fused_copy_plan,
    fused_copy_rows,
    fused_copy_rows_cpu,
)

logger = init_logger(__name__)

__all__ = [
    "BANK_SCHEMAS",
    "ExpertSlotCache",
    "get_global_slot_cache",
    "set_global_slot_cache",
    "clear_global_slot_cache",
]

# quant_format -> bank names, in registration order (mirrors FreeToken's
# _BANK_SCHEMAS). The cache machinery iterates banks in this order; phase 1 only
# implements dense bf16. Quantized bank layouts (fp8_block / nvfp4 / mxfp4 /
# ds_fp4 / q4_0) plug in here later, each adding its scale/packed banks.
BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights: fused gate_up (w13) + down (w2)
    "bf16": ("w13", "w2"),
    # future (out of scope, phase 1):
    # "fp8_block": ("w13", "w13_scale", "w2", "w2_scale"),
    # "nvfp4": ("w13_packed", "w13_scale", "w13_global", "w2_packed",
    #           "w2_scale", "w2_global"),
    # "mxfp4": ("w13_blocks", "w13_scales", "w13_bias", "w2_blocks",
    #           "w2_scales", "w2_bias"),
}

# bank name -> the RoutedExperts parameter it is diverted from (bf16 only).
BANK_TO_PARAM: dict[str, str] = {
    "w13": "w13_weight",
    "w2": "w2_weight",
}


@dataclass
class ExpertSlotCache:
    """Global GPU slot cache + pinned host banks for offloaded MoE experts.

    Args:
        num_layers: number of MoE layers sharing this cache.
        num_experts: experts per layer (all layers must match).
        cache_size: number of GPU expert slots (>= num_experts).
        device: the CUDA device the slot caches/bookkeeping live on.
        quant_format: bank layout key into ``BANK_SCHEMAS`` (bf16 only today).
        pin_memory: whether host banks are page-locked (required for fast async
            H2D miss copies).
    """

    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    quant_format: str = "bf16"
    pin_memory: bool = True

    def __post_init__(self) -> None:
        assert self.quant_format in BANK_SCHEMAS, (
            f"unknown quant_format {self.quant_format!r}"
        )
        if self.cache_size < self.num_experts:
            raise ValueError(
                f"cache_size {self.cache_size} < num_experts {self.num_experts}"
            )

        dev = self.device
        # Device-side LRU bookkeeping (flat id space: id = layer * E + expert).
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int32, device=dev
        )
        self.id_of_slot = torch.full(
            (self.cache_size,), -1, dtype=torch.int32, device=dev
        )
        self.usage = torch.zeros((self.cache_size,), dtype=torch.int64, device=dev)
        self.step = torch.zeros((), dtype=torch.int64, device=dev)
        self.active_mask = torch.zeros(
            (self.num_experts,), dtype=torch.int32, device=dev
        )
        # Miss staging buffers (shared by ensure and materialize). Sized to the
        # largest possible single-call miss list.
        plan_slots = max(self.num_experts, self.cache_size)
        self.evict_slots = torch.empty((plan_slots,), dtype=torch.int32, device=dev)
        self.src_indices = torch.empty((plan_slots,), dtype=torch.int32, device=dev)
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=dev)

        self.bank_schema = BANK_SCHEMAS[self.quant_format]
        # name -> list of per-layer host source tensors [num_experts, ...]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        # name -> unified GPU slot cache [cache_size, ...]
        self.bank_caches: dict[str, torch.Tensor] = {}
        # (per-layer sources, cache) pairs in schema order -- every piece of
        # machinery that moves bank bytes iterates this, so the cache is
        # bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []

        # Dedicated copy stream + event for host->device miss streaming. Only
        # created on CUDA (tests on CPU run copy_missing synchronously).
        self.copy_stream: torch.cuda.Stream | None = None
        self.copy_done_event: torch.cuda.Event | None = None
        if dev.type == "cuda":
            self.copy_stream = torch.cuda.Stream(device=dev)
            self.copy_done_event = torch.cuda.Event()

        # The layer whose misses were staged most recently by
        # ensure_experts/materialize_layer; consumed by copy_missing.
        self._pending_src_layer: int | None = None
        self._pending_whole_layer = False
        # Device-driven fused miss-copy descriptors (built by
        # set_bank_sources); None -> legacy host-count copy fallback.
        self._fused_plan: FusedCopyPlan | None = None

    # ------------------------------------------------------------------
    # Bank registration
    # ------------------------------------------------------------------

    def set_bank_sources(self, sources: dict[str, list[torch.Tensor]]) -> None:
        """Attach the per-layer pinned host banks and allocate the GPU slot caches.

        ``sources[name]`` is a list of ``num_layers`` tensors, each
        ``[num_experts, ...]`` (row = one expert's matrix). Every bank must have
        identical per-layer shape/dtype. Allocates one unified GPU slot cache per
        bank: ``[cache_size, ...]``.
        """
        assert set(sources) == set(self.bank_schema), (
            f"banks {sorted(sources)} do not match the {self.quant_format!r} "
            f"schema {self.bank_schema}"
        )
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            for layer_id, source in enumerate(per_layer):
                assert source.size(0) == self.num_experts, (
                    name,
                    layer_id,
                    source.shape,
                )
                assert source.shape == head.shape and source.dtype == head.dtype, (
                    name,
                    layer_id,
                    source.shape,
                    head.shape,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.empty(
                (self.cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
        self.banks = [
            (self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema
        ]
        # Slot-cache allocations are now fixed: build the device-driven fused
        # miss-copy descriptors (None -> legacy host-count copy fallback).
        self._fused_plan = build_fused_copy_plan(self)

    # ------------------------------------------------------------------
    # Geometry / budget helpers
    # ------------------------------------------------------------------

    def bytes_per_slot(self) -> int:
        """GPU bytes one expert slot occupies (summed over all banks)."""
        total = 0
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            total += int(head[0].numel()) * head.element_size()
        return total

    def slot_cache_bytes(self) -> int:
        """Total GPU bytes occupied by the slot caches (cache_size slots)."""
        return self.cache_size * self.bytes_per_slot()

    def bank_row_bytes(self) -> dict[str, int]:
        """Per-bank row (one expert) byte size."""
        out = {}
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            out[name] = int(head[0].numel()) * head.element_size()
        return out

    # ------------------------------------------------------------------
    # Movement entry points (bookkeeping + H2D copy)
    # ------------------------------------------------------------------

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Make this layer's routed experts resident; rewrite ``expert_ids`` in
        place to slot ids and stage the miss list for :meth:`copy_missing`."""
        from vllm.model_executor.layers.fused_moe.offload.lru import lru_ensure

        self._pending_src_layer = layer_id
        self._pending_whole_layer = False
        lru_ensure(self, layer_id, expert_ids)

    def materialize_layer(self, layer_id: int) -> None:
        """Stage a full-layer materialize (simple synchronous prefill path): all
        ``num_experts`` experts of ``layer_id`` land in slots ``0 .. E-1``
        (position == expert id), so routing ids pass through unmapped."""
        from vllm.model_executor.layers.fused_moe.offload.lru import materialize_layer

        self._pending_src_layer = layer_id
        self._pending_whole_layer = True
        materialize_layer(self, layer_id)

    def copy_missing(self) -> None:
        """Stream the staged miss list host -> device into the slot caches.

        Must be called after :meth:`ensure_experts` or :meth:`materialize_layer`.
        On CUDA the copies run on a dedicated copy stream and the compute stream
        is made to wait on ``copy_done_event``; on CPU (tests) it is synchronous.

        When a fused copy plan is available the miss count is read on the
        device and this call performs NO host synchronization (FULL CUDA-graph
        capturable). Otherwise it falls back to the legacy host-count gather,
        which is correct for eager/piecewise but not capturable.
        """
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure/materialize first)"

        if self.device.type != "cuda":
            self._copy_missing_reference(layer_id)
            return

        assert self.copy_stream is not None and self.copy_done_event is not None
        compute_stream = torch.cuda.current_stream(self.device)
        # The bookkeeping (evict_slots/src_indices/num_indices) was written on the
        # compute stream; order the copy stream behind it.
        self.copy_stream.wait_stream(compute_stream)
        with torch.cuda.stream(self.copy_stream):
            if self._fused_plan is not None:
                fused_copy_rows(
                    self._fused_plan,
                    layer_id,
                    self.evict_slots,
                    self.src_indices,
                    self.num_indices,
                )
            else:
                self._copy_missing_reference(layer_id)
            self.copy_done_event.record(self.copy_stream)
        compute_stream.wait_event(self.copy_done_event)

    def _copy_missing_reference(self, layer_id: int) -> None:
        """Host-count gather on the current stream (reference + legacy fallback).

        Reads the miss count on the host, so it is NOT CUDA-graph capturable. It
        serves as the CPU-test reference implementation and as the fallback when
        the device-driven fused copy plan is unavailable.
        """
        fused_copy_rows_cpu(
            self.banks,
            layer_id,
            self.evict_slots,
            self.src_indices,
            self.num_indices,
        )

    # ------------------------------------------------------------------
    # Views / lifecycle
    # ------------------------------------------------------------------

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank slot-cache views in registration order: the full ``[S]`` slot
        cache (decode) or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def reset(self) -> None:
        """Cold-start the cache (drop all residency)."""
        from vllm.model_executor.layers.fused_moe.offload.lru import reset_cache

        reset_cache(self)
        self._pending_src_layer = None
        self._pending_whole_layer = False


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------

_global_slot_cache: ExpertSlotCache | None = None


def get_global_slot_cache() -> ExpertSlotCache:
    assert _global_slot_cache is not None, (
        "expert slot cache not initialized (expert_cache offload not finalized)"
    )
    return _global_slot_cache


def set_global_slot_cache(cache: ExpertSlotCache | None) -> None:
    global _global_slot_cache
    _global_slot_cache = cache


def clear_global_slot_cache() -> None:
    set_global_slot_cache(None)
