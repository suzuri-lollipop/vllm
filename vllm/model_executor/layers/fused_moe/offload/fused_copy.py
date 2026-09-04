# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device-driven fused miss copy for the expert slot cache.

Port of FreeToken's ``fast_index_copy_multi`` (the fused multi-bank gather that
streams missed expert rows host -> device). One kernel launch copies the SAME
miss list (``evict_slots[i] <- src_indices[i]``) for every registered bank, with
the per-bank base addresses and row byte sizes passed as small precomputed
device tensors, and the valid count ``num_indices`` read ON THE DEVICE -- no
host round trip anywhere in the copy. That makes the decode movement path
(ensure -> copy -> slot GEMM) fixed-shape and host-sync free, hence capturable
into FULL CUDA graphs.

Design (mirrors FreeToken's ``MultiIndexCopyKernel``):
  * grid = ``num_banks * BLOCKS_PER_BANK`` 1-D programs; program ``pid`` serves
    bank ``pid // BLOCKS_PER_BANK`` and grid-strides over that bank's work items
    (one item = one 8-byte unit of one miss row), masking items at and past
    ``num_indices * units_per_row``;
  * source rows live in pinned host memory and are dereferenced from the kernel
    through their UVA device alias (``_build_fused_copy_plan`` resolves each bank
    via ``cudaHostGetDevicePointer``, which is also correct on Windows/WDDM where
    the host VA is not device-dereferenceable);
  * rows are copied in 8-byte units, so the kernel is dtype-agnostic: quantized
    bank schemas (more, narrower banks) slot in without redesign as long as every
    bank's row bytes are a multiple of 16 (validated at plan-build time; the
    cache falls back to the legacy host-count copy otherwise).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    import triton

    from vllm.triton_utils import tl

logger = init_logger(__name__)

__all__ = [
    "FUSED_COPY_BLOCKS_PER_BANK",
    "FUSED_COPY_BLOCK_UNITS",
    "FusedCopyPlan",
    "build_fused_copy_plan",
    "fused_copy_rows",
    "fused_copy_rows_cpu",
]

# Programs per bank in the fused launch. FreeToken measures the PCIe-bound copy
# knee at ~4096 in-flight threads per bank; 8 programs x BLOCK_UNITS int32
# units x 8 warps covers it with one launch regardless of bank count.
FUSED_COPY_BLOCKS_PER_BANK = 8
# 4-byte units copied per program per loop iteration (2048 * 4B = 8 KiB, the
# same bytes per iteration as the original 1024 x 8B).
FUSED_COPY_BLOCK_UNITS = 2048

# Bank rows are copied in 4-byte units. FreeToken's vectorized kernel wanted
# 16-byte rows (uint4 accesses), but a quantized schema banks per-expert scalar
# scales alongside the weights -- NVFP4's w{13,2}_gscale2 / a{13,2}_gscale rows
# are a single fp32, i.e. 4 bytes -- and a single such bank used to disqualify
# the WHOLE plan, dropping every bank onto the legacy host-count copy. That path
# host-synchronizes once per bank per MoE layer per step, which on real hardware
# made decode so slow that vLLM's own `sample_tokens` RPC timed out. The copy is
# PCIe-bound (host bank -> device slot), so narrowing the unit costs
# little, while keeping the device-driven path is worth a great deal.
_BANK_ROW_ALIGN_BYTES = 4


@dataclass
class FusedCopyPlan:
    """Precomputed pointer descriptors for the fused multi-bank miss copy.

    Built once after the banks are registered (addresses are fixed for the
    cache's lifetime), so the per-step launch is CUDA-graph safe.

    Attributes:
        num_banks: number of registered banks.
        dst_ptrs: ``[num_banks]`` int64 device tensor; base address of each
            bank's GPU slot cache.
        src_ptrs: per-layer ``[num_banks]`` int64 device tensors; the
            GPU-visible (UVA alias) base address of each bank's host source.
        feat_bytes: ``[num_banks]`` int64 device tensor; bytes per expert row.
    """

    num_banks: int
    dst_ptrs: torch.Tensor
    src_ptrs: list[torch.Tensor]
    feat_bytes: torch.Tensor
    # Keep the UVA view tensors alive for the cache's lifetime: their deleters
    # hold the mapping (and a reference to the host bank) for each alias.
    _views: list[torch.Tensor] = field(default_factory=list, repr=False)


def build_fused_copy_plan(cache) -> FusedCopyPlan | None:
    """Build the fused multi-bank copy descriptor for ``cache``.

    Returns ``None`` (fall back to the legacy host-count copy, eager/piecewise
    only) if the device is not CUDA, a bank's row bytes or base address is not
    suitably aligned, or a host bank is not pinned (the fused copy dereferences
    pinned memory from the kernel).
    """
    if not HAS_TRITON or cache.device.type != "cuda" or not cache.banks:
        return None

    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

    dst_ptrs: list[int] = []
    feat: list[int] = []
    for _, slot_cache in cache.banks:
        row_bytes = int(slot_cache[0].numel()) * slot_cache.element_size()
        if row_bytes % _BANK_ROW_ALIGN_BYTES != 0:
            logger.warning_once(
                "expert_cache offload: bank rows are %d bytes (not a multiple "
                "of %d); using the legacy host-count miss copy (FULL CUDA-graph "
                "capture of the offloaded MoE is unavailable).",
                row_bytes,
                _BANK_ROW_ALIGN_BYTES,
            )
            return None
        if slot_cache.data_ptr() % _BANK_ROW_ALIGN_BYTES != 0:
            logger.warning_once(
                "expert_cache offload: unaligned slot-cache base address; "
                "using the legacy host-count miss copy."
            )
            return None
        dst_ptrs.append(slot_cache.data_ptr())
        feat.append(row_bytes)

    views: list[torch.Tensor] = []
    src_ptrs: list[torch.Tensor] = []
    try:
        for layer_id in range(cache.num_layers):
            row: list[int] = []
            for per_layer, _ in cache.banks:
                source = per_layer[layer_id]
                if not source.is_pinned():
                    logger.warning_once(
                        "expert_cache offload: host bank layer %d is not "
                        "pinned; using the legacy host-count miss copy "
                        "(enable --moe-cache-pin-memory for the device-driven "
                        "copy).",
                        layer_id,
                    )
                    return None
                # UVA alias of the pinned bank: the address the copy kernel
                # dereferences (cudaHostGetDevicePointer -- also correct on
                # Windows/WDDM, where the host VA differs from the device VA).
                view = get_accelerator_view_from_cpu_tensor(source)
                if view.data_ptr() % _BANK_ROW_ALIGN_BYTES != 0:
                    logger.warning_once(
                        "expert_cache offload: unaligned host bank alias; "
                        "using the legacy host-count miss copy."
                    )
                    return None
                views.append(view)
                row.append(view.data_ptr())
            src_ptrs.append(torch.tensor(row, dtype=torch.int64, device=cache.device))
    except Exception as exc:  # noqa: BLE001 - any UVA gap => legacy path
        logger.warning_once(
            "expert_cache offload: UVA aliasing unavailable (%s); using the "
            "legacy host-count miss copy.",
            exc,
        )
        return None

    return FusedCopyPlan(
        num_banks=len(cache.banks),
        dst_ptrs=torch.tensor(dst_ptrs, dtype=torch.int64, device=cache.device),
        src_ptrs=src_ptrs,
        feat_bytes=torch.tensor(feat, dtype=torch.int64, device=cache.device),
        _views=views,
    )


def fused_copy_rows_cpu(
    banks: list[tuple[list[torch.Tensor], torch.Tensor]],
    layer_id: int,
    evict_slots: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: int | torch.Tensor,
) -> None:
    """CPU reference mirror of the fused copy kernel semantics.

    Gathers the first ``num_indices`` miss rows (``src_indices[i]`` of
    ``layer_id``'s host bank) into slot rows ``evict_slots[i]`` of each bank's
    cache. Entries at and past ``num_indices`` are ignored, whatever
    ``evict_slots`` / ``src_indices`` contain there. Used by the unit tests and
    as the legacy (host-count) copy implementation.
    """
    if isinstance(num_indices, torch.Tensor):
        n = int(num_indices.item())
    else:
        n = int(num_indices)
    if n <= 0:
        return
    slots = evict_slots[:n].to(torch.long)
    idx = src_indices[:n].to(torch.long)
    for per_layer, cache_tensor in banks:
        source = per_layer[layer_id]
        # Gather the missed rows on the source's device, then move them to the
        # slot cache's device if they differ (host bank -> GPU slot cache).
        rows = source[idx.to(source.device)]
        if rows.device != cache_tensor.device:
            rows = rows.to(cache_tensor.device, non_blocking=True)
        # index_copy_ has no CUDA kernel for some narrow dtypes (confirmed on
        # real hardware: NotImplementedError for float8_e4m3fn scale banks,
        # e.g. the NVFP4 offload format's per-block/per-expert scale rows).
        # Plain indexed assignment lowers to index_put_, which has broader
        # dtype coverage; safe here because `slots` are unique LRU slot
        # assignments (no duplicate-index accumulation to worry about, unlike
        # a general index_put_ use).
        cache_tensor[slots.to(cache_tensor.device)] = rows


if HAS_TRITON:

    @triton.jit
    def _fused_copy_rows_kernel(
        dst_ptrs_ptr,
        src_ptrs_ptr,
        feat_bytes_ptr,
        evict_slots_ptr,
        src_indices_ptr,
        num_indices_ptr,
        BLOCKS_PER_BANK: tl.constexpr,
        BLOCK_UNITS: tl.constexpr,
    ):
        # Program pid serves bank pid // BLOCKS_PER_BANK, grid-striding over that
        # bank's miss work items (one 8-byte unit of one miss row each). The
        # valid count is read on the device, so the launch is graph-capturable.
        pid = tl.program_id(0)
        b = pid // BLOCKS_PER_BANK
        num_banks = tl.num_programs(0) // BLOCKS_PER_BANK
        if b >= num_banks:
            return
        s = pid % BLOCKS_PER_BANK

        dst_base = tl.load(dst_ptrs_ptr + b)
        src_base = tl.load(src_ptrs_ptr + b)
        feat = tl.load(feat_bytes_ptr + b)
        n = tl.load(num_indices_ptr)

        units = feat // 4  # 4-byte units per row (see _BANK_ROW_ALIGN_BYTES)
        total = n * units
        start = s.to(tl.int64) * BLOCK_UNITS
        stride = BLOCKS_PER_BANK * BLOCK_UNITS

        num_iters = (total - start + stride - 1) // stride
        num_iters = tl.maximum(num_iters, 0).to(tl.int32)

        dst = dst_base.to(tl.pointer_type(tl.int32))
        src = src_base.to(tl.pointer_type(tl.int32))

        for it in tl.range(num_iters):
            u = (
                start
                + it.to(tl.int64) * stride
                + tl.arange(0, BLOCK_UNITS).to(tl.int64)
            )
            mask = u < total
            row = u // units
            unit = u - row * units
            dst_row = tl.load(evict_slots_ptr + row, mask=mask, other=0).to(tl.int64)
            src_row = tl.load(src_indices_ptr + row, mask=mask, other=0).to(tl.int64)
            vals = tl.load(src + src_row * units + unit, mask=mask)
            tl.store(dst + dst_row * units + unit, vals, mask=mask)


def fused_copy_rows(
    plan: FusedCopyPlan,
    layer_id: int,
    evict_slots: torch.Tensor,
    src_indices: torch.Tensor,
    num_indices: torch.Tensor,
) -> None:
    """Launch the fused multi-bank miss gather on the current stream.

    ``num_indices`` is a one-element DEVICE tensor; it is passed to the kernel
    as a pointer and never read on the host, so this call performs no device
    synchronization and is safe to capture into a CUDA graph.
    """
    grid = (plan.num_banks * FUSED_COPY_BLOCKS_PER_BANK,)
    _fused_copy_rows_kernel[grid](
        plan.dst_ptrs,
        plan.src_ptrs[layer_id],
        plan.feat_bytes,
        evict_slots,
        src_indices,
        num_indices,
        BLOCKS_PER_BANK=FUSED_COPY_BLOCKS_PER_BANK,
        BLOCK_UNITS=FUSED_COPY_BLOCK_UNITS,
        num_warps=8,
    )
