# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure GPU-memory budget policy for auto-sizing the expert slot cache.

Ported from FreeToken's ``engine/cache_budget.py``. No torch/GPU side effects:
every function here is integer/byte arithmetic over already-measured quantities,
so it is unit-testable without a device.

The intended use (auto-sizing follow-up): after the model weights are loaded,
measure the free VRAM, subtract non-paged fixed caches, then split the remainder
MoE-first -- expert slots greedily fill the budget after reserving a few KV pages,
and KV pages take whatever is left. Phase 1 exposes the arithmetic only; wiring
it into the worker's KV-cache profiling is a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    import torch


def expert_bytes_per_slot(sources: dict[str, list[torch.Tensor]]) -> int:
    """Bytes one expert slot occupies on GPU: summed row bytes over all banks.

    Each bank source is per-layer ``[num_experts, *row_shape]`` tensors and is
    already TP-sharded upstream, so the per-row byte count is the per-rank slot
    size. ``tensor[0][0].numel()`` is the per-row element count (one expert slot).
    """
    return sum(t[0][0].numel() * t[0].element_size() for t in sources.values())


def net_cache_budget_bytes(
    memory_ratio: float, baseline_free: int, weights_bytes: int, fixed_cache_size: int
) -> int:
    """Net GPU bytes available for the MoE + KV pools: ``memory_ratio`` of the
    pre-model baseline minus weights and fixed (non-paged) cache. The
    ``(1 - memory_ratio)`` remainder is the CUDA-graph/activation headroom."""
    return int(memory_ratio * baseline_free) - weights_bytes - fixed_cache_size


def required_bytes(
    moe_cache_size: int, num_pages: int, per_expert_bytes: int, cache_per_page: int
) -> int:
    """GPU bytes a ``(moe_cache_size, num_pages)`` geometry occupies."""
    return moe_cache_size * per_expert_bytes + num_pages * cache_per_page


def plan_cache_budget(
    budget_bytes: int,
    per_expert_bytes: int,
    cache_per_page: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_pages: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Split ``budget_bytes`` MoE-first into ``(moe_cache_size, num_pages, overlap)``.

    ``budget_bytes`` is the net pool for MoE cache + KV cache (caller already
    subtracted weights + fixed_cache_size; the ``(1 - memory_ratio)`` remainder is
    the graph headroom). Experts greedily fill the budget after reserving
    ``kv_reserve_pages`` for KV, clamped to ``[floor, min(total_experts,
    max_slots)]`` (floor is ``2 * num_experts`` when prefill overlap is feasible,
    else ``num_experts``); KV pages take whatever remains.
    """
    assert per_expert_bytes > 0, "per_expert_bytes must be positive"
    assert cache_per_page > 0, (
        "cache_per_page must be positive (owned-KV models unsupported here)"
    )

    hi = min(total_experts, max_slots)
    # Prefill overlap borrows two full expert-layer buffers, so it needs
    # >= 2 * num_experts slots; disable it (and lower the floor) if the cap
    # cannot fit that.
    overlap = prefill_overlap and hi >= 2 * num_experts
    lo = 2 * num_experts if overlap else num_experts
    assert hi >= lo, f"slot cap {hi} below the minimum {lo} slots"

    kv_reserve_bytes = kv_reserve_pages * cache_per_page
    # MoE-priority: reserve KV first, then experts greedily take the rest.
    raw = (budget_bytes - kv_reserve_bytes) // per_expert_bytes
    moe_cache_size = max(lo, min(raw, hi))
    # A tiny budget may have forced moe_cache_size below 2E even with overlap on.
    overlap = overlap and moe_cache_size >= 2 * num_experts

    remaining = budget_bytes - moe_cache_size * per_expert_bytes
    num_pages = max(remaining // cache_per_page, kv_reserve_pages)
    # A tiny budget can floor num_pages at kv_reserve_pages even when ``remaining``
    # is below the reserve (or negative), yielding a plan that exceeds budget_bytes.
    # Reject here instead of OOMing in a later CUDA allocation.
    total = moe_cache_size * per_expert_bytes + num_pages * cache_per_page
    assert total <= budget_bytes, (
        f"cache budget too small: minimum plan (moe={moe_cache_size} slots, "
        f"kv={num_pages} pages) needs {total} B > budget {budget_bytes} B "
        "(raise memory_ratio, lower kv_reserve_tokens, or free GPU memory)"
    )
    assert num_pages > 1, "not enough memory for KV cache after MoE allocation"
    return moe_cache_size, num_pages, overlap


def resolve_moe_cache_auto(
    *,
    baseline_free: int,
    weights_bytes: int,
    memory_ratio: float,
    cache_per_page: int,
    fixed_cache_size: int,
    per_expert_bytes: int,
    num_experts: int,
    total_experts: int,
    prefill_overlap: bool,
    kv_reserve_tokens: int,
    page_size: int,
    max_slots: int,
) -> tuple[int, int, bool]:
    """Resolve auto-sizing into ``(moe_cache_size, num_pages, prefill_overlap)``.

    Applies ``memory_ratio`` to the persisted pre-model baseline exactly once, then
    defers the MoE-vs-KV split to :func:`plan_cache_budget`. The
    ``(1 - memory_ratio)`` remainder is the CUDA-graph/activation headroom (not
    subtracted here).
    """
    budget_bytes = net_cache_budget_bytes(
        memory_ratio, baseline_free, weights_bytes, fixed_cache_size
    )
    kv_reserve_pages = cdiv(kv_reserve_tokens, page_size)
    return plan_cache_budget(
        budget_bytes=budget_bytes,
        per_expert_bytes=per_expert_bytes,
        cache_per_page=cache_per_page,
        num_experts=num_experts,
        total_experts=total_experts,
        prefill_overlap=prefill_overlap,
        kv_reserve_pages=kv_reserve_pages,
        max_slots=max_slots,
    )
