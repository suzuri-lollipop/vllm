# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global-LRU "ensure" logic for the MoE expert slot cache.

This is an in-tree port of FreeToken's ``lru_ensure`` (the flashlib slot-cache
primitive). FreeToken's authoritative spec is the Triton
``_ensure_experts_hybrid_kernel`` together with its bit-identical CPU mirror
``_ensure_experts_hybrid_cpu``; this module implements the *non-hybrid* variant
used for pure GPU offload:

* every routed expert that is not resident (a "miss") is fetched this step,
* victims are chosen by ``argmin(usage)`` over the slots, excluding slots that
  hold an expert routed this step (so a live expert is never evicted under us)
  and slots already chosen as victims this step,
* ``topk_ids`` is rewritten IN PLACE from raw expert ids to cache slot ids,
* the miss list is emitted (``evict_slots`` / ``src_indices`` / ``num_indices``)
  so the caller can stream the missing expert rows host -> device.

The bookkeeping tensors (see :class:`ExpertSlotCache`) are:

* ``slot_for_id`` ``[num_layers, num_experts]`` int32 -- slot holding each
  (layer, expert), ``-1`` when absent,
* ``id_of_slot`` ``[cache_size]`` int32 -- flat id (``layer * E + expert``)
  stored in each slot, ``-1`` when the slot is empty,
* ``usage`` ``[cache_size]`` int64 -- LRU timestamps,
* ``step`` scalar int64 -- global step counter, incremented once per ensure
  call.

Both a pure-Python CPU reference (:func:`lru_ensure_cpu`) and a Triton GPU
kernel (:func:`lru_ensure`) are provided; they make identical eviction/fetch
decisions and are cross-tested on GPU in ``tests`` (the CPU path is also
unit-tested standalone, since it runs without a device).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.triton_utils import HAS_TRITON

if HAS_TRITON:
    import triton

    from vllm.triton_utils import tl

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.offload.slot_cache import (
        ExpertSlotCache,
    )

__all__ = [
    "lru_ensure",
    "lru_ensure_cpu",
    "materialize_layer_cpu",
    "materialize_layer",
    "reset_cache_cpu",
    "reset_cache",
]


def _unique_in_order(values: list[int]) -> list[int]:
    seen: list[int] = []
    for value in values:
        if value not in seen:
            seen.append(value)
    return seen


def lru_ensure_cpu(
    cache: ExpertSlotCache,
    layer_id: int,
    expert_ids: torch.Tensor,
) -> None:
    """CPU reference for :func:`lru_ensure`.

    Rewrites ``expert_ids`` (in place) from raw expert ids to cache slot ids and
    stages the miss list for the host -> device copy. Decisions are bit-identical
    to the Triton kernel: misses are fetched in ascending expert-id order and
    victims are picked by ``argmin(usage)`` with the lowest slot index winning
    ties, excluding slots that hold an expert routed this step and slots already
    chosen as victims this step.

    This is the authoritative spec; it runs on any device but is exercised on
    CPU in the unit tests.
    """
    num_experts = cache.num_experts
    cache_size = cache.cache_size
    base = layer_id * num_experts

    flat = expert_ids.view(-1)
    actives = _unique_in_order(flat.tolist())

    step = int(cache.step.item()) + 1
    cache.step.fill_(step)

    # Hits: refresh the LRU timestamp of every resident active expert.
    for expert in actives:
        slot = int(cache.slot_for_id[layer_id, expert].item())
        if slot != -1:
            cache.usage[slot] = step

    # Misses, in ascending expert-id order (deterministic victim pairing).
    missing = sorted(
        e for e in actives if int(cache.slot_for_id[layer_id, e].item()) == -1
    )
    cache.num_indices.fill_(len(missing))

    # Slots holding an expert routed this step are protected from eviction.
    active_flat_ids = {base + e for e in actives}
    usage = cache.usage.tolist()
    used_victims: set[int] = set()

    for idx, expert in enumerate(missing):
        victim = -1
        best_usage = None
        for slot in range(cache_size):
            if slot in used_victims:
                continue
            owner = int(cache.id_of_slot[slot].item())
            if owner in active_flat_ids:
                continue
            u = usage[slot]
            if best_usage is None or u < best_usage:
                best_usage = u
                victim = slot
        assert victim >= 0, "no evictable slot (cache_size < active experts?)"

        old_id = int(cache.id_of_slot[victim].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1
        cache.id_of_slot[victim] = base + expert
        cache.slot_for_id[layer_id, expert] = victim
        cache.usage[victim] = step
        usage[victim] = step
        used_victims.add(victim)
        cache.evict_slots[idx] = victim
        cache.src_indices[idx] = expert  # layer-local expert row

    # Rewrite raw expert ids -> slot ids (every active is resident now).
    for i in range(flat.numel()):
        flat[i] = int(cache.slot_for_id[layer_id, int(flat[i].item())].item())


def materialize_layer_cpu(
    cache: ExpertSlotCache,
    layer_id: int,
) -> None:
    """CPU reference for :func:`materialize_layer`.

    Synchronously makes the whole ``layer_id`` expert layer resident in the first
    ``num_experts`` slots (``position == expert id``), evicting whatever was there.
    Used as the simple prefill path: the routing ids then pass through unmapped and
    the staged copy moves all ``num_experts`` rows.
    """
    num_experts = cache.num_experts
    base = layer_id * num_experts

    # Drop any slot currently owned by this layer (its bytes are about to be
    # overwritten positionally) and clear the residency of any other-layer expert
    # living in the first num_experts slots.
    for slot in range(num_experts):
        old_id = int(cache.id_of_slot[slot].item())
        if old_id >= 0:
            cache.slot_for_id.view(-1)[old_id] = -1

    step = int(cache.step.item()) + 1
    cache.step.fill_(step)

    for expert in range(num_experts):
        slot = expert
        cache.id_of_slot[slot] = base + expert
        cache.slot_for_id[layer_id, expert] = slot
        cache.usage[slot] = step
        cache.evict_slots[expert] = slot
        cache.src_indices[expert] = expert  # layer-local row == expert id
    cache.num_indices.fill_(num_experts)


def reset_cache_cpu(cache: ExpertSlotCache) -> None:
    """CPU reference for :func:`reset_cache` (cold-start the slot cache)."""
    cache.slot_for_id.fill_(-1)
    cache.id_of_slot.fill_(-1)
    cache.usage.zero_()
    cache.step.zero_()
    if cache.active_mask is not None:
        cache.active_mask.zero_()
    cache.num_indices.zero_()


# ---------------------------------------------------------------------------
# Triton (GPU) implementations
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit(do_not_specialize=["layer_id", "num_active"])
    def _lru_ensure_kernel(
        expert_ids_ptr,
        slot_for_id_ptr,
        id_of_slot_ptr,
        usage_ptr,
        step_ptr,
        evict_slots_ptr,
        src_indices_ptr,
        num_indices_ptr,
        layer_id,
        num_active,
        num_experts: tl.constexpr,
        cache_size: tl.constexpr,
        BLOCK_E: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        # Non-hybrid timestamp-LRU: fetch ALL misses this step.
        step = tl.load(step_ptr) + 1
        tl.store(step_ptr, step)
        base = layer_id * num_experts

        # ---- Phase 1: active + missing over experts ----
        off_e = tl.arange(0, BLOCK_E)
        e_mask = off_e < num_experts
        is_active = tl.zeros((BLOCK_E,), dtype=tl.int1)
        for i in tl.range(num_active):
            e = tl.load(expert_ids_ptr + i)
            is_active = is_active | (off_e == e)
        slot = tl.load(slot_for_id_ptr + base + off_e, mask=e_mask, other=-1)
        is_missing = is_active & (slot == -1) & e_mask
        num_missing = tl.sum(is_missing.to(tl.int32))
        tl.store(num_indices_ptr, num_missing.to(tl.int64))
        is_hit = is_active & (slot >= 0)
        tl.store(usage_ptr + slot, step, mask=is_hit)

        # Miss ordering: ascending expert id (cumsum rank over off_e).
        missing_rank = tl.cumsum(is_missing.to(tl.int32)) - 1

        # ---- Phase 2: evict victims by argmin(usage) for every miss ----
        if num_missing > 0:
            off_c = tl.arange(0, BLOCK_C)
            c_mask = off_c < cache_size
            oid = tl.load(id_of_slot_ptr + off_c, mask=c_mask, other=-1)
            u = tl.load(usage_ptr + off_c, mask=c_mask, other=9223372036854775807)
            u = u.to(tl.int64)
            # Protect slots that hold an expert routed this step.
            owner_active = c_mask & False
            for i in tl.range(num_active):
                ei = tl.load(expert_ids_ptr + i)
                owner_active = owner_active | (oid == base + ei)
            u = tl.where(owner_active | (~c_mask), 9223372036854775807, u)
            for i in tl.range(num_missing):
                victim = tl.argmin(u, axis=0).to(tl.int32)
                old_id = tl.sum(tl.where(off_c == victim, oid, 0))
                if old_id >= 0:
                    tl.store(slot_for_id_ptr + old_id, -1)
                e = tl.sum(tl.where((missing_rank == i) & is_missing, off_e, 0))
                tl.store(id_of_slot_ptr + victim, base + e)
                tl.store(slot_for_id_ptr + base + e, victim)
                tl.store(usage_ptr + victim, step)
                tl.store(evict_slots_ptr + i, victim)
                tl.store(src_indices_ptr + i, e)  # layer-local row
                u = tl.where(off_c == victim, 9223372036854775807, u)

        # ---- Phase 3: rewrite expert_ids -> slot id ----
        for i in tl.range(num_active):
            e = tl.load(expert_ids_ptr + i)
            s = tl.load(slot_for_id_ptr + base + e)
            tl.store(expert_ids_ptr + i, s)

    @triton.jit(do_not_specialize=["layer_id"])
    def _materialize_layer_kernel(
        slot_for_id_ptr,
        id_of_slot_ptr,
        usage_ptr,
        step_ptr,
        evict_slots_ptr,
        src_indices_ptr,
        num_indices_ptr,
        layer_id,
        num_experts: tl.constexpr,
        cache_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        off = tl.arange(0, BLOCK)
        expert_mask = off < num_experts
        slot_mask = off < cache_size
        slot = off

        base = layer_id * num_experts
        old_id = tl.load(id_of_slot_ptr + slot, mask=slot_mask, other=-1)
        # Flat ids make "belongs to this layer" a range check.
        same_layer = slot_mask & (old_id >= base) & (old_id < base + num_experts)
        tl.store(id_of_slot_ptr + slot, -1, mask=same_layer)
        tl.store(usage_ptr + slot, 0, mask=same_layer)

        old_valid = expert_mask & (old_id >= 0) & (~same_layer)
        tl.store(slot_for_id_ptr + old_id, -1, mask=old_valid)

        step = tl.load(step_ptr) + 1
        tl.store(step_ptr, step)
        tl.store(id_of_slot_ptr + slot, base + off, mask=expert_mask)
        tl.store(slot_for_id_ptr + base + off, slot, mask=expert_mask)
        tl.store(usage_ptr + slot, step, mask=expert_mask)
        tl.store(evict_slots_ptr + off, slot, mask=expert_mask)
        tl.store(src_indices_ptr + off, off, mask=expert_mask)  # layer-local row
        tl.store(num_indices_ptr, num_experts)

    @triton.jit
    def _reset_cache_kernel(
        slot_for_id_ptr,
        id_of_slot_ptr,
        usage_ptr,
        step_ptr,
        active_mask_ptr,
        num_indices_ptr,
        total_ids: tl.constexpr,
        num_experts: tl.constexpr,
        cache_size: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(slot_for_id_ptr + off, -1, mask=off < total_ids)
        tl.store(id_of_slot_ptr + off, -1, mask=off < cache_size)
        tl.store(usage_ptr + off, 0, mask=off < cache_size)
        if active_mask_ptr is not None:
            tl.store(active_mask_ptr + off, 0, mask=off < num_experts)
        if tl.program_id(0) == 0:
            tl.store(step_ptr, 0)
            tl.store(num_indices_ptr, 0)


def lru_ensure(
    cache: ExpertSlotCache,
    layer_id: int,
    expert_ids: torch.Tensor,
) -> None:
    """Make this layer's routed experts resident; rewrite ``expert_ids`` to slots.

    Dispatches to the CPU reference when the bookkeeping lives off-CUDA (tests)
    and to the Triton kernel on the GPU hot path.
    """
    if not expert_ids.is_cuda:
        return lru_ensure_cpu(cache, layer_id, expert_ids)

    block_e = triton.next_power_of_2(cache.num_experts)
    block_c = triton.next_power_of_2(cache.cache_size)
    num_warps = 8 if block_c >= 2048 else 4
    _lru_ensure_kernel[(1,)](
        expert_ids,
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        expert_ids.numel(),
        cache.num_experts,
        cache.cache_size,
        BLOCK_E=block_e,
        BLOCK_C=block_c,
        num_warps=num_warps,
    )


def materialize_layer(cache: ExpertSlotCache, layer_id: int) -> None:
    """Stage a full-layer materialize (simple synchronous prefill path)."""
    if cache.device.type != "cuda":
        return materialize_layer_cpu(cache, layer_id)

    block = triton.next_power_of_2(max(cache.num_experts, cache.cache_size))
    _materialize_layer_kernel[(1,)](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.evict_slots,
        cache.src_indices,
        cache.num_indices,
        layer_id,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )


def reset_cache(cache: ExpertSlotCache) -> None:
    """Cold-start the global slot cache (drop all residency)."""
    if cache.device.type != "cuda":
        return reset_cache_cpu(cache)

    block = 256
    total_ids = cache.num_layers * cache.num_experts
    grid = (triton.cdiv(max(total_ids, cache.cache_size), block),)
    _reset_cache_kernel[grid](
        cache.slot_for_id,
        cache.id_of_slot,
        cache.usage,
        cache.step,
        cache.active_mask,
        cache.num_indices,
        total_ids,
        cache.num_experts,
        cache.cache_size,
        BLOCK=block,
    )
