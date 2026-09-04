# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for expert-granularity MoE offloading (expert_cache backend).

Covers the CPU-testable core of the FreeToken port:
  * the LRU ensure policy (eviction choice, in-place id rewrite, usage/step),
  * the materialize (prefill) and reset paths,
  * bank geometry / sizing,
  * the host->device miss-copy plan,
  * config resolution + rejection,
  * weight diversion into pinned host banks (small synthetic MoE layer).

GPU-only paths (the Triton ensure/materialize kernels and the slot-space GEMM)
are cross-checked against the CPU reference and skip-guarded when CUDA/Triton are
unavailable.

Run `pytest tests/kernels/moe/test_moe_expert_offload.py`.
"""

import types

import pytest
import torch
import torch.nn as nn

from vllm.config import (
    EXPERT_CACHE_MAX_SLOTS,
    ExpertCacheOffloadConfig,
    OffloadConfig,
)
from vllm.model_executor.layers.fused_moe.offload.lru import (
    lru_ensure_cpu,
    materialize_layer_cpu,
    reset_cache_cpu,
)
from vllm.model_executor.layers.fused_moe.offload.slot_cache import (
    BANK_SCHEMAS,
    ExpertSlotCache,
    clear_global_slot_cache,
)
from vllm.model_executor.offloader.base import NoopOffloader, create_offloader
from vllm.model_executor.offloader.expert_cache import ExpertCacheOffloader
from vllm.triton_utils import HAS_TRITON

CPU = torch.device("cpu")

# Small, fast shapes for the policy tests.
NUM_EXPERTS = 8
CACHE_SIZE = 8  # >= num_experts (cache invariant)
HIDDEN = 16
INTER = 24


def make_cache(
    num_layers: int = 2,
    num_experts: int = NUM_EXPERTS,
    cache_size: int = CACHE_SIZE,
    device: torch.device = CPU,
    dtype: torch.dtype = torch.float32,
) -> ExpertSlotCache:
    """Build a slot cache with random host banks registered."""
    cache = ExpertSlotCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=device,
    )
    w13 = [
        torch.randn(num_experts, 2 * INTER, HIDDEN, dtype=dtype)
        for _ in range(num_layers)
    ]
    w2 = [
        torch.randn(num_experts, HIDDEN, INTER, dtype=dtype) for _ in range(num_layers)
    ]
    cache.set_bank_sources({"w13": w13, "w2": w2})
    return cache


# ---------------------------------------------------------------------------
# LRU ensure policy (CPU reference)
# ---------------------------------------------------------------------------


def test_ensure_passes_through_non_local_experts_under_ep():
    """Expert parallelism: -1 (another rank's expert) must survive untouched.

    _offload_forward maps global routing ids through expert_map before the
    ensure, so -1 marks a token this rank contributes nothing for. Resolving it
    against slot_for_id would silently compute the LAST expert instead (negative
    indexing), i.e. wrong numerics rather than a crash.
    """
    cache = make_cache(num_layers=1)
    ids = torch.tensor([3, -1, 2, -1], dtype=torch.int32)
    lru_ensure_cpu(cache, 0, ids)

    # Only the two real experts were fetched; -1 is not a residency candidate.
    assert int(cache.num_indices.item()) == 2
    assert int(cache.slot_for_id[0, 3].item()) >= 0
    assert int(cache.slot_for_id[0, 2].item()) >= 0
    # -1 passes through; the real ids became slot ids.
    assert ids.tolist() == [
        int(cache.slot_for_id[0, 3].item()),
        -1,
        int(cache.slot_for_id[0, 2].item()),
        -1,
    ]


def test_ensure_all_non_local_is_a_noop_under_ep():
    """A batch routed entirely to other ranks' experts fetches nothing."""
    cache = make_cache(num_layers=1)
    ids = torch.full((4,), -1, dtype=torch.int32)
    lru_ensure_cpu(cache, 0, ids)

    assert int(cache.num_indices.item()) == 0
    assert ids.tolist() == [-1, -1, -1, -1]
    # Nothing became resident.
    assert int((cache.slot_for_id >= 0).sum().item()) == 0


def test_ensure_non_local_does_not_protect_previous_layer_slot_under_ep():
    """-1 must not shield flat id ``base - 1`` (the previous layer's last expert).

    Eviction protection is keyed on ``base + expert_id``; an unguarded -1 marks
    the preceding layer's last expert as "routed this step", needlessly pinning
    it in the cache.
    """
    cache = make_cache(num_layers=2, num_experts=4, cache_size=4)
    # Fill the cache with layer 0's experts, oldest usage on expert 3.
    lru_ensure_cpu(cache, 0, torch.tensor([3, 0, 1, 2], dtype=torch.int32))
    cache.usage[int(cache.slot_for_id[0, 3].item())] = 0

    # Layer 1 (base = 4) routes one real expert plus a non-local -1.
    victim_slot = int(cache.slot_for_id[0, 3].item())
    lru_ensure_cpu(cache, 1, torch.tensor([0, -1], dtype=torch.int32))

    # Layer 0's expert 3 (flat id base-1 == 3) was the argmin victim and must
    # have been evicted rather than protected by the -1.
    assert int(cache.slot_for_id[1, 0].item()) == victim_slot
    assert int(cache.slot_for_id[0, 3].item()) == -1


def test_ensure_cold_start_assigns_and_rewrites_in_place():
    cache = make_cache(num_layers=1)
    ids = torch.tensor([3, 1, 2, 1], dtype=torch.int32)
    lru_ensure_cpu(cache, 0, ids)
    # Unique actives {1,2,3} were all misses on a cold cache.
    assert int(cache.num_indices.item()) == 3
    # ids rewritten in place to slot ids.
    assert ids.dtype == torch.int32
    # Every active expert is now resident and ids map to its slot.
    for e in (1, 2, 3):
        slot = int(cache.slot_for_id[0, e].item())
        assert slot >= 0
    # Rewritten ids equal the slot of the original expert.
    assert ids.tolist() == [int(cache.slot_for_id[0, e].item()) for e in (3, 1, 2, 1)]
    # step advanced exactly once.
    assert int(cache.step.item()) == 1


def test_ensure_hit_refreshes_usage_no_miss():
    cache = make_cache(num_layers=1)
    a = torch.tensor([0, 1], dtype=torch.int32)
    lru_ensure_cpu(cache, 0, a)
    # Second call routes the same experts -> all hits, no misses.
    b = torch.tensor([1, 0, 1], dtype=torch.int32)
    lru_ensure_cpu(cache, 0, b)
    assert int(cache.num_indices.item()) == 0
    assert int(cache.step.item()) == 2
    # The slots for experts 0 and 1 carry the newest timestamp.
    s0 = int(cache.slot_for_id[0, 0].item())
    s1 = int(cache.slot_for_id[0, 1].item())
    assert int(cache.usage[s0].item()) == 2
    assert int(cache.usage[s1].item()) == 2


def test_eviction_picks_argmin_usage_and_protects_actives():
    # Two layers share the global cache; evictions happen across layers.
    cache = make_cache(num_layers=2, num_experts=4, cache_size=4)
    # Layer 0 fills all 4 slots.
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert cache.slot_for_id[0].tolist() == [0, 1, 2, 3]
    # Layer 1 routes experts {0,1}: both misses; every slot holds a layer-0
    # (non-active) expert with equal usage, so the lowest-index slots win.
    ids = torch.tensor([0, 1], dtype=torch.int32)
    lru_ensure_cpu(cache, 1, ids)
    assert int(cache.num_indices.item()) == 2
    # Layer-0 experts 0,1 evicted; layer-1 experts 0,1 resident in slots 0,1.
    assert cache.slot_for_id[0].tolist() == [-1, -1, 2, 3]
    assert cache.slot_for_id[1].tolist() == [0, 1, -1, -1]
    # Flat ids: layer 1 expert e -> 1*4 + e.
    assert cache.id_of_slot.tolist() == [4, 5, 2, 3]
    # ids rewritten to slots.
    assert ids.tolist() == [0, 1]


def test_eviction_lru_order_across_steps():
    # Expert used less recently is evicted first.
    cache = make_cache(num_layers=2, num_experts=4, cache_size=4)
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    # Touch layer-0 experts 0 and 1 again (refresh their usage).
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1], dtype=torch.int32))
    # Now layer 1 needs 2 slots: least-recently-used are layer-0 experts 2,3.
    lru_ensure_cpu(cache, 1, torch.tensor([0, 1], dtype=torch.int32))
    # Experts 2,3 of layer 0 evicted; 0,1 retained.
    assert cache.slot_for_id[0].tolist() == [0, 1, -1, -1]
    assert cache.slot_for_id[1, 0].item() >= 0
    assert cache.slot_for_id[1, 1].item() >= 0


def test_ensure_usage_and_step_semantics_monotonic():
    cache = make_cache(num_layers=1)
    for i in range(3):
        lru_ensure_cpu(cache, 0, torch.tensor([i % NUM_EXPERTS], dtype=torch.int32))
    assert int(cache.step.item()) == 3
    # All recorded usage timestamps are <= the current step and > 0.
    usage = cache.usage.tolist()
    assert all(0 <= u <= 3 for u in usage)


# ---------------------------------------------------------------------------
# Materialize (prefill) and reset
# ---------------------------------------------------------------------------


def test_materialize_places_layer_in_first_slots():
    cache = make_cache(num_layers=2, num_experts=4, cache_size=4)
    # Put layer 0 in the cache first.
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    # Materialize layer 1: all experts into slots 0..E-1 (position == expert id).
    materialize_layer_cpu(cache, 1)
    assert int(cache.num_indices.item()) == 4
    assert cache.slot_for_id[1].tolist() == [0, 1, 2, 3]
    # Layer-0 residency was cleared from the slots it lost.
    assert cache.id_of_slot.tolist() == [4, 5, 6, 7]
    # Staged copy indices are the identity (position == expert id).
    assert cache.src_indices[:4].tolist() == [0, 1, 2, 3]
    assert cache.evict_slots[:4].tolist() == [0, 1, 2, 3]


def test_reset_cold_starts():
    cache = make_cache(num_layers=2, num_experts=4, cache_size=4)
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1], dtype=torch.int32))
    reset_cache_cpu(cache)
    assert int(cache.step.item()) == 0
    assert torch.all(cache.slot_for_id == -1)
    assert torch.all(cache.id_of_slot == -1)
    assert torch.all(cache.usage == 0)
    assert int(cache.num_indices.item()) == 0


# ---------------------------------------------------------------------------
# Miss copy plan
# ---------------------------------------------------------------------------


def test_copy_missing_moves_missed_rows_into_slots():
    cache = make_cache(num_layers=1, num_experts=4, cache_size=4)
    src_w13 = cache.bank_sources["w13"][0]
    src_w2 = cache.bank_sources["w2"][0]
    ids = torch.tensor([2, 3], dtype=torch.int32)
    cache.ensure_experts(0, ids)
    cache.copy_missing()
    # The routed experts' rows now live at their assigned slots.
    for e in (2, 3):
        slot = int(cache.slot_for_id[0, e].item())
        assert torch.equal(cache.bank_caches["w13"][slot], src_w13[e])
        assert torch.equal(cache.bank_caches["w2"][slot], src_w2[e])


def _force_stage_buffers(cache, rows: int) -> None:
    """Install unpinned stand-ins for the legacy copy's staging buffers.

    _alloc_stage_buffers page-locks, which needs a CUDA context the CPU test
    machine does not have; the staged gather itself is device-independent, so
    plain host buffers exercise exactly the same code.
    """
    cache._stage_buffers = [
        torch.empty((rows, *c.shape[1:]), dtype=c.dtype) for _, c in cache.banks
    ]


def test_staged_miss_copy_matches_the_pageable_gather():
    """The staging buffers must not change which rows land in which slots."""
    ids = torch.tensor([2, 3, 0], dtype=torch.int32)

    expected = make_cache(num_layers=1, num_experts=4, cache_size=4)
    assert expected._stage_buffers is None, "CPU banks should not be page-locked"
    expected.ensure_experts(0, ids.clone())
    expected.copy_missing()

    staged = make_cache(num_layers=1, num_experts=4, cache_size=4)
    # Same random banks, so the two caches are comparable row for row.
    staged.set_bank_sources(
        {name: list(expected.bank_sources[name]) for name in expected.bank_schema}
    )
    _force_stage_buffers(staged, rows=4)
    staged.ensure_experts(0, ids.clone())
    staged.copy_missing()

    for name in expected.bank_schema:
        assert torch.equal(staged.bank_caches[name], expected.bank_caches[name])


def test_staged_miss_copy_handles_narrow_float8_scale_banks():
    """float8 scale rows have no index_select kernel; the byte view must work."""
    cache = ExpertSlotCache(num_layers=1, num_experts=4, cache_size=4, device=CPU)
    scales = torch.arange(4 * INTER, dtype=torch.float32).reshape(4, INTER)
    cache.set_bank_sources(
        {
            "w13": [torch.randn(4, 2 * INTER, HIDDEN)],
            "w2": [scales.to(torch.float8_e4m3fn)],
        }
    )
    _force_stage_buffers(cache, rows=4)

    src = cache.bank_sources["w2"][0]
    ids = torch.tensor([3, 1], dtype=torch.int32)
    cache.ensure_experts(0, ids)
    cache.copy_missing()

    for e in (3, 1):
        slot = int(cache.slot_for_id[0, e].item())
        got = cache.bank_caches["w2"][slot].view(torch.uint8)
        assert torch.equal(got, src[e].view(torch.uint8))


def test_staged_miss_copy_is_skipped_for_a_whole_layer_materialize():
    """A materialize stages every expert, which overflows the small buffer."""
    cache = make_cache(num_layers=1, num_experts=8, cache_size=8)
    _force_stage_buffers(cache, rows=4)
    src_w13 = cache.bank_sources["w13"][0]

    cache.materialize_layer(0)
    cache.copy_missing()

    for e in range(8):
        assert torch.equal(cache.bank_caches["w13"][e], src_w13[e])


def test_copy_missing_materialize_copies_whole_layer():
    cache = make_cache(num_layers=1, num_experts=4, cache_size=4)
    src_w13 = cache.bank_sources["w13"][0]
    cache.materialize_layer(0)
    cache.copy_missing()
    for e in range(4):
        assert torch.equal(cache.bank_caches["w13"][e], src_w13[e])


def test_copy_missing_noop_when_all_hits():
    cache = make_cache(num_layers=1, num_experts=4, cache_size=4)
    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32))
    cache.copy_missing()
    # Poison a slot, then route the same experts again (all hits): the copy must
    # not overwrite anything (num_indices == 0).
    cache.bank_caches["w13"][0].fill_(123.0)
    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32))
    cache.copy_missing()
    assert torch.all(cache.bank_caches["w13"][0] == 123.0)


# ---------------------------------------------------------------------------
# Fused (device-driven) miss copy
# ---------------------------------------------------------------------------


def _make_identifiable_cache(num_experts: int = 4, cache_size: int = 4):
    """Slot cache whose host-bank rows are filled with per-expert constants so
    gathered rows are trivially recognizable, and whose slot caches start at a
    sentinel so untouched slots are detectable."""
    cache = make_cache(num_layers=1, num_experts=num_experts, cache_size=cache_size)
    # Source row e is filled with (e + 1) in both banks.
    for name in cache.bank_schema:
        src = cache.bank_sources[name][0]
        for e in range(num_experts):
            src[e].fill_(float(e + 1))
        # Slot caches start at a sentinel distinct from any source row value.
        cache.bank_caches[name].fill_(-999.0)
    return cache


def test_fused_copy_rows_cpu_gather_mapping_and_multibank():
    """CPU mirror of the fused copy: row gather mapping + fan-out to every bank."""
    from vllm.model_executor.layers.fused_moe.offload.fused_copy import (
        fused_copy_rows_cpu,
    )

    cache = _make_identifiable_cache()
    # src row 1 -> slot 2, src row 3 -> slot 0, src row 2 -> slot 3.
    cache.evict_slots[:4] = torch.tensor([2, 0, 3, 3], dtype=torch.int32)
    cache.src_indices[:4] = torch.tensor([1, 3, 2, 0], dtype=torch.int32)
    num_indices = torch.tensor([3], dtype=torch.int64)
    fused_copy_rows_cpu(
        cache.banks, 0, cache.evict_slots, cache.src_indices, num_indices
    )
    for name in cache.bank_schema:
        c = cache.bank_caches[name]
        assert torch.all(c[2] == 2.0)  # source row 1 -> value 2
        assert torch.all(c[0] == 4.0)  # source row 3 -> value 4
        assert torch.all(c[3] == 3.0)  # source row 2 -> value 3
        # Slot 1 was never a destination: still the sentinel.
        assert torch.all(c[1] == -999.0)


def test_fused_copy_rows_cpu_masks_past_num_indices():
    """Entries at/after num_indices are ignored, whatever garbage they hold."""
    from vllm.model_executor.layers.fused_moe.offload.fused_copy import (
        fused_copy_rows_cpu,
    )

    cache = _make_identifiable_cache()
    # Only the first 2 entries are valid; the tail holds an OUT-OF-RANGE source
    # row (99) that must NOT be read, and a destination that must NOT be written.
    cache.evict_slots[:4] = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cache.src_indices[:4] = torch.tensor([0, 1, 99, 99], dtype=torch.int32)
    num_indices = torch.tensor([2], dtype=torch.int64)
    fused_copy_rows_cpu(
        cache.banks, 0, cache.evict_slots, cache.src_indices, num_indices
    )
    for name in cache.bank_schema:
        c = cache.bank_caches[name]
        assert torch.all(c[0] == 1.0)
        assert torch.all(c[1] == 2.0)
        # Slots 2 and 3 untouched (the garbage tail was masked off).
        assert torch.all(c[2] == -999.0)
        assert torch.all(c[3] == -999.0)


def test_fused_copy_rows_cpu_zero_misses_noop():
    from vllm.model_executor.layers.fused_moe.offload.fused_copy import (
        fused_copy_rows_cpu,
    )

    cache = _make_identifiable_cache()
    cache.evict_slots[:4].fill_(3)
    cache.src_indices[:4].fill_(3)
    fused_copy_rows_cpu(
        cache.banks,
        0,
        cache.evict_slots,
        cache.src_indices,
        torch.zeros(1, dtype=torch.int64),
    )
    for name in cache.bank_schema:
        assert torch.all(cache.bank_caches[name] == -999.0)


def test_copy_missing_masks_past_num_indices_end_to_end():
    """copy_missing (CPU reference path) honors num_indices: garbage staged past
    the count must not be copied (an out-of-range src row would raise if read)."""
    cache = _make_identifiable_cache()
    # Stage a valid miss list then corrupt the tail beyond num_indices.
    cache.ensure_experts(0, torch.tensor([0, 1], dtype=torch.int32))
    n = int(cache.num_indices.item())
    assert n == 2
    tail = slice(n, cache.evict_slots.numel())
    cache.evict_slots[tail].fill_(3)
    cache.src_indices[tail].fill_(99)  # out of range; must never be gathered
    cache.copy_missing()
    for name in cache.bank_schema:
        c = cache.bank_caches[name]
        assert torch.all(c[0] == 1.0) or torch.all(c[0] == 2.0)
        assert torch.all(c[1] == 1.0) or torch.all(c[1] == 2.0)


def test_build_fused_copy_plan_none_on_cpu():
    """No CUDA device -> no fused plan (copy falls back to the reference path)."""
    cache = make_cache(num_layers=1, num_experts=4, cache_size=4, device=CPU)
    assert cache._fused_plan is None


def test_reset_offload_state_cold_starts_via_offloader():
    off = ExpertCacheOffloader(cache_size=CACHE_SIZE, pin_memory=False)
    off._divert_routed_experts(_FakeRoutedExperts(NUM_EXPERTS))
    off.post_init()
    cache = off.cache
    assert cache is not None
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1], dtype=torch.int32))
    assert int(cache.step.item()) == 1
    off.reset_offload_state()
    assert int(cache.step.item()) == 0
    assert torch.all(cache.slot_for_id == -1)
    clear_global_slot_cache()


def test_fused_copy_launcher_does_not_read_count_on_host():
    """The device-driven copy must not pull num_indices to the host (that is the
    whole point: FULL-graph capturability). Guard at the source level: the fused
    launcher (and kernel, when Triton is present) never calls
    .item()/.cpu()/.tolist()/numpy() on any argument."""
    import inspect

    from vllm.model_executor.layers.fused_moe.offload import fused_copy

    src = inspect.getsource(fused_copy.fused_copy_rows)
    kernel = getattr(fused_copy, "_fused_copy_rows_kernel", None)
    if kernel is not None:
        src += inspect.getsource(kernel)
    for bad in (".item(", ".cpu(", ".tolist(", ".numpy("):
        assert bad not in src, f"fused copy reads a host value via {bad}"


# ---------------------------------------------------------------------------
# Geometry / sizing
# ---------------------------------------------------------------------------


def test_bank_geometry_and_sizing():
    cache = make_cache(num_layers=2, num_experts=4, cache_size=6)
    # One slot holds one expert's w13 row plus one expert's w2 row.
    per_expert = (2 * INTER * HIDDEN + HIDDEN * INTER) * 4  # float32 elems*bytes
    assert cache.bytes_per_slot() == per_expert
    assert cache.slot_cache_bytes() == 6 * per_expert
    row_bytes = cache.bank_row_bytes()
    assert row_bytes["w13"] == 2 * INTER * HIDDEN * 4
    assert row_bytes["w2"] == HIDDEN * INTER * 4
    # Slot caches have the right shapes.
    assert cache.bank_caches["w13"].shape == (6, 2 * INTER, HIDDEN)
    assert cache.bank_caches["w2"].shape == (6, HIDDEN, INTER)
    assert set(BANK_SCHEMAS["bf16"]) == {"w13", "w2"}


def test_cache_size_below_num_experts_rejected():
    with pytest.raises(ValueError, match="cache_size"):
        ExpertSlotCache(num_layers=1, num_experts=8, cache_size=4, device=CPU)


def test_set_bank_sources_shape_mismatch_rejected():
    cache = ExpertSlotCache(num_layers=2, num_experts=4, cache_size=4, device=CPU)
    with pytest.raises(AssertionError):
        cache.set_bank_sources(
            {
                "w13": [torch.randn(4, 8, 6), torch.randn(3, 8, 6)],  # bad E
                "w2": [torch.randn(4, 6, 8), torch.randn(4, 6, 8)],
            }
        )


# ---------------------------------------------------------------------------
# Config resolution + rejection
# ---------------------------------------------------------------------------


def test_offload_config_expert_cache_requires_size():
    with pytest.raises(ValueError):
        OffloadConfig(offload_backend="expert_cache")


def test_offload_config_expert_cache_size_cap():
    with pytest.raises(ValueError):
        OffloadConfig(
            offload_backend="expert_cache",
            expert_cache=ExpertCacheOffloadConfig(moe_cache_size=10**6),
        )
    assert EXPERT_CACHE_MAX_SLOTS == 992


def test_offload_config_expert_cache_valid():
    cfg = OffloadConfig(
        offload_backend="expert_cache",
        expert_cache=ExpertCacheOffloadConfig(moe_cache_size=128),
    )
    assert cfg.expert_cache.moe_cache_size == 128
    assert cfg.expert_cache.moe_cache_pin_memory is True


def test_create_offloader_dispatch():
    # expert_cache -> ExpertCacheOffloader
    cfg = OffloadConfig(
        offload_backend="expert_cache",
        expert_cache=ExpertCacheOffloadConfig(moe_cache_size=64),
    )
    off = create_offloader(cfg)
    assert isinstance(off, ExpertCacheOffloader)
    assert off.cache_size == 64
    # auto with nothing set -> Noop
    assert isinstance(create_offloader(OffloadConfig()), NoopOffloader)


# ---------------------------------------------------------------------------
# Weight diversion (small synthetic MoE layer, CPU)
# ---------------------------------------------------------------------------


class _FakeRoutedExperts(nn.Module):
    def __init__(self, num_experts, dtype=torch.float32):
        super().__init__()
        self.layer_name = "model.layers.0.mlp.experts.routed_experts"
        self.w13_weight = nn.Parameter(
            torch.randn(num_experts, 2 * INTER, HIDDEN, dtype=dtype),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.randn(num_experts, HIDDEN, INTER, dtype=dtype),
            requires_grad=False,
        )


def test_offloader_diverts_weights_to_host_and_registers():
    off = ExpertCacheOffloader(cache_size=CACHE_SIZE, pin_memory=False)
    layer = _FakeRoutedExperts(NUM_EXPERTS)
    off._divert_routed_experts(layer)
    # Weights moved to CPU and marked.
    assert layer.w13_weight.device.type == "cpu"
    assert layer.w2_weight.device.type == "cpu"
    assert getattr(layer.w13_weight, "_vllm_is_expert_offloaded", False)
    assert layer._offload_layer_id == 0
    assert off._num_experts == NUM_EXPERTS
    assert len(off._layer_banks) == 1
    assert set(off._layer_banks[0]) == {"w13", "w2"}


def test_offloader_assigns_layer_ids_and_validates_uniform_experts():
    off = ExpertCacheOffloader(cache_size=CACHE_SIZE, pin_memory=False)
    l0 = _FakeRoutedExperts(NUM_EXPERTS)
    l1 = _FakeRoutedExperts(NUM_EXPERTS)
    off._divert_routed_experts(l0)
    off._divert_routed_experts(l1)
    assert l0._offload_layer_id == 0 and l1._offload_layer_id == 1
    # A layer with a different expert count is rejected.
    bad = _FakeRoutedExperts(NUM_EXPERTS + 1)
    with pytest.raises(ValueError, match="uniform expert count"):
        off._divert_routed_experts(bad)


def test_offloader_post_init_builds_global_cache():
    from vllm.model_executor.layers.fused_moe.offload.slot_cache import (
        get_global_slot_cache,
    )

    clear_global_slot_cache()
    off = ExpertCacheOffloader(cache_size=CACHE_SIZE, pin_memory=False)
    off._divert_routed_experts(_FakeRoutedExperts(NUM_EXPERTS))
    off._divert_routed_experts(_FakeRoutedExperts(NUM_EXPERTS))
    off.post_init()
    cache = get_global_slot_cache()
    assert cache.num_layers == 2
    assert cache.num_experts == NUM_EXPERTS
    assert cache.cache_size == CACHE_SIZE
    assert len(cache.banks) == 2
    clear_global_slot_cache()


# ---------------------------------------------------------------------------
# GPU: Triton kernels vs CPU reference (skip-guarded)
# ---------------------------------------------------------------------------

_CUDA_AND_TRITON = torch.cuda.is_available() and HAS_TRITON


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_triton_lru_ensure_matches_cpu_reference(seed):
    torch.manual_seed(seed)
    num_layers, num_experts, cache_size = 3, 16, 32
    dev = torch.device("cuda")

    gpu_cache = make_cache(num_layers, num_experts, cache_size, device=dev)
    cpu_cache = make_cache(num_layers, num_experts, cache_size, device=CPU)
    # Mirror bookkeeping state (both start cold/identical).

    from vllm.model_executor.layers.fused_moe.offload.lru import lru_ensure

    rng = torch.Generator().manual_seed(seed)
    for step in range(20):
        layer_id = int(torch.randint(0, num_layers, (1,), generator=rng).item())
        k = int(torch.randint(1, num_experts, (1,), generator=rng).item())
        raw = torch.randint(0, num_experts, (k,), generator=rng, dtype=torch.int32)
        gpu_ids = raw.to(dev).clone()
        cpu_ids = raw.clone()
        lru_ensure(gpu_cache, layer_id, gpu_ids)
        lru_ensure_cpu(cpu_cache, layer_id, cpu_ids)
        assert torch.equal(gpu_ids.cpu(), cpu_ids), f"step {step} ids diverged"
        assert torch.equal(gpu_cache.slot_for_id.cpu(), cpu_cache.slot_for_id), (
            f"step {step} slot_for_id diverged"
        )
        assert torch.equal(gpu_cache.id_of_slot.cpu(), cpu_cache.id_of_slot)
        assert torch.equal(gpu_cache.usage.cpu(), cpu_cache.usage)
        assert int(gpu_cache.step.item()) == int(cpu_cache.step.item())
        assert int(gpu_cache.num_indices.item()) == int(cpu_cache.num_indices.item())


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
def test_triton_materialize_matches_cpu():
    from vllm.model_executor.layers.fused_moe.offload.lru import materialize_layer

    num_layers, num_experts, cache_size = 2, 8, 8
    dev = torch.device("cuda")
    gpu_cache = make_cache(num_layers, num_experts, cache_size, device=dev)
    cpu_cache = make_cache(num_layers, num_experts, cache_size, device=CPU)
    materialize_layer(gpu_cache, 1)
    materialize_layer_cpu(cpu_cache, 1)
    assert torch.equal(gpu_cache.slot_for_id.cpu(), cpu_cache.slot_for_id)
    assert torch.equal(gpu_cache.id_of_slot.cpu(), cpu_cache.id_of_slot)
    assert torch.equal(gpu_cache.evict_slots.cpu(), cpu_cache.evict_slots)
    assert torch.equal(gpu_cache.src_indices.cpu(), cpu_cache.src_indices)
    assert int(gpu_cache.num_indices.item()) == int(cpu_cache.num_indices.item())


# ---------------------------------------------------------------------------
# Cache budget arithmetic (P2)
# ---------------------------------------------------------------------------


def test_expert_bytes_per_slot():
    from vllm.model_executor.layers.fused_moe.offload.cache_budget import (
        expert_bytes_per_slot,
    )

    sources = {
        "w13": [torch.randn(4, 2 * INTER, HIDDEN, dtype=torch.bfloat16)],
        "w2": [torch.randn(4, HIDDEN, INTER, dtype=torch.bfloat16)],
    }
    expected = (2 * INTER * HIDDEN + HIDDEN * INTER) * 2  # bf16
    assert expert_bytes_per_slot(sources) == expected


def test_plan_cache_budget_moe_first():
    from vllm.model_executor.layers.fused_moe.offload.cache_budget import (
        plan_cache_budget,
        required_bytes,
    )

    per_expert = 1024
    page = 512
    # Budget fits 100 slots after reserving 4 KV pages; total experts 64 caps it.
    budget = 100 * per_expert + 4 * page
    slots, pages, overlap = plan_cache_budget(
        budget_bytes=budget,
        per_expert_bytes=per_expert,
        cache_per_page=page,
        num_experts=8,
        total_experts=64,
        prefill_overlap=False,
        kv_reserve_pages=4,
        max_slots=992,
    )
    assert slots == 64  # clamped to total_experts
    assert not overlap
    assert required_bytes(slots, pages, per_expert, page) <= budget
    assert pages >= 4


def test_plan_cache_budget_floor_and_kv_remainder():
    from vllm.model_executor.layers.fused_moe.offload.cache_budget import (
        plan_cache_budget,
    )

    per_expert = 1000
    page = 100
    # Huge budget: slots clamp at max_slots, KV gets everything left.
    slots, pages, overlap = plan_cache_budget(
        budget_bytes=10**9,
        per_expert_bytes=per_expert,
        cache_per_page=page,
        num_experts=8,
        total_experts=128,
        prefill_overlap=False,
        kv_reserve_pages=2,
        max_slots=96,
    )
    assert slots == 96
    assert not overlap
    assert pages == (10**9 - 96 * per_expert) // page


def test_plan_cache_budget_rejects_tiny_budget():
    from vllm.model_executor.layers.fused_moe.offload.cache_budget import (
        plan_cache_budget,
    )

    with pytest.raises(AssertionError):
        plan_cache_budget(
            budget_bytes=10,  # far too small for the minimum plan
            per_expert_bytes=1000,
            cache_per_page=100,
            num_experts=8,
            total_experts=64,
            prefill_overlap=False,
            kv_reserve_pages=4,
            max_slots=992,
        )


def test_resolve_moe_cache_auto():
    from vllm.model_executor.layers.fused_moe.offload.cache_budget import (
        resolve_moe_cache_auto,
    )

    slots, pages, overlap = resolve_moe_cache_auto(
        baseline_free=16 << 30,
        weights_bytes=2 << 30,
        memory_ratio=0.9,
        cache_per_page=4096,
        fixed_cache_size=0,
        per_expert_bytes=1 << 20,
        num_experts=8,
        total_experts=128,
        prefill_overlap=False,
        kv_reserve_tokens=8192,
        page_size=16,
        max_slots=992,
    )
    assert slots >= 8
    assert slots <= 128
    assert pages > 1
    assert not overlap


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
def test_triton_reset_matches_cpu():
    from vllm.model_executor.layers.fused_moe.offload.lru import reset_cache

    dev = torch.device("cuda")
    gpu_cache = make_cache(2, 8, 8, device=dev)
    cpu_cache = make_cache(2, 8, 8, device=CPU)
    # Dirty the GPU cache a little, then reset both.
    from vllm.model_executor.layers.fused_moe.offload.lru import lru_ensure

    lru_ensure(gpu_cache, 0, torch.tensor([0, 1, 2], dtype=torch.int32, device=dev))
    reset_cache(gpu_cache)
    reset_cache_cpu(cpu_cache)
    assert torch.equal(gpu_cache.slot_for_id.cpu(), cpu_cache.slot_for_id)
    assert torch.equal(gpu_cache.id_of_slot.cpu(), cpu_cache.id_of_slot)
    assert int(gpu_cache.step.item()) == 0


def _make_cuda_pinned_cache(
    num_experts: int = 8, cache_size: int = 8, num_layers: int = 1
):
    """CUDA slot cache with pinned, per-expert-identifiable host banks."""
    dev = torch.device("cuda")
    cache = ExpertSlotCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=dev,
    )
    shapes = {
        "w13": (num_experts, 2 * INTER, HIDDEN),
        "w2": (num_experts, HIDDEN, INTER),
    }
    sources = {}
    for name in cache.bank_schema:
        t = torch.randn(*shapes[name]).pin_memory()
        for e in range(num_experts):
            t[e].fill_(float(e + 1))
        sources[name] = [t] * num_layers
    cache.set_bank_sources(sources)
    for name in cache.bank_schema:
        cache.bank_caches[name].fill_(-999.0)
    return cache


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
def test_fused_copy_plan_built_on_cuda():
    cache = _make_cuda_pinned_cache(num_experts=4, cache_size=4)
    assert cache._fused_plan is not None
    plan = cache._fused_plan
    assert plan.num_banks == len(cache.bank_schema) == 2
    assert plan.dst_ptrs.dtype == torch.int64
    assert plan.feat_bytes.dtype == torch.int64
    assert len(plan.src_ptrs) == cache.num_layers
    row_bytes = {
        name: cache.bank_caches[name][0].numel()
        * cache.bank_caches[name].element_size()
        for name in cache.bank_schema
    }
    assert sorted(plan.feat_bytes.tolist()) == sorted(row_bytes.values())


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
def test_triton_fused_copy_matches_cpu_reference():
    cache = _make_cuda_pinned_cache(num_experts=8, cache_size=8)
    assert cache._fused_plan is not None
    dev = cache.device
    # Stage a miss list manually (isolating the copy from the LRU ensure).
    cache.evict_slots[:4].copy_(
        torch.tensor([2, 0, 3, 5], dtype=torch.int32, device=dev)
    )
    cache.src_indices[:4].copy_(
        torch.tensor([1, 3, 2, 6], dtype=torch.int32, device=dev)
    )
    cache.num_indices.fill_(4)
    cache._pending_src_layer = 0
    cache._pending_whole_layer = False
    # Poison the tail past num_indices: it must be masked off (src 99 is out of
    # range and would be an OOB read if not masked).
    cache.evict_slots[4:].fill_(7)
    cache.src_indices[4:].fill_(99)

    cache.copy_missing()
    torch.cuda.synchronize()

    # slot -> src expert (value = src expert + 1)
    expected_map = {2: 1, 0: 3, 3: 2, 5: 6}
    for name in cache.bank_schema:
        c = cache.bank_caches[name].cpu()
        for slot in range(cache.cache_size):
            if slot in expected_map:
                assert torch.all(c[slot] == float(expected_map[slot] + 1)), (
                    name,
                    slot,
                )
            else:
                assert torch.all(c[slot] == -999.0), (name, slot)


# ---------------------------------------------------------------------------
# FP8 (block-quantized) expert banks
# ---------------------------------------------------------------------------

FP8 = torch.float8_e4m3fn
FP8_BLOCK = 8  # small block divisor so scale rows stay 16-byte aligned in tests


def _make_moe_config():
    """Minimal FusedMoEConfig for constructing Fp8MoEMethod on CPU."""
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )

    pc = FusedMoEParallelConfig(
        tp_size=1,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        tp_rank=0,
        pcp_rank=0,
        dp_rank=0,
        ep_rank=0,
        sp_size=1,
        use_ep=False,
        all2all_backend="naive",
        enable_eplb=False,
    )
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=16,
        intermediate_size=32,
        num_local_experts=8,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=pc,
        in_dtype=torch.bfloat16,
    )


def _fp8_bank_shapes(num_experts, hidden=16, inter=24, blk=FP8_BLOCK):
    return {
        "w13": (num_experts, 2 * inter, hidden),
        "w2": (num_experts, hidden, inter),
        "w13_scale": (num_experts, (2 * inter) // blk, hidden // blk),
        "w2_scale": (num_experts, hidden // blk, inter // blk),
    }


def _make_fp8_cache(num_experts=4, cache_size=4):
    shapes = _fp8_bank_shapes(num_experts)
    dtypes = {
        "w13": FP8,
        "w2": FP8,
        "w13_scale": torch.float32,
        "w2_scale": torch.float32,
    }
    cache = ExpertSlotCache(
        num_layers=1,
        num_experts=num_experts,
        cache_size=cache_size,
        device=CPU,
        quant_format="fp8_block",
    )
    sources = {n: [torch.zeros(shapes[n], dtype=dtypes[n])] for n in cache.bank_schema}
    cache.set_bank_sources(sources)
    return cache


def test_fp8_bank_schema_and_param_mapping():
    assert BANK_SCHEMAS["fp8_block"] == ("w13", "w2", "w13_scale", "w2_scale")
    from vllm.model_executor.layers.fused_moe.offload.slot_cache import BANK_TO_PARAM

    assert BANK_TO_PARAM["w13_scale"] == "w13_weight_scale_inv"
    assert BANK_TO_PARAM["w2_scale"] == "w2_weight_scale_inv"


def test_fp8_bank_geometry_bytes():
    cache = _make_fp8_cache(num_experts=4, cache_size=4)
    hidden, inter = 16, 24
    blk = FP8_BLOCK
    # 1 byte/elem for fp8 weights, 4 bytes/elem for fp32 scales.
    w13_row = 2 * inter * hidden * 1
    w2_row = hidden * inter * 1
    s13_row = ((2 * inter) // blk) * (hidden // blk) * 4
    s2_row = (hidden // blk) * (inter // blk) * 4
    assert cache.bytes_per_slot() == w13_row + w2_row + s13_row + s2_row
    assert cache.slot_cache_bytes() == 4 * cache.bytes_per_slot()
    # The fp8 slot weight caches hold 1-byte elements.
    assert cache.bank_caches["w13"].dtype == FP8
    assert cache.bank_caches["w2"].dtype == FP8
    assert cache.bank_caches["w13_scale"].dtype == torch.float32


def test_nvfp4_bank_schema_and_param_mapping():
    assert BANK_SCHEMAS["nvfp4"] == (
        "w13",
        "w13_qscale",
        "w13_gscale2",
        "a13_gscale",
        "w2",
        "w2_qscale",
        "w2_gscale2",
        "a2_gscale",
    )
    from vllm.model_executor.layers.fused_moe.offload.slot_cache import BANK_TO_PARAM

    assert BANK_TO_PARAM["w13_qscale"] == "w13_weight_scale"
    assert BANK_TO_PARAM["w13_gscale2"] == "w13_weight_scale_2"
    assert BANK_TO_PARAM["a13_gscale"] == "w13_input_scale"
    assert BANK_TO_PARAM["w2_qscale"] == "w2_weight_scale"
    assert BANK_TO_PARAM["w2_gscale2"] == "w2_weight_scale_2"
    assert BANK_TO_PARAM["a2_gscale"] == "w2_input_scale"
    # No collision with the fp8_block bank names sharing the same dict.
    assert BANK_TO_PARAM["w13_scale"] == "w13_weight_scale_inv"


def _make_nvfp4_cache(num_experts: int, cache_size: int) -> ExpertSlotCache:
    hidden, inter, group = 16, 24, 16
    NVFP4_U8 = torch.uint8
    NVFP4_SCALE = torch.float8_e4m3fn
    shapes = {
        "w13": (num_experts, 2 * inter, hidden // 2),
        "w2": (num_experts, hidden, inter // 2),
        "w13_qscale": (num_experts, 2 * inter, hidden // group),
        "w2_qscale": (num_experts, hidden, inter // group),
        "w13_gscale2": (num_experts, 1),
        "w2_gscale2": (num_experts, 1),
        "a13_gscale": (num_experts, 1),
        "a2_gscale": (num_experts, 1),
    }
    dtypes = {
        "w13": NVFP4_U8,
        "w2": NVFP4_U8,
        "w13_qscale": NVFP4_SCALE,
        "w2_qscale": NVFP4_SCALE,
        "w13_gscale2": torch.float32,
        "w2_gscale2": torch.float32,
        "a13_gscale": torch.float32,
        "a2_gscale": torch.float32,
    }
    cache = ExpertSlotCache(
        num_layers=1,
        num_experts=num_experts,
        cache_size=cache_size,
        device=CPU,
        quant_format="nvfp4",
    )
    sources = {n: [torch.zeros(shapes[n], dtype=dtypes[n])] for n in cache.bank_schema}
    cache.set_bank_sources(sources)
    return cache


def test_nvfp4_bank_geometry_bytes():
    cache = _make_nvfp4_cache(num_experts=4, cache_size=4)
    hidden, inter, group = 16, 24, 16
    # 1 byte/elem for packed fp4 weights (2 items/byte -> hidden/2 columns).
    w13_row = 2 * inter * (hidden // 2) * 1
    w2_row = hidden * (inter // 2) * 1
    # 1 byte/elem for the fp8_e4m3 per-block scales.
    q13_row = 2 * inter * (hidden // group) * 1
    q2_row = hidden * (inter // group) * 1
    # 4 bytes/elem for the four per-expert fp32 scalar rows (1 element each):
    # w13_gscale2, a13_gscale, w2_gscale2, a2_gscale.
    scalar_row = 1 * 4
    assert cache.bytes_per_slot() == (
        w13_row + q13_row + w2_row + q2_row + 4 * scalar_row
    )
    assert cache.slot_cache_bytes() == 4 * cache.bytes_per_slot()
    assert cache.bank_caches["w13"].dtype == torch.uint8
    assert cache.bank_caches["w13_qscale"].dtype == torch.float8_e4m3fn
    assert cache.bank_caches["a13_gscale"].shape == (4, 1)


def test_every_bank_row_keeps_the_fused_copy_plan_eligible():
    """No bank may have a row size the device-driven copy cannot handle.

    build_fused_copy_plan rejects the WHOLE plan if any single bank's row is
    not a multiple of _BANK_ROW_ALIGN_BYTES, dropping every bank onto the
    legacy host-count copy -- which host-synchronizes per bank per MoE layer
    per step. On real hardware that alone made decode slow enough for vLLM's
    `sample_tokens` RPC to time out, so a new bank whose row breaks the
    alignment is a serious regression, not a cosmetic one.
    """
    from vllm.model_executor.layers.fused_moe.offload.fused_copy import (
        _BANK_ROW_ALIGN_BYTES,
    )

    # The narrowest row any schema banks today: one fp32 per expert.
    assert 4 % _BANK_ROW_ALIGN_BYTES == 0, (
        "per-expert fp32 scalar rows (NVFP4's g/a scales) must stay eligible"
    )

    # "nvfp4_emulation" banks a subset of "nvfp4"'s names, so the nvfp4 cache
    # already covers every row size it can produce.
    caches = {
        "bf16": make_cache(num_layers=1),
        "fp8_block": _make_fp8_cache(),
        "nvfp4": _make_nvfp4_cache(num_experts=4, cache_size=4),
    }
    for fmt, cache in caches.items():
        for name, row_bytes in cache.bank_row_bytes().items():
            assert row_bytes % _BANK_ROW_ALIGN_BYTES == 0, (
                f"{fmt!r} bank {name!r} row is {row_bytes} bytes, which "
                f"disqualifies the fused copy plan for every bank"
            )


class _MockQuantConfig:
    def __init__(self, method):
        self._method = method

    def get_quant_method(self, layer, prefix):
        return self._method


def _make_fp8_method(block: bool):
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod

    cfg = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=[128, 128] if block else None,
    )

    class _Layer:
        pass

    layer = _Layer()
    layer.moe_config = _make_moe_config()
    return Fp8MoEMethod(cfg, layer)


def test_routing_unquantized_format_marker():
    """The bf16 branch is the phase-1 default; constructing the CustomOp-derived
    method needs a live vllm config + platform, so assert the format marker here
    and cover the branch shape (quant_config is None -> bf16 method)."""
    from vllm.model_executor.layers.fused_moe.offload.routed_experts import (
        OffloadUnquantizedFusedMoEMethod,
    )

    assert OffloadUnquantizedFusedMoEMethod.offload_quant_format == "bf16"


def test_routing_accepts_block_fp8():
    from vllm.model_executor.layers.fused_moe.offload.routed_experts import (
        OffloadRoutedExperts,
    )

    method = _make_fp8_method(block=True)
    qc = _MockQuantConfig(method)
    m = OffloadRoutedExperts._get_quant_method(
        object.__new__(OffloadRoutedExperts), "x", qc, _make_moe_config()
    )
    assert getattr(m, "offload_quant_format", None) == "fp8_block"
    assert m.moe_block_shape == [128, 128]


def test_routing_rejects_pertensor_fp8_loudly():
    from vllm.model_executor.layers.fused_moe.offload.routed_experts import (
        OffloadRoutedExperts,
    )

    method = _make_fp8_method(block=False)
    qc = _MockQuantConfig(method)
    with pytest.raises(NotImplementedError, match="block-quantized FP8"):
        OffloadRoutedExperts._get_quant_method(
            object.__new__(OffloadRoutedExperts), "x", qc, _make_moe_config()
        )


def test_routing_rejects_other_quant_loudly():
    from vllm.model_executor.layers.fused_moe.offload.routed_experts import (
        OffloadRoutedExperts,
    )

    class _OtherMethod:
        pass

    qc = _MockQuantConfig(_OtherMethod())
    with pytest.raises(NotImplementedError, match="MXFP4, AWQ"):
        OffloadRoutedExperts._get_quant_method(
            object.__new__(OffloadRoutedExperts), "x", qc, _make_moe_config()
        )


class _FakeFp8RoutedExperts(nn.Module):
    def __init__(self, num_experts):
        super().__init__()
        self.layer_name = "model.layers.0.mlp.experts.routed_experts"
        shapes = _fp8_bank_shapes(num_experts)
        self.w13_weight = nn.Parameter(
            torch.zeros(shapes["w13"], dtype=FP8), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.zeros(shapes["w2"], dtype=FP8), requires_grad=False
        )
        self.w13_weight_scale_inv = nn.Parameter(
            torch.zeros(shapes["w13_scale"], dtype=torch.float32), requires_grad=False
        )
        self.w2_weight_scale_inv = nn.Parameter(
            torch.zeros(shapes["w2_scale"], dtype=torch.float32), requires_grad=False
        )
        self.quant_method = types.SimpleNamespace(offload_quant_format="fp8_block")


def test_offloader_diverts_fp8_weights_and_scales():
    off = ExpertCacheOffloader(cache_size=CACHE_SIZE, pin_memory=False)
    layer = _FakeFp8RoutedExperts(NUM_EXPERTS)
    off._divert_routed_experts(layer)
    assert off.quant_format == "fp8_block"
    assert set(off._layer_banks[0]) == {"w13", "w2", "w13_scale", "w2_scale"}
    for name in ("w13_weight", "w2_weight"):
        p = getattr(layer, name)
        assert p.device.type == "cpu"
        assert getattr(p, "_vllm_is_expert_offloaded", False)
    for name in ("w13_weight_scale_inv", "w2_weight_scale_inv"):
        p = getattr(layer, name)
        assert p.device.type == "cpu"
        assert getattr(p, "_vllm_is_expert_offloaded", False)


def test_offloader_rejects_mixed_formats():
    off = ExpertCacheOffloader(cache_size=CACHE_SIZE, pin_memory=False)
    off._divert_routed_experts(_FakeFp8RoutedExperts(NUM_EXPERTS))
    # A bf16 layer after an fp8 layer must be rejected.
    bf16_layer = _FakeRoutedExperts(NUM_EXPERTS)
    with pytest.raises(ValueError, match="uniform expert format"):
        off._divert_routed_experts(bf16_layer)


def test_fp8_fused_copy_plan_none_on_cpu_but_schema_ok():
    cache = _make_fp8_cache(num_experts=4, cache_size=4)
    # No CUDA device -> no fused plan; the schema itself is still registered.
    assert cache._fused_plan is None
    assert cache.bank_schema == BANK_SCHEMAS["fp8_block"]


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
def test_fp8_fused_copy_plan_built_on_cuda():
    # hidden=32, inter=32 -> every bank row (incl. the fp32 scale rows) is a
    # multiple of 16 bytes, so the fused plan builds.
    num_experts = 4
    cache = ExpertSlotCache(
        num_layers=1,
        num_experts=num_experts,
        cache_size=num_experts,
        device=torch.device("cuda"),
        quant_format="fp8_block",
    )
    shapes = _fp8_bank_shapes(num_experts, hidden=32, inter=32)
    dtypes = {
        "w13": FP8,
        "w2": FP8,
        "w13_scale": torch.float32,
        "w2_scale": torch.float32,
    }
    sources = {
        n: [torch.zeros(shapes[n], dtype=dtypes[n]).pin_memory()]
        for n in cache.bank_schema
    }
    cache.set_bank_sources(sources)
    plan = cache._fused_plan
    assert plan is not None
    assert plan.num_banks == 4
    assert sorted(plan.feat_bytes.tolist()) == sorted(cache.bank_row_bytes().values())


# ---------------------------------------------------------------------------
# Double-buffered prefill streaming (state machine, CPU reference)
# ---------------------------------------------------------------------------
#
# On a CPU cache ``prefill_copy_stream`` is None, so the prefetch copy runs
# synchronously on the current stream -- exercising the exact buffer alternation,
# invalidation and release/re-borrow bookkeeping without a device.


def _make_overlap_cache(num_layers: int = 4, num_experts: int = 4):
    cache = make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=2 * num_experts,  # exactly enough for the two buffers
        device=CPU,
    )
    assert cache.prefill_overlap
    assert len(cache.prefill_bank_buffers) == len(cache.bank_schema)
    return cache


def test_prefill_overlap_buffers_borrow_first_2e_slots():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    for name, buffer in zip(cache.bank_schema, cache.prefill_bank_buffers):
        slot_cache = cache.bank_caches[name]
        # The buffer is a [2, E, ...] view over the first 2E slots.
        assert buffer.shape == (2, 4, *slot_cache.shape[1:])
        assert buffer.data_ptr() == slot_cache.data_ptr()


def test_prefill_overlap_alternation_and_contents():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    assert cache._prefill_buffer_layer == [0, 1]

    v0 = cache.wait_prefill_layer(0)
    v1 = cache.wait_prefill_layer(1)
    for name_idx, name in enumerate(cache.bank_schema):
        # position == expert id: the buffer rows ARE the layer's expert rows.
        assert torch.equal(v0[name_idx], cache.bank_sources[name][0])
        assert torch.equal(v1[name_idx], cache.bank_sources[name][1])


def test_prefill_overlap_reuse_before_release_is_rejected():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)  # buffer 0 <- layer 0
    cache.prefetch_prefill_layer(1)  # buffer 1 <- layer 1
    # Layer 2 maps to buffer 0, which still holds unreleased layer 0.
    with pytest.raises(AssertionError, match="before release"):
        cache.prefetch_prefill_layer(2)
    # After release, the re-borrow succeeds and carries layer 2's rows.
    cache.release_prefill_layer(0)
    assert cache._prefill_buffer_released[0] is True
    cache.prefetch_prefill_layer(2)
    assert cache._prefill_buffer_layer == [2, 1]
    v2 = cache.wait_prefill_layer(2)
    for name_idx, name in enumerate(cache.bank_schema):
        assert torch.equal(v2[name_idx], cache.bank_sources[name][2])


def test_prefill_overlap_invalidates_borrowed_slots():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    # Make layer-0 experts resident in the to-be-borrowed slots [0..4).
    lru_ensure_cpu(cache, 0, torch.tensor([0, 1, 2, 3], dtype=torch.int32))
    assert torch.all(cache.slot_for_id[0] >= 0)
    assert torch.all(cache.usage[:4] == 1)

    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)  # borrows slots [0..4)
    # Residency dropped and usage zeroed so decode evicts these slots first.
    assert torch.all(cache.id_of_slot[:4] == -1)
    assert torch.all(cache.slot_for_id[0] == -1)
    assert torch.all(cache.usage[:4] == 0)


def test_prefill_overlap_begin_resets_tracking():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    assert cache._prefill_buffer_layer[0] == 0
    assert cache._prefill_buffer_released[0] is False
    # A new prefill batch clears the tenant/release tracking.
    cache.begin_prefill()
    assert cache._prefill_buffer_layer == [None, None]
    assert cache._prefill_buffer_released == [True, True]


def test_prefill_overlap_prefetch_is_idempotent_and_bounds_checked():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(0)  # already resident -> no-op
    assert cache._prefill_buffer_layer[0] == 0
    cache.prefetch_prefill_layer(cache.num_layers)  # >= num_layers -> no-op
    with pytest.raises(ValueError, match="Invalid prefill layer id"):
        cache.prefetch_prefill_layer(-1)


def test_prefill_overlap_degrades_when_cache_too_small():
    # cache_size == num_experts < 2*num_experts -> overlap must disable itself.
    cache = make_cache(num_layers=2, num_experts=4, cache_size=4, device=CPU)
    assert cache.prefill_overlap is False
    assert cache.prefill_bank_buffers == []
    assert cache.prefill_copy_stream is None


def test_prefill_overlap_eligibility_by_format():
    """Overlap prefill is only valid for scale-free formats: the GEMM reads
    per-row weight scales by expert id from the fixed slot-scale cache, which
    cannot follow the buffer_id*E weight offset of the alternating buffers."""
    bf16_cache = _make_overlap_cache(num_layers=2, num_experts=4)
    assert not any("scale" in n for n in bf16_cache.bank_schema)

    fp8_cache = _make_fp8_cache(num_experts=4, cache_size=8)
    assert any("scale" in n for n in fp8_cache.bank_schema)


def test_prefill_overlap_disabled_by_config():
    cache = ExpertSlotCache(
        num_layers=2,
        num_experts=4,
        cache_size=8,
        device=CPU,
        prefill_overlap=False,
    )
    cache.set_bank_sources(
        {
            "w13": [torch.randn(4, 2 * INTER, HIDDEN) for _ in range(2)],
            "w2": [torch.randn(4, HIDDEN, INTER) for _ in range(2)],
        }
    )
    assert cache.prefill_overlap is False
    assert cache.prefill_bank_buffers == []
    # begin/prefetch/wait/release are all no-ops (or guarded) when disabled.
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    cache.release_prefill_layer(0)


def test_prefill_overlap_config_and_offloader_wiring():
    from vllm.config import ExpertCacheOffloadConfig, OffloadConfig

    cfg = ExpertCacheOffloadConfig(moe_cache_size=64)
    assert cfg.moe_prefill_overlap is True  # default on

    off = create_offloader(
        OffloadConfig(
            offload_backend="expert_cache",
            expert_cache=ExpertCacheOffloadConfig(
                moe_cache_size=64, moe_prefill_overlap=False
            ),
        )
    )
    assert off.prefill_overlap is False
    off2 = create_offloader(
        OffloadConfig(
            offload_backend="expert_cache",
            expert_cache=ExpertCacheOffloadConfig(moe_cache_size=64),
        )
    )
    assert off2.prefill_overlap is True


def test_reset_clears_prefill_tracking():
    cache = _make_overlap_cache(num_layers=4, num_experts=4)
    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    assert cache._prefill_buffer_layer[0] == 0
    cache.reset()
    assert cache._prefill_buffer_layer == [None, None]
    assert cache._prefill_buffer_released == [True, True]
    assert cache._prefill_buffer_has_release_event == [False, False]


def test_ple_table_is_fp8_under_non_fp8_quant():
    """The PLE n-gram table is FP8 whenever the config declares it so, even when
    the global quant config is not Fp8Config (e.g. ModelOpt NVFP4 checkpoints keep
    ``*.ple.*`` out of the NVFP4 quant but store it as float8_e4m3fn)."""
    from vllm.models.qwen4_exp.nvidia.ple_layer import _ple_table_is_fp8

    # Declared FP8 by dtype string or torch dtype -> True regardless of quant cfg.
    assert _ple_table_is_fp8(None, "x.ple", "float8_e4m3fn")
    assert _ple_table_is_fp8(None, "x.ple", torch.float8_e4m3fn)
    # Non-FP8 dtype -> False.
    assert not _ple_table_is_fp8(None, "x.ple", "bfloat16")
    # No dtype hint and no Fp8Config -> False.
    assert not _ple_table_is_fp8(None, "x.ple", None)


@pytest.mark.skipif(not _CUDA_AND_TRITON, reason="requires CUDA + Triton")
def test_prefill_overlap_cuda_stream_and_events():
    dev = torch.device("cuda")
    num_experts = 4
    cache = ExpertSlotCache(
        num_layers=4,
        num_experts=num_experts,
        cache_size=2 * num_experts,
        device=dev,
    )
    sources = {
        "w13": [
            torch.randn(num_experts, 2 * INTER, HIDDEN).pin_memory() for _ in range(4)
        ],
        "w2": [torch.randn(num_experts, HIDDEN, INTER).pin_memory() for _ in range(4)],
    }
    cache.set_bank_sources(sources)
    assert cache.prefill_copy_stream is not None
    assert cache.prefill_begin_event is not None
    assert len(cache.prefill_ready_events) == 2
    assert len(cache.prefill_release_events) == 2

    cache.begin_prefill()
    cache.prefetch_prefill_layer(0)
    cache.prefetch_prefill_layer(1)
    v0 = cache.wait_prefill_layer(0)
    v1 = cache.wait_prefill_layer(1)
    torch.cuda.synchronize()
    for name_idx, name in enumerate(cache.bank_schema):
        assert torch.equal(v0[name_idx].cpu(), cache.bank_sources[name][0])
        assert torch.equal(v1[name_idx].cpu(), cache.bank_sources[name][1])
    cache.release_prefill_layer(0)
    cache.release_prefill_layer(1)
    assert cache._prefill_buffer_has_release_event == [True, True]
