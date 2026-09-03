# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Expert-granularity MoE offloading (host expert banks + global LRU slot cache).

This package ports FreeToken's low-VRAM MoE core into vLLM: routed expert weights
live in pinned host memory and only the experts actually routed are streamed into
a global GPU slot cache (LRU), so a MoE model larger than VRAM can decode.

Modules:
    slot_cache      -- the global slot cache + host banks (``ExpertSlotCache``).
    lru             -- the LRU ensure/materialize/reset logic (CPU reference and
                       Triton kernels).
    routed_experts  -- the offload-mode ``RoutedExperts`` subclass + quant method
                       that run the ensure/copy/GEMM forward.

Phase-1 scope: NVIDIA, unquantized (bf16) experts, decode on the LRU slot path and
prefill on the synchronous materialize path. Quantized banks, CPU/hybrid
co-execution, double-buffered prefill streaming, and FULL CUDA-graph capture are
follow-ups (see the report).
"""

from vllm.model_executor.layers.fused_moe.offload.slot_cache import (
    BANK_SCHEMAS,
    ExpertSlotCache,
    clear_global_slot_cache,
    get_global_slot_cache,
    set_global_slot_cache,
)

__all__ = [
    "BANK_SCHEMAS",
    "ExpertSlotCache",
    "clear_global_slot_cache",
    "get_global_slot_cache",
    "set_global_slot_cache",
]
