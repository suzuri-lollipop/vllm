# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch", "expert_cache"]

# vLLM's moe_align_block_size requires round_up(num_experts, 32) < 1024, so
# the slot cache (whose size plays the role of the local expert count in
# the slot-space grouped GEMM) caps at 992 slots.
EXPERT_CACHE_MAX_SLOTS = 992


@config
class UVAOffloadConfig:
    """Configuration for UVA (Unified Virtual Addressing) CPU offloading.

    Uses zero-copy access from CPU-pinned memory. Simple but requires
    fast CPU-GPU interconnect.
    """

    cpu_offload_gb: float = Field(default=0, ge=0)
    """The space in GiB to offload to CPU, per GPU. Default is 0, which means
    no offloading. Intuitively, this argument can be seen as a virtual way to
    increase the GPU memory size. For example, if you have one 24 GB GPU and
    set this to 10, virtually you can think of it as a 34 GB GPU. Then you can
    load a 13B model with BF16 weight, which requires at least 26GB GPU memory.
    Note that this requires fast CPU-GPU interconnect, as part of the model is
    loaded from CPU memory to GPU memory on the fly in each model forward pass.
    This uses UVA (Unified Virtual Addressing) for zero-copy access.
    """

    cpu_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for CPU offloading.
    Unmatched parameters are not offloaded. If this set is empty, parameters
    are offloaded non-selectively until the memory limit defined by
    `cpu_offload_gb` is reached.
    Examples:
        - For parameter name "mlp.experts.w2_weight":
            - "experts" or "experts.w2_weight" will match.
            - "expert" or "w2" will NOT match (must be exact segments).
    This allows distinguishing parameters like "w2_weight" and "w2_weight_scale".
    """


@config
class PrefetchOffloadConfig:
    """Configuration for prefetch-based CPU offloading.

    Groups layers and uses async H2D prefetch to hide transfer latency.
    """

    offload_group_size: int = Field(default=0, ge=0)
    """Group every N layers together. Offload last `offload_num_in_group`
    layers of each group. Default is 0 (disabled).
    Example: group_size=8, num_in_group=2 offloads layers 6,7,14,15,22,23,...
    Unlike cpu_offload_gb, this uses explicit async prefetching to hide transfer
    latency.
    """

    offload_num_in_group: int = Field(default=1, ge=1)
    """Number of layers to offload per group.
    Must be <= offload_group_size. Default is 1."""

    offload_prefetch_step: int = Field(default=1, ge=0)
    """Number of layers to prefetch ahead.
    Higher values hide more latency but use more GPU memory. Default is 1."""

    offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to target for prefetch offloading.
    Unmatched parameters are not offloaded. If this set is empty, ALL
    parameters of each offloaded layer are offloaded.
    Uses segment matching: "w13_weight" matches "mlp.experts.w13_weight"
    but not "mlp.experts.w13_weight_scale".
    """


@config
class ExpertCacheOffloadConfig:
    """Configuration for expert-granularity MoE offloading ("expert_cache").

    Keeps the routed expert weights (w13/w2 of every MoE layer) in pinned
    host memory and streams only the routed experts that are actually hit
    into a global GPU slot cache, so a MoE model larger than VRAM can run.
    Decode (and prefill) routing is remapped onto cache slots with a global
    LRU eviction policy; misses are copied host->device on a dedicated copy
    stream. Phase 1 supports unquantized (BF16) experts on NVIDIA only.
    """

    moe_cache_size: int = Field(default=0, ge=0)
    """Number of expert slots in the global GPU slot cache (one slot holds
    one expert's w13 rows and one expert's w2 rows, TP-sharded). Shared
    across ALL MoE layers with a global LRU policy. Must be at least the
    number of experts per layer and at most 992. Required (non-zero) when
    offload_backend is "expert_cache". Rule of thumb: size it so that
    slots * per-expert-bytes fills the VRAM left over after weights/KV."""

    moe_cache_pin_memory: bool = True
    """Pin the host expert banks (page-locked CPU memory). Pinning is
    required for the async host->device miss copies; disable only on hosts
    where CUDA pinning is unavailable (miss copies then go through pageable
    staging and the feature is not supported on the GPU path)."""


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    - "expert_cache": Expert-granularity MoE offloading with a global LRU
      GPU slot cache (see ExpertCacheOffloadConfig). MoE models larger than
      VRAM: unquantized experts on NVIDIA only.
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    expert_cache: ExpertCacheOffloadConfig = Field(
        default_factory=ExpertCacheOffloadConfig
    )
    """Parameters for the expert_cache offloading backend."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        if self.offload_backend == "expert_cache":
            if self.expert_cache.moe_cache_size < 1:
                raise ValueError(
                    "offload_backend='expert_cache' requires a positive "
                    "moe_cache_size (number of expert slots in the global "
                    "GPU slot cache); set --moe-cache-size"
                )
            if self.expert_cache.moe_cache_size > EXPERT_CACHE_MAX_SLOTS:
                raise ValueError(
                    f"moe_cache_size ({self.expert_cache.moe_cache_size}) "
                    f"exceeds the maximum of {EXPERT_CACHE_MAX_SLOTS} slots "
                    "(moe_align_block_size caps padded experts at 1024)"
                )
            uva_active = self.uva.cpu_offload_gb > 0
            prefetch_active = self.prefetch.offload_group_size > 0
            if uva_active or prefetch_active:
                warnings.warn(
                    "UVA/prefetch offload fields are set but "
                    "offload_backend='expert_cache'. The expert cache "
                    "offloads MoE expert weights itself; the other "
                    "offload settings apply to the remaining parameters "
                    "only if you also keep them enabled explicitly.",
                    stacklevel=2,
                )
        if self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0:
            if self.prefetch.offload_num_in_group > self.prefetch.offload_group_size:
                raise ValueError(
                    f"offload_num_in_group ({self.prefetch.offload_num_in_group})"
                    f" must be <= offload_group_size"
                    f" ({self.prefetch.offload_group_size})"
                )
            if self.prefetch.offload_prefetch_step < 1:
                raise ValueError(
                    f"offload_prefetch_step"
                    f" ({self.prefetch.offload_prefetch_step})"
                    f" must be >= 1 when prefetch offloading is enabled"
                    f" (offload_group_size > 0)"
                )

        # Warn if both backends have non-default values
        uva_active = self.uva.cpu_offload_gb > 0
        prefetch_active = self.prefetch.offload_group_size > 0
        if self.offload_backend == "uva" and prefetch_active:
            warnings.warn(
                "Prefetch offload fields are set but offload_backend='uva'. "
                "Prefetch settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "prefetch" and uva_active:
            warnings.warn(
                "UVA offload fields are set but offload_backend='prefetch'. "
                "UVA settings will be ignored.",
                stacklevel=2,
            )
        elif self.offload_backend == "auto" and uva_active and prefetch_active:
            warnings.warn(
                "Both UVA and prefetch offload fields are set with "
                "offload_backend='auto'. Prefetch backend will be selected. "
                "Set offload_backend explicitly to suppress this warning.",
                stacklevel=2,
            )
        return self

    def compute_hash(self) -> str:
        """
        Provide a hash that uniquely identifies all the offload configs.

        All fields are included because PrefetchOffloader patches module
        forwards and inserts custom ops (wait_prefetch, start_prefetch)
        into the computation graph. Changing any offload setting can
        alter which layers are hooked and how prefetch indices are
        computed, so the compilation cache must distinguish them.
        """
        from vllm.config.utils import get_hash_factors, hash_factors

        factors = get_hash_factors(self, ignored_factors=set())
        hash_str = hash_factors(factors)
        return hash_str
