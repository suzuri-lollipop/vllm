# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for model weight offloading."""

import warnings
from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config

OffloadBackend = Literal["auto", "uva", "prefetch"]


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
class DiskOffloadConfig:
    """Configuration for disk-backed offloading of large embedding tables.

    Memory-maps the selected parameters from a file instead of holding them in
    device or host memory, and gathers the needed rows on the host. Intended
    for sparsely read tables that do not fit anywhere else, such as the n-gram
    hash embeddings of per-layer embedding (PLE) models.

    This is applied in addition to `offload_backend`, not instead of it.
    """

    disk_offload_path: str = ""
    """Directory holding the memory-mapped parameter files. Point it at an
    NVMe SSD mount, or at an Intel Optane persistent memory device exposed
    through a DAX filesystem, to keep the tables off host and device memory.
    Empty (the default) disables disk offloading."""

    disk_offload_params: set[str] = Field(default_factory=set)
    """The set of parameter name segments to memory-map. Required when
    `disk_offload_path` is set; disk offloading is never applied
    non-selectively. Uses the same full-segment matching as
    `cpu_offload_params`, e.g. "ngram_embedding" matches
    "ple.ngram.ngram_embedding.weight"."""

    disk_offload_layers: str = ""
    """Decoder layer indices whose matching parameters are memory-mapped, as a
    comma-separated list of indices and inclusive ranges, e.g. "0,2,24-47".
    Indices are global and unaffected by pipeline parallelism. Empty (the
    default) memory-maps every matching parameter."""

    disk_offload_keep_files: bool = False
    """Keep the backing files on exit instead of deleting them. Useful on a
    dedicated Optane or SSD partition where the space is reserved anyway."""


@config
class OffloadConfig:
    """Configuration for model weight offloading to reduce GPU memory usage."""

    offload_backend: OffloadBackend = "auto"
    """The backend for weight offloading. Options:
    - "auto": Selects based on which sub-config has non-default values
      (prefetch if offload_group_size > 0, uva if cpu_offload_gb > 0).
    - "uva": UVA (Unified Virtual Addressing) zero-copy offloading.
    - "prefetch": Async prefetch with group-based layer offloading.
    """

    uva: UVAOffloadConfig = Field(default_factory=UVAOffloadConfig)
    """Parameters for UVA offloading backend."""

    prefetch: PrefetchOffloadConfig = Field(default_factory=PrefetchOffloadConfig)
    """Parameters for prefetch offloading backend."""

    disk: DiskOffloadConfig = Field(default_factory=DiskOffloadConfig)
    """Parameters for disk-backed embedding offloading. Applied in addition to
    the backend selected by `offload_backend`."""

    offload_layers: str = ""
    """Decoder layer indices to offload, as a comma-separated list of indices
    and inclusive ranges, e.g. "0,2,24-47". Indices are global and unaffected
    by pipeline parallelism. This overrides the layer selection of the active
    backend: the UVA backend offloads only these layers (up to
    `cpu_offload_gb`, or without a byte limit when it is 0) and the prefetch
    backend ignores its group pattern. Combine it with `cpu_offload_params` /
    `offload_params` to pin, for example, the experts of the second half of
    the model to host memory while the rest stays resident in device memory.
    Empty (the default) keeps each backend's own selection."""

    @model_validator(mode="after")
    def validate_offload_config(self) -> "OffloadConfig":
        """Validate offload configuration constraints."""
        from vllm.model_executor.offloader.layer_selection import parse_layer_spec

        # Surface a malformed layer selection at config time.
        parse_layer_spec(self.offload_layers)
        parse_layer_spec(self.disk.disk_offload_layers)

        if self.disk.disk_offload_path and not self.disk.disk_offload_params:
            raise ValueError(
                "disk_offload_path requires disk_offload_params: disk "
                "offloading is only applied to explicitly named embedding "
                "tables."
            )
        if self.disk.disk_offload_params and not self.disk.disk_offload_path:
            warnings.warn(
                "disk_offload_params is set but disk_offload_path is empty. "
                "Disk offloading will not be applied.",
                stacklevel=2,
            )

        prefetch_selected = (
            self.offload_backend == "prefetch" or self.prefetch.offload_group_size > 0
        )
        # An explicit layer selection replaces the group pattern, so the group
        # fields are unused and need not be consistent.
        if prefetch_selected and not self.offload_layers:
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
