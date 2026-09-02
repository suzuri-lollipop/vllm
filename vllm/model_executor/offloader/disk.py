# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed offloading of large embedding tables.

Targets parameters that are far too large for device memory and are read
sparsely, such as the n-gram hash embedding tables of per-layer embedding
(PLE) models. The table is memory-mapped from an SSD or an Intel Optane DAX
mount and looked up on the host; only the gathered rows are copied to the
device.
"""

from collections.abc import Generator

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.layer_selection import matches_param
from vllm.model_executor.offloader.mmap_store import MmapTensorStore
from vllm.utils.mem_utils import format_gib
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# FP8 and other sub-byte-exponent types that `index_select` does not implement
# on CPU. They are gathered through a `uint8` bit-cast instead.
_BITCAST_DTYPES = tuple(
    dtype
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if (dtype := getattr(torch, name, None)) is not None
)


def host_gather_rows(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather embedding rows from a host-resident table.

    Args:
        table: 2D CPU tensor of embedding rows.
        indices: Row indices, on any device and of any integer dtype.

    Returns:
        The gathered rows, on the device of `indices`.
    """
    flat_indices = indices.reshape(-1).to(device="cpu", dtype=torch.long)
    if table.dtype in _BITCAST_DTYPES:
        rows = table.view(torch.uint8).index_select(0, flat_indices).view(table.dtype)
    else:
        rows = table.index_select(0, flat_indices)
    rows = rows.to(indices.device, non_blocking=True)
    return rows.view(*indices.shape, table.shape[-1])


def host_gather_rows_fake(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (*indices.shape, table.shape[-1]),
        dtype=table.dtype,
        device=indices.device,
    )


# The gather crosses the host/device boundary, which Dynamo must not trace into.
direct_register_custom_op(
    op_name="host_gather_embedding_rows",
    op_func=host_gather_rows,
    fake_impl=host_gather_rows_fake,
)


class HostGatherEmbeddingMethod(QuantizeMethodBase):
    """Quant method wrapper that looks embeddings up in host memory.

    Delegates everything except `embedding` to the wrapped method, so the
    weight loading and quantization behaviour of the original layer is kept.
    """

    def __init__(self, inner: QuantizeMethodBase):
        super().__init__()
        self.inner = inner
        self.uses_meta_device = inner.uses_meta_device

    def create_weights(self, *args, **kwargs):
        return self.inner.create_weights(*args, **kwargs)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        self.inner.process_weights_after_loading(layer)

    def apply(self, *args, **kwargs):
        return self.inner.apply(*args, **kwargs)

    def tie_weights(self, layer: nn.Module, embed_tokens: nn.Module):
        return self.inner.tie_weights(layer, embed_tokens)

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        weight = layer.weight
        if weight.device.type != "cpu":
            return self.inner.embedding(layer, input_)
        return torch.ops.vllm.host_gather_embedding_rows(weight, input_)


class DiskOffloader(BaseOffloader):
    """Offloader that memory-maps selected embedding tables from disk.

    Only embedding parameters are supported: a disk-backed weight is never
    materialized on the device, so the layer using it must consume it through
    a sparse row lookup rather than a dense matmul.

    Args:
        path: Directory on the target device (SSD or Optane DAX mount).
        offload_params: Parameter name segments to memory-map.
        offload_layers: Global decoder layer indices to restrict the offload
            to. None offloads every matching parameter.
        keep_files: Keep the backing files when the store is cleaned up.
    """

    supports_tower_offload = True

    def __init__(
        self,
        path: str,
        offload_params: set[str],
        offload_layers: frozenset[int] | None = None,
        keep_files: bool = False,
    ):
        if not offload_params:
            raise ValueError(
                "Disk offloading requires an explicit set of parameters to "
                "offload; mapping every parameter from disk is not supported."
            )
        self.offload_params = offload_params
        self.offload_layers = offload_layers
        self.store = MmapTensorStore(path, keep_files=keep_files)
        self.offloaded_params: list[str] = []

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
        prefix: str = "",
        start_index: int = 0,
    ) -> list[nn.Module]:
        """Memory-map the matching parameters of each module."""
        qualifier = f"{prefix}." if prefix else ""
        modules = []
        for offset, module in enumerate(modules_generator):
            modules.append(module)
            layer_index = start_index + offset
            if (
                self.offload_layers is not None
                and layer_index not in self.offload_layers
            ):
                continue
            self._offload_module(module, f"{qualifier}{layer_index}")

        if self.offloaded_params:
            logger.info(
                "Memory-mapped %s of parameters from %s: %s",
                format_gib(self.store.total_bytes),
                self.store.root,
                ", ".join(self.offloaded_params),
            )
        return modules

    def _offload_module(self, module: nn.Module, layer_prefix: str) -> None:
        for submodule_name, submodule in module.named_modules():
            for param_name, param in submodule.named_parameters(recurse=False):
                name = (
                    f"{submodule_name}.{param_name}" if submodule_name else param_name
                )
                if not matches_param(name, self.offload_params):
                    continue
                if getattr(param, "_vllm_is_disk_offloaded", False):
                    continue
                self._offload_param(submodule, name, param_name, param, layer_prefix)

    def _offload_param(
        self,
        submodule: nn.Module,
        name: str,
        param_name: str,
        param: nn.Parameter,
        layer_prefix: str,
    ) -> None:
        quant_method = getattr(submodule, "quant_method", None)
        supports_embedding = quant_method is not None and (
            isinstance(quant_method, HostGatherEmbeddingMethod)
            or method_has_implemented_embedding(type(quant_method))
        )
        if param_name != "weight" or not supports_embedding:
            raise ValueError(
                f"Cannot disk-offload {layer_prefix}.{name}: disk offloading "
                "only supports embedding tables, whose owning layer exposes a "
                "`quant_method.embedding` lookup. Use --cpu-offload-gb or "
                "--offload-group-size for dense parameters."
            )
        if param.dim() != 2:
            raise ValueError(
                f"Cannot disk-offload {layer_prefix}.{name}: expected a 2D "
                f"embedding table, got shape {tuple(param.shape)}."
            )

        key = f"{layer_prefix}.{name}"
        mapped = self.store.allocate(key, tuple(param.shape), param.dtype)
        param.data = mapped
        param._vllm_is_disk_offloaded = True
        if not isinstance(quant_method, HostGatherEmbeddingMethod):
            submodule.quant_method = HostGatherEmbeddingMethod(quant_method)
        # The fused vocab-parallel kernel reads the weight on the device, which
        # a memory-mapped table cannot satisfy.
        if getattr(submodule, "use_fused_embedding", False):
            submodule.use_fused_embedding = False
        self.offloaded_params.append(key)
