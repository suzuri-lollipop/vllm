# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for explicit layer placement and disk-backed embedding offloading."""

import pytest
import torch
import torch.nn as nn

from vllm.config import DiskOffloadConfig, OffloadConfig, UVAOffloadConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.offloader import (
    CompositeOffloader,
    DiskOffloader,
    NoopOffloader,
    UVAOffloader,
    create_offloader,
    parse_layer_spec,
)
from vllm.model_executor.offloader.disk import (
    HostGatherEmbeddingMethod,
    host_gather_rows,
)


class _Embedding(nn.Module):
    """Minimal stand-in for `VocabParallelEmbedding`."""

    class _Method(QuantizeMethodBase):
        def create_weights(self, layer, *args, **kwargs):
            raise NotImplementedError

        def apply(self, layer, *args, **kwargs):
            raise NotImplementedError

        def embedding(self, layer, input_):
            return torch.nn.functional.embedding(input_, layer.weight)

    def __init__(self, num_embeddings: int, dim: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.arange(num_embeddings * dim, dtype=torch.float32).view(
                num_embeddings, dim
            ),
            requires_grad=False,
        )
        self.quant_method = self._Method()

    def forward(self, ids):
        return self.quant_method.embedding(self, ids)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.Linear(4, 4, bias=False)
        self.ngram_embedding = _Embedding(8, 3)


def _layers(count: int) -> list[nn.Module]:
    return [_Layer() for _ in range(count)]


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("3", {3}),
        ("0,2", {0, 2}),
        ("2-4", {2, 3, 4}),
        (" 0 , 5-7 ", {0, 5, 6, 7}),
    ],
)
def test_parse_layer_spec(spec, expected):
    assert parse_layer_spec(spec) == expected


def test_parse_layer_spec_empty_is_none():
    assert parse_layer_spec("") is None
    assert parse_layer_spec("   ") is None


@pytest.mark.parametrize("spec", ["4-2", "-1", "a", "1-b", ","])
def test_parse_layer_spec_rejects_invalid(spec):
    with pytest.raises(ValueError):
        parse_layer_spec(spec)


def _offloaded_positions(
    monkeypatch, spec: str, num_layers: int, start_index: int = 0
) -> list[int]:
    """Positions in the stack that a UVA offloader hands to the CPU path.

    The parameters themselves already live on CPU on a machine without an
    accelerator, so the selection is observed at the call boundary instead.
    """
    offloader = UVAOffloader(
        cpu_offload_max_bytes=0,
        cpu_offload_params={"experts"},
        offload_layers=parse_layer_spec(spec),
    )
    modules = _layers(num_layers)
    considered = []
    monkeypatch.setattr(
        UVAOffloader,
        "_maybe_offload_to_cpu",
        lambda self, module, prefix="": (
            considered.append(modules.index(module)) or module
        ),
    )
    offloader.wrap_modules(iter(modules), start_index=start_index)
    return considered


def test_uva_offloader_honors_explicit_layers(monkeypatch):
    """Only the selected layers are offloaded, in stack order."""
    assert _offloaded_positions(monkeypatch, "1,3", 4) == [1, 3]


def test_uva_offloader_layer_selection_is_global_under_pp(monkeypatch):
    """`start_index` makes the selection refer to global layer indices."""
    assert _offloaded_positions(monkeypatch, "5", 4, start_index=4) == [1]


def test_uva_offloader_without_selection_considers_every_layer(monkeypatch):
    assert _offloaded_positions(monkeypatch, "", 3) == [0, 1, 2]


def test_disk_offloader_maps_embedding_and_preserves_lookup(tmp_path):
    """A memory-mapped table still returns the rows that were loaded into it."""
    layers = _layers(2)
    reference = layers[0].ngram_embedding.weight.detach().clone()

    offloader = DiskOffloader(
        path=str(tmp_path),
        offload_params={"ngram_embedding"},
    )
    modules = offloader.wrap_modules(iter(layers))
    embedding = modules[0].ngram_embedding

    assert isinstance(embedding.quant_method, HostGatherEmbeddingMethod)
    assert embedding.weight._vllm_is_disk_offloaded
    # The mapping starts zeroed; weight loading writes through it to the file.
    assert torch.equal(embedding.weight, torch.zeros_like(reference))
    embedding.weight.data.copy_(reference)

    ids = torch.tensor([[7, 0], [3, 3]])
    assert torch.equal(embedding(ids), reference[ids])

    offloader.store.cleanup()


def test_disk_offloader_rejects_non_embedding_params(tmp_path):
    offloader = DiskOffloader(path=str(tmp_path), offload_params={"experts"})
    with pytest.raises(ValueError, match="only supports embedding tables"):
        offloader.wrap_modules(iter(_layers(1)))
    offloader.store.cleanup()


def test_disk_offloader_requires_explicit_params(tmp_path):
    with pytest.raises(ValueError, match="explicit set of parameters"):
        DiskOffloader(path=str(tmp_path), offload_params=set())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_host_gather_rows_matches_dense_lookup(dtype):
    table = torch.randn(16, 5).to(dtype)
    ids = torch.tensor([[0, 15], [3, 3]])
    expected = torch.nn.functional.embedding(ids, table)
    assert torch.equal(host_gather_rows(table, ids), expected)


def test_host_gather_rows_supports_fp8():
    """FP8 tables are gathered through a uint8 bit-cast."""
    table = torch.randn(16, 4).to(torch.float8_e4m3fn)
    ids = torch.tensor([1, 9, 9])
    gathered = host_gather_rows(table, ids)
    assert gathered.dtype == torch.float8_e4m3fn
    assert torch.equal(gathered.float(), table.float()[ids])


def test_create_offloader_selects_uva_for_layer_selection_alone():
    config = OffloadConfig(
        offload_layers="0-3", uva=UVAOffloadConfig(cpu_offload_params={"experts"})
    )
    offloader = create_offloader(config)
    assert isinstance(offloader, UVAOffloader)
    # No byte budget means "offload exactly the selected layers".
    assert offloader.cpu_offload_max_bytes > 0


def test_create_offloader_composes_disk_and_cpu_backends(tmp_path):
    config = OffloadConfig(
        offload_layers="0-3",
        uva=UVAOffloadConfig(cpu_offload_params={"experts"}),
        disk=DiskOffloadConfig(
            disk_offload_path=str(tmp_path),
            disk_offload_params={"ngram_embedding"},
        ),
    )
    offloader = create_offloader(config)
    assert isinstance(offloader, CompositeOffloader)
    assert isinstance(offloader.offloaders[0], DiskOffloader)
    assert isinstance(offloader.offloaders[1], UVAOffloader)

    modules = offloader.wrap_modules(iter(_layers(4)))
    for module in modules:
        assert module.ngram_embedding.weight._vllm_is_disk_offloaded
    offloader.offloaders[0].store.cleanup()


def test_create_offloader_defaults_to_noop():
    assert isinstance(create_offloader(OffloadConfig()), NoopOffloader)


def test_disk_offload_path_requires_params(tmp_path):
    with pytest.raises(ValueError, match="requires disk_offload_params"):
        OffloadConfig(disk=DiskOffloadConfig(disk_offload_path=str(tmp_path)))
