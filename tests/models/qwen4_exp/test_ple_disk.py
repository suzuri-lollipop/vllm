# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import vllm.model_executor.layers.vocab_parallel_embedding as embedding_module
import vllm.model_executor.parameter as parameter_module
from vllm.models.qwen4_exp.common.ple_disk import (
    PLEDiskConfig,
    PLEDiskRowStore,
    PLERowSource,
    PLERowSpool,
    Qwen4ExpPLEDiskEmbeddingMethod,
    aligned_span,
    resolve_ple_disk_config,
)

ROW_DTYPE = torch.float8_e4m3fn
HEAD_DIM = 4
ROW_BYTES = HEAD_DIM  # one byte per fp8 element


def _table(num_rows: int) -> torch.Tensor:
    """A deterministic fp8 table whose rows are all distinct."""
    values = torch.arange(num_rows * HEAD_DIM, dtype=torch.float32) % 100
    return values.reshape(num_rows, HEAD_DIM).to(ROW_DTYPE)


def _spool_table(tmp_path, table: torch.Tensor, *, rows_per_shard: int) -> PLERowSource:
    spool = PLERowSpool(
        str(tmp_path / "ple.plerows"),
        num_rows=table.shape[0],
        row_bytes=ROW_BYTES,
        dtype=ROW_DTYPE,
    )
    for start in range(0, table.shape[0], rows_per_shard):
        spool.write_rows(start, table[start : start + rows_per_shard])
    return spool.finalize()


def _store(source: PLERowSource, *, max_rows: int, **kwargs) -> PLEDiskRowStore:
    return PLEDiskRowStore(
        source,
        dtype=ROW_DTYPE,
        device=torch.device("cpu"),
        max_rows=max_rows,
        io_threads=kwargs.pop("io_threads", 2),
        direct_io=kwargs.pop("direct_io", False),
        **kwargs,
    )


def test_resolve_ple_disk_config_defaults_to_the_resident_table() -> None:
    assert resolve_ple_disk_config(SimpleNamespace(additional_config=None)) is None
    assert resolve_ple_disk_config(SimpleNamespace(additional_config={})) is None
    assert (
        resolve_ple_disk_config(
            SimpleNamespace(additional_config={"ple_table_backend": "resident"})
        )
        is None
    )


def test_resolve_ple_disk_config_parses_the_disk_backend() -> None:
    config = resolve_ple_disk_config(
        SimpleNamespace(
            additional_config={
                "ple_table_backend": "disk",
                "ple_disk_cache_dir": "/mnt/nvme/ple",
                "ple_disk_io_threads": 4,
                "ple_disk_direct_io": False,
            }
        )
    )
    assert config == PLEDiskConfig(
        cache_dir="/mnt/nvme/ple", io_threads=4, direct_io=False
    )


@pytest.mark.parametrize(
    "additional_config",
    [
        {"ple_table_backend": "disk"},
        {"ple_table_backend": "ssd", "ple_disk_cache_dir": "/tmp/x"},
    ],
)
def test_resolve_ple_disk_config_rejects_unusable_requests(additional_config) -> None:
    with pytest.raises(ValueError):
        resolve_ple_disk_config(SimpleNamespace(additional_config=additional_config))


@pytest.mark.parametrize(
    ("offset", "length", "expected"),
    [
        (0, 160, (0, 4096)),
        (4000, 160, (0, 8192)),
        (4096, 4096, (4096, 4096)),
        (8191, 1, (4096, 8192 - 4096)),
    ],
)
def test_aligned_span_covers_the_payload(offset, length, expected) -> None:
    start, size = aligned_span(offset, length)
    assert (start, size) == expected
    assert start % 4096 == 0 and size % 4096 == 0
    assert start <= offset and start + size >= offset + length


def test_row_source_locates_rows_across_extents() -> None:
    source = PLERowSource(
        paths=("a", "b"),
        extent_file=(0, 1, 0),
        extent_base=(100, 200, 300),
        rows_per_extent=4,
        row_bytes=8,
        row_stride=8,
    )
    assert source.total_rows == 12
    extent, offset = source.locate(torch.tensor([0, 5, 11]).numpy())
    assert extent.tolist() == [0, 1, 2]
    assert offset.tolist() == [100, 200 + 8, 300 + 24]


def test_row_source_rejects_inconsistent_geometry() -> None:
    with pytest.raises(ValueError):
        PLERowSource(
            paths=("a",),
            extent_file=(0,),
            extent_base=(0,),
            rows_per_extent=4,
            row_bytes=8,
            row_stride=4,
        )


def test_spool_round_trips_the_table(tmp_path) -> None:
    table = _table(37)
    source = _spool_table(tmp_path, table, rows_per_shard=8)
    assert source.total_rows == 37

    store = _store(source, max_rows=37)
    try:
        ids = torch.tensor([[36, 0], [17, 5]])
        gathered = store.gather(ids)
    finally:
        store.close()

    assert gathered.dtype == ROW_DTYPE
    assert gathered.shape == (2, 2, HEAD_DIM)
    torch.testing.assert_close(gathered.float(), table.float()[ids])


def test_spool_is_adopted_only_when_the_geometry_matches(tmp_path) -> None:
    table = _table(16)
    path = str(tmp_path / "ple.plerows")
    spool = PLERowSpool(path, num_rows=16, row_bytes=ROW_BYTES, dtype=ROW_DTYPE)
    spool.write_rows(0, table)

    assert (
        PLERowSpool.adopt(path, num_rows=16, row_bytes=ROW_BYTES, dtype=ROW_DTYPE)
        is None
    ), "an unfinalized spool has no sidecar and must not be adopted"

    spool.finalize()
    adopted = PLERowSpool.adopt(path, num_rows=16, row_bytes=ROW_BYTES, dtype=ROW_DTYPE)
    assert adopted is not None and adopted.total_rows == 16
    assert (
        PLERowSpool.adopt(path, num_rows=17, row_bytes=ROW_BYTES, dtype=ROW_DTYPE)
        is None
    )
    assert (
        PLERowSpool.adopt(path, num_rows=16, row_bytes=ROW_BYTES, dtype=torch.bfloat16)
        is None
    )


def test_spool_rejects_rows_outside_its_range(tmp_path) -> None:
    spool = PLERowSpool(
        str(tmp_path / "ple.plerows"),
        num_rows=4,
        row_bytes=ROW_BYTES,
        dtype=ROW_DTYPE,
    )
    with pytest.raises(ValueError):
        spool.write_rows(2, _table(4))
    with pytest.raises(ValueError):
        spool.write_rows(0, _table(2).to(torch.bfloat16))


def test_plan_reads_coalesces_consecutive_rows(tmp_path) -> None:
    source = _spool_table(tmp_path, _table(32), rows_per_shard=32)
    store = _store(source, max_rows=32)
    try:
        runs = store.plan_reads(torch.tensor([1, 2, 3, 9, 10, 30]).numpy())
    finally:
        store.close()

    assert [(run.offset, run.row_count, run.destination_row) for run in runs] == [
        (1 * ROW_BYTES, 3, 0),
        (9 * ROW_BYTES, 2, 3),
        (30 * ROW_BYTES, 1, 5),
    ]


def test_plan_reads_splits_runs_at_the_read_size_cap(tmp_path) -> None:
    source = _spool_table(tmp_path, _table(32), rows_per_shard=32)
    store = _store(source, max_rows=32, max_read_bytes=2 * ROW_BYTES)
    try:
        runs = store.plan_reads(torch.tensor([0, 1, 2, 3, 4]).numpy())
    finally:
        store.close()

    assert [run.row_count for run in runs] == [2, 2, 1]


def test_gather_reads_each_distinct_row_once(tmp_path, monkeypatch) -> None:
    table = _table(24)
    source = _spool_table(tmp_path, table, rows_per_shard=24)
    store = _store(source, max_rows=24, io_threads=1)
    reads: list[int] = []
    original = type(store._files[0]).read_into

    def counting_read(self, destination, offset):
        reads.append(len(destination))
        original(self, destination, offset)

    monkeypatch.setattr(type(store._files[0]), "read_into", counting_read)
    try:
        ids = torch.tensor([[7, 7, 7], [7, 8, 23]])
        gathered = store.gather(ids)
    finally:
        store.close()

    torch.testing.assert_close(gathered.float(), table.float()[ids])
    # rows {7, 8, 23}: 7 and 8 coalesce into one read, 23 is its own.
    assert sum(reads) == 3 * ROW_BYTES
    assert len(reads) == 2


def test_gather_is_correct_with_a_thread_pool(tmp_path) -> None:
    """Workers must not share a seek position (the non-``pread`` platforms)."""
    table = _table(512)
    source = _spool_table(tmp_path, table, rows_per_shard=128)
    store = _store(source, max_rows=512, io_threads=8, max_read_bytes=ROW_BYTES)
    try:
        ids = torch.arange(511, -1, -1).reshape(64, 8)
        torch.testing.assert_close(store.gather(ids).float(), table.float()[ids])
    finally:
        store.close()


def test_gather_rejects_out_of_range_rows(tmp_path) -> None:
    source = _spool_table(tmp_path, _table(8), rows_per_shard=8)
    store = _store(source, max_rows=8)
    try:
        with pytest.raises(IndexError):
            store.gather(torch.tensor([8]))
    finally:
        store.close()


def test_gather_rejects_more_distinct_rows_than_it_was_sized_for(tmp_path) -> None:
    source = _spool_table(tmp_path, _table(8), rows_per_shard=8)
    store = _store(source, max_rows=2)
    try:
        with pytest.raises(ValueError, match="sized for"):
            store.gather(torch.tensor([0, 1, 2]))
    finally:
        store.close()


@pytest.mark.skipif(
    not hasattr(__import__("os"), "O_DIRECT"), reason="O_DIRECT is Linux-only"
)
def test_direct_io_reads_match_buffered_reads(tmp_path) -> None:
    table = _table(64)
    source = _spool_table(tmp_path, table, rows_per_shard=64)
    store = _store(source, max_rows=64, direct_io=True)
    try:
        ids = torch.tensor([0, 1, 33, 63])
        torch.testing.assert_close(store.gather(ids).float(), table.float()[ids])
    finally:
        store.close()


def _disk_embedding_layer(
    monkeypatch: pytest.MonkeyPatch,
    method: Qwen4ExpPLEDiskEmbeddingMethod,
    num_embeddings: int,
) -> embedding_module.VocabParallelEmbedding:
    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    from vllm.models.qwen4_exp.common.ple import PLEVocabParallelEmbedding

    return PLEVocabParallelEmbedding(
        num_embeddings,
        HEAD_DIM,
        params_dtype=torch.bfloat16,
        padding_size=1,
        quant_method=method,
    )


def _disk_method(tmp_path, **kwargs) -> Qwen4ExpPLEDiskEmbeddingMethod:
    return Qwen4ExpPLEDiskEmbeddingMethod(
        PLEDiskConfig(cache_dir=str(tmp_path), io_threads=1, direct_io=False),
        spool_name=kwargs.pop("spool_name", "model.layers.1.ple.ngram_embedding"),
        max_gathered_rows=kwargs.pop("max_gathered_rows", 32),
        **kwargs,
    )


def test_disk_method_spools_shards_instead_of_materializing_the_table(
    tmp_path, monkeypatch
) -> None:
    table = _table(8)
    method = _disk_method(tmp_path)
    layer = _disk_embedding_layer(monkeypatch, method, table.shape[0])

    assert layer.weight.shape == (0, HEAD_DIM), (
        "the disk backend must not reserve the table in device memory"
    )

    for shard, start in ((table[:4], 0), (table[4:], 4)):
        layer.weight_loader(layer.weight, shard, checkpoint_start=start)
    # The FP8 global scale still loads through the normal parameter path.
    layer.weight_loader(layer.weight_scale, torch.tensor(0.25, dtype=torch.bfloat16))
    method.process_weights_after_loading(layer)
    try:
        ids = torch.tensor([[0, 7], [3, 3]])
        output = layer(ids)
    finally:
        method.store.close()

    assert output.dtype == ROW_DTYPE
    assert layer.weight_scale.item() == 0.25
    torch.testing.assert_close(output.float(), table.float()[ids])


def test_disk_method_reuses_a_previous_spool(tmp_path, monkeypatch) -> None:
    table = _table(8)
    first = _disk_method(tmp_path)
    layer = _disk_embedding_layer(monkeypatch, first, table.shape[0])
    for shard, start in ((table[:4], 0), (table[4:], 4)):
        layer.weight_loader(layer.weight, shard, checkpoint_start=start)
    first.process_weights_after_loading(layer)
    first.store.close()

    second = _disk_method(tmp_path)
    reused_layer = _disk_embedding_layer(monkeypatch, second, table.shape[0])
    # A restart adopts the spool, so the loader's shards are dropped.
    assert (
        second.load_ple_shard(
            table[:4], checkpoint_start=0, tp_start=0, tp_end=table.shape[0]
        )
        == 0
    )
    second.process_weights_after_loading(reused_layer)
    try:
        ids = torch.tensor([1, 6])
        torch.testing.assert_close(
            second.embedding(reused_layer, ids).float(), table.float()[ids]
        )
    finally:
        second.store.close()


def test_disk_method_fails_loudly_when_no_shards_arrive(tmp_path, monkeypatch) -> None:
    method = _disk_method(tmp_path, spool_name="empty")
    layer = _disk_embedding_layer(monkeypatch, method, 8)
    with pytest.raises(RuntimeError, match="No PLE rows"):
        method.process_weights_after_loading(layer)


def test_disk_method_spools_only_the_local_tp_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 2
    )
    table = _table(8)
    method = _disk_method(tmp_path)
    # Rank 1 of 2 owns rows [4, 8); a shard covering [2, 6) contributes 2 rows.
    method.create_weights(
        SimpleNamespace(tp_size=2, tp_rank=1, register_parameter=lambda *_: None),
        HEAD_DIM,
        [4],
        HEAD_DIM,
        8,
        torch.bfloat16,
        weight_loader=None,
    )
    written = method.load_ple_shard(
        table[2:6], checkpoint_start=2, tp_start=4, tp_end=8
    )
    assert written == 2
    assert method.spool_path.endswith("_tp1of2.plerows")


def test_ple_disk_gather_op_dispatches_to_the_owning_layer(monkeypatch) -> None:
    from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module

    class RecordingNGramEmbedding(nn.Module):
        def gather_disk_rows(self, ngram_ids, output) -> None:
            output.copy_(ngram_ids.unsqueeze(-1).expand_as(output).float())

    layer = SimpleNamespace(ple_embedding=RecordingNGramEmbedding())
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"ple": layer}),
    )
    ngram_ids = torch.tensor([[1, 2], [3, 4]])
    output = torch.zeros(2, 2, 3)

    ple_layer_module.qwen4_exp_gather_ple_disk_rows(ngram_ids, output, "ple")

    torch.testing.assert_close(
        output, ngram_ids.unsqueeze(-1).expand(2, 2, 3).float().contiguous()
    )


def _ngram_embedding(monkeypatch, prefix: str, disk_config: PLEDiskConfig | None):
    from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
    from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig

    for module in (embedding_module, parameter_module):
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
    config = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=8,
        num_hidden_layers=1,
        hc_count=2,
        ple_layer_ids=[1],
        ple_embed_dim=2 * HEAD_DIM,
        ngram_size=2,
        heads_per_ngram=2,
        ngram_vocab_size_base=16,
        make_ngram_vocab_size_divisible_by=4,
        eos_token_id=3,
    )
    return Qwen4ExpNGramEmbedding(
        config,
        2 * HEAD_DIM,
        0,
        max_total_tokens=8,
        max_num_reqs=2,
        prefix=prefix,
        layer_name="ple",
        params_dtype=torch.bfloat16,
        ple_disk_config=disk_config,
    )


def test_ngram_embedding_forward_matches_the_resident_table(
    tmp_path, monkeypatch
) -> None:
    """The disk backend is a drop-in for the VRAM-resident PLE table."""
    from vllm.models.qwen4_exp.nvidia import ple_layer as ple_layer_module

    resident = _ngram_embedding(monkeypatch, "resident", None)
    disk = _ngram_embedding(
        monkeypatch,
        "disk",
        PLEDiskConfig(cache_dir=str(tmp_path), io_threads=1, direct_io=False),
    )
    assert not resident.gathers_from_disk
    assert disk.gathers_from_disk
    assert disk.ngram_embedding.weight.shape[0] == 0

    num_rows = resident.ngram_embedding.org_vocab_size
    table = torch.arange(num_rows * HEAD_DIM, dtype=torch.float32)
    table = (table % 17).reshape(num_rows, HEAD_DIM).to(torch.bfloat16)
    shard_size = (num_rows + 1) // 2
    shards = [
        (f"ngram_embedding.shard_{index}.weight", table[start : start + shard_size])
        for index, start in enumerate(range(0, num_rows, shard_size))
    ]
    for module in (resident, disk):
        module.split_ngram_parts = 2
        module.load_weights([(name, shard.clone()) for name, shard in shards])
    disk.ngram_embedding.quant_method.process_weights_after_loading(
        disk.ngram_embedding
    )

    input_ids = torch.tensor([5, 6, 7, 8])
    query_start_loc = torch.tensor([0, 2, 4])
    ngram_context = torch.zeros(2, 1, dtype=torch.long)
    running: dict[str, nn.Module] = {}
    monkeypatch.setattr(
        ple_layer_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            no_compile_layers={"ple": SimpleNamespace(ple_embedding=running["module"])}
        ),
    )
    try:
        running["module"] = resident
        expected = resident(input_ids, query_start_loc, ngram_context)
        running["module"] = disk
        actual = disk(input_ids, query_start_loc, ngram_context)
    finally:
        disk.ngram_embedding.quant_method.store.close()

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.float(), expected.float())


def test_ple_disk_gather_op_is_a_compilation_split_point() -> None:
    from vllm.config.compilation import CompilationConfig

    assert "vllm::qwen4_exp_gather_ple_disk_rows" in CompilationConfig._attention_ops, (
        "the host-side gather must not be captured into a piecewise CUDA graph"
    )
