# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk-backed row store for the Qwen4Exp (Qwen3.8-Flash-Next) PLE n-gram table.

Qwen3.8-Flash-Next addresses a ~47.7 GiB FP8 n-gram embedding table with 16
hashed row ids per token. Keeping the whole table resident costs more VRAM than
the rest of the model, but a decode step only ever touches
``num_tokens * ngram_heads`` rows of 160 bytes, so the table is a natural
candidate for SSD residency.

This is a port of FreeToken's ``freetoken.models.qwen4_exp.ple_disk``
(``--ple-backend disk``). FreeToken maps the checkpoint's fp8 shard tensors in
place and hashes the n-gram windows host-side in C++; vLLM already hashes the
windows on device (``qwen4_exp_compute_ple_ngram_ids``), so the port keeps
FreeToken's row-source/row-store split and replaces the hashing front end with a
row-id gather:

* [`PLERowSource`][] is FreeToken's ``PleRowSource``: equal-size extents of
  fixed-stride rows. FreeToken's ``resolve_row_source`` calls a repacked flat
  file "the seam where a repacked format would plug in"; that is exactly what
  [`PLERowSpool`][] writes, so vLLM never has to parse safetensors headers or
  care which loader produced the shards.
* [`PLEDiskRowStore`][] is ``PleStore``: it dedups the requested rows, coalesces
  them into positioned reads, fans the payload back out, and drops the pages it
  read so a 47.7 GiB table cannot squat in the page cache.
* [`Qwen4ExpPLEDiskEmbeddingMethod`][] is the ``PLETableBackend`` protocol,
  expressed through the seam vLLM already uses for this table: the PLE
  embedding's `QuantizeMethodBase`.
"""

import json
import mmap
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
)
from vllm.model_executor.parameter import ModelWeightParameter, PerTensorScaleParameter
from vllm.utils.platform_utils import is_pin_memory_available

from .ple import compute_ple_shard_overlap

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

PAGE_SIZE = 4096
SPOOL_FORMAT_VERSION = 1
# fio on consumer NVMe: pread saturates near 16 threads, more only adds latency.
DEFAULT_IO_THREADS = 16
# Cap a coalesced read so a large batch still spreads over the thread pool.
DEFAULT_MAX_READ_BYTES = 1 << 20

_HAS_PREAD = hasattr(os, "pread")
_HAS_FADVISE = hasattr(os, "posix_fadvise")
O_DIRECT = getattr(os, "O_DIRECT", 0)

SYNC_MODES = frozenset({"auto", "forward", "gate", "memops"})

# cuStreamWaitValue64 flags; CU_STREAM_WAIT_VALUE_GEQ is 0x0.
_CU_WAIT_VALUE_GEQ = 0x0
_CU_WRITE_VALUE_DEFAULT = 0x0
_CUDA_SUCCESS = 0


class PLEStreamMemops:
    """CUDA stream memory ops, resolved from the driver with `ctypes`.

    Port of FreeToken's ``cumemop_resolve`` / ``memop_wait_reset`` /
    ``signal_flag``. They let a captured graph block on a host-filled pinned
    flag, which is the only way the disk gather can run under full CUDA graph
    capture: the graph emits ``WAIT(flag >= 1); WRITE(flag, 0)`` before it
    consumes the staging buffer, and the host signals once the rows have landed.

    vLLM ships no binding for these, and building a C extension for one model
    is not worth it, so the two driver entry points are looked up directly.
    """

    _instance: "PLEStreamMemops | None" = None

    def __init__(self) -> None:
        self._write = None
        self._wait = None
        self._resolve()

    @classmethod
    def get(cls) -> "PLEStreamMemops":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def available(self) -> bool:
        return self._write is not None and self._wait is not None

    def _resolve(self) -> None:
        import ctypes
        import ctypes.util

        candidates = ["libcuda.so.1", "libcuda.so", "nvcuda.dll"]
        found = ctypes.util.find_library("cuda")
        if found:
            candidates.append(found)
        for name in candidates:
            try:
                library = ctypes.CDLL(name)
            except OSError:
                continue
            write = self._lookup(library, "cuStreamWriteValue64_v2")
            wait = self._lookup(library, "cuStreamWaitValue64_v2")
            if write is None or wait is None:
                write = write or self._lookup(library, "cuStreamWriteValue64")
                wait = wait or self._lookup(library, "cuStreamWaitValue64")
            if write is not None and wait is not None:
                self._write, self._wait = write, wait
                return

    @staticmethod
    def _lookup(library, symbol: str):
        import ctypes

        try:
            function = getattr(library, symbol)
        except AttributeError:
            return None
        function.restype = ctypes.c_int
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulonglong,
            ctypes.c_ulonglong,
            ctypes.c_uint,
        ]
        return function

    def write(self, stream: int, address: int, value: int) -> int:
        if self._write is None:
            return -1
        return int(self._write(stream, address, value, _CU_WRITE_VALUE_DEFAULT))

    def wait_geq(self, stream: int, address: int, value: int) -> int:
        if self._wait is None:
            return -1
        return int(self._wait(stream, address, value, _CU_WAIT_VALUE_GEQ))

    def wait_and_reset(self, stream: int, address: int) -> None:
        """Block the stream until the host signals, then re-arm the flag.

        Resetting before the wait would race a fast host signal and deadlock
        the stream, so the order matters (as in FreeToken).
        """
        if (
            self.wait_geq(stream, address, 1) != _CUDA_SUCCESS
            or self.write(stream, address, 0) != _CUDA_SUCCESS
        ):
            raise RuntimeError(
                "CUDA stream memory ops were rejected; set "
                'additional_config["ple_disk_sync_mode"]="gate"'
            )

    @staticmethod
    def signal_flag(flag: torch.Tensor) -> None:
        """Release-store 1 to a pinned flag once the staged rows have landed.

        Port of FreeToken's ``signal_flag``. The caller must have joined the
        I/O reads and finished the staging scatter on this thread before
        calling, so the plain store only has to publish the value to the
        device-side ``WAIT(flag >= 1)``; on x86 (the consumer target) stores
        are already release-ordered. The flag is pinned host memory, so the
        write goes through the ordinary numpy view.
        """
        flag.numpy().reshape(-1)[0] = 1

    def probe(self, device: torch.device) -> bool:
        """Round-trip a write and a wait to check the driver really allows them."""
        if not self.available or device.type != "cuda":
            return False
        scratch = torch.zeros(1, dtype=torch.int64, pin_memory=True)
        stream = torch.cuda.current_stream(device)
        handle = stream.cuda_stream
        address = scratch.data_ptr()
        if (
            self.write(handle, address, 7) != _CUDA_SUCCESS
            or self.wait_geq(handle, address, 7) != _CUDA_SUCCESS
        ):
            return False
        stream.synchronize()
        return int(scratch[0]) == 7


@dataclass(frozen=True)
class PLEDiskConfig:
    """User-facing knobs for the disk PLE backend.

    Read from ``--additional-config``, mirroring how other models in
    `vllm.models` select a model-specific backend (e.g. Kimi K3's
    ``kda_prefill_backend``)::

        --additional-config '{"ple_table_backend": "disk",
                              "ple_disk_cache_dir": "/mnt/nvme/vllm-ple"}'
    """

    cache_dir: str
    io_threads: int = DEFAULT_IO_THREADS
    direct_io: bool = True
    reuse_cache: bool = True
    max_read_bytes: int = DEFAULT_MAX_READ_BYTES
    sync_mode: str = "auto"
    """How the disk read is ordered against the forward.

    ``forward`` reads inside the split op (simplest, stalls the forward);
    ``gate`` hashes host-side before dispatch so the reads overlap the forward
    (FreeToken's launch-gating); ``memops`` additionally arms a pinned flag with
    CUDA stream memory ops so the gather survives full CUDA graph capture;
    ``auto`` picks ``memops`` when the driver supports it, else ``gate``.
    """

    max_staged_rows: int = 0
    """Hard cap on distinct rows held in the pinned/device staging buffers.

    0 sizes them for the worst case (``max_num_batched_tokens * ngram_heads``).
    Low-VRAM hosts can trade a smaller buffer for a rare fallback re-read.
    """

    verify_host_hash: bool = False
    """Compare host-computed row ids against the device ids every forward.

    Costs a D2H per step; used to validate the host hasher on new hardware.
    """


def resolve_ple_disk_config(vllm_config: "VllmConfig") -> PLEDiskConfig | None:
    """Build a [`PLEDiskConfig`][] when the user asked for the disk backend.

    Args:
        vllm_config: The engine config whose ``additional_config`` carries the
            model-specific knobs.

    Returns:
        The parsed config, or `None` when the PLE table should stay resident.

    Raises:
        ValueError: If the disk backend is requested without a cache directory
            or with an unknown backend name.
    """
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return None
    backend = additional_config.get("ple_table_backend", "auto")
    if backend in ("auto", "resident"):
        return None
    if backend != "disk":
        raise ValueError(
            f"Unknown ple_table_backend {backend!r}; expected 'auto', "
            "'resident' or 'disk'"
        )
    cache_dir = additional_config.get("ple_disk_cache_dir")
    if not cache_dir:
        raise ValueError(
            "ple_table_backend='disk' requires ple_disk_cache_dir to point at "
            "a directory on the SSD that should hold the PLE row spool"
        )
    sync_mode = str(additional_config.get("ple_disk_sync_mode", "auto"))
    if sync_mode not in SYNC_MODES:
        raise ValueError(
            f"Unknown ple_disk_sync_mode {sync_mode!r}; expected one of "
            f"{sorted(SYNC_MODES)}"
        )
    return PLEDiskConfig(
        cache_dir=str(cache_dir),
        io_threads=int(
            additional_config.get("ple_disk_io_threads", DEFAULT_IO_THREADS)
        ),
        direct_io=bool(additional_config.get("ple_disk_direct_io", True)),
        reuse_cache=bool(additional_config.get("ple_disk_reuse_cache", True)),
        max_read_bytes=int(
            additional_config.get("ple_disk_max_read_bytes", DEFAULT_MAX_READ_BYTES)
        ),
        sync_mode=sync_mode,
        max_staged_rows=int(additional_config.get("ple_disk_max_staged_rows", 0)),
        verify_host_hash=bool(
            additional_config.get("ple_disk_verify_host_hash", False)
        ),
    )


@dataclass(frozen=True)
class PLEHostHasher:
    """Computes PLE row ids on the host, from token ids the host already has.

    This is FreeToken's ``PleStore::hash_rows``: it removes the device round
    trip that an on-GPU hash forces on the disk backend, because the ids are
    what name the disk reads. Knowing them before the forward is dispatched is
    what lets the reads overlap the forward instead of stalling it.

    It is a transcription of
    [`Qwen4ExpNGramEmbedding.compute_ngram_ids`][], including the EOS barrier
    that stops a hash window from crossing a sequence boundary; the two are
    asserted bit-identical in the tests.
    """

    multipliers: np.ndarray
    vocab_sizes: np.ndarray
    offsets: np.ndarray
    ngram_size: int
    heads_per_ngram: int
    eos_token_id: int

    @classmethod
    def from_embedding(cls, embedding) -> "PLEHostHasher":
        """Build a hasher from a live `Qwen4ExpNGramEmbedding`."""
        to_np = lambda buffer: buffer.detach().to("cpu").numpy().astype(np.int64)
        return cls(
            multipliers=to_np(embedding.layer_multipliers),
            vocab_sizes=to_np(embedding.ngram_heads_vocab_sizes),
            offsets=to_np(embedding.ngram_heads_offsets),
            ngram_size=int(embedding.ngram_size),
            heads_per_ngram=int(embedding.heads_per_ngram),
            eos_token_id=int(embedding.eos_token_id),
        )

    @property
    def ngram_heads(self) -> int:
        return (self.ngram_size - 1) * self.heads_per_ngram

    def row_ids(
        self,
        token_ids: np.ndarray,
        query_start_loc: np.ndarray,
        ngram_context: np.ndarray,
    ) -> np.ndarray:
        """Row ids for a ragged batch.

        Args:
            token_ids: ``[num_tokens]`` this forward's tokens, concatenated in
                request order.
            query_start_loc: ``[num_reqs + 1]`` token offsets per request.
            ngram_context: ``[num_reqs, ngram_size - 1]`` the tokens
                immediately before each request's first token this forward.

        Returns:
            ``[num_tokens, ngram_heads]`` int64 global row ids.
        """
        tokens = np.asarray(token_ids, dtype=np.int64).reshape(-1)
        starts = np.asarray(query_start_loc, dtype=np.int64).reshape(-1)
        num_reqs = starts.size - 1
        context = np.asarray(ngram_context, dtype=np.int64)[:num_reqs]
        context_len = self.ngram_size - 1
        if context.shape != (num_reqs, context_len):
            raise ValueError(
                f"ngram_context must be [{num_reqs}, {context_len}], got "
                f"{context.shape}"
            )
        num_tokens = tokens.size
        if num_tokens == 0:
            return np.zeros((0, self.ngram_heads), dtype=np.int64)

        positions = np.arange(num_tokens, dtype=np.int64)
        request: np.ndarray = (
            np.searchsorted(starts, positions, side="right") - 1
        ).clip(0, num_reqs - 1)
        columns = positions - starts[request]
        width = context_len + int(columns.max()) + 1
        packed = np.full((num_reqs, width), self.eos_token_id, dtype=np.int64)
        packed[:, :context_len] = context
        packed[request, columns + context_len] = tokens

        window = self._shift_with_eos_barrier(packed)
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = window[0] * self.multipliers[0]
            for index in range(1, ngram):
                mixed = mixed ^ (window[index] * self.multipliers[index])
            taken = mixed[request, columns + context_len]
            ids = np.mod(taken[:, None], self.vocab_sizes[start:end])
            blocks.append(ids + self.offsets[start:end])
        return np.concatenate(blocks, axis=-1)

    def _shift_with_eos_barrier(self, packed: np.ndarray) -> list[np.ndarray]:
        """``out[s][b, p]`` is the token ``s`` left of ``p``, or EOS at a boundary."""
        num_reqs, width = packed.shape
        positions = np.arange(width, dtype=np.int64)
        eos_positions = np.where(packed == self.eos_token_id, positions, -1)
        previous_eos = np.maximum.accumulate(eos_positions, axis=1)
        previous_eos = np.concatenate(
            [np.full((num_reqs, 1), -1, dtype=np.int64), previous_eos[:, :-1]], axis=1
        )
        in_segment = positions[None, :] - previous_eos - 1

        shifted = [packed]
        for shift in range(1, self.ngram_size):
            source: np.ndarray = positions - shift
            gathered = packed[:, np.clip(source, 0, None)]
            valid = (source[None, :] >= 0) & (in_segment >= shift)
            shifted.append(np.where(valid, gathered, self.eos_token_id))
        return shifted


@dataclass(frozen=True)
class PLERowSource:
    """On-disk row layout: equal extents of fixed-stride rows.

    Row ``i`` of extent ``e`` lives at ``extent_base[e] + i * row_stride`` in
    ``paths[extent_file[e]]``. A repacked flat file is one extent with its own
    stride; a mapped checkpoint is one extent per shard tensor.
    """

    paths: tuple[str, ...]
    extent_file: tuple[int, ...]
    extent_base: tuple[int, ...]
    rows_per_extent: int
    row_bytes: int
    row_stride: int

    def __post_init__(self) -> None:
        if len(self.extent_file) != len(self.extent_base):
            raise ValueError("extent_file and extent_base must have equal length")
        if not self.extent_file:
            raise ValueError("a row source needs at least one extent")
        if self.rows_per_extent <= 0 or self.row_bytes <= 0:
            raise ValueError("rows_per_extent and row_bytes must be positive")
        if self.row_stride < self.row_bytes:
            raise ValueError("row_stride must be at least row_bytes")
        if max(self.extent_file) >= len(self.paths):
            raise ValueError("extent_file references a path that does not exist")

    @property
    def total_rows(self) -> int:
        return len(self.extent_base) * self.rows_per_extent

    def locate(self, row_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map global row ids to ``(extent index, byte offset)``."""
        extent: np.ndarray = row_ids // self.rows_per_extent
        base = np.asarray(self.extent_base, dtype=np.int64)[extent]
        return extent, base + (row_ids % self.rows_per_extent) * self.row_stride


def _spool_metadata_path(path: str) -> str:
    return f"{path}.json"


class PLERowSpool:
    """Writes the TP-local PLE rows into one flat file and describes it.

    vLLM hands weights to a layer as materialized tensors, so instead of
    mapping the checkpoint (FreeToken's ``source_from_safetensors``) the spool
    repacks the rows it is given. The sidecar records the geometry so a restart
    can adopt an existing spool instead of rewriting tens of gigabytes.
    """

    def __init__(
        self,
        path: str,
        *,
        num_rows: int,
        row_bytes: int,
        dtype: torch.dtype,
    ) -> None:
        self.path = path
        self.num_rows = int(num_rows)
        self.row_bytes = int(row_bytes)
        self.dtype = dtype
        self._complete = False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0))
        os.ftruncate(self._fd, self.num_rows * self.row_bytes)

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "version": SPOOL_FORMAT_VERSION,
            "num_rows": self.num_rows,
            "row_bytes": self.row_bytes,
            "dtype": str(self.dtype).removeprefix("torch."),
        }

    @staticmethod
    def adopt(
        path: str,
        *,
        num_rows: int,
        row_bytes: int,
        dtype: torch.dtype,
    ) -> PLERowSource | None:
        """Return a source for an existing, complete spool with this geometry."""
        metadata_path = _spool_metadata_path(path)
        if not os.path.exists(path) or not os.path.exists(metadata_path):
            return None
        try:
            with open(metadata_path, encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, ValueError):
            return None
        expected = {
            "version": SPOOL_FORMAT_VERSION,
            "num_rows": int(num_rows),
            "row_bytes": int(row_bytes),
            "dtype": str(dtype).removeprefix("torch."),
        }
        if {key: metadata.get(key) for key in expected} != expected:
            return None
        if os.path.getsize(path) != int(num_rows) * int(row_bytes):
            return None
        return PLERowSource(
            paths=(path,),
            extent_file=(0,),
            extent_base=(0,),
            rows_per_extent=int(num_rows),
            row_bytes=int(row_bytes),
            row_stride=int(row_bytes),
        )

    def write_rows(self, start_row: int, rows: torch.Tensor) -> int:
        """Write ``rows`` at ``start_row``; returns the number of rows written."""
        if self._complete:
            raise RuntimeError("cannot write to a finalized PLE spool")
        if rows.numel() == 0:
            return 0
        if rows.ndim != 2:
            raise ValueError(f"expected a 2D row block, got shape {tuple(rows.shape)}")
        if rows.dtype != self.dtype:
            raise ValueError(
                f"spool holds {self.dtype} rows, got {rows.dtype}; the PLE disk "
                "backend does not requantize"
            )
        if start_row < 0 or start_row + rows.shape[0] > self.num_rows:
            raise ValueError(
                f"rows [{start_row}, {start_row + rows.shape[0]}) fall outside "
                f"the {self.num_rows}-row spool"
            )
        payload = rows.detach().to("cpu").contiguous()
        if payload.shape[1] * payload.element_size() != self.row_bytes:
            raise ValueError(
                f"spool row is {self.row_bytes} bytes, got "
                f"{payload.shape[1] * payload.element_size()}"
            )
        buffer = payload.view(torch.uint8).reshape(-1).numpy().tobytes()
        _pwrite_all(self._fd, buffer, start_row * self.row_bytes)
        return int(payload.shape[0])

    def finalize(self) -> PLERowSource:
        """Flush the spool, publish its sidecar and describe the rows."""
        if not self._complete:
            os.fsync(self._fd)
            os.close(self._fd)
            with open(_spool_metadata_path(self.path), "w", encoding="utf-8") as handle:
                json.dump(self.metadata, handle)
            self._complete = True
        return PLERowSource(
            paths=(self.path,),
            extent_file=(0,),
            extent_base=(0,),
            rows_per_extent=self.num_rows,
            row_bytes=self.row_bytes,
            row_stride=self.row_bytes,
        )


def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        if _HAS_PREAD:
            done = os.pwrite(fd, view[written:], offset + written)
        else:
            os.lseek(fd, offset + written, os.SEEK_SET)
            done = os.write(fd, view[written:])
        if done <= 0:
            raise OSError(f"short write at offset {offset + written}")
        written += done


def aligned_span(
    offset: int, length: int, alignment: int = PAGE_SIZE
) -> tuple[int, int]:
    """Return the ``(start, size)`` of the aligned span covering the payload.

    Direct I/O rejects unaligned offsets and lengths, so a packed row has to be
    read through the aligned window that contains it and copied out.
    """
    if offset < 0 or length <= 0:
        raise ValueError("offset must be non-negative and length positive")
    start = offset - (offset % alignment)
    end = offset + length
    end += (-end) % alignment
    return start, end - start


class _FileReader:
    """Positioned reads for one row file; the platform seam of the store.

    Mirrors FreeToken's ``TableFile``: prefer ``O_DIRECT`` so a table larger
    than RAM cannot evict everything else, and drop the pages behind buffered
    reads when it is unavailable. Windows has neither ``os.pread`` nor
    ``posix_fadvise``, so reads there fall back to thread-local seek+read
    handles, which is what keeps the store's thread pool safe.
    """

    def __init__(self, path: str, *, direct_io: bool) -> None:
        self.path = path
        self._flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        self.direct_io = False
        if direct_io and O_DIRECT:
            try:
                self._fd = os.open(path, self._flags | O_DIRECT)
                self.direct_io = True
            except OSError:
                self._fd = os.open(path, self._flags)
        else:
            self._fd = os.open(path, self._flags)
        self.size = os.fstat(self._fd).st_size
        if not self.direct_io and _HAS_FADVISE:
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)
        self._local = threading.local()
        self._extra_fds: list[int] = []
        self._lock = threading.Lock()

    def close(self) -> None:
        for fd in (self._fd, *self._extra_fds):
            if fd >= 0:
                os.close(fd)
        self._fd = -1
        self._extra_fds = []

    def read_into(self, destination: memoryview, offset: int) -> None:
        """Read ``len(destination)`` bytes at ``offset`` into ``destination``."""
        if self.direct_io:
            self._read_direct(destination, offset)
            return
        self._read_buffered(destination, offset)
        if _HAS_FADVISE:
            os.posix_fadvise(self._fd, offset, len(destination), os.POSIX_FADV_DONTNEED)

    def _thread_fd(self) -> int:
        """A private handle per worker; ``lseek`` + ``read`` is not shareable."""
        fd = getattr(self._local, "fd", None)
        if fd is None:
            fd = os.open(self.path, self._flags)
            self._local.fd = fd
            with self._lock:
                self._extra_fds.append(fd)
        return fd

    def _read_buffered(self, destination: memoryview, offset: int) -> None:
        done = 0
        need = len(destination)
        fd = self._fd if _HAS_PREAD else self._thread_fd()
        while done < need:
            if _HAS_PREAD:
                chunk = os.pread(fd, need - done, offset + done)
            else:
                os.lseek(fd, offset + done, os.SEEK_SET)
                chunk = os.read(fd, need - done)
            if not chunk:
                break
            destination[done : done + len(chunk)] = chunk
            done += len(chunk)
        if done < need:
            raise OSError(f"{self.path}: short read at {offset}: got {done} of {need}")

    def _bounce(self, size: int) -> memoryview:
        """A page-aligned, thread-local landing buffer for direct reads."""
        bounce = getattr(self._local, "bounce", None)
        if bounce is None or len(bounce) < size:
            # mmap is page-aligned, which is what O_DIRECT requires of the buffer.
            bounce = memoryview(mmap.mmap(-1, size))
            self._local.bounce = bounce
        return bounce[:size]

    def _read_direct(self, destination: memoryview, offset: int) -> None:
        start, size = aligned_span(offset, len(destination))
        bounce = self._bounce(size)
        done = 0
        while done < size:
            # preadv lands in the aligned buffer; os.pread would allocate an
            # unaligned one and the kernel would reject the request.
            got = os.preadv(self._fd, [bounce[done:]], start + done)
            if got <= 0:
                break
            done += got
        payload = offset - start
        if done < payload + len(destination):
            raise OSError(
                f"{self.path}: short direct read at {offset}: got {done} of "
                f"{payload + len(destination)}"
            )
        destination[:] = bounce[payload : payload + len(destination)]


@dataclass(frozen=True)
class _ReadRun:
    """One coalesced positioned read: consecutive rows of one extent."""

    extent: int
    offset: int
    row_count: int
    destination_row: int


@dataclass
class StagedRows:
    """Handle to an in-flight batch of staged reads.

    Returned by [`PLEDiskRowStore.stage`][] and consumed by
    [`PLEDiskRowStore.wait`][] / [`PLEDiskRowStore.fetch`][] (FreeToken's
    ``stage`` / ``flush`` split): the reads run on the I/O thread pool while
    the caller dispatches GPU work.
    """

    shape: tuple[int, ...]
    """Shape of the requested row-id tensor (before flattening)."""

    num_unique: int
    """Number of distinct rows being read; they land in staging rows [0, n)."""

    inverse: torch.Tensor
    """``[prod(shape)]`` int64 host tensor mapping each requested row to its
    unique staging row (the dedup fan-out)."""

    pending: list[Future] = field(default_factory=list)
    done: bool = False


class PLEDiskRowStore:
    """Gathers PLE table rows from disk into device memory.

    The store is the runtime half of FreeToken's ``PleStore``: duplicate row
    ids collapse to one read, sorted ids coalesce into contiguous reads, and the
    payload lands in a pinned staging buffer that is copied to the device in one
    shot. The ``gather`` path fans duplicates back out on device with a single
    `index_select`; the staged sync modes instead scatter the deduplicated rows
    into request order in pinned memory ([`scatter_ordered`][]) so the forward
    only does a contiguous fixed-shape H2D copy.

    Batch reader: the I/O fan-out is FreeToken's portable shape — the
    ``ThreadPoolExecutor`` + ``pread`` pool is exactly the fallback its C++
    ``BatchReader`` uses when io_uring is unavailable, and fio says the two
    saturate the same consumer NVMe at ~16 outstanding reads. The port keeps
    vLLM pure-Python on purpose (no ``csrc``/build-system surface for one
    model), so the seam is [`_submit_reads`][] returning joinable handles and
    [`wait`][] draining them. FreeToken's io_uring reader was NOT ported:
    Python has no stdlib binding, vLLM's only batched-file extension
    (``vllm.fs_io_C.batch_load_block``) reads whole files rather than offsets
    inside one, and adding a ``csrc``/liburing build dependency for one model
    is not worth it here. An io_uring reader (Linux-only, the C++
    ``IoUringBatchReader`` shape: one submission per coalesced run, no threads,
    lower per-read CPU on weak cores) would slot behind that same submit/wait
    pair without touching the hashing, staging or sync machinery.
    """

    def __init__(
        self,
        source: PLERowSource,
        *,
        dtype: torch.dtype,
        device: torch.device,
        max_rows: int,
        io_threads: int = DEFAULT_IO_THREADS,
        direct_io: bool = True,
        max_read_bytes: int = DEFAULT_MAX_READ_BYTES,
        ordered_staging: bool = False,
    ) -> None:
        """Initialize the store.

        Args:
            ordered_staging: Allocate the token-ordered pinned buffer the
                pre-dispatch fill needs ([`scatter_ordered`][] /
                [`copy_ordered`][]). Only the staged sync modes use it;
                ``gather``-only stores save the pinned memory.
        """
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        element_size = torch.empty((), dtype=dtype).element_size()
        if source.row_bytes % element_size:
            raise ValueError(
                f"row of {source.row_bytes} bytes does not divide into {dtype}"
            )
        self.source = source
        self.dtype = dtype
        self.device = device
        self.row_elements = source.row_bytes // element_size
        self.max_rows = int(max_rows)
        self.max_read_rows = max(1, max_read_bytes // source.row_stride)
        self._files = [_FileReader(path, direct_io=direct_io) for path in source.paths]
        # These threads block on I/O rather than compute, but on a weak CPU
        # more of them than cores is pure context-switch cost, so the pool is
        # capped by the core count (FreeToken's `min(kReaderThreads,
        # hardware_concurrency())`).
        self.io_threads = max(1, min(int(io_threads), os.cpu_count() or 1))
        self._pool = (
            ThreadPoolExecutor(
                max_workers=self.io_threads, thread_name_prefix="ple-disk"
            )
            if self.io_threads > 1
            else None
        )
        pin = device.type == "cuda" and is_pin_memory_available()
        # Zeroed, not uninitialized: padded graph lanes and the warmup prefill
        # stage no rows and read whatever sits here, and the PLE prefill conv
        # packs every request into one conv1d, so an FP8 NaN left in a padded
        # lane can bleed into a real request's output window.
        self._staging = torch.zeros(
            (self.max_rows, source.row_bytes), dtype=torch.uint8, pin_memory=pin
        )
        self._device_rows = torch.zeros(
            (self.max_rows, source.row_bytes), dtype=torch.uint8, device=device
        )
        # Pre-dispatch fills land rows in REQUEST order (duplicates expanded),
        # so the in-graph H2D is one contiguous copy, right-sized per graph
        # shape by `copy_ordered` (FreeToken keeps separate decode/prefill
        # buffers for the same effect). Allocated up front; pinned allocation
        # inside a stream capture is illegal.
        self._ordered = (
            torch.zeros(
                (self.max_rows, source.row_bytes), dtype=torch.uint8, pin_memory=pin
            )
            if ordered_staging
            else None
        )
        self._staging_np = self._staging.numpy()
        logger.info_once(
            "Qwen4Exp PLE disk backend: %d rows over %d file(s), %s I/O, "
            "%d thread(s), %.1f MiB pinned",
            source.total_rows,
            len(source.paths),
            "direct" if self._files[0].direct_io else "buffered",
            self.io_threads,
            self._staging.numel() * (2 if self._ordered is not None else 1) / 2**20,
        )

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        for file in self._files:
            file.close()

    def plan_reads(self, sorted_rows: np.ndarray) -> list[_ReadRun]:
        """Coalesce sorted, unique row ids into positioned reads."""
        if sorted_rows.size == 0:
            return []
        extent, offset = self.source.locate(sorted_rows)
        packed = self.source.row_stride == self.source.row_bytes
        if not packed:
            return [
                _ReadRun(int(extent[i]), int(offset[i]), 1, i)
                for i in range(sorted_rows.size)
            ]
        contiguous = (extent[1:] == extent[:-1]) & (
            offset[1:] == offset[:-1] + self.source.row_stride
        )
        breaks: np.ndarray = np.flatnonzero(~contiguous) + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [sorted_rows.size]))
        runs: list[_ReadRun] = []
        for start, end in zip(starts.tolist(), ends.tolist()):
            for chunk in range(start, end, self.max_read_rows):
                count = min(self.max_read_rows, end - chunk)
                runs.append(
                    _ReadRun(int(extent[chunk]), int(offset[chunk]), count, chunk)
                )
        return runs

    def _submit_reads(self, runs: list[_ReadRun], num_unique: int) -> list[Future]:
        staging = self._staging[:num_unique]
        buffer = memoryview(staging.numpy().reshape(-1).data).cast("B")
        row_bytes = self.source.row_bytes

        def execute(run: _ReadRun) -> None:
            file = self._files[self.source.extent_file[run.extent]]
            start = run.destination_row * row_bytes
            file.read_into(
                buffer[start : start + run.row_count * row_bytes], run.offset
            )

        if self._pool is None or len(runs) < 2:
            for run in runs:
                execute(run)
            return []
        return [self._pool.submit(execute, run) for run in runs]

    def stage(self, row_ids: np.ndarray | torch.Tensor) -> StagedRows:
        """Start the disk reads for ``row_ids`` without touching the device.

        This is FreeToken's ``PleStore::stage``: it dedups, plans coalesced
        positioned reads and hands them to the I/O threads, then returns
        immediately so the caller can dispatch GPU work while the SSD is busy.

        Args:
            row_ids: Global row ids with any leading shape, on the host.

        Returns:
            A handle to pass to [`wait`][] and [`fetch`][].
        """
        if isinstance(row_ids, torch.Tensor):
            host_ids = row_ids.reshape(-1).to(device="cpu", dtype=torch.int64)
        else:
            host_ids = torch.from_numpy(
                np.ascontiguousarray(row_ids, dtype=np.int64).reshape(-1)
            )
        shape = tuple(row_ids.shape)
        unique, inverse = torch.unique(host_ids, return_inverse=True)
        num_unique = int(unique.numel())
        if num_unique > self.max_rows:
            raise ValueError(
                f"PLE disk gather needs {num_unique} distinct rows but the "
                f"store was sized for {self.max_rows}; raise "
                'additional_config["ple_disk_max_staged_rows"]'
            )
        ids = unique.numpy()
        if ids.size and (ids[0] < 0 or ids[-1] >= self.source.total_rows):
            raise IndexError(
                f"PLE row id out of range [0, {self.source.total_rows}): "
                f"{int(ids[0])}..{int(ids[-1])}"
            )
        pending = self._submit_reads(self.plan_reads(ids), num_unique)
        return StagedRows(shape, num_unique, inverse, pending)

    @staticmethod
    def wait(staged: StagedRows) -> None:
        """Join the staged reads; ``flush`` in FreeToken."""
        if staged.done:
            return
        try:
            for future in staged.pending:
                future.result()
        finally:
            staged.pending = []
            staged.done = True

    def fetch(self, staged: StagedRows) -> torch.Tensor:
        """Copy staged rows to the device and fan duplicates back out."""
        self.wait(staged)
        rows = self._device_rows[: staged.num_unique]
        rows.copy_(
            self._staging[: staged.num_unique],
            non_blocking=self.device.type == "cuda",
        )
        gathered = rows.index_select(0, staged.inverse.to(self.device))
        return gathered.view(self.dtype).reshape(*staged.shape, self.row_elements)

    def scatter_ordered(self, staged: StagedRows) -> int:
        """Fan staged rows out into request order in the pinned buffer.

        Joins the reads, then gathers the deduplicated staging rows through
        ``staged.inverse`` so ``_ordered[i]`` holds the ``i``-th requested row
        (duplicates expanded, in token order). The device-side consume is then
        a single contiguous fixed-shape H2D copy, which is what survives CUDA
        graph capture. Returns the number of ordered rows written.
        """
        if self._ordered is None:
            raise RuntimeError(
                "the store was created without ordered_staging; staged sync "
                "modes need ordered_staging=True"
            )
        self.wait(staged)
        total = 1
        for dim in staged.shape:
            total *= int(dim)
        if total > self.max_rows:
            raise ValueError(
                f"PLE disk fill needs {total} ordered rows but the store was "
                f"sized for {self.max_rows}; raise "
                'additional_config["ple_disk_max_staged_rows"]'
            )
        ordered_np = self._ordered.numpy()
        inverse_np = staged.inverse.numpy().reshape(-1)
        ordered_np[:total] = self._staging_np[inverse_np]
        return total

    def copy_ordered(self, num_rows: int) -> torch.Tensor:
        """Copy the first ``num_rows`` ordered rows to the device.

        The staged-mode lookup: one contiguous pinned->device copy, then a
        typed view. Capture-safe because both endpoints and the byte count are
        fixed for a given graph shape.
        """
        if self._ordered is None:
            raise RuntimeError(
                "the store was created without ordered_staging; staged sync "
                "modes need ordered_staging=True"
            )
        if num_rows > self.max_rows:
            raise ValueError(
                f"PLE disk fill copied {num_rows} ordered rows but the store "
                f"was sized for {self.max_rows}"
            )
        device_rows = self._device_rows[:num_rows]
        device_rows.copy_(
            self._ordered[:num_rows], non_blocking=self.device.type == "cuda"
        )
        return device_rows

    def gather(self, row_ids: torch.Tensor) -> torch.Tensor:
        """Read the rows named by ``row_ids`` and return them in that order.

        Args:
            row_ids: Integer tensor of global row ids with any leading shape.

        Returns:
            A tensor of shape ``(*row_ids.shape, row_elements)`` in the store's
            dtype, on the store's device.
        """
        self.assert_not_capturing()
        return self.fetch(self.stage(row_ids))

    @staticmethod
    def assert_not_capturing() -> None:
        """Reject in-forward gathers during stream capture.

        Only reachable with ``sync_mode="forward"``: the staged modes
        (``gate`` / ``memops``) read before dispatch and never call
        [`gather`][], so a capture-time trip here means the in-forward path
        was captured on purpose or by a FULL graph over a forward-mode store.
        """
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The Qwen4Exp PLE disk backend was asked to read from the "
                'host inside a CUDA graph capture; sync_mode="forward" '
                "cannot be captured. Set "
                'additional_config["ple_disk_sync_mode"]="memops" (or '
                '"gate") so rows are staged before dispatch, or keep the '
                "forward gather out of graphs (cudagraph_mode=PIECEWISE / "
                "--enforce-eager)."
            )


@dataclass
class _PLEDiskLayerEntry:
    """Per-PLE-layer state owned by the stager."""

    hasher: PLEHostHasher
    store: PLEDiskRowStore
    flag: torch.Tensor
    """Pinned int64 flag this layer's captured graph WAITs on (memops)."""


@dataclass
class _PLEDeferredFill:
    """A fill deferred until after the forward was dispatched (memops)."""

    num_tokens: int
    query_start_loc: np.ndarray
    num_reqs: int
    event: Any


class PLEDiskStager:
    """Pre-dispatch staging for the disk PLE backend.

    Port of the low-spec half of FreeToken's ``DiskRowTable``: rows are hashed
    and read on the HOST before the forward is dispatched, so the device-side
    lookup shrinks to a fixed-shape pinned->device copy and never synchronizes
    inside the forward. That is what makes the lookup capturable into FULL CUDA
    graphs and what lets the SSD reads overlap the forward instead of stalling
    it.

    The fill needs this forward's tokens and the per-request n-gram context.
    Both are staged device-side by the runner's input preparation, so the
    stager reads them back through pinned buffers + a stream event (FreeToken's
    decode readback) instead of asking the scheduler to mirror token state:

    * ``gate`` (launch-gating): the readback is awaited and the fill completes
      before dispatch; the lookup is a plain fixed-shape H2D copy.
    * ``memops`` (flag-sync): the fill is deferred until after dispatch and the
      captured lookup starts with ``WAIT(flag >= 1); WRITE(flag, 0)``, so the
      replay launches immediately and blocks at the PLE consume until the host
      signals the staged rows. No device-wide sync anywhere. The fill for step
      N+1 waits step N's readback event, which is ordered after step N's replay
      on the stream, so one shared staging buffer and one flag per layer can
      never be overwritten mid-consume (FreeToken's readback-serialization
      invariant).
    """

    def __init__(
        self,
        config: PLEDiskConfig,
        *,
        device: torch.device,
        max_num_tokens: int,
        max_num_reqs: int,
        ngram_context_len: int,
        memops: PLEStreamMemops | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.max_num_tokens = int(max_num_tokens)
        self.max_num_reqs = int(max_num_reqs)
        self.ngram_context_len = int(ngram_context_len)
        self._memops = memops
        pin = device.type == "cuda" and is_pin_memory_available()
        self._readback_tokens = torch.zeros(
            self.max_num_tokens, dtype=torch.int32, pin_memory=pin
        )
        self._readback_context = torch.zeros(
            (self.max_num_reqs, self.ngram_context_len),
            dtype=torch.int32,
            pin_memory=pin,
        )
        self._readback_event: torch.cuda.Event | None = None
        self._layers: dict[str, _PLEDiskLayerEntry] = {}
        self._resolved_mode: str | None = None
        self._deferred: _PLEDeferredFill | None = None
        self._host_row_ids: dict[str, np.ndarray] = {}

    # ---- setup ----

    def register_layer(self, layer_name: str, embedding: Any) -> None:
        """Adopt one PLE layer's hasher, store and (memops) flag."""
        store = embedding.ngram_embedding.quant_method.store
        if store._ordered is None:
            raise RuntimeError(
                "PLE disk store for "
                f"{layer_name} has no ordered staging; the staged sync modes "
                "need the store created with ordered_staging=True"
            )
        pin = self.device.type == "cuda" and is_pin_memory_available()
        flag = torch.zeros(1, dtype=torch.int64, pin_memory=pin)
        self._layers[layer_name] = _PLEDiskLayerEntry(
            hasher=PLEHostHasher.from_embedding(embedding),
            store=store,
            flag=flag,
        )

    def attach_model(self, model: nn.Module) -> int:
        """Find every staged disk PLE layer in ``model`` and adopt it.

        Returns the number of layers attached. Resolves the sync mode eagerly
        (the memops probe runs stream ops, which must never first run inside a
        capture).
        """
        for module in model.modules():
            embedding = getattr(module, "ple_embedding", None)
            if embedding is None or not getattr(embedding, "staged_disk_lookup", False):
                continue
            layer_name = embedding.layer_name
            self.register_layer(layer_name, embedding)
            embedding._disk_stager = self
        if self._layers:
            self.resolve_sync_mode()
        return len(self._layers)

    def _get_memops(self) -> PLEStreamMemops:
        if self._memops is None:
            self._memops = PLEStreamMemops.get()
        return self._memops

    def resolve_sync_mode(self) -> str:
        """Resolve ``auto`` once; mirrors FreeToken's ``_probe_wait_sync``."""
        if self._resolved_mode is not None:
            return self._resolved_mode
        mode = self.config.sync_mode
        memops_ok = self._probe_memops() if mode in ("auto", "memops") else False
        if mode == "forward":
            resolved = "forward"
        elif mode == "gate":
            resolved = "gate"
        elif mode == "memops":
            if not memops_ok:
                raise RuntimeError(
                    'ple_disk_sync_mode="memops" was requested but CUDA '
                    "stream memory ops are unavailable on this device"
                )
            resolved = "memops"
        else:  # auto
            resolved = "memops" if memops_ok else "gate"
        if self.config.verify_host_hash and resolved == "memops":
            raise ValueError(
                "ple_disk_verify_host_hash needs the host fill to complete "
                'before the forward; use ple_disk_sync_mode="gate" (or '
                '"forward"), not memops'
            )
        self._resolved_mode = resolved
        logger.info_once(
            "Qwen4Exp PLE disk sync: %s (requested %s, memops probe %s)",
            resolved,
            mode,
            "passed" if memops_ok else "unavailable",
        )
        return resolved

    def _probe_memops(self) -> bool:
        if self.device.type != "cuda":
            return False
        return self._get_memops().probe(self.device)

    @property
    def sync_mode(self) -> str:
        if self._resolved_mode is None:
            raise RuntimeError(
                "PLE disk sync mode was never resolved; attach_model must run "
                "before the first forward"
            )
        return self._resolved_mode

    @property
    def verify_host_hash(self) -> bool:
        return self.config.verify_host_hash

    # ---- host side (model-runner thread, before/after dispatch) ----

    def prepare(
        self,
        *,
        input_ids: torch.Tensor,
        num_tokens: int,
        query_start_loc: np.ndarray,
        num_reqs: int,
        ngram_context: torch.Tensor,
        use_graph: bool,
    ) -> None:
        """Stage this batch's rows; defer the fill only under memops graphs.

        ``input_ids`` / ``ngram_context`` are the runner's device-side input
        buffers (already populated for this step); both are read back through
        pinned buffers so the fill never needs scheduler token state.
        """
        if self.sync_mode == "forward" or not self._layers:
            return
        cuda = self.device.type == "cuda"
        self._readback_tokens[:num_tokens].copy_(
            input_ids[:num_tokens], non_blocking=cuda
        )
        self._readback_context[:num_reqs].copy_(
            ngram_context[:num_reqs], non_blocking=cuda
        )
        event = None
        if cuda:
            if self._readback_event is None:
                self._readback_event = torch.cuda.Event()
            event = self._readback_event
            event.record(torch.cuda.current_stream(self.device))
        if use_graph and self.sync_mode == "memops":
            # Flag-sync: launch first, fill + signal after dispatch.
            self._deferred = _PLEDeferredFill(
                num_tokens,
                np.asarray(query_start_loc, dtype=np.int64),
                num_reqs,
                event,
            )
            return
        self._fill(
            num_tokens, np.asarray(query_start_loc, dtype=np.int64), num_reqs, event
        )

    def prepare_dummy(self, num_reqs: int, num_tokens: int) -> None:
        """Synchronous fill for dummy/capture runs (no device readback)."""
        if self.sync_mode == "forward" or not self._layers:
            return
        starts: np.ndarray = np.zeros(num_reqs + 1, dtype=np.int64)
        if num_reqs > 0:
            per_req, extra = divmod(num_tokens, num_reqs)
            lengths: np.ndarray = np.full(num_reqs, per_req, dtype=np.int64)
            if extra:
                lengths[-extra:] += 1
            np.cumsum(lengths, out=starts[1:])
        self._fill(num_tokens, starts, num_reqs, None)

    def post_dispatch(self) -> None:
        """Run a deferred fill after the forward was dispatched (memops)."""
        deferred, self._deferred = self._deferred, None
        if deferred is None:
            return
        try:
            self._fill(
                deferred.num_tokens,
                deferred.query_start_loc,
                deferred.num_reqs,
                deferred.event,
            )
        except BaseException:
            # Unblock the stream before surfacing; this step's output is discarded.
            self._signal_all()
            raise
        self._signal_all()

    def _fill(
        self,
        num_tokens: int,
        query_start_loc: np.ndarray,
        num_reqs: int,
        event: Any,
    ) -> None:
        if event is not None:
            event.synchronize()
        tokens = self._readback_tokens[:num_tokens].numpy().astype(np.int64)
        context = self._readback_context[:num_reqs].numpy().astype(np.int64)
        staged = []
        for layer_name, entry in self._layers.items():
            row_ids = entry.hasher.row_ids(tokens, query_start_loc, context)
            if self.verify_host_hash:
                self._host_row_ids[layer_name] = row_ids
            staged.append(entry.store.stage(row_ids))
        for entry, handle in zip(self._layers.values(), staged):
            entry.store.scatter_ordered(handle)

    def _signal_all(self) -> None:
        memops = self._get_memops()
        for entry in self._layers.values():
            memops.signal_flag(entry.flag)

    # ---- device side (inside the forward / captured graph) ----

    def lookup(self, layer_name: str, output: torch.Tensor) -> None:
        """Serve this layer's rows from staging into ``output``.

        ``output`` is ``[num_tokens, ngram_heads, head_dim]`` in the table
        dtype. Under a memops capture this emits the flag WAIT/RESET first, so
        the replayed graph blocks at the consume until the host fill signals.
        """
        entry = self._layers[layer_name]
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if capturing and self.sync_mode == "memops":
            stream = torch.cuda.current_stream(self.device)
            self._get_memops().wait_and_reset(stream.cuda_stream, entry.flag.data_ptr())
        num_rows = output.shape[0] * output.shape[1]
        device_rows = entry.store.copy_ordered(num_rows)
        output.copy_(device_rows.view(entry.store.dtype).reshape(output.shape))

    def verify_host_ids(self, layer_name: str, device_ids: torch.Tensor) -> None:
        """Compare device-hashed row ids against the host fill's ids.

        Debug aid (``ple_disk_verify_host_hash``); costs a D2H per forward.
        Only valid once the fill has landed, i.e. not under memops (rejected at
        resolution) and not during capture.
        """
        host_ids = self._host_row_ids.get(layer_name)
        if host_ids is None:
            raise RuntimeError(
                "verify_host_hash has no host row ids for "
                f"{layer_name}; the fill must run before the forward"
            )
        actual = device_ids.detach().to("cpu", dtype=torch.int64).numpy()
        expected = host_ids.reshape(actual.shape)
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"verify_host_hash shape mismatch for {layer_name}: device "
                f"{actual.shape} vs host {expected.shape}"
            )
        if not np.array_equal(actual, expected):
            mismatch = int(np.count_nonzero((actual != expected).any(axis=-1)))
            raise RuntimeError(
                f"verify_host_hash: {mismatch}/{len(actual)} tokens of "
                f"{layer_name} hashed differently on host and device"
            )


class Qwen4ExpPLEDiskEmbeddingMethod(QuantizeMethodBase):
    """PLE embedding whose rows live on the SSD instead of in VRAM.

    Selected by ``--additional-config '{"ple_table_backend": "disk", ...}'``.
    ``create_weights`` registers a row-less placeholder so the table never
    reaches device memory; the checkpoint shards are spooled to disk by
    [`load_ple_shard`][vllm.models.qwen4_exp.common.ple_disk.Qwen4ExpPLEDiskEmbeddingMethod.load_ple_shard]
    as they arrive, and ``embedding`` gathers rows per forward.

    The lookup output keeps the checkpoint dtype (FP8 for Qwen3.8-Flash-Next),
    so the caller's existing dequantization and TP all-reduce paths are unchanged.
    """

    def __init__(
        self,
        config: PLEDiskConfig,
        *,
        spool_name: str,
        max_gathered_rows: int,
        table_dtype: torch.dtype = torch.float8_e4m3fn,
        has_weight_scale: bool = True,
    ) -> None:
        self.config = config
        self.spool_name = spool_name
        self.max_gathered_rows = int(max_gathered_rows)
        self.table_dtype = table_dtype
        self.has_weight_scale = has_weight_scale
        self._spool: PLERowSpool | None = None
        self._source: PLERowSource | None = None
        self._store: PLEDiskRowStore | None = None
        self._rows_written = 0
        self._num_rows = 0
        self._row_bytes = 0
        self._shard_tag = ""

    @property
    def spool_path(self) -> str:
        """Per-layer, per-TP-shard spool file; each rank owns its own rows."""
        name = self.spool_name.replace(".", "_").replace(os.sep, "_")
        return os.path.join(self.config.cache_dir, f"{name}{self._shard_tag}.plerows")

    @property
    def store(self) -> PLEDiskRowStore:
        if self._store is None:
            raise RuntimeError(
                "PLE disk store is not ready; process_weights_after_loading "
                "has not run for this layer"
            )
        return self._store

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size, params_dtype
        weight_loader = extra_weight_attrs.get("weight_loader")
        tp_size = int(getattr(layer, "tp_size", 1))
        if tp_size > 1:
            self._shard_tag = f"_tp{int(getattr(layer, 'tp_rank', 0))}of{tp_size}"
        self._num_rows = sum(output_partition_sizes)
        self._row_bytes = (
            input_size_per_partition
            * torch.empty((), dtype=self.table_dtype).element_size()
        )
        # A row-less placeholder: the parameter has to exist for the weight
        # loader and the state dict, but must not reserve the table's VRAM.
        layer.register_parameter(
            "weight",
            ModelWeightParameter(
                data=torch.empty(0, input_size_per_partition, dtype=self.table_dtype),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        if self.has_weight_scale:
            layer.register_parameter(
                "weight_scale",
                create_fp8_scale_parameter(
                    PerTensorScaleParameter,
                    output_partition_sizes,
                    input_size_per_partition,
                    None,
                    weight_loader,
                    scale_dtype=torch.bfloat16,
                ),
            )
        source = None
        if self.config.reuse_cache:
            source = PLERowSpool.adopt(
                self.spool_path,
                num_rows=self._num_rows,
                row_bytes=self._row_bytes,
                dtype=self.table_dtype,
            )
        if source is not None:
            logger.info_once("Reusing PLE row spool at %s", self.spool_path)
            self._source = source
        else:
            self._spool = PLERowSpool(
                self.spool_path,
                num_rows=self._num_rows,
                row_bytes=self._row_bytes,
                dtype=self.table_dtype,
            )

    def load_ple_shard(
        self,
        loaded_weight: torch.Tensor,
        *,
        checkpoint_start: int,
        tp_start: int,
        tp_end: int,
    ) -> int:
        """Spool the TP-local rows of one checkpoint shard; returns row count.

        Called instead of the resident copy by
        [`PLEVocabParallelEmbedding.weight_loader`][vllm.models.qwen4_exp.common.ple.PLEVocabParallelEmbedding.weight_loader].
        """
        if self._spool is None:
            # An adopted spool already holds these rows.
            return 0
        overlap = compute_ple_shard_overlap(
            checkpoint_start=checkpoint_start,
            checkpoint_rows=loaded_weight.shape[0],
            tp_start=tp_start,
            tp_end=tp_end,
        )
        if overlap is None:
            return 0
        rows = loaded_weight.narrow(0, overlap.source_start, overlap.row_count)
        written = self._spool.write_rows(overlap.destination_start, rows)
        self._rows_written += written
        return written

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if self._store is not None:
            return
        if self._spool is not None:
            if self._rows_written == 0:
                raise RuntimeError(
                    f"No PLE rows reached the spool at {self.spool_path}; the "
                    "checkpoint has no ngram_embedding shards"
                )
            self._source = self._spool.finalize()
            self._spool = None
        if self._source is None:
            raise RuntimeError("PLE disk backend has no row source")
        max_rows = self.max_gathered_rows
        if self.config.max_staged_rows > 0:
            max_rows = min(max_rows, self.config.max_staged_rows)
        self._store = PLEDiskRowStore(
            self._source,
            dtype=self.table_dtype,
            device=layer.weight.device,
            max_rows=max_rows,
            io_threads=self.config.io_threads,
            direct_io=self.config.direct_io,
            max_read_bytes=self.config.max_read_bytes,
            # The staged sync modes serve the forward from a token-ordered
            # pinned buffer filled before dispatch; only "forward" keeps the
            # gather-in-forward shape.
            ordered_staging=self.config.sync_mode != "forward",
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE disk weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        """Legacy in-forward gather (``sync_mode="forward"`` only).

        The staged sync modes never call this: their rows are filled before
        dispatch and served by [`PLEDiskStager.lookup`][] through the
        ``qwen4_exp_lookup_ple_disk_rows`` op, so no D2H sync happens inside
        the forward.
        """
        del layer
        return self.store.gather(input_)


__all__ = [
    "DEFAULT_IO_THREADS",
    "DEFAULT_MAX_READ_BYTES",
    "PAGE_SIZE",
    "PLEDiskConfig",
    "PLEDiskRowStore",
    "PLEDiskStager",
    "PLEHostHasher",
    "PLERowSource",
    "PLERowSpool",
    "PLEStreamMemops",
    "Qwen4ExpPLEDiskEmbeddingMethod",
    "StagedRows",
    "aligned_span",
    "resolve_ple_disk_config",
]
