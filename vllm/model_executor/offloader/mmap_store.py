# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Memory-mapped file storage for model parameters.

Backs parameters with files on a fast block device (NVMe SSD) or a
byte-addressable persistent memory device (Intel Optane exposed through a
DAX filesystem). The mapping is shared, so the kernel serves reads straight
from the device or the page cache instead of consuming anonymous host memory
or device memory.
"""

import os
import shutil

import torch

from vllm.logger import init_logger
from vllm.utils.mem_utils import format_gib

logger = init_logger(__name__)


def _sanitize(name: str) -> str:
    return name.replace("/", "_").replace(os.sep, "_")


class MmapTensorStore:
    """Allocates memory-mapped tensors under a directory.

    Args:
        root: Directory holding the backing files. Point it at an SSD or an
            Optane DAX mount to keep the parameters off host and device RAM.
        keep_files: Keep the backing files after `cleanup`, so a later run can
            reuse the same device space without re-allocating it.
    """

    def __init__(self, root: str, keep_files: bool = False):
        self.root = os.path.join(os.path.abspath(root), f"vllm-mmap-{os.getpid()}")
        self.keep_files = keep_files
        self.total_bytes = 0
        self._files: list[str] = []
        os.makedirs(self.root, exist_ok=True)

    def allocate(
        self,
        key: str,
        shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Allocate a zero-filled memory-mapped tensor.

        Args:
            key: Unique name for the tensor; used as the file name.
            shape: Shape of the returned tensor.
            dtype: Data type of the returned tensor.

        Returns:
            A CPU tensor whose storage is a shared mapping of a file in `root`.
        """
        numel = 1
        for dim in shape:
            numel *= dim
        num_bytes = numel * torch.empty(0, dtype=dtype).element_size()

        path = os.path.join(self.root, f"{_sanitize(key)}.bin")
        # Map as bytes and bit-cast, so dtypes without a `torch.from_file`
        # mapping (notably the FP8 types) are supported too.
        storage = torch.from_file(path, shared=True, size=num_bytes, dtype=torch.uint8)
        if self.keep_files:
            self._files.append(path)
        else:
            # The mapping outlives the directory entry, so unlinking now
            # reclaims the space even if the process dies without cleanup.
            os.unlink(path)
        self.total_bytes += num_bytes
        return storage.view(dtype).view(shape)

    def cleanup(self) -> None:
        """Drop the backing directory.

        Individual files are already unlinked at allocation time unless
        `keep_files` is set, in which case this is a no-op that only reports
        what was left behind.
        """
        if self.keep_files:
            logger.info(
                "Keeping %s of memory-mapped parameters in %s",
                format_gib(self.total_bytes),
                self.root,
            )
            return
        shutil.rmtree(self.root, ignore_errors=True)
        self.total_bytes = 0
