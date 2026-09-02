# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model parameter offloading infrastructure."""

from vllm.model_executor.offloader.base import (
    BaseOffloader,
    CompositeOffloader,
    NoopOffloader,
    create_offloader,
    get_offloader,
    set_offloader,
    should_pin_memory,
)
from vllm.model_executor.offloader.disk import DiskOffloader
from vllm.model_executor.offloader.layer_selection import parse_layer_spec
from vllm.model_executor.offloader.prefetch import PrefetchOffloader
from vllm.model_executor.offloader.uva import UVAOffloader

__all__ = [
    "BaseOffloader",
    "CompositeOffloader",
    "DiskOffloader",
    "NoopOffloader",
    "UVAOffloader",
    "PrefetchOffloader",
    "create_offloader",
    "get_offloader",
    "parse_layer_spec",
    "set_offloader",
    "should_pin_memory",
]
