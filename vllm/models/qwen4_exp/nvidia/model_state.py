# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState

from ..common.ple_disk import PLEDiskStager, resolve_ple_disk_config


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

        # Pre-dispatch disk staging (FreeToken's host_fill_batch seam). Built
        # only for the staged sync modes; "forward" keeps the in-forward
        # gather. The model is already loaded here, so the layer stores exist.
        self.ple_disk_stager: PLEDiskStager | None = None
        disk_config = resolve_ple_disk_config(vllm_config)
        if disk_config is not None and disk_config.sync_mode != "forward":
            stager = PLEDiskStager(
                disk_config,
                device=self.device,
                max_num_tokens=self.max_num_tokens,
                max_num_reqs=self.max_num_reqs,
                ngram_context_len=self.ngram_context_len,
            )
            if stager.attach_model(self.model) > 0:
                self.ple_disk_stager = stager

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        num_reqs_padded = input_batch.num_reqs_after_padding
        context = self.ngram_context[:num_reqs_padded]
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc[: num_reqs_padded + 1]
        query_start_loc.copy_(input_batch.query_start_loc[: num_reqs_padded + 1])
        ngram_context = self._prepare_ngram_context(input_batch, req_states)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

        stager = getattr(self, "ple_disk_stager", None)
        if stager is not None:
            # Stage this forward's rows on the host before dispatch; the fill
            # reads the tokens back through pinned buffers, so it needs no
            # scheduler token state. Deferred (post-dispatch) only for FULL
            # graphs under memops.
            stager.prepare(
                input_ids=input_batch.input_ids,
                num_tokens=input_batch.num_tokens_after_padding,
                query_start_loc=input_batch.query_start_loc_np,
                num_reqs=num_reqs_padded,
                ngram_context=ngram_context,
                use_graph=input_batch.cg_mode == CUDAGraphMode.FULL,
            )
        return model_inputs

    def post_dispatch(self) -> None:
        """Run a deferred disk fill after the forward was dispatched."""
        stager = getattr(self, "ple_disk_stager", None)
        if stager is not None:
            stager.post_dispatch()

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc[: num_reqs + 1]
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1:])

        ngram_context = self.ngram_context[:num_reqs]
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )

        stager = getattr(self, "ple_disk_stager", None)
        if stager is not None:
            # Dummy/capture runs still need the staged buffers filled and (for
            # memops) the flag armed, so the captured WAIT/H2D has valid data.
            stager.prepare_dummy(num_reqs, num_tokens)
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
