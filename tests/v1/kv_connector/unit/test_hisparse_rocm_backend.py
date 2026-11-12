# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm HiSparse backend policy, checked without a device."""

import mmap

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.hisparse import (
    worker as hisparse_worker,
)
from vllm.utils.host_memory import host_registration_ranges
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.hisparse.backend import CUDAHiSparseBackend, ROCmHiSparseBackend
from vllm.v1.hisparse.runtime import PagedCacheView

PAGE = mmap.PAGESIZE


def test_rocm_platform_selects_rocm_backend():
    from vllm.platforms.rocm import RocmPlatform

    backend_cls = resolve_obj_by_qualname(RocmPlatform.get_hisparse_backend_cls())
    assert backend_cls is ROCmHiSparseBackend


def test_rocm_backend_policy_differs_from_cuda_only_where_hip_needs_it():
    rocm, cuda = ROCmHiSparseBackend, CUDAHiSparseBackend
    assert rocm.supports_shared_host_pool and cuda.supports_shared_host_pool
    assert not rocm.supports_ipc_events and cuda.supports_ipc_events
    assert rocm.register_whole_host_tensors and not cuda.register_whole_host_tensors
    assert rocm.pin_dma_descriptors and not cuda.pin_dma_descriptors
    assert rocm.host_register_chunk_bytes < cuda.host_register_chunk_bytes
    with pytest.raises(NotImplementedError):
        ROCmHiSparseBackend().create_ipc_event()


def test_rocm_swap_blocks_batch_forces_one_copy_kernel(monkeypatch):
    from vllm.v1.kv_offload.cpu import swap_blocks_triton

    calls = []
    monkeypatch.setattr(
        swap_blocks_triton,
        "swap_blocks_batch",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    src, dst, sizes = (torch.zeros(1, dtype=torch.int64) for _ in range(3))
    ROCmHiSparseBackend().swap_blocks_batch(src, dst, sizes)
    assert len(calls) == 1
    assert calls[0][1]["force_kernel"] is True


def test_rocm_host_dma_view_rejects_unaligned_rows():
    host = torch.zeros(4, 3, dtype=torch.uint8)
    with pytest.raises(ValueError, match="8-byte aligned"):
        ROCmHiSparseBackend().host_dma_view(host)


def test_without_ipc_events_shared_pool_uses_local_events(monkeypatch):
    monkeypatch.setattr(
        hisparse_worker, "get_hisparse_backend", lambda: ROCmHiSparseBackend()
    )
    events = hisparse_worker._create_hisparse_host_events(
        shared_host_region=object(),  # type: ignore[arg-type]
        is_host_writer=True,
        device=torch.device("cpu"),
    )
    assert len(events) == 2


def test_whole_tensor_registrations_reject_a_split_layer():
    # A legal whole-block seam would still split a layer's device alias.
    with pytest.raises(ValueError, match="must each fit"):
        host_registration_ranges([8 * PAGE], 8, 8 * PAGE, 4 * PAGE, whole_tensors=True)


def test_whole_tensor_registrations_group_unaligned_layers():
    ranges = host_registration_ranges(
        [3 * PAGE // 2] * 4, 2, 6 * PAGE, 4 * PAGE, whole_tensors=True
    )
    assert ranges == ((0, 3 * PAGE), (3 * PAGE, 6 * PAGE))


def test_block_seam_registrations_still_split_layers_without_whole_tensors():
    ranges = host_registration_ranges([8 * PAGE], 8, 8 * PAGE, 4 * PAGE)
    assert ranges == ((0, 4 * PAGE), (4 * PAGE, 8 * PAGE))


def test_paged_cache_view_binds_page_padded_backing():
    """A backing padded to a page leaves a partial trailing block."""
    block_size, row_width, num_blocks = 16, 576, 3
    block_stride = block_size * row_width
    raw = torch.zeros(num_blocks * block_stride + 2048, dtype=torch.uint8)

    view = PagedCacheView.bind(
        raw,
        dtype=torch.uint8,
        row_width=row_width,
        byte_offset=0,
        block_stride=block_stride,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    assert view.attention_cache.shape == (num_blocks, block_size, row_width)
