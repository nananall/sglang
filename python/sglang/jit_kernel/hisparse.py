from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_PTX_STATIC_SMEM_LIMIT_BYTES = 48 * 1024


def _hash_size(num_top_k: int, hot_buffer_size: int) -> int:
    if hot_buffer_size >= 8192 and num_top_k >= 2048:
        return (num_top_k * 3) // 2
    return num_top_k * 2


def _estimate_static_shared_mem_bytes(num_top_k: int, hot_buffer_size: int) -> int:
    num_buffer_chunks = (hot_buffer_size + 31) // 32
    hash_size = _hash_size(num_top_k, hot_buffer_size)
    total = 0
    total += 4 * num_top_k  # s_top_k_tokens
    total += 2 * (num_buffer_chunks + 1)  # s_chunk_offset
    total += 2 * (num_buffer_chunks + 1)  # s_evict_chunk_offset
    total += 2 * hot_buffer_size  # s_lru_slots_out
    total += 4 * hash_size  # s_hash_keys
    total += 2 * hash_size  # s_hash_vals
    total += 12  # s_total_hits / s_newest_hit / s_total_misses
    # Conservatively round up to the next 16-byte boundary to reflect ptxas
    # alignment of static shared allocations.
    return ((total + 15) // 16) * 16


@functools.cache
def _jit_sparse_module(
    item_size_bytes: int,
    block_size: int,
    num_top_k: int,
    hot_buffer_size: int,
    is_mla: bool = False,
) -> Module:
    template_args = make_cpp_args(block_size, num_top_k, hot_buffer_size, is_mla)
    cache_args = make_cpp_args(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla
    )
    return load_jit(
        "sparse_cache",
        *cache_args,
        cuda_files=["hisparse.cuh"],
        cuda_wrappers=[
            (
                "load_cache_to_device_buffer",
                f"load_cache_to_device_buffer<{template_args}>",
            )
        ],
    )


def load_cache_to_device_buffer_mla(
    top_k_tokens: torch.Tensor,
    device_buffer_tokens: torch.Tensor,
    host_cache_locs: torch.Tensor,
    device_buffer_locs: torch.Tensor,
    host_cache: torch.Tensor,
    device_buffer: torch.Tensor,
    top_k_device_locs: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    lru_slots: torch.Tensor,
    item_size_bytes: int,
    num_top_k: int,
    hot_buffer_size: int,
    page_size: int = 1,
    block_size: int = 256,
    num_real_reqs: torch.Tensor | None = None,
) -> None:
    assert (
        hot_buffer_size >= num_top_k
    ), f"hot_buffer_size ({hot_buffer_size}) must be >= num_top_k ({num_top_k})"
    estimated_smem = _estimate_static_shared_mem_bytes(num_top_k, hot_buffer_size)
    if estimated_smem > _PTX_STATIC_SMEM_LIMIT_BYTES:
        raise ValueError(
            "HiSparse kernel configuration exceeds nvcc's static shared-memory "
            f"limit: top_k={num_top_k}, hot_buffer_size={hot_buffer_size}, "
            f"estimated_static_smem={estimated_smem} bytes > "
            f"{_PTX_STATIC_SMEM_LIMIT_BYTES}. "
            "Reduce device_buffer_size/top_k or update the kernel layout."
        )

    module = _jit_sparse_module(
        item_size_bytes, block_size, num_top_k, hot_buffer_size, is_mla=True
    )

    empty = torch.empty(0)

    if num_real_reqs is None:
        num_real_reqs = torch.tensor(
            [top_k_tokens.size(0)], dtype=torch.int32, device=top_k_tokens.device
        )

    module.load_cache_to_device_buffer(
        top_k_tokens,
        device_buffer_tokens,
        host_cache_locs,
        device_buffer_locs,
        host_cache,
        empty,
        device_buffer,
        empty,
        top_k_device_locs,
        req_pool_indices,
        seq_lens,
        lru_slots,
        num_real_reqs,
        page_size,
        item_size_bytes,
    )
