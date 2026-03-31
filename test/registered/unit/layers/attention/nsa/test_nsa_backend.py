import torch

from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend


def test_sanitize_flashmla_page_table_clamps_invalid_indices():
    backend = NativeSparseAttnBackend.__new__(NativeSparseAttnBackend)
    backend.real_page_size = 64

    kv_cache = torch.empty((64 * 3, 8), dtype=torch.float32)
    page_table_1 = torch.tensor(
        [[-1, 0, 1, 2, 3, 999]], dtype=torch.int32
    )

    result = backend._sanitize_flashmla_page_table(page_table_1, kv_cache)

    assert result.tolist() == [[0, 0, 1, 2, 2, 2]]


def test_sanitize_flashmla_page_table_uses_block_dim_for_flashmla_kv():
    backend = NativeSparseAttnBackend.__new__(NativeSparseAttnBackend)
    backend.real_page_size = 64

    kv_cache = torch.empty((3, 64, 1, 8), dtype=torch.float32)
    page_table_1 = torch.tensor(
        [[-1, 0, 1, 2, 3, 999]], dtype=torch.int32
    )

    result = backend._sanitize_flashmla_page_table(page_table_1, kv_cache)

    assert result.tolist() == [[0, 0, 1, 2, 2, 2]]
