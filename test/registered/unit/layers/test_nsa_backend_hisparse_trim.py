import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend


class TestNativeSparseAttnBackendHiSparseTrim(unittest.TestCase):
    def _make_backend(self, decode_impl: str) -> NativeSparseAttnBackend:
        backend = NativeSparseAttnBackend.__new__(NativeSparseAttnBackend)
        backend.nsa_decode_impl = decode_impl
        return backend

    def test_trim_hisparse_decode_topk_suffix_buckets_shared_invalid_tail(self):
        backend = self._make_backend("fa3")
        forward_batch = SimpleNamespace(hisparse_coordinator=object())
        topk_indices = torch.full((2, 2048), -1, dtype=torch.int32)
        topk_indices[0, :1401] = torch.arange(1401, dtype=torch.int32)
        topk_indices[1, :1390] = torch.arange(1390, dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.nsa_backend.get_global_server_args",
            return_value=SimpleNamespace(
                disable_cuda_graph=True, disable_piecewise_cuda_graph=True
            ),
        ):
            trimmed = backend._trim_hisparse_decode_topk_suffix(
                topk_indices, forward_batch
            )

        self.assertEqual(trimmed.shape, (2, 1536))
        self.assertTrue(torch.equal(trimmed[:, :1401], topk_indices[:, :1401]))

    def test_trim_hisparse_decode_topk_suffix_keeps_fixed_width_backends(self):
        backend = self._make_backend("flashmla_kv")
        forward_batch = SimpleNamespace(hisparse_coordinator=object())
        topk_indices = torch.tensor(
            [[10, 11, -1, -1], [20, 21, -1, -1]], dtype=torch.int32
        )

        trimmed = backend._trim_hisparse_decode_topk_suffix(
            topk_indices, forward_batch
        )

        self.assertTrue(torch.equal(trimmed, topk_indices))

    def test_trim_hisparse_decode_topk_suffix_skips_when_cuda_graph_enabled(self):
        backend = self._make_backend("fa3")
        forward_batch = SimpleNamespace(hisparse_coordinator=object())
        topk_indices = torch.full((2, 2048), -1, dtype=torch.int32)
        topk_indices[0, :1200] = torch.arange(1200, dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.nsa_backend.get_global_server_args",
            return_value=SimpleNamespace(
                disable_cuda_graph=False, disable_piecewise_cuda_graph=False
            ),
        ):
            trimmed = backend._trim_hisparse_decode_topk_suffix(
                topk_indices, forward_batch
            )

        self.assertTrue(torch.equal(trimmed, topk_indices))

    def test_get_runtime_decode_nsa_k_metadata_clamps_fa3_width(self):
        backend = self._make_backend("fa3")
        backend.nsa_index_topk = 2048
        metadata = SimpleNamespace(
            nsa_cache_seqlens_int32=torch.tensor([2048, 1600], dtype=torch.int32),
            nsa_cu_seqlens_k=torch.tensor([0, 2048, 3648], dtype=torch.int32),
        )
        page_table = torch.full((2, 1536), -1, dtype=torch.int32)

        runtime_cache_seqlens, runtime_cu_seqlens_k = (
            backend._get_runtime_decode_nsa_k_metadata(metadata, page_table)
        )

        self.assertTrue(
            torch.equal(
                runtime_cache_seqlens,
                torch.tensor([1536, 1536], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(
                runtime_cu_seqlens_k,
                torch.tensor([0, 1536, 3072], dtype=torch.int32),
            )
        )
