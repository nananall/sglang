import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestHiSparseCoordinator(unittest.TestCase):
    def _make_coordinator(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        coordinator.top_k = 4
        coordinator.device_buffer_size = 4
        coordinator.top_k_device_locs_buffer = torch.full(
            (2, 4), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.zeros(
            (1, 2, 4), dtype=torch.int32
        )
        coordinator.req_to_host_pool = torch.zeros((2, 16), dtype=torch.int64)
        coordinator.req_device_buffer_token_locs = torch.zeros(
            (1, 2, 4), dtype=torch.int32
        )
        coordinator.mem_pool_host = SimpleNamespace(
            kv_buffer=[torch.zeros((1,), dtype=torch.uint8)],
            token_stride_size=1,
        )
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=[torch.zeros((1,), dtype=torch.uint8)]
        )
        coordinator.lru_slots = torch.zeros((1, 2, 4), dtype=torch.int16)
        coordinator.num_real_reqs = torch.zeros(1, dtype=torch.int32)
        return coordinator

    def test_swap_in_selected_pages_casts_int64_seq_lens_to_int32(self):
        coordinator = self._make_coordinator()
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        seq_lens = torch.tensor([11, 12], dtype=torch.int64)
        top_k_result = torch.tensor(
            [[1, 2, -1, -1], [3, 4, -1, -1]], dtype=torch.int32
        )
        captured = {}

        def _fake_kernel(**kwargs):
            captured["seq_lens_dtype"] = kwargs["seq_lens"].dtype
            captured["seq_lens"] = kwargs["seq_lens"].clone()

        with patch(
            "sglang.srt.managers.hisparse_coordinator.load_cache_to_device_buffer_mla",
            side_effect=_fake_kernel,
        ):
            result = HiSparseCoordinator.swap_in_selected_pages(
                coordinator,
                req_pool_indices,
                seq_lens,
                top_k_result,
                layer_id=0,
            )

        self.assertEqual(captured["seq_lens_dtype"], torch.int32)
        self.assertTrue(
            torch.equal(captured["seq_lens"], torch.tensor([11, 12], dtype=torch.int32))
        )
        self.assertTrue(torch.equal(result, coordinator.top_k_device_locs_buffer[:2]))

    def test_swap_in_selected_pages_rejects_int64_seq_lens_over_int32_range(self):
        coordinator = self._make_coordinator()
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([torch.iinfo(torch.int32).max + 1], dtype=torch.int64)
        top_k_result = torch.tensor([[1, -1, -1, -1]], dtype=torch.int32)

        with self.assertRaisesRegex((RuntimeError, ValueError), "int32 range"):
            HiSparseCoordinator.swap_in_selected_pages(
                coordinator,
                req_pool_indices,
                seq_lens,
                top_k_result,
                layer_id=0,
            )
