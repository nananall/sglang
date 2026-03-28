import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        coordinator.device = "cpu"
        coordinator.decode_backup_stream = SimpleNamespace(wait_stream=MagicMock())
        coordinator.decode_producer_stream = None
        coordinator.pending_decode_backup_event = None
        coordinator._skip_first_backup = torch.zeros(2, dtype=torch.bool)
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

    def test_swap_in_selected_pages_uses_adaptive_block_size(self):
        coordinator = self._make_coordinator()
        req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        seq_lens = torch.tensor([11, 12], dtype=torch.int32)
        top_k_result = torch.tensor([[1, 2], [3, -1]], dtype=torch.int32)
        captured = {}

        def _fake_kernel(**kwargs):
            captured["block_size"] = kwargs["block_size"]
            captured["num_top_k"] = kwargs["num_top_k"]
            captured["top_k_device_locs_shape"] = tuple(kwargs["top_k_device_locs"].shape)

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

        self.assertEqual(captured["block_size"], 256)
        self.assertEqual(captured["num_top_k"], 2)
        self.assertEqual(captured["top_k_device_locs_shape"], (2, 2))
        self.assertEqual(tuple(result.shape), (2, 2))

    def test_wait_pending_decode_backup_waits_and_clears_event(self):
        coordinator = self._make_coordinator()
        pending_event = object()
        coordinator.pending_decode_backup_event = pending_event
        fake_stream = SimpleNamespace(wait_event=MagicMock())

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.current_stream",
            return_value=fake_stream,
        ):
            HiSparseCoordinator._wait_pending_decode_backup(coordinator)

        fake_stream.wait_event.assert_called_once_with(pending_event)
        self.assertIsNone(coordinator.pending_decode_backup_event)

    def test_maybe_wait_pending_decode_backup_waits_once_when_event_present(self):
        coordinator = self._make_coordinator()
        pending_event = object()
        coordinator.pending_decode_backup_event = pending_event
        fake_stream = SimpleNamespace(wait_event=MagicMock())

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.current_stream",
            return_value=fake_stream,
        ):
            HiSparseCoordinator._maybe_wait_pending_decode_backup(
                coordinator,
                seq_lens=torch.tensor([6], dtype=torch.int32),
                top_k_result=torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
            )

        fake_stream.wait_event.assert_called_once_with(pending_event)
        self.assertIsNone(coordinator.pending_decode_backup_event)

    def test_maybe_wait_pending_decode_backup_skips_when_no_event_is_pending(self):
        coordinator = self._make_coordinator()
        fake_stream = SimpleNamespace(wait_event=MagicMock())

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.current_stream",
            return_value=fake_stream,
        ):
            HiSparseCoordinator._maybe_wait_pending_decode_backup(
                coordinator,
                seq_lens=torch.tensor([6], dtype=torch.int32),
                top_k_result=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
            )

        fake_stream.wait_event.assert_not_called()
        self.assertIsNone(coordinator.pending_decode_backup_event)

    def test_eager_backup_previous_token_launches_async_backup_on_dedicated_stream(self):
        coordinator = self._make_coordinator()
        coordinator.decode_producer_stream = object()
        coordinator.req_to_device_buffer = torch.tensor([[101, 102, 103, 104, 105]], dtype=torch.int64)
        coordinator.req_to_host_pool = torch.full((1, 16), -1, dtype=torch.int64)
        coordinator.mem_pool_host = SimpleNamespace(
            alloc=MagicMock(return_value=torch.tensor([7], dtype=torch.int64)),
            backup_from_device_all_layer=MagicMock(),
        )
        fake_event = MagicMock()
        fake_current_stream = object()

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.stream",
            return_value=nullcontext(),
        ), patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.Event",
            return_value=fake_event,
        ), patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.current_stream",
            return_value=fake_current_stream,
        ):
            HiSparseCoordinator._eager_backup_previous_token(
                coordinator,
                seq_lens=torch.tensor([6], dtype=torch.int64),
                req_pool_indices=torch.tensor([0], dtype=torch.int64),
                seq_lens_cpu=torch.tensor([6], dtype=torch.int64),
                req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
            )

        self.assertEqual(coordinator.decode_backup_stream.wait_stream.call_count, 2)
        coordinator.decode_backup_stream.wait_stream.assert_any_call(fake_current_stream)
        coordinator.decode_backup_stream.wait_stream.assert_any_call(
            coordinator.decode_producer_stream
        )
        coordinator.mem_pool_host.backup_from_device_all_layer.assert_called_once()
        fake_event.record.assert_called_once()
        self.assertIs(coordinator.pending_decode_backup_event, fake_event)

    def test_eager_backup_previous_token_skips_tokens_still_in_hot_buffer(self):
        coordinator = self._make_coordinator()
        coordinator.mem_pool_host = SimpleNamespace(
            alloc=MagicMock(),
            backup_from_device_all_layer=MagicMock(),
        )

        HiSparseCoordinator._eager_backup_previous_token(
            coordinator,
            seq_lens=torch.tensor([5], dtype=torch.int64),
            req_pool_indices=torch.tensor([0], dtype=torch.int64),
            seq_lens_cpu=torch.tensor([5], dtype=torch.int64),
            req_pool_indices_cpu=torch.tensor([0], dtype=torch.int64),
        )

        coordinator.mem_pool_host.alloc.assert_not_called()
        coordinator.mem_pool_host.backup_from_device_all_layer.assert_not_called()
        self.assertIsNone(coordinator.pending_decode_backup_event)
