import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseAct, HiSparseCoordinator
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestHiSparseCoordinator(unittest.TestCase):
    def _make_coordinator(self):
        coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
        mapping = torch.zeros(64, dtype=torch.int64)
        coordinator.top_k = 4
        coordinator.device_buffer_size = 4
        coordinator.padded_buffer_size = 5
        coordinator.top_k_device_locs_buffer = torch.full(
            (2, 4), -1, dtype=torch.int32
        )
        coordinator.req_device_buffer_tokens = torch.full(
            (1, 2, 4), -1, dtype=torch.int32
        )
        coordinator.req_to_host_pool = torch.full((2, 16), -1, dtype=torch.int64)
        coordinator.req_device_buffer_token_locs = torch.full(
            (1, 2, 5), -1, dtype=torch.int32
        )
        coordinator.req_to_device_buffer = torch.zeros((2, 5), dtype=torch.int64)
        coordinator.req_device_buffer_size = torch.zeros(2, dtype=torch.int64)
        coordinator.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.full((2, 16), -1, dtype=torch.int64)
        )
        coordinator.mem_pool_host = SimpleNamespace(
            kv_buffer=torch.zeros((1, 16, 1, 4), dtype=torch.float32),
            token_stride_size=1,
            kv_cache_dim=4,
        )
        coordinator.mem_pool_device = SimpleNamespace(
            kv_buffer=[torch.zeros((1,), dtype=torch.uint8)],
            page_size=1,
            layer_num=1,
            kv_cache_dim=4,
            full_to_hisparse_device_index_mapping=mapping,
        )
        coordinator.lru_slots = torch.zeros((1, 2, 4), dtype=torch.int16)
        coordinator._lru_init = torch.zeros(4, dtype=torch.int16)
        coordinator.num_real_reqs = torch.zeros(1, dtype=torch.int32)
        coordinator.device = "cpu"
        coordinator.write_staging_stream = SimpleNamespace(synchronize=MagicMock())
        coordinator.decode_backup_stream = SimpleNamespace(wait_stream=MagicMock())
        coordinator.decode_producer_stream = None
        coordinator.pending_decode_backup_event = None
        coordinator._skip_first_backup = torch.zeros(2, dtype=torch.bool)
        coordinator.ack_staging_queue = []
        coordinator._direct_staging_req_pool_indices = set()
        coordinator.tp_world_size = 1
        coordinator.token_to_kv_pool_allocator = SimpleNamespace(
            hisparse_attn_allocator=SimpleNamespace(alloc=MagicMock()),
            free_hisparse_indices=MagicMock(),
            full_to_hisparse_device_index_mapping=mapping,
        )
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

    def test_collect_ready_reqs_skips_alloc_for_direct_staging(self):
        coordinator = self._make_coordinator()
        req = SimpleNamespace(req_pool_idx=1, staging=True)
        finish_event = MagicMock()
        finish_event.query.return_value = True
        coordinator.ack_staging_queue = [
            HiSparseAct(
                start_event=MagicMock(),
                finish_event=finish_event,
                req=req,
                needs_alloc_device_buffer=False,
            )
        ]
        coordinator.alloc_device_buffer = MagicMock()
        coordinator._direct_staging_req_pool_indices.add(req.req_pool_idx)

        ready_reqs = HiSparseCoordinator.collect_ready_reqs(coordinator)

        coordinator.alloc_device_buffer.assert_not_called()
        self.assertEqual(ready_reqs, [req])
        self.assertFalse(req.staging)
        self.assertTrue(coordinator._skip_first_backup[req.req_pool_idx].item())
        self.assertNotIn(
            req.req_pool_idx, coordinator._direct_staging_req_pool_indices
        )

    def test_admit_request_direct_cleans_up_on_preload_failure(self):
        coordinator = self._make_coordinator()
        req = SimpleNamespace(req_pool_idx=1, kv_allocated_len=2, rid="req-1", staging=False)
        coordinator.req_to_token_pool.req_to_token[1, :2] = torch.tensor([3, 4])
        coordinator.mem_pool_host.alloc = MagicMock(
            return_value=torch.tensor([7, 8], dtype=torch.int64)
        )
        coordinator.mem_pool_host.free = MagicMock()
        coordinator.mem_pool_host.load_to_device_per_layer = MagicMock(
            side_effect=RuntimeError("preload failed")
        )
        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc.return_value = (
            torch.tensor([101, 102], dtype=torch.int64)
        )
        source_host_pool = SimpleNamespace(
            layout="layer_first",
            page_size=1,
            kv_cache_dim=4,
            size=16,
            kv_buffer=torch.ones((1, 16, 1, 4), dtype=torch.float32),
        )

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.Event"
        ) as event_cls, patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.stream",
            return_value=nullcontext(),
        ):
            event_cls.side_effect = [MagicMock(), MagicMock()]
            with self.assertRaisesRegex(RuntimeError, "preload failed"):
                HiSparseCoordinator.admit_request_direct(
                    coordinator, req, source_host_pool
                )

        coordinator.write_staging_stream.synchronize.assert_called_once()
        coordinator.mem_pool_host.free.assert_called_once()
        coordinator.token_to_kv_pool_allocator.free_hisparse_indices.assert_called_once()
        self.assertFalse(req.staging)
        self.assertEqual(int(coordinator.req_device_buffer_size[req.req_pool_idx]), 0)
        self.assertTrue(
            torch.equal(
                coordinator.req_to_host_pool[req.req_pool_idx, :2],
                torch.full((2,), -1, dtype=torch.int64),
            )
        )
        self.assertNotIn(
            req.req_pool_idx, coordinator._direct_staging_req_pool_indices
        )
        self.assertEqual(
            int(coordinator.mem_pool_device.full_to_hisparse_device_index_mapping[3]),
            0,
        )
        self.assertEqual(
            int(coordinator.mem_pool_device.full_to_hisparse_device_index_mapping[4]),
            0,
        )

    def test_admit_request_direct_maps_resident_prompt_tokens(self):
        coordinator = self._make_coordinator()
        req = SimpleNamespace(
            req_pool_idx=1,
            kv_allocated_len=2,
            rid="req-1",
            staging=False,
        )
        coordinator.req_to_token_pool.req_to_token[1, :2] = torch.tensor([3, 4])
        coordinator.mem_pool_host.alloc = MagicMock(
            return_value=torch.tensor([7, 8], dtype=torch.int64)
        )
        coordinator.mem_pool_host.load_to_device_per_layer = MagicMock()
        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc.return_value = (
            torch.tensor([101, 102], dtype=torch.int64)
        )
        source_host_pool = SimpleNamespace(
            layout="layer_first",
            page_size=1,
            kv_cache_dim=4,
            size=16,
            kv_buffer=torch.ones((1, 16, 1, 4), dtype=torch.float32),
        )

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.Event"
        ) as event_cls, patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.stream",
            return_value=nullcontext(),
        ):
            start_event = MagicMock()
            finish_event = MagicMock()
            event_cls.side_effect = [start_event, finish_event]

            HiSparseCoordinator.admit_request_direct(coordinator, req, source_host_pool)

        self.assertEqual(
            int(coordinator.mem_pool_device.full_to_hisparse_device_index_mapping[3]),
            101,
        )
        self.assertEqual(
            int(coordinator.mem_pool_device.full_to_hisparse_device_index_mapping[4]),
            102,
        )
        self.assertEqual(len(coordinator.ack_staging_queue), 1)

    def test_admit_request_direct_copies_transferred_prompt_into_host_pool(self):
        coordinator = self._make_coordinator()
        req = SimpleNamespace(
            req_pool_idx=1,
            kv_allocated_len=2,
            rid="req-1",
            staging=False,
        )
        coordinator.req_to_token_pool.req_to_token[1, :2] = torch.tensor([3, 4])
        coordinator.mem_pool_host.alloc = MagicMock(
            return_value=torch.tensor([7, 8], dtype=torch.int64)
        )
        coordinator.mem_pool_host.load_to_device_per_layer = MagicMock()
        coordinator.token_to_kv_pool_allocator.hisparse_attn_allocator.alloc.return_value = (
            torch.tensor([101, 102], dtype=torch.int64)
        )
        source_host_pool = SimpleNamespace(
            layout="layer_first",
            page_size=1,
            kv_cache_dim=4,
            size=16,
            kv_buffer=torch.zeros((1, 16, 1, 4), dtype=torch.float32),
        )
        source_host_pool.kv_buffer[0, 3, 0] = torch.tensor([3.0, 30.0, 300.0, 3000.0])
        source_host_pool.kv_buffer[0, 4, 0] = torch.tensor([4.0, 40.0, 400.0, 4000.0])

        with patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.Event"
        ) as event_cls, patch(
            "sglang.srt.managers.hisparse_coordinator.device_module.stream",
            return_value=nullcontext(),
        ):
            start_event = MagicMock()
            finish_event = MagicMock()
            event_cls.side_effect = [start_event, finish_event]

            HiSparseCoordinator.admit_request_direct(coordinator, req, source_host_pool)

        torch.testing.assert_close(
            coordinator.mem_pool_host.kv_buffer[0, 7, 0],
            torch.tensor([3.0, 30.0, 300.0, 3000.0]),
        )
        torch.testing.assert_close(
            coordinator.mem_pool_host.kv_buffer[0, 8, 0],
            torch.tensor([4.0, 40.0, 400.0, 4000.0]),
        )
