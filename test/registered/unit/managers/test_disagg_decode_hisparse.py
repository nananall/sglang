import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseNSATokenToKVPool


class TestDisaggDecodeHiSparse(unittest.TestCase):
    def test_process_decode_queue_direct_admits_transferred_reqs_for_hisparse(self):
        transferred_req = SimpleNamespace(rid="req-1")
        ready_req = SimpleNamespace(rid="req-ready")
        transfer_host_pool = object()
        hisparse = SimpleNamespace(
            collect_ready_reqs=MagicMock(side_effect=[[], [ready_req]]),
            admit_request_direct=MagicMock(),
        )
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False,
                disaggregation_decode_polling_interval=1,
            ),
            enable_hisparse=True,
            hisparse_coordinator=hisparse,
            waiting_queue=[],
            disagg_decode_prealloc_queue=SimpleNamespace(
                hisparse_transfer_host_pool=transfer_host_pool,
                resume_retracted_reqs=MagicMock(return_value=[]),
                retracted_queue=[],
                pop_preallocated=MagicMock(return_value=([], [])),
            ),
            disagg_decode_transfer_queue=SimpleNamespace(
                extend=MagicMock(),
                pop_transferred=MagicMock(return_value=[transferred_req]),
            ),
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        hisparse.admit_request_direct.assert_called_once_with(
            transferred_req, transfer_host_pool
        )
        self.assertEqual(scheduler.waiting_queue, [ready_req])

    def test_process_decode_queue_keeps_non_hisparse_behavior(self):
        transferred_req = SimpleNamespace(rid="req-1")
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False,
                disaggregation_decode_polling_interval=1,
            ),
            enable_hisparse=False,
            waiting_queue=[],
            disagg_decode_prealloc_queue=SimpleNamespace(
                resume_retracted_reqs=MagicMock(return_value=[]),
                retracted_queue=[],
                pop_preallocated=MagicMock(return_value=([], [])),
            ),
            disagg_decode_transfer_queue=SimpleNamespace(
                extend=MagicMock(),
                pop_transferred=MagicMock(return_value=[transferred_req]),
            ),
        )

        SchedulerDisaggregationDecodeMixin.process_decode_queue(scheduler)

        self.assertEqual(scheduler.waiting_queue, [transferred_req])

    def test_pre_alloc_uses_logical_only_allocator_for_hisparse(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        hisparse_pool = HiSparseNSATokenToKVPool.__new__(HiSparseNSATokenToKVPool)
        req = SimpleNamespace(
            origin_input_ids=[1, 2, 3],
            output_ids=[],
            req_pool_idx=None,
            kv_allocated_len=0,
            kv_committed_len=0,
            fill_ids=[],
            set_extend_input_len=MagicMock(),
        )
        def _alloc(reqs):
            reqs[0].req_pool_idx = 7
            return [7]
        queue.req_to_token_pool = SimpleNamespace(
            alloc=MagicMock(side_effect=_alloc),
            write=MagicMock(),
        )
        queue.token_to_kv_pool = hisparse_pool
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=64,
            device="cpu",
            alloc_logical_only=MagicMock(return_value=torch.tensor([11, 12, 13])),
        )

        kv_loc = DecodePreallocQueue._pre_alloc(queue, req)

        queue.token_to_kv_pool_allocator.alloc_logical_only.assert_called_once()
        queue.req_to_token_pool.write.assert_called_once()
        self.assertIsNotNone(kv_loc)

    def test_init_kv_manager_uses_transfer_host_pool_buffer_infos(self):
        queue = DecodePreallocQueue.__new__(DecodePreallocQueue)
        captured = {}

        class _DummyArgs:
            pass

        class _DummyManager:
            def __init__(self, args, mode, server_args, is_mla_backend):
                captured["args"] = args
                captured["mode"] = mode
                captured["server_args"] = server_args
                captured["is_mla_backend"] = is_mla_backend

        queue.tp_rank = 0
        queue.pp_rank = 0
        queue.is_mla_backend = True
        queue.scheduler = SimpleNamespace(
            dp_rank=0,
            gpu_id=0,
            server_args=SimpleNamespace(disaggregation_ib_device=None),
        )
        queue.token_to_kv_pool = SimpleNamespace(page_size=64)
        queue.draft_token_to_kv_pool = None
        queue.metadata_buffers = SimpleNamespace(
            get_buf_infos=MagicMock(return_value=([], [], []))
        )
        queue.hisparse_transfer_host_pool = SimpleNamespace(
            get_contiguous_buf_infos=MagicMock(return_value=([11], [22], [33]))
        )

        with patch(
            "sglang.srt.disaggregation.decode.get_attention_tp_size",
            return_value=1,
        ), patch(
            "sglang.srt.disaggregation.decode.get_kv_class",
            side_effect=[_DummyArgs, _DummyManager],
        ):
            DecodePreallocQueue._init_kv_manager(queue)

        queue.hisparse_transfer_host_pool.get_contiguous_buf_infos.assert_called_once()
        self.assertEqual(captured["args"].kv_data_ptrs, [11])
        self.assertEqual(captured["args"].kv_data_lens, [22])
        self.assertEqual(captured["args"].kv_item_lens, [33])
        self.assertEqual(captured["args"].page_size, 64)
