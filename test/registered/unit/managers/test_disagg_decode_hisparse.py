import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    SchedulerDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.mem_cache.hisparse_memory_pool import HiSparseNSATokenToKVPool
from sglang.srt.managers.scheduler_output_processor_mixin import (
    SchedulerOutputProcessorMixin,
)
from sglang.srt.managers.scheduler_runtime_checker_mixin import (
    SchedulerRuntimeCheckerMixin,
)


class TestDisaggDecodeHiSparse(unittest.TestCase):
    def test_prepare_for_prebuilt_populates_req_pool_indices_cpu(self):
        req = SimpleNamespace(
            req_pool_idx=7,
            fill_ids=[101, 102, 103, 104],
            prefix_indices=[11, 12, 13],
            extend_input_len=1,
            origin_input_ids=[101, 102, 103, 104],
            output_ids=[999],
            retracted_stain=False,
            cached_tokens=0,
            already_computed=0,
            is_retracted=False,
            extend_logprob_start_len=0,
            multimodal_inputs=None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            device="cpu",
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor([[555]], dtype=torch.int64)
            ),
            return_logprob=False,
            model_config=SimpleNamespace(vocab_size=32000),
            tree_cache=object(),
            spec_algorithm=SimpleNamespace(
                is_eagle=MagicMock(return_value=False),
            ),
            enable_overlap=False,
        )

        with patch(
            "sglang.srt.disaggregation.decode_schedule_batch_mixin.SamplingBatchInfo.from_schedule_batch",
            return_value=MagicMock(),
        ):
            ScheduleBatchDisaggregationDecodeMixin.prepare_for_prebuilt(batch)

        self.assertTrue(torch.equal(batch.req_pool_indices, torch.tensor([7])))
        self.assertTrue(torch.equal(batch.req_pool_indices_cpu, torch.tensor([7])))

    def test_get_new_prebuilt_batch_attaches_hisparse_coordinator(self):
        req = SimpleNamespace(init_next_round_input=MagicMock())
        new_batch = SimpleNamespace(
            prepare_for_prebuilt=MagicMock(),
            process_prebuilt=MagicMock(),
            hisparse_coordinator=None,
        )
        hisparse = object()
        scheduler = SimpleNamespace(
            grammar_manager=SimpleNamespace(
                has_waiting_grammars=MagicMock(return_value=False)
            ),
            waiting_queue=[req],
            running_batch=SimpleNamespace(batch_size=MagicMock(return_value=0)),
            req_to_token_pool=SimpleNamespace(size=8),
            max_running_requests=8,
            tree_cache=object(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            enable_hisparse=True,
            hisparse_coordinator=hisparse,
            server_args=object(),
            future_map=object(),
        )

        with patch(
            "sglang.srt.disaggregation.decode.ScheduleBatch.init_new",
            return_value=new_batch,
        ):
            ret = SchedulerDisaggregationDecodeMixin.get_new_prebuilt_batch(scheduler)

        req.init_next_round_input.assert_called_once_with(scheduler.tree_cache)
        self.assertIs(ret, new_batch)
        self.assertIs(new_batch.hisparse_coordinator, hisparse)
        new_batch.prepare_for_prebuilt.assert_called_once_with()
        new_batch.process_prebuilt.assert_called_once_with(
            scheduler.server_args, scheduler.future_map
        )

    def test_get_next_disagg_decode_batch_to_run_preserves_hisparse_coordinator(self):
        hisparse = object()
        new_prebuilt_batch = SimpleNamespace(
            filter_batch=MagicMock(),
            is_empty=MagicMock(return_value=False),
            hisparse_coordinator=None,
        )
        running_batch = SimpleNamespace(
            is_empty=MagicMock(return_value=False),
            merge_batch=MagicMock(),
            hisparse_coordinator=None,
        )
        scheduler = SimpleNamespace(
            get_new_prebuilt_batch=MagicMock(return_value=new_prebuilt_batch),
            chunked_req=None,
            process_batch_result_prebuilt=MagicMock(),
            running_batch=running_batch,
            enable_hisparse=True,
            hisparse_coordinator=hisparse,
            update_running_batch=MagicMock(return_value=running_batch),
            maybe_prepare_mlp_sync_batch=MagicMock(side_effect=lambda batch: batch),
        )

        with patch("sglang.srt.disaggregation.decode.set_schedule_time_batch"):
            ret = (
                SchedulerDisaggregationDecodeMixin.get_next_disagg_decode_batch_to_run(
                    scheduler
                )
            )

        scheduler.process_batch_result_prebuilt.assert_called_once_with(
            new_prebuilt_batch
        )
        running_batch.merge_batch.assert_called_once_with(new_prebuilt_batch)
        self.assertIs(running_batch.hisparse_coordinator, hisparse)
        self.assertIs(ret, running_batch)

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

    def test_idle_self_check_skips_while_hisparse_staging_is_ongoing(self):
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            waiting_queue=[],
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            disagg_decode_prealloc_queue=SimpleNamespace(queue=[]),
            server_args=SimpleNamespace(
                disaggregation_decode_enable_offload_kvcache=False
            ),
            enable_hisparse=True,
            hisparse_coordinator=SimpleNamespace(
                has_ongoing_staging=MagicMock(return_value=True)
            ),
            check_memory=MagicMock(),
            check_tree_cache=MagicMock(),
            maybe_sleep_on_idle=MagicMock(),
            new_token_ratio=None,
            init_new_token_ratio=0.42,
        )

        SchedulerRuntimeCheckerMixin.self_check_during_idle(scheduler)

        scheduler.hisparse_coordinator.has_ongoing_staging.assert_called_once_with()
        scheduler.check_memory.assert_not_called()
        scheduler.check_tree_cache.assert_not_called()
        scheduler.maybe_sleep_on_idle.assert_not_called()

    def test_release_disagg_decode_req_uses_hisparse_request_finished(self):
        req = SimpleNamespace(staging=False)
        hisparse = SimpleNamespace(
            request_finished=MagicMock(),
            abort_staging_request=MagicMock(),
        )
        scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=hisparse,
            tree_cache=object(),
        )

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as mock_release:
            SchedulerOutputProcessorMixin._release_disagg_decode_req(scheduler, req)

        hisparse.request_finished.assert_called_once_with(req)
        hisparse.abort_staging_request.assert_not_called()
        mock_release.assert_called_once_with(req, scheduler.tree_cache, is_insert=True)

    def test_release_disagg_decode_req_aborts_hisparse_staging_req(self):
        req = SimpleNamespace(staging=True)
        hisparse = SimpleNamespace(
            request_finished=MagicMock(),
            abort_staging_request=MagicMock(),
        )
        scheduler = SimpleNamespace(
            enable_hisparse=True,
            hisparse_coordinator=hisparse,
            tree_cache=object(),
        )

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as mock_release:
            SchedulerOutputProcessorMixin._release_disagg_decode_req(scheduler, req)

        hisparse.abort_staging_request.assert_called_once_with(req)
        hisparse.request_finished.assert_not_called()
        mock_release.assert_called_once_with(req, scheduler.tree_cache, is_insert=True)

    def test_process_batch_result_prebuilt_releases_hisparse_state(self):
        req = SimpleNamespace(
            staging=False,
            time_stats=SimpleNamespace(
                set_decode_prebuilt_finish_time=MagicMock(),
                set_quick_finish_time=MagicMock(),
            ),
            check_finished=MagicMock(),
            finished=MagicMock(return_value=True),
        )
        hisparse = SimpleNamespace(
            request_finished=MagicMock(),
            abort_staging_request=MagicMock(),
        )
        scheduler = SimpleNamespace(
            disaggregation_mode=DisaggregationMode.DECODE,
            enable_hisparse=True,
            hisparse_coordinator=hisparse,
            tree_cache=object(),
            stream_output=MagicMock(),
        )
        batch = SimpleNamespace(reqs=[req], return_logprob=False)

        with patch(
            "sglang.srt.managers.scheduler_output_processor_mixin.release_kv_cache"
        ) as mock_release:
            SchedulerOutputProcessorMixin.process_batch_result_prebuilt(
                scheduler, batch
            )

        req.time_stats.set_decode_prebuilt_finish_time.assert_called_once_with()
        req.check_finished.assert_called_once_with()
        req.time_stats.set_quick_finish_time.assert_called_once_with()
        hisparse.request_finished.assert_called_once_with(req)
        hisparse.abort_staging_request.assert_not_called()
        mock_release.assert_called_once_with(req, scheduler.tree_cache, is_insert=True)
        scheduler.stream_output.assert_called_once_with(batch.reqs, batch.return_logprob)
