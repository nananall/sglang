import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.disaggregation.decode import SchedulerDisaggregationDecodeMixin


class TestDisaggDecodeHiSparse(unittest.TestCase):
    def test_process_decode_queue_stages_transferred_reqs_for_hisparse(self):
        transferred_req = SimpleNamespace(rid="req-1")
        ready_req = SimpleNamespace(rid="req-ready")
        hisparse = SimpleNamespace(
            collect_ready_reqs=MagicMock(side_effect=[[], [ready_req]]),
            admit_request_into_staging=MagicMock(),
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

        hisparse.admit_request_into_staging.assert_called_once_with(transferred_req)
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
