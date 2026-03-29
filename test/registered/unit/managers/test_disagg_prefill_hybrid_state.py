import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import kv_to_page_indices
from sglang.srt.mem_cache.memory_pool import NSATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool


class TestDisaggPrefillHybridState(unittest.TestCase):
    def test_send_kv_chunk_uses_prompt_length_for_nsa_state(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            start_send_idx=0,
            fill_ids=[1, 2, 3, 4, 5, 6, 7],
            origin_input_ids=[1, 2, 3, 4, 5, 6],
            disagg_kv_sender=MagicMock(),
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=4,
                get_kvcache=MagicMock(
                    return_value=NSATokenToKVPool.__new__(NSATokenToKVPool)
                ),
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor(
                    [[100, 101, 102, 103, 104, 105, 106]], dtype=torch.int64
                )
            ),
            disagg_metadata_buffers=SimpleNamespace(set_buf=MagicMock()),
        )

        SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            scheduler, req, last_chunk=True
        )

        expected_prompt_indices = np.array([100, 101, 102, 103, 104, 105])
        expected_pages = kv_to_page_indices(expected_prompt_indices, 4)
        sent_pages, sent_state_indices = req.disagg_kv_sender.send.call_args.args

        np.testing.assert_array_equal(sent_pages, expected_pages)
        np.testing.assert_array_equal(sent_state_indices, expected_pages)
        self.assertEqual(req.start_send_idx, len(req.origin_input_ids))

    def test_send_kv_chunk_uses_prompt_length_for_swa_state(self):
        req = SimpleNamespace(
            req_pool_idx=0,
            start_send_idx=0,
            fill_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9],
            origin_input_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            disagg_kv_sender=MagicMock(),
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=4,
                get_kvcache=MagicMock(
                    return_value=SWAKVPool.__new__(SWAKVPool)
                ),
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.tensor(
                    [[100, 101, 102, 103, 104, 105, 106, 107, 108]],
                    dtype=torch.int64,
                )
            ),
            disagg_metadata_buffers=SimpleNamespace(set_buf=MagicMock()),
            sliding_window_size=4,
        )

        SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            scheduler, req, last_chunk=True
        )

        expected_state_indices = kv_to_page_indices(
            np.array([104, 105, 106, 107]), 4
        )
        _, sent_state_indices = req.disagg_kv_sender.send.call_args.args

        np.testing.assert_array_equal(sent_state_indices, expected_state_indices)

