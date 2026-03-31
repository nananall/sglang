from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.disaggregation.decode_schedule_batch_mixin import (
    ScheduleBatchDisaggregationDecodeMixin,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


class DummyBatch(ScheduleBatchDisaggregationDecodeMixin):
    pass


def _make_req(
    *,
    rid: str,
    req_pool_idx: int,
    origin_input_ids: list[int],
    output_ids: list[int],
    prefix_indices: list[int],
    kv_committed_len: int,
):
    req = SimpleNamespace()
    req.rid = rid
    req.req_pool_idx = req_pool_idx
    req.origin_input_ids = origin_input_ids
    req.output_ids = output_ids
    req.prefix_indices = prefix_indices
    req.kv_committed_len = kv_committed_len
    req.retracted_stain = False
    req.cached_tokens = 0
    req.already_computed = 0
    req.is_retracted = True
    req.extend_logprob_start_len = -1
    req.top_logprobs_num = 0
    req.token_ids_logprob = None
    req.multimodal_inputs = None

    def set_extend_input_len(v: int) -> None:
        req.extend_input_len = v

    req.set_extend_input_len = set_extend_input_len
    return req


def test_prepare_for_prebuilt_uses_only_committed_kv_range():
    batch = DummyBatch()
    batch.device = "cpu"
    batch.reqs = [
        _make_req(
            rid="r0",
            req_pool_idx=0,
            origin_input_ids=[1, 2, 3, 4],
            output_ids=[9],
            prefix_indices=[10, 11],
            kv_committed_len=4,
        )
    ]
    batch.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.tensor([[100, 101, 102, 103, 104]], dtype=torch.int64)
    )
    batch.return_logprob = False
    batch.model_config = SimpleNamespace(vocab_size=32000)

    with patch(
        "sglang.srt.disaggregation.decode_schedule_batch_mixin.SamplingBatchInfo.from_schedule_batch",
        return_value=SimpleNamespace(),
    ):
        batch.prepare_for_prebuilt()

    req = batch.reqs[0]
    assert batch.forward_mode == ForwardMode.PREBUILT
    assert req.fill_ids == [1, 2, 3, 4]
    assert req.extend_input_len == 2
    assert batch.input_ids.tolist() == [3, 4]
    assert batch.seq_lens.tolist() == [4]
    assert batch.seq_lens_cpu.tolist() == [4]
    assert batch.out_cache_loc.tolist() == [102, 103]
    assert req.cached_tokens == 2
    assert req.already_computed == 4
    assert req.is_retracted is False
