from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.nsa.nsa_indexer import Indexer


class DummyForwardMode:
    def is_decode(self) -> bool:
        return True

    def is_decode_or_idle(self) -> bool:
        return True

    def is_extend_without_speculative(self) -> bool:
        return False


def test_build_decode_k_only_topk_returns_safe_recent_window():
    indexer = Indexer.__new__(Indexer)
    indexer.index_topk = 4

    result = indexer._build_decode_k_only_topk(
        torch.tensor([2, 4, 6], dtype=torch.int32),
        device=torch.device("cpu"),
    )

    assert result.tolist() == [
        [0, 1, -1, -1],
        [0, 1, 2, 3],
        [2, 3, 4, 5],
    ]


def test_forward_cuda_uses_k_only_fallback_for_hisparse_warmup():
    indexer = Indexer.__new__(Indexer)
    indexer.alt_stream = None
    indexer.nsa_enable_prefill_cp = False
    indexer.index_topk = 4

    called = {}

    def fake_forward_cuda_k_only(
        x,
        positions,
        forward_batch,
        layer_id,
        act_quant,
        enable_dual_stream,
        metadata,
        return_indices,
    ):
        called["layer_id"] = layer_id
        called["return_indices"] = return_indices
        called["enable_dual_stream"] = enable_dual_stream
        called["req_pool_indices"] = forward_batch.req_pool_indices.clone()
        return torch.tensor([[0, -1, -1, -1]], dtype=torch.int32)

    indexer._forward_cuda_k_only = fake_forward_cuda_k_only

    forward_batch = SimpleNamespace(
        token_to_kv_pool=SimpleNamespace(),
        attn_backend=SimpleNamespace(
            get_indexer_metadata=lambda layer_id, _: SimpleNamespace()
        ),
        forward_mode=DummyForwardMode(),
        hisparse_coordinator=SimpleNamespace(
            should_force_nsa_k_only=lambda req_pool_indices: True
        ),
        req_pool_indices=torch.tensor([7], dtype=torch.int64),
    )

    result = indexer.forward_cuda(
        x=torch.zeros((1, 16), dtype=torch.bfloat16),
        q_lora=torch.zeros((1, 16), dtype=torch.bfloat16),
        positions=torch.zeros((1,), dtype=torch.int64),
        forward_batch=forward_batch,
        layer_id=3,
        return_indices=True,
    )

    assert result.tolist() == [[0, -1, -1, -1]]
    assert called["layer_id"] == 3
    assert called["return_indices"] is True
    assert called["enable_dual_stream"] is False
    assert torch.equal(called["req_pool_indices"], torch.tensor([7], dtype=torch.int64))
