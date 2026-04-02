from collections import deque
from types import SimpleNamespace

import torch

from sglang.srt.managers.hisparse_coordinator import HiSparseAct, HiSparseCoordinator


class DummyEvent:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self) -> bool:
        return self.ready


class DummyStream:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self) -> None:
        self.sync_calls += 1


class DummyHostPool:
    def __init__(self):
        self.freed = []
        self.backups = []
        self.loads = []

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())

    def alloc(self, size: int) -> torch.Tensor:
        return torch.arange(100, 100 + size, dtype=torch.int64)

    def backup_from_device_all_layer(self, *args, **kwargs) -> None:
        self.backups.append((args, kwargs))

    def load_to_device_per_layer(self, *args, **kwargs) -> None:
        self.loads.append((args, kwargs))


def test_abort_staging_request_preserves_deque_and_frees_host_pool():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    req = SimpleNamespace(req_pool_idx=0, kv_allocated_len=3, staging=True)
    other_req = SimpleNamespace(req_pool_idx=1, kv_allocated_len=1, staging=True)
    coordinator.ack_staging_queue = deque(
        [
            HiSparseAct(None, DummyEvent(True), req),
            HiSparseAct(None, DummyEvent(False), other_req),
        ]
    )
    coordinator.write_staging_stream = DummyStream()
    coordinator.req_to_host_pool = torch.tensor(
        [[11, 12, -1], [21, -1, -1]], dtype=torch.int64
    )
    coordinator.mem_pool_host = DummyHostPool()
    coordinator._skip_first_backup = [True, True]
    coordinator._nsa_k_only_warmup_steps = [0, 0]
    coordinator._naive_swap_in_steps = [1, 1]

    coordinator.abort_staging_request(req)

    assert isinstance(coordinator.ack_staging_queue, deque)
    assert [act.req for act in coordinator.ack_staging_queue] == [other_req]
    assert coordinator.write_staging_stream.sync_calls == 1
    assert coordinator.mem_pool_host.freed[0].tolist() == [11, 12]
    assert coordinator.req_to_host_pool[0].tolist() == [-1, -1, -1]
    assert coordinator._skip_first_backup[0] is False
    assert coordinator._naive_swap_in_steps[0] == 0
    assert req.staging is False


def test_collect_ready_reqs_pops_ready_prefix_from_deque():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    ready_req = SimpleNamespace(req_pool_idx=0, staging=True)
    pending_req = SimpleNamespace(req_pool_idx=1, staging=True)
    coordinator.ack_staging_queue = deque(
        [
            HiSparseAct(None, DummyEvent(True), ready_req),
            HiSparseAct(None, DummyEvent(False), pending_req),
        ]
    )
    coordinator.tp_world_size = 1
    coordinator.tp_group = None
    coordinator._skip_first_backup = [False, False]
    allocated = []

    def alloc_device_buffer(req):
        allocated.append(req.req_pool_idx)

    coordinator.alloc_device_buffer = alloc_device_buffer

    ready_reqs = coordinator.collect_ready_reqs()

    assert ready_reqs == [ready_req]
    assert allocated == [0]
    assert ready_req.staging is False
    assert coordinator._skip_first_backup[0] is True
    assert [act.req for act in coordinator.ack_staging_queue] == [pending_req]


def test_direct_admit_keeps_two_step_nsa_warmup_for_long_sequences():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.device = "cpu"
    coordinator.device_buffer_size = 4
    coordinator.top_k = 4
    coordinator.decode_producer_stream = None
    coordinator.mem_pool_device = SimpleNamespace(layer_num=1)
    coordinator.mem_pool_host = DummyHostPool()
    coordinator.req_to_host_pool = torch.tensor(
        [[100, 101, 102, 103, 104, 105, 106, 107] + [-1] * 8], dtype=torch.int64
    )
    coordinator.req_to_device_buffer = torch.arange(16, dtype=torch.int64).view(1, 16)
    coordinator._skip_first_backup = [False]
    coordinator._nsa_k_only_warmup_steps = [0]
    coordinator._naive_swap_in_steps = [0]
    coordinator.req_device_buffer_tokens = torch.zeros((1, 1, 5), dtype=torch.int32)

    def alloc_device_buffer(req):
        coordinator.req_device_buffer_tokens[:, req.req_pool_idx, :4] = torch.arange(
            4, dtype=torch.int32
        )

    coordinator.alloc_device_buffer = alloc_device_buffer

    req = SimpleNamespace(req_pool_idx=0, kv_allocated_len=8, rid="r0", staging=True)
    coordinator.admit_request_direct(req)

    assert req.staging is False
    assert coordinator.should_force_nsa_k_only(torch.tensor([0], dtype=torch.int64))
    assert coordinator._skip_first_backup == [True]
    assert coordinator._nsa_k_only_warmup_steps == [2]
    assert coordinator._naive_swap_in_steps == [2]
    assert coordinator.req_device_buffer_tokens[0, 0, :4].tolist() == [5, 6, 7, -1]
    assert len(coordinator.mem_pool_host.loads) == 1

    seq_lens = torch.tensor([8], dtype=torch.int64)
    req_pool_indices = torch.tensor([0], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([8], dtype=torch.int64)
    req_pool_indices_cpu = torch.tensor([0], dtype=torch.int64)

    coordinator._eager_backup_previous_token(
        seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
    )
    assert coordinator._skip_first_backup == [False]
    assert coordinator._nsa_k_only_warmup_steps == [2]

    coordinator._eager_backup_previous_token(
        seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
    )
    assert coordinator._nsa_k_only_warmup_steps == [2]
    assert coordinator.mem_pool_host.backups

    coordinator.finish_nsa_k_only_warmup(req_pool_indices)
    assert coordinator._nsa_k_only_warmup_steps == [1]
    assert coordinator.should_force_nsa_k_only(req_pool_indices)

    coordinator.finish_nsa_k_only_warmup(req_pool_indices)
    assert coordinator._nsa_k_only_warmup_steps == [0]
    assert not coordinator.should_force_nsa_k_only(req_pool_indices)


def test_swap_in_selected_pages_uses_naive_fallback_for_flagged_req():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.device = "cpu"
    coordinator.top_k = 4
    coordinator._naive_swap_in_steps = [1]

    called = {}

    def fake_naive_load_topk(req_pool_indices, seq_lens, top_k_tokens, layer_id):
        called["args"] = (
            req_pool_indices.clone(),
            seq_lens.clone(),
            top_k_tokens.clone(),
            layer_id,
        )
        return torch.full((1, 4), 7, dtype=torch.int32)

    coordinator.naive_load_topk = fake_naive_load_topk

    result = coordinator.swap_in_selected_pages(
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        top_k_result=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        layer_id=2,
    )

    assert result.tolist() == [[7, 7, 7, 7]]
    req_pool_indices, seq_lens, top_k_tokens, layer_id = called["args"]
    assert req_pool_indices.tolist() == [0]
    assert seq_lens.tolist() == [8]
    assert top_k_tokens.tolist() == [[1, 2, 3, 4]]
    assert layer_id == 2
    assert coordinator._naive_swap_in_steps == [1]


def test_naive_load_topk_tolerates_invalid_or_missing_host_tokens():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.device = "cpu"
    coordinator.top_k = 4
    coordinator.device_buffer_size = 4
    coordinator.req_to_device_buffer = torch.tensor([[10, 11, 12, 13, 99]])
    coordinator.req_to_host_pool = torch.tensor([[30, 31, 32, 33, 34, -1, 36, 37]])
    coordinator.mem_pool_device = SimpleNamespace()
    coordinator.mem_pool_host = DummyHostPool()

    result = coordinator.naive_load_topk(
        req_pool_indices=torch.tensor([0], dtype=torch.int64),
        seq_lens=torch.tensor([8], dtype=torch.int32),
        top_k_tokens=torch.tensor([[7, -1, 8, 5]], dtype=torch.int32),
        layer_id=0,
    )

    assert result.tolist() == [[99, -1, -1, -1]]
    assert coordinator.mem_pool_host.loads == []
