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

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())

    def alloc(self, size: int) -> torch.Tensor:
        return torch.arange(100, 100 + size, dtype=torch.int64)

    def backup_from_device_all_layer(self, *args, **kwargs) -> None:
        self.backups.append((args, kwargs))


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

    coordinator.abort_staging_request(req)

    assert isinstance(coordinator.ack_staging_queue, deque)
    assert [act.req for act in coordinator.ack_staging_queue] == [other_req]
    assert coordinator.write_staging_stream.sync_calls == 1
    assert coordinator.mem_pool_host.freed[0].tolist() == [11, 12]
    assert coordinator.req_to_host_pool[0].tolist() == [-1, -1, -1]
    assert coordinator._skip_first_backup[0] is False
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


def test_direct_admit_keeps_nsa_warmup_until_decode_forward_finishes():
    coordinator = HiSparseCoordinator.__new__(HiSparseCoordinator)
    coordinator.device = "cpu"
    coordinator.device_buffer_size = 4
    coordinator.decode_producer_stream = None
    coordinator.mem_pool_device = SimpleNamespace()
    coordinator.mem_pool_host = DummyHostPool()
    coordinator.req_to_host_pool = torch.full((1, 16), -1, dtype=torch.int64)
    coordinator.req_to_device_buffer = torch.arange(16, dtype=torch.int64).view(1, 16)
    coordinator._skip_first_backup = [False]
    coordinator._needs_nsa_k_only_warmup = [False]
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
    assert coordinator._needs_nsa_k_only_warmup == [True]

    seq_lens = torch.tensor([8], dtype=torch.int64)
    req_pool_indices = torch.tensor([0], dtype=torch.int64)
    seq_lens_cpu = torch.tensor([8], dtype=torch.int64)
    req_pool_indices_cpu = torch.tensor([0], dtype=torch.int64)

    coordinator._eager_backup_previous_token(
        seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
    )
    assert coordinator._skip_first_backup == [False]
    assert coordinator._needs_nsa_k_only_warmup == [True]

    coordinator._eager_backup_previous_token(
        seq_lens, req_pool_indices, seq_lens_cpu, req_pool_indices_cpu
    )
    assert coordinator._needs_nsa_k_only_warmup == [True]
    assert coordinator.mem_pool_host.backups

    coordinator.finish_nsa_k_only_warmup(req_pool_indices)
    assert coordinator._needs_nsa_k_only_warmup == [False]
    assert not coordinator.should_force_nsa_k_only(req_pool_indices)
