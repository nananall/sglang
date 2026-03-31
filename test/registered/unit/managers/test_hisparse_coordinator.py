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

    def free(self, indices: torch.Tensor) -> None:
        self.freed.append(indices.clone())


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
