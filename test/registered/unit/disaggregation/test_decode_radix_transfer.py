"""Unit tests for decode-radix transfer bookkeeping."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

try:
    from sglang.test.ci.ci_register import register_cpu_ci
    from sglang.test.test_utils import CustomTestCase
except Exception:
    # Local lightweight environments may not have the full SGLang test deps.
    def register_cpu_ci(*args, **kwargs):
        pass

    class CustomTestCase(unittest.TestCase):
        pass


register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class _KVPoll:
    Bootstrapping = 0
    WaitingForInput = 1
    Transferring = 2
    Success = 3
    Failed = 4


class _EvictParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _TorchStub(types.ModuleType):
    int64 = "int64"

    def no_grad(self):
        def decorator(fn):
            return fn

        return decorator

    def tensor(self, value, **kwargs):
        return value

    def empty(self, shape, **kwargs):
        return _Tensor(np.empty(shape, dtype=np.int64))


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.int32)

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def astype(self, dtype):
        return self.values.astype(dtype)

    def __getitem__(self, index):
        return _Tensor(self.values[index])

    def __len__(self):
        return len(self.values)

    def __add__(self, other):
        return _Tensor(self.values + other)


class _ReqToToken:
    def __init__(self, rows):
        self.rows = [np.asarray(row, dtype=np.int32) for row in rows]

    def __getitem__(self, index):
        if isinstance(index, tuple):
            row, col = index
            return _Tensor(self.rows[row][col])
        return _Tensor(self.rows[index])


class _MetadataAllocator:
    def available_size(self):
        return 1

    def alloc(self):
        return 7


class _TimeStats:
    def set_bootstrap_done_time(self):
        pass

    def set_wait_queue_entry_time(self):
        pass

    def set_decode_transfer_queue_entry_time(self):
        pass


def _kv_to_page_indices(kv_indices, page_size):
    kv_indices = np.asarray(kv_indices, dtype=np.int32)
    if page_size == 1:
        return kv_indices
    return kv_indices[::page_size] // page_size


def _kv_to_page_num(num_kv_indices, page_size):
    return (num_kv_indices + page_size - 1) // page_size


@contextmanager
def _stub_sglang_imports():
    modules = {
        "torch": _TorchStub("torch"),
        "torch.distributed": types.ModuleType("torch.distributed"),
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.configs": types.ModuleType("sglang.srt.configs"),
        "sglang.srt.configs.mamba_utils": types.ModuleType(
            "sglang.srt.configs.mamba_utils"
        ),
        "sglang.srt.constants": types.ModuleType("sglang.srt.constants"),
        "sglang.srt.disaggregation": types.ModuleType("sglang.srt.disaggregation"),
        "sglang.srt.disaggregation.base": types.ModuleType(
            "sglang.srt.disaggregation.base"
        ),
        "sglang.srt.disaggregation.common": types.ModuleType(
            "sglang.srt.disaggregation.common"
        ),
        "sglang.srt.disaggregation.common.conn": types.ModuleType(
            "sglang.srt.disaggregation.common.conn"
        ),
        "sglang.srt.disaggregation.utils": types.ModuleType(
            "sglang.srt.disaggregation.utils"
        ),
        "sglang.srt.environ": types.ModuleType("sglang.srt.environ"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.dp_attention": types.ModuleType(
            "sglang.srt.layers.dp_attention"
        ),
        "sglang.srt.managers": types.ModuleType("sglang.srt.managers"),
        "sglang.srt.managers.schedule_batch": types.ModuleType(
            "sglang.srt.managers.schedule_batch"
        ),
        "sglang.srt.managers.utils": types.ModuleType("sglang.srt.managers.utils"),
        "sglang.srt.mem_cache": types.ModuleType("sglang.srt.mem_cache"),
        "sglang.srt.mem_cache.allocator": types.ModuleType(
            "sglang.srt.mem_cache.allocator"
        ),
        "sglang.srt.mem_cache.base_prefix_cache": types.ModuleType(
            "sglang.srt.mem_cache.base_prefix_cache"
        ),
        "sglang.srt.mem_cache.common": types.ModuleType("sglang.srt.mem_cache.common"),
        "sglang.srt.mem_cache.memory_pool": types.ModuleType(
            "sglang.srt.mem_cache.memory_pool"
        ),
        "sglang.srt.mem_cache.swa_memory_pool": types.ModuleType(
            "sglang.srt.mem_cache.swa_memory_pool"
        ),
        "sglang.srt.observability": types.ModuleType("sglang.srt.observability"),
        "sglang.srt.observability.req_time_stats": types.ModuleType(
            "sglang.srt.observability.req_time_stats"
        ),
        "sglang.srt.utils": types.ModuleType("sglang.srt.utils"),
        "sglang.srt.utils.network": types.ModuleType("sglang.srt.utils.network"),
        "sglang.srt.utils.torch_memory_saver_adapter": types.ModuleType(
            "sglang.srt.utils.torch_memory_saver_adapter"
        ),
    }

    modules["torch.distributed"].ProcessGroup = object
    modules["sglang.srt.configs.mamba_utils"].Mamba2CacheParams = object
    modules["sglang.srt.constants"].GPU_MEMORY_TYPE_KV_CACHE = "kv"
    modules["sglang.srt.disaggregation.base"].KVPoll = _KVPoll
    modules["sglang.srt.disaggregation.common.conn"].CommonKVManager = object
    modules["sglang.srt.disaggregation.common.conn"].CommonKVReceiver = object

    disagg_utils = modules["sglang.srt.disaggregation.utils"]
    disagg_utils.FAKE_BOOTSTRAP_HOST = "fake"
    disagg_utils.DisaggregationMode = SimpleNamespace(PREFILL="prefill", DECODE="decode")
    disagg_utils.KVClassType = SimpleNamespace()
    disagg_utils.MetadataBuffers = object
    disagg_utils.ReqToMetadataIdxAllocator = object
    disagg_utils.TransferBackend = SimpleNamespace(FAKE="fake")
    disagg_utils.get_kv_class = MagicMock()
    disagg_utils.is_mla_backend = MagicMock(return_value=False)
    disagg_utils.poll_and_all_reduce = MagicMock()
    disagg_utils.poll_and_all_reduce_with_staging = MagicMock()
    disagg_utils.poll_and_all_reduce_attn_cp_tp_group = MagicMock()
    disagg_utils.prepare_abort = MagicMock()

    class _EnvValue:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    modules["sglang.srt.environ"].envs = SimpleNamespace(
        SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=_EnvValue(1024),
        SGLANG_DISAGG_RADIX_DEBUG=_EnvValue(False),
        SGLANG_DISAGG_STAGING_BUFFER=_EnvValue(False),
    )

    modules["sglang.srt.layers.dp_attention"].get_attention_tp_size = MagicMock(
        return_value=1
    )

    schedule_batch = modules["sglang.srt.managers.schedule_batch"]
    schedule_batch.FINISH_ABORT = type("FINISH_ABORT", (), {})
    schedule_batch.FINISH_LENGTH = type("FINISH_LENGTH", (), {})
    schedule_batch.Req = object
    schedule_batch.ScheduleBatch = object
    modules["sglang.srt.managers.utils"].GenerationBatchResult = object

    modules["sglang.srt.mem_cache.allocator"].BaseTokenToKVPoolAllocator = object
    base_prefix_cache = modules["sglang.srt.mem_cache.base_prefix_cache"]
    base_prefix_cache.BasePrefixCache = object
    base_prefix_cache.EvictParams = _EvictParams
    base_prefix_cache.MatchPrefixParams = object

    mem_common = modules["sglang.srt.mem_cache.common"]
    mem_common.kv_to_page_indices = _kv_to_page_indices
    mem_common.kv_to_page_num = _kv_to_page_num
    mem_common.maybe_cache_unfinished_req = MagicMock()
    mem_common.page_align_floor = lambda length, page_size: (length // page_size) * page_size
    mem_common.release_kv_cache = MagicMock()

    memory_pool = modules["sglang.srt.mem_cache.memory_pool"]
    memory_pool.HybridLinearKVPool = type("HybridLinearKVPool", (), {})
    memory_pool.HybridReqToTokenPool = object
    memory_pool.KVCache = object
    memory_pool.NSATokenToKVPool = type("NSATokenToKVPool", (), {})
    memory_pool.ReqToTokenPool = object
    modules["sglang.srt.mem_cache.swa_memory_pool"].SWAKVPool = type("SWAKVPool", (), {})

    req_time_stats = modules["sglang.srt.observability.req_time_stats"]
    req_time_stats.set_schedule_time_batch = MagicMock()
    req_time_stats.set_time_batch = MagicMock()

    modules["sglang.srt.utils"].get_num_new_pages = MagicMock(return_value=1)
    modules["sglang.srt.utils.network"].NetworkAddress = object
    modules[
        "sglang.srt.utils.torch_memory_saver_adapter"
    ].TorchMemorySaverAdapter = object

    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def _load_module(relative_path, module_name):
    path = Path(__file__).resolve().parents[4] / relative_path
    with _stub_sglang_imports():
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        sys.modules.pop(module_name, None)
    return module


class TestDecodeRadixTransfer(CustomTestCase):
    def test_prefill_initializes_sender_for_delta_pages_after_decode_prefix(self):
        prefill = _load_module(
            "python/sglang/srt/disaggregation/prefill.py", "prefill_under_test"
        )
        prefill.poll_and_all_reduce_attn_cp_tp_group = MagicMock(
            return_value=[_KVPoll.WaitingForInput]
        )

        sender = SimpleNamespace(
            init=MagicMock(),
        )
        req = SimpleNamespace(
            rid="req-1",
            origin_input_ids=list(range(150)),
            bootstrap_room=11,
            disagg_kv_sender=sender,
            metadata_buffer_index=-1,
            time_stats=_TimeStats(),
        )
        queue = object.__new__(prefill.PrefillBootstrapQueue)
        queue.queue = [req]
        queue.scheduler = SimpleNamespace(
            attn_cp_cpu_group=None,
            attn_tp_cpu_group=None,
            enable_metrics=False,
            enable_hicache_storage=False,
        )
        queue.req_to_metadata_buffer_idx_allocator = _MetadataAllocator()
        queue.token_to_kv_pool = SimpleNamespace(page_size=64)
        queue.kv_manager = SimpleNamespace(
            transfer_infos={
                11: {"decode-rank": SimpleNamespace(decode_prefix_len=128)}
            }
        )
        queue.tp_rank = 0

        bootstrapped = queue.pop_bootstrapped()

        self.assertEqual(bootstrapped, [req])
        self.assertEqual(req.start_send_idx, 128)
        sender.init.assert_called_once_with(1, 7)

    def test_prefill_sends_empty_last_chunk_for_mooncake_completion_signal(self):
        prefill = _load_module(
            "python/sglang/srt/disaggregation/prefill.py", "prefill_under_test"
        )

        sender = SimpleNamespace(
            should_send_kv_chunk=MagicMock(return_value=True),
            send=MagicMock(),
        )
        req = SimpleNamespace(
            rid="req-2",
            bootstrap_room=12,
            start_send_idx=128,
            fill_ids=list(range(128)),
            origin_input_ids=list(range(128)),
            req_pool_idx=0,
            disagg_kv_sender=sender,
        )
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(
                page_size=64,
                get_kvcache=MagicMock(return_value=object()),
            ),
            req_to_token_pool=SimpleNamespace(
                req_to_token=_ReqToToken([np.arange(128, dtype=np.int32)])
            ),
            disagg_metadata_buffers=SimpleNamespace(set_buf=MagicMock()),
        )

        prefill.SchedulerDisaggregationPrefillMixin.send_kv_chunk(
            scheduler, req, last_chunk=True
        )

        sender.should_send_kv_chunk.assert_called_once_with(0, True)
        sender.send.assert_called_once()
        np.testing.assert_array_equal(sender.send.call_args.args[0], np.array([], dtype=np.int32))

    def test_decode_sends_only_delta_page_indices_to_prefill(self):
        decode = _load_module(
            "python/sglang/srt/disaggregation/decode.py", "decode_under_test"
        )

        receiver = SimpleNamespace(send_metadata=MagicMock())
        req = SimpleNamespace(
            rid="req-3",
            origin_input_ids=list(range(150)),
            output_ids=[],
            finished_reason=None,
            sampling_params=SimpleNamespace(max_new_tokens=16),
            req_pool_idx=0,
            time_stats=_TimeStats(),
        )
        decode_req = decode.DecodeRequest(req=req, kv_receiver=receiver)
        decode_req.waiting_for_input = True

        queue = object.__new__(decode.DecodePreallocQueue)
        queue.queue = [decode_req]
        queue.pending_reqs = []
        queue.retracted_queue = []
        queue.num_reserved_decode_tokens = 0
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
            enable_hisparse=False,
            running_batch=SimpleNamespace(reqs=[]),
            waiting_queue=[],
            last_batch=None,
        )
        queue.req_to_token_pool = SimpleNamespace(
            available_size=MagicMock(return_value=1),
            req_to_token=_ReqToToken([np.arange(150, dtype=np.int32)]),
        )
        queue.req_to_metadata_buffer_idx_allocator = _MetadataAllocator()
        queue.token_to_kv_pool_allocator = SimpleNamespace(page_size=64)
        queue.token_to_kv_pool = object()
        queue.tree_cache = SimpleNamespace(dec_lock_ref=MagicMock())
        queue.transfer_queue = SimpleNamespace(queue=[], enable_staging=False)
        queue._resolve_pending_reqs = MagicMock()
        queue._update_handshake_waiters = MagicMock()
        queue._match_prefix_and_lock = MagicMock(return_value=(_Tensor(np.arange(128)), 128))
        queue._required_alloc_tokens = MagicMock(return_value=22)
        queue._allocatable_tokens = MagicMock(return_value=1000)
        queue._pre_alloc = MagicMock(return_value=_Tensor(np.arange(22)))

        preallocated, failed = queue.pop_preallocated()

        self.assertEqual(preallocated, [decode_req])
        self.assertEqual(failed, [])
        sent_page_indices = receiver.send_metadata.call_args.args[0]
        np.testing.assert_array_equal(sent_page_indices, np.array([2], dtype=np.int32))

    def test_decode_pre_alloc_evicts_radix_cache_before_delta_allocation(self):
        decode = _load_module(
            "python/sglang/srt/disaggregation/decode.py", "decode_under_test"
        )

        req = SimpleNamespace(
            rid="req-4",
            origin_input_ids=list(range(10)),
            output_ids=[],
            req_pool_idx=0,
            set_extend_input_len=MagicMock(),
        )
        queue = object.__new__(decode.DecodePreallocQueue)
        queue.scheduler = SimpleNamespace(
            server_args=SimpleNamespace(disaggregation_decode_enable_radix_cache=True),
            enable_hisparse=False,
        )
        queue.req_to_token_pool = SimpleNamespace(
            alloc=MagicMock(return_value=[0]),
            write=MagicMock(),
        )
        queue.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=1,
            available_size=MagicMock(side_effect=[0, 10]),
            alloc=MagicMock(return_value=_Tensor(np.arange(10, dtype=np.int32))),
        )
        queue.tree_cache = SimpleNamespace(
            evict=MagicMock(return_value=SimpleNamespace(num_tokens_evicted=10)),
            evictable_size=MagicMock(return_value=10),
            protected_size=MagicMock(return_value=0),
        )

        kv_loc = queue._pre_alloc(req, prefix_len=0)

        self.assertEqual(len(kv_loc), 10)
        queue.tree_cache.evict.assert_called_once()
        self.assertEqual(queue.tree_cache.evict.call_args.args[0].num_tokens, 10)


if __name__ == "__main__":
    unittest.main()
