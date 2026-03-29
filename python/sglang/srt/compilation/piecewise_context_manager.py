from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_in_piecewise_cuda_graph = False
_in_pcg_torch_compile = False
_pcg_capture_stream = None


def is_in_piecewise_cuda_graph():
    return _in_piecewise_cuda_graph


def is_in_pcg_torch_compile():
    return _in_pcg_torch_compile


def get_pcg_capture_stream():
    return _pcg_capture_stream


@contextmanager
def enable_piecewise_cuda_graph_compile():
    global _in_pcg_torch_compile
    _in_pcg_torch_compile = True
    yield
    _in_pcg_torch_compile = False


@contextmanager
def enable_piecewise_cuda_graph():
    global _in_piecewise_cuda_graph
    _in_piecewise_cuda_graph = True
    try:
        yield
    except Exception as e:
        logger.error(
            "Piecewise CUDA Graph failed with error: %s\n%s",
            e,
            PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG,
        )
        raise
    finally:
        _in_piecewise_cuda_graph = False


@contextmanager
def set_pcg_capture_stream(stream: torch.cuda.Stream):
    global _pcg_capture_stream
    _pcg_capture_stream = stream
    yield
    _pcg_capture_stream = None


@dataclass
class ForwardContext:
    def __init__(self):
        self.forward_batch = None
        self.attention_layers = None
        self.quant_config = None
        self.moe_layers = None
        self.moe_fusions = None
        self.num_tokens: Optional[int] = None
        # Layer-wise KV transfer callback: called after each attention layer
        # during disagg prefill extend. Signature: fn(layer_id: int) -> None
        self.disagg_layerwise_fn: Optional[Any] = None

    def set_forward_batch(self, forward_batch: ForwardBatch):
        self.forward_batch = forward_batch

    def set_attention_layers(self, layers: List[Any]):
        self.attention_layers = layers

    def set_quant_config(self, quant_config: Any):
        self.quant_config = quant_config

    def set_moe_layers(self, layers: List[Any]):
        self.moe_layers = layers

    def set_moe_fusions(self, fusions: List[Any]):
        self.moe_fusions = fusions


_forward_context: Optional[ForwardContext] = None
# Layerwise KV transfer callback to be picked up by the next set_forward_context call.
# Set via register_disagg_layerwise_fn() before calling run_batch on a disagg prefill batch.
_pending_disagg_layerwise_fn: Optional[Any] = None


def get_forward_context() -> Optional[ForwardContext]:
    if _forward_context is None:
        return None
    return _forward_context


def register_disagg_layerwise_fn(fn: Optional[Any]) -> None:
    """Register a layer-wise KV transfer callback for the next forward pass.

    Must be called before run_batch() on a disagg prefill extend batch.
    The callback will be set on ForwardContext.disagg_layerwise_fn by
    set_forward_context() and cleared immediately afterwards.
    """
    global _pending_disagg_layerwise_fn
    _pending_disagg_layerwise_fn = fn


@contextmanager
def set_forward_context(
    forward_batch: ForwardBatch,
    attention_layers: List[Any],
    quant_config: Any,
    moe_layers: List[Any],
    moe_fusions: List[Any],
    num_tokens: Optional[int] = None,
):
    global _forward_context, _pending_disagg_layerwise_fn
    _forward_context = ForwardContext()
    _forward_context.set_forward_batch(forward_batch)
    _forward_context.set_attention_layers(attention_layers)
    _forward_context.set_quant_config(quant_config)
    _forward_context.set_moe_layers(moe_layers)
    _forward_context.set_moe_fusions(moe_fusions)
    _forward_context.num_tokens = num_tokens
    # Pick up pending layer-wise KV transfer callback (disagg prefill only)
    if _pending_disagg_layerwise_fn is not None:
        _forward_context.disagg_layerwise_fn = _pending_disagg_layerwise_fn
        _pending_disagg_layerwise_fn = None
    try:
        yield
    finally:
        _forward_context = None


PIECEWISE_CUDA_GRAPH_CAPTURE_FAILED_MSG = (
    "Piecewise CUDA Graph is enabled by default as an experimental feature.\n"
    "To work around this error, add --disable-piecewise-cuda-graph to your launch command.\n"
    "Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose"
)
