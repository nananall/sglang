import json
import logging
from typing import Optional

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.deepseek_nsa import DeepSeekNSAAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.quest_algorithm import QuestAlgorithm
from sglang.srt.mem_cache.sparsity.backend.backend_adaptor import (
    FlashAttentionAdaptor,
    NSABackendAdaptor,
)
from sglang.srt.mem_cache.sparsity.core.sparse_coordinator import (
    SparseConfig,
    SparseCoordinator,
)

logger = logging.getLogger(__name__)

_global_sparse_coordinator: Optional[SparseCoordinator] = None

_ALGORITHM_REGISTRY = {
    "quest": lambda config, device, **kw: QuestAlgorithm(config, device, **kw),
    "deepseek_nsa": lambda config, device, **kw: DeepSeekNSAAlgorithm(
        config, device, **kw
    ),
}


def _create_sparse_algorithm(
    config: SparseConfig,
    device: torch.device,
    **kwargs,
) -> BaseSparseAlgorithm:
    algorithm_name = config.algorithm.lower()
    factory = _ALGORITHM_REGISTRY.get(algorithm_name)

    if factory is None:
        raise ValueError(f"Unknown sparse algorithm: {algorithm_name}")

    return factory(config, device, **kwargs)


def _create_backend_adaptor(
    backend: str,
    device: torch.device,
    sparse_algorithm: BaseSparseAlgorithm,
    req_to_token_pool,
):
    """Create backend adaptor."""
    if isinstance(sparse_algorithm, DeepSeekNSAAlgorithm):
        return NSABackendAdaptor(device, req_to_token_pool)

    if backend in ["fa3", "flashattention"]:
        return FlashAttentionAdaptor(device)

    raise ValueError(f"Unknown attention backend: {backend}")


def _parse_sparse_config(server_args) -> SparseConfig:
    """Parse hierarchical sparse config"""
    # Parse extra config if provided
    extra_config_str = server_args.hisparse_config
    if extra_config_str is not None:
        try:
            extra_config = json.loads(extra_config_str)

            # Extract algorithm and backend
            algorithm = extra_config.pop("algorithm", "quest")
            backend = extra_config.pop("backend", "flashattention")
            min_sparse_prompt_len = extra_config.pop("min_sparse_prompt_len", 2048)
            top_k = extra_config.pop("top_k", 2048)
            device_buffer_size = extra_config.pop("device_buffer_size", 2 * top_k)
            host_to_device_ratio = extra_config.pop("host_to_device_ratio", 2)

            if device_buffer_size <= top_k:
                raise ValueError(
                    f"device_buffer_size ({device_buffer_size}) must be larger than "
                    f"top_k ({top_k})"
                )

            # Everything else goes to algorithm_extra_config
            sparse_extra_config = extra_config
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse hisparse_config: {e}")

    config = SparseConfig(
        algorithm=algorithm,
        backend=backend,
        page_size=server_args.page_size,
        min_sparse_prompt_len=min_sparse_prompt_len,
        top_k=top_k,
        device_buffer_size=device_buffer_size,
        host_to_device_ratio=host_to_device_ratio,
        sparse_extra_config=sparse_extra_config,
    )
    return config


def parse_hisparse_config(server_args) -> SparseConfig:
    """Parse hisparse config from server_args, returning defaults if no config provided."""
    if server_args.hisparse_config is not None:
        return _parse_sparse_config(server_args)

    # Return defaults when no hisparse_config JSON is provided
    return SparseConfig(
        algorithm="quest",
        backend="flashattention",
        page_size=server_args.page_size,
    )


def create_sparse_coordinator(
    device: torch.device,
    req_to_token_pool,
    token_to_kv_pool,
    start_layer: int,
    end_layer: int,
    server_args,
    **kwargs,
) -> SparseCoordinator:
    config = _parse_sparse_config(server_args)
    algorithm = _create_sparse_algorithm(config, device, **kwargs)
    backend_adaptor = _create_backend_adaptor(
        config.backend, device, algorithm, req_to_token_pool
    )

    coordinator = SparseCoordinator(
        config=config,
        algorithm=algorithm,
        backend_adaptor=backend_adaptor,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool=token_to_kv_pool,
        start_layer=start_layer,
        end_layer=end_layer,
        device=device,
    )
    register_sparse_coordinator(coordinator)
    return coordinator


def register_sparse_coordinator(coordinator: SparseCoordinator) -> None:
    global _global_sparse_coordinator
    _global_sparse_coordinator = coordinator


def get_sparse_coordinator() -> Optional[SparseCoordinator]:
    return _global_sparse_coordinator
