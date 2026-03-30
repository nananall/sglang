from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=1, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=1, suite="stage-b-test-small-1-gpu-amd")


class TestHiSparseConfigValidation(TestCase):
    def _make_runner(self, *, architecture="DeepseekV32ForCausalLM", index_topk=2048):
        return SimpleNamespace(
            server_args=SimpleNamespace(hisparse_config="{}"),
            use_mla_backend=True,
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=[architecture],
                    index_topk=index_topk,
                )
            ),
        )

    def test_rejects_hisparse_top_k_smaller_than_model_nsa_top_k(self):
        runner = self._make_runner(index_topk=2048)
        hisparse_cfg = SimpleNamespace(
            top_k=1536,
            device_buffer_size=3072,
            host_to_device_ratio=2,
        )

        with patch(
            "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
            return_value=hisparse_cfg,
        ):
            with self.assertRaisesRegex(ValueError, "model NSA index_topk"):
                ModelRunnerKVCacheMixin.get_validated_hisparse_config(runner)

    def test_allows_hisparse_top_k_at_least_model_nsa_top_k(self):
        runner = self._make_runner(index_topk=2048)
        hisparse_cfg = SimpleNamespace(
            top_k=2048,
            device_buffer_size=4096,
            host_to_device_ratio=2,
        )

        with patch(
            "sglang.srt.mem_cache.sparsity.parse_hisparse_config",
            return_value=hisparse_cfg,
        ):
            validated = ModelRunnerKVCacheMixin.get_validated_hisparse_config(runner)

        self.assertIs(validated, hisparse_cfg)
