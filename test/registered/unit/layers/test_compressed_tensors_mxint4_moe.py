import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class TestCompressedTensorsMxInt4MoEImport(unittest.TestCase):
    TARGET_MODULE = (
        "sglang.srt.layers.quantization.compressed_tensors.schemes."
        "compressed_tensors_w4a4_mxint4_moe"
    )

    def _reload_with_partial_flashinfer(self):
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_fp4_quantization = types.ModuleType("flashinfer.fp4_quantization")
        fake_fp4_quantization.block_scale_interleave = MagicMock()

        fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
        fake_fused_moe.convert_to_block_layout = MagicMock()
        # Intentionally omit `trtllm_mxint4_block_scale_moe` to simulate an
        # older flashinfer build that is importable but lacks mxint4 support.

        fake_fused_moe_core = types.ModuleType("flashinfer.fused_moe.core")
        fake_fused_moe_core._maybe_get_cached_w3_w1_permute_indices = MagicMock()
        fake_fused_moe_core.get_w2_permute_indices_with_cache = MagicMock()

        with patch.dict(
            sys.modules,
            {
                "numpy": MagicMock(),
                "flashinfer": fake_flashinfer,
                "flashinfer.fp4_quantization": fake_fp4_quantization,
                "flashinfer.fused_moe": fake_fused_moe,
                "flashinfer.fused_moe.core": fake_fused_moe_core,
            },
        ):
            with patch("sglang.srt.utils.is_flashinfer_available", return_value=True):
                sys.modules.pop(self.TARGET_MODULE, None)
                module = importlib.import_module(self.TARGET_MODULE)
                return importlib.reload(module)

    def test_import_succeeds_with_partial_flashinfer(self):
        module = self._reload_with_partial_flashinfer()

        self.assertTrue(hasattr(module, "CompressedTensorsMxInt4MoE"))
        self.assertIsNotNone(module._FLASHINFER_MXINT4_IMPORT_ERROR)

    def test_scheme_raises_clear_error_when_mxint4_kernel_missing(self):
        module = self._reload_with_partial_flashinfer()

        with self.assertRaises(ImportError) as exc_info:
            module.CompressedTensorsMxInt4MoE(MagicMock())

        self.assertIn("trtllm_mxint4_block_scale_moe", str(exc_info.exception))


if __name__ == "__main__":
    unittest.main()
