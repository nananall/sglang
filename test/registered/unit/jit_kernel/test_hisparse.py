import unittest

from sglang.jit_kernel.hisparse import (
    _estimate_static_shared_mem_bytes,
    _hash_size,
)


class TestHiSparseKernelConfig(unittest.TestCase):
    def test_large_hot_buffer_uses_smaller_hash_table(self):
        self.assertEqual(_hash_size(2048, 4096), 4096)
        self.assertEqual(_hash_size(2048, 8192), 3072)

    def test_8192_hot_buffer_fits_static_shared_memory_budget(self):
        self.assertLessEqual(_estimate_static_shared_mem_bytes(2048, 8192), 48 * 1024)

    def test_extreme_configuration_is_rejected_by_estimator(self):
        self.assertGreater(_estimate_static_shared_mem_bytes(4096, 8192), 48 * 1024)


if __name__ == "__main__":
    unittest.main()
