import unittest

import numpy as np

from sglang.srt.disaggregation.fake.conn import FakeKVReceiver


class TestFakeKVReceiver(unittest.TestCase):
    def test_send_metadata_accepts_device_kv_indices(self):
        receiver = FakeKVReceiver(None, "")
        receiver.init(prefill_dp_rank=0)

        receiver.send_metadata(
            np.array([1, 2], dtype=np.int32),
            aux_index=0,
            device_kv_indices=np.array([3, 4], dtype=np.int32),
        )

        self.assertTrue(receiver.has_sent_metadata)


if __name__ == "__main__":
    unittest.main()
