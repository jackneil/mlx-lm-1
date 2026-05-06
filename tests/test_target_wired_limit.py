# Copyright © 2026 Apple Inc.

"""Tests for _target_wired_limit env coordination.

Background: each mlx-lm process raises Metal's wired-limit to ~75% of
physical RAM at startup. When multiple processes share a host (e.g. an
arena fronting 9 vllm-mlx servers), the requests sum past physical
capacity and the kernel can't satisfy them — the proximate cause of the
cascading hang in mlx-lm/issues/883. The MLX_LM_NUM_SERVERS env var
divides the per-process target so co-resident processes share fairly.
"""

import os
import unittest
from unittest.mock import patch

from mlx_lm.generate import _target_wired_limit


class TestTargetWiredLimit(unittest.TestCase):
    def test_default_returns_full_max_recommended(self):
        """No env set → full max_recommended_working_set_size."""
        with patch("mlx_lm.generate.mx.device_info",
                   return_value={"max_recommended_working_set_size": 384_000_000_000}), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MLX_LM_NUM_SERVERS", None)
            self.assertEqual(_target_wired_limit(), 384_000_000_000)

    def test_env_divisor_applied(self):
        """MLX_LM_NUM_SERVERS=9 → max_recommended / 9."""
        with patch("mlx_lm.generate.mx.device_info",
                   return_value={"max_recommended_working_set_size": 384_000_000_000}), \
             patch.dict(os.environ, {"MLX_LM_NUM_SERVERS": "9"}):
            self.assertEqual(_target_wired_limit(), 384_000_000_000 // 9)

    def test_env_zero_clamped_to_one(self):
        """MLX_LM_NUM_SERVERS=0 must not produce a divide-by-zero or a
        wired limit of zero — clamp to 1."""
        with patch("mlx_lm.generate.mx.device_info",
                   return_value={"max_recommended_working_set_size": 384_000_000_000}), \
             patch.dict(os.environ, {"MLX_LM_NUM_SERVERS": "0"}):
            self.assertEqual(_target_wired_limit(), 384_000_000_000)

    def test_env_negative_clamped_to_one(self):
        with patch("mlx_lm.generate.mx.device_info",
                   return_value={"max_recommended_working_set_size": 384_000_000_000}), \
             patch.dict(os.environ, {"MLX_LM_NUM_SERVERS": "-3"}):
            self.assertEqual(_target_wired_limit(), 384_000_000_000)


if __name__ == "__main__":
    unittest.main()
