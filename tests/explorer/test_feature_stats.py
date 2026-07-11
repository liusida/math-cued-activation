from __future__ import annotations

import unittest

import torch

from ica_lens_v9.features.stats import (
    accumulate_feature_side_stats,
    empty_signed_feature_stats,
    finish_feature_side_stats,
)


def _stats_for(values: torch.Tensor) -> dict[str, torch.Tensor]:
    stats = empty_signed_feature_stats(n_components=1)["pos"]
    accumulate_feature_side_stats(stats, values.reshape(-1, 1))
    return finish_feature_side_stats(stats, total_rows=int(values.numel()))


class ActiveMirroredKurtosisTest(unittest.TestCase):
    def test_half_normal_has_gaussian_kurtosis(self) -> None:
        generator = torch.Generator().manual_seed(0)
        scores = torch.randn(500_000, generator=generator, dtype=torch.float64)
        values = torch.relu(scores)

        result = _stats_for(values)

        self.assertAlmostEqual(float(result["activation_frequency"][0]), 0.5, delta=0.003)
        self.assertAlmostEqual(float(result["kurtosis"][0]), 3.0, delta=0.08)
        self.assertAlmostEqual(float(result["excess_kurtosis"][0]), 0.0, delta=0.08)

    def test_constant_active_magnitudes_have_kurtosis_one(self) -> None:
        values = torch.tensor([0.0, 2.0, 0.0, 2.0, 2.0], dtype=torch.float64)

        result = _stats_for(values)

        self.assertAlmostEqual(float(result["activation_frequency"][0]), 0.6)
        self.assertAlmostEqual(float(result["kurtosis"][0]), 1.0)
        self.assertAlmostEqual(float(result["excess_kurtosis"][0]), -2.0)

    def test_rare_active_outlier_is_high_kurtosis(self) -> None:
        values = torch.ones(1000, dtype=torch.float64)
        values[0] = 100.0

        result = _stats_for(values)

        self.assertGreater(float(result["kurtosis"][0]), 500.0)

    def test_no_active_tokens_is_finite_low_kurtosis(self) -> None:
        values = torch.zeros(100, dtype=torch.float64)

        result = _stats_for(values)

        self.assertEqual(float(result["activation_frequency"][0]), 0.0)
        self.assertEqual(float(result["kurtosis"][0]), 0.0)
        self.assertEqual(float(result["excess_kurtosis"][0]), -3.0)


if __name__ == "__main__":
    unittest.main()
