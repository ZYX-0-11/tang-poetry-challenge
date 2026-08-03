"""Lightweight numerical tests; run with: python -m unittest test_tbr_modules.py"""

import unittest

import numpy as np

from tbr_modules import calculate_window_periodograms, ewma_smooth, extract_tbr


class TBRModuleTests(unittest.TestCase):
    def test_ewma_matches_requested_formula(self):
        actual = ewma_smooth(np.array([1.0, 2.0, 4.0]), alpha=0.3)
        np.testing.assert_allclose(actual, [1.0, 1.3, 2.11])

    def test_tbr_is_theta_mean_over_beta_mean(self):
        frequencies = np.arange(0.0, 31.0, 1.0)
        psd = np.ones((2, 3, frequencies.size))
        psd[..., (frequencies >= 4) & (frequencies <= 8)] = 6.0
        psd[..., (frequencies >= 13) & (frequencies <= 30)] = 2.0
        theta, beta, tbr = extract_tbr(frequencies, psd)
        np.testing.assert_allclose(theta, 6.0)
        np.testing.assert_allclose(beta, 2.0)
        np.testing.assert_allclose(tbr, 3.0)

    def test_two_second_window_half_second_step(self):
        fs = 250.0
        eeg = np.zeros((2, int(60 * fs)))
        times, _frequencies, psd = calculate_window_periodograms(eeg, fs)
        self.assertEqual(psd.shape[0], 117)
        self.assertAlmostEqual(times[0], 1.0)
        self.assertAlmostEqual(times[-1], 59.0)


if __name__ == "__main__":
    unittest.main()
