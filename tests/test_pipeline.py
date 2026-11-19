import numpy as np
import pytest

from prototype_deconv import deconvolve_gcms, generate_synthetic_gcms, match_components
from src.preprocessing.raman_arpls import arpls


def test_gcms_resolves_three_overlapping_components():
    rt, _, signal, truth = generate_synthetic_gcms(snr=5.0)
    resolved = deconvolve_gcms(rt, signal)

    assert len(resolved) == len(truth) == 3
    for known, result, similarity in match_components(truth, resolved):
        assert similarity >= 0.92, known["name"]
        assert result["rt"] == pytest.approx(known["rt"], abs=0.08)
        assert np.all(result["raw_spectrum"] >= 0)


def test_gcms_no_components_for_constant_background():
    rt = np.linspace(5, 15, 20)
    assert deconvolve_gcms(rt, np.ones((rt.size, 4))) == []


def test_raman_arpls_recovers_baseline_and_peaks():
    x = np.linspace(0, 1, 900)
    fluorescence = 2 + 0.6 * np.cos(2 * np.pi * x) + 2.5 * np.exp(-4 * x**2)
    centers = np.array([0.09, 0.43, 0.83, 0.94])
    heights = np.array([1.2, 0.9, 1.8, 1.3])
    widths = np.array([0.008, 0.005, 0.012, 0.006])
    peaks = np.exp(-0.5 * ((x[:, None] - centers) / widths) ** 2) @ heights
    noise = np.random.default_rng(24).normal(0, 0.015, x.size)
    measured = fluorescence + peaks + noise

    corrected, estimated = arpls(measured)
    np.testing.assert_allclose(corrected + estimated, measured)
    assert np.sqrt(np.mean((estimated - fluorescence) ** 2)) < 0.04
    assert np.sqrt(np.mean((corrected - peaks) ** 2)) < 0.05
    peak_indices = np.argmin(np.abs(x[:, None] - centers), axis=0)
    np.testing.assert_allclose(corrected[peak_indices], peaks[peak_indices], atol=0.08)


def test_raman_arpls_preserves_constant_baseline():
    corrected, baseline = arpls(np.full(30, 2.0))
    np.testing.assert_allclose(baseline, 2.0, atol=1e-8)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-8)
