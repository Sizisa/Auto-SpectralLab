"""Simulate, resolve, and benchmark overlapping GC-MS components."""

import time

import numpy as np
from scipy.cluster.hierarchy import fclusterdata
from scipy.optimize import curve_fit, linear_sum_assignment, nnls
from scipy.signal import find_peaks, savgol_filter

_SIGMA_TO_FWHM = 2 * np.sqrt(2 * np.log(2))
_MIN_SIMILARITY = 0.92
_FRAGMENTS = {
    "Compound_A (Target_1)": {91: 1000, 105: 450, 144: 820, 214: 610, 313: 240},
    "Compound_B (Co-eluting_Isomer)": {91: 300, 131: 950, 144: 400, 228: 780, 327: 190},
    "Compound_C (Matrix_Interference)": {73: 1000, 147: 650, 207: 890, 281: 520},
}


def _gaussian(rt, amplitude, center, width):
    return amplitude * np.exp(-0.5 * ((rt - center) / width) ** 2)


def generate_synthetic_gcms(n_scans=400, max_mz=350, snr=5.0, seed=42):
    """Return retention times, integer m/z bins, noisy scans, and ground truth.

    SNR is the strongest clean ion intensity divided by the pre-clipping
    noise SD at that scan and m/z bin. Shot noise is Poisson-distributed.
    """
    if n_scans < 7 or max_mz < 327 or not np.isfinite(snr) or snr <= 0:
        raise ValueError("Require at least 7 scans, m/z bins through 327, and positive SNR.")

    rt = np.linspace(5.0, 15.0, n_scans)
    mz_axis = np.arange(1, max_mz + 1)
    width = 0.15
    fwhm = _SIGMA_TO_FWHM * width
    centers = 10.0 + 0.45 * fwhm * np.arange(3)
    spectra = np.zeros((len(_FRAGMENTS), max_mz))
    truth = []
    for index, (name, peaks) in enumerate(_FRAGMENTS.items()):
        spectra[index, np.array(list(peaks)) - 1] = list(peaks.values())
        truth.append({
            "name": name,
            "rt": centers[index],
            "spectrum": spectra[index] / np.linalg.norm(spectra[index]),
        })

    elution = _gaussian(rt[:, None], np.array([1.0, 0.85, 1.2]), centers, width)
    clean = elution @ spectra
    peak = clean.max()
    noise_sd = peak / snr
    read_sd = 0.12 * noise_sd
    shot_gain = (noise_sd**2 - read_sd**2) / peak
    rng = np.random.default_rng(seed)
    noisy = shot_gain * rng.poisson(clean / shot_gain)
    noisy += rng.normal(0, read_sd, size=clean.shape)
    return rt, mz_axis, noisy.clip(min=0), truth


def deconvolve_gcms(rt, X, min_snr=4.0, rt_cluster_tol=0.08):
    """Resolve overlapping Gaussian chromatographic profiles in a scans × m/z matrix."""
    rt = np.asarray(rt, dtype=float)
    X = np.asarray(X, dtype=float)
    if (
        rt.ndim != 1 or rt.size < 7 or X.ndim != 2 or X.shape[0] != rt.size
        or not np.all(np.isfinite(rt)) or not np.all(np.isfinite(X))
        or np.any(np.diff(rt) <= 0) or np.any(X < 0)
    ):
        raise ValueError("Supply increasing retention times and a finite, nonnegative scan matrix.")
    if not np.isfinite(min_snr) or min_snr <= 0 or not np.isfinite(rt_cluster_tol) or rt_cluster_tol <= 0:
        raise ValueError("SNR and RT clustering tolerance must be finite and positive.")

    differences = np.diff(X, axis=0)
    centered = differences - np.median(differences, axis=0)
    noise = np.median(np.abs(centered), axis=0) / (0.67448975 * np.sqrt(2))
    baseline = np.median(X, axis=0)
    # Zero clipping suppresses first-difference MAD. The upper one-sigma
    # quantile remains informative when the lower tail is censored.
    upper_noise = np.quantile(X, 0.84134475, axis=0) - baseline
    threshold = min_snr * np.maximum(np.maximum(noise, upper_noise), np.finfo(float).eps)
    signal = X - baseline
    smooth = savgol_filter(signal, window_length=7, polyorder=2, axis=0)
    active = np.flatnonzero(smooth.max(axis=0) >= threshold)
    scan_step = np.median(np.diff(rt))
    candidates = []

    for channel in active:
        peaks, properties = find_peaks(
            smooth[:, channel], height=threshold[channel],
            prominence=0.8 * threshold[channel], width=3,
        )
        for peak, width, height in zip(peaks, properties["widths"], properties["peak_heights"]):
            # Fit the raw local EIC on the time axis to refine the apex and width.
            sigma = width * scan_step / _SIGMA_TO_FWHM
            radius = int(np.ceil(width * 1.5))
            region = slice(max(0, peak - radius), peak + radius + 1)
            try:
                params, _ = curve_fit(
                    _gaussian, rt[region], signal[region, channel],
                    p0=(height, rt[peak], sigma),
                    bounds=(
                        (0, rt[region][0], scan_step / 2),
                        (np.inf, rt[region][-1], 3 * sigma),
                    ),
                )
            except RuntimeError:
                continue
            amplitude, center, fitted_sigma = params
            if fitted_sigma >= scan_step and amplitude >= threshold[channel]:
                candidates.append((channel, center, fitted_sigma, amplitude))

    if len(candidates) < 2:
        return []

    channels, centers, widths, amplitudes = np.asarray(candidates).T
    groups = fclusterdata(centers[:, None], rt_cluster_tol, criterion="distance", method="average")
    models = []
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        # Keep one peak per ion when estimating a component profile.
        strongest = members[np.argsort(-amplitudes[members])]
        _, first = np.unique(channels[strongest], return_index=True)
        members = strongest[first]
        if members.size < 2:
            continue
        models.append((np.median(centers[members]), np.median(widths[members])))

    if not models:
        return []

    component_rts, component_widths = np.asarray(sorted(models)).T
    profiles = _gaussian(rt[:, None], 1.0, component_rts, component_widths)
    spectra = np.zeros((len(models), X.shape[1]))
    for channel in active:
        spectra[:, channel], _ = nnls(profiles, signal[:, channel])

    return [
        {"component_id": index + 1, "rt": center, "spectrum": spectrum / norm,
         "raw_spectrum": spectrum}
        for index, (center, spectrum, norm) in enumerate(
            zip(component_rts, spectra, np.linalg.norm(spectra, axis=1))
        )
        if norm > 0
    ]


def cosine_similarity(spec_a, spec_b):
    """Return cosine similarity, or zero when either spectrum is empty."""
    norms = np.linalg.norm(spec_a) * np.linalg.norm(spec_b)
    return float(np.clip(np.dot(spec_a, spec_b) / norms, -1, 1)) if norms else 0.0


def match_components(ground_truth, resolved):
    """Return one-to-one matches in ground-truth order, including misses."""
    scores = np.array([
        [cosine_similarity(known["spectrum"], result["spectrum"]) for result in resolved]
        for known in ground_truth
    ]).reshape(len(ground_truth), len(resolved))
    rows, columns = linear_sum_assignment(scores, maximize=True)
    assignment = dict(zip(rows, columns))
    return [
        (known, resolved[assignment[index]], scores[index, assignment[index]])
        if index in assignment else (known, None, 0.0)
        for index, known in enumerate(ground_truth)
    ]


def run_benchmark():
    """Print the synthetic validation audit and assert the acceptance target."""
    rt, mz_axis, noisy, truth = generate_synthetic_gcms()
    print("Auto-SpectralLab | GC-MS Co-elution Deconvolution Benchmark")
    print(f"[INFO] {rt.size} scans × {mz_axis.size} m/z bins | SNR = 5.0 | seed = 42")

    start = time.perf_counter()
    resolved = deconvolve_gcms(rt, noisy)
    elapsed_ms = (time.perf_counter() - start) * 1000
    matches = match_components(truth, resolved)
    print(f"[DONE] Deconvolution took {elapsed_ms:.2f} ms | Resolved {len(resolved)} components")
    print()
    print(f"{'Ground-truth compound':<32} {'True RT':>8} {'Est. RT':>8} "
          f"{'RT error':>8} {'Cosine':>8}  Status")
    print("-" * 83)
    for known, result, score in matches:
        estimated = f"{result['rt']:.4f}" if result else "—"
        error = f"{abs(result['rt'] - known['rt']):.4f}" if result else "—"
        status = "PASS" if score >= _MIN_SIMILARITY else "FAIL"
        print(f"{known['name']:<32} {known['rt']:8.4f} {estimated:>8} "
              f"{error:>8} {score:8.4f}  [{status}]")

    assert len(resolved) == len(truth), "Unexpected number of resolved components."
    assert all(score >= _MIN_SIMILARITY for _, _, score in matches), (
        f"Each component must reach cosine similarity ≥ {_MIN_SIMILARITY}."
    )
    print(f"[SUCCESS] All {len(truth)} components reached cosine similarity ≥ {_MIN_SIMILARITY}.")


if __name__ == "__main__":
    run_benchmark()
