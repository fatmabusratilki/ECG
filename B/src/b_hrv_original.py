#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRV All-in-One (single file)
----------------------------
Implements:
• Time-domain: HR stats, % time <50 bpm + episodes, SDNN, RMSSD, pNN50/pNN20, triangular index
• PRSA: Deceleration capacity (DC), Acceleration capacity (AC)
• Frequency-domain (Welch or AR/Burg; RR resampled to ~4 Hz): VLF/LF/HF power (ms²), LF/HF, normalized LF/HF, HF peak & resp rate
• Non-linear: Poincaré SD1/SD2 (+ratio), SampEn/ApEn, DFA α1/α2, Multiscale Entropy
• Rolling windows: default 5-min windows, 1-min step

Inputs:
• HDF5 (auto RR/IBI/R-peaks; fallback RR≈60000/HR from BedMaster HR stream)
• Direct RR array (ms) via analyze_rr()

Exports:
• *_summary.csv, *_segments.csv, *_psd.csv, *_mse.csv

Dependencies: numpy, pandas, h5py, scipy (signal, interpolate)
"""

from __future__ import annotations
import argparse, json, math, warnings, os, tempfile, re
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

# Default paths so the script runs with NO CLI args
DEFAULT_H5 = r"\\MGBMAD3ISILON1-smb.partners.org\MGH-BTSDATA-ARCHIVE\2019\06\3254689660.hd5"
DEFAULT_OUT_PREFIX = r"\\MGBMAD3ISILON1-smb.partners.org\MGH-BTSDATA-ARCHIVE\2019\06\3254689660_HRV"
DEFAULT_DATASET = None  # set to an HDF5 dataset path if you know exact RR/IBI location

# Optional deps
try:
    import h5py as h5
except Exception:
    h5 = None

try:
    from scipy import signal as sig
    from scipy import interpolate as interp
except Exception:
    sig = None
    interp = None


# ------------------------------- Utilities ------------------------------------

def _sliding_window_view(x: np.ndarray, window: int) -> np.ndarray:
    """Compat sliding window when np.lib.stride_tricks.sliding_window_view is missing."""
    try:
        return np.lib.stride_tricks.sliding_window_view(x, window, writeable=False)
    except Exception:
        n = x.shape[0]
        if n < window:
            return np.empty((0, window), dtype=x.dtype)
        return np.vstack([x[i:i + window] for i in range(n - window + 1)])


def _as_ms(rr, assume_seconds_if_lt_10: bool = True) -> np.ndarray:
    """Convert RR-like input to ms. If median <10, assume seconds→ms."""
    rr = np.asarray(rr, dtype=np.float64).copy()
    if rr.size == 0:
        return rr
    return rr * 1000.0 if (assume_seconds_if_lt_10 and np.nanmedian(rr) < 10.0) else rr


def _valid_mask_rr(rr_ms: np.ndarray, min_ms: float = 300.0, max_ms: float = 2000.0, rel_jump: float = 0.2) -> np.ndarray:
    """Artifact filter: physio range + relative jump vs 9-beat moving mean."""
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    m = (rr_ms >= min_ms) & (rr_ms <= max_ms)
    if rr_ms.size >= 9:
        med = np.convolve(rr_ms, np.ones(9) / 9.0, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.abs(rr_ms - med) / np.maximum(1e-9, med)
        m &= (rel <= rel_jump)
    return m


def _cumsums(rr_ms: np.ndarray) -> np.ndarray:
    """Cumulative time (s) at each RR boundary, starting at 0."""
    t = np.cumsum(np.asarray(rr_ms, dtype=np.float64)) / 1000.0
    t -= t[0] if t.size else 0.0
    return t


def _interp_even(rr_ms: np.ndarray, fs: float = 4.0, kind: str = "pchip") -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate uneven RR (ms) to an evenly-sampled tachogram (ms) at fs Hz."""
    if rr_ms is None or len(rr_ms) < 3:
        return np.array([]), np.array([])
    t = _cumsums(rr_ms)
    x = np.asarray(rr_ms, dtype=np.float64)
    t_even = np.arange(t[0], t[-1], 1.0 / fs)
    if t_even.size < 8:
        return np.array([]), np.array([])
    if interp is not None and hasattr(interp, "PchipInterpolator") and kind == "pchip":
        f = interp.PchipInterpolator(t, x, extrapolate=False)
        x_even = f(t_even)
        if np.isnan(x_even[0]):
            first_valid = np.where(~np.isnan(x_even))[0]
            if first_valid.size:
                x_even[:first_valid[0]] = x_even[first_valid[0]]
        if np.isnan(x_even[-1]):
            last_valid = np.where(~np.isnan(x_even))[0]
            if last_valid.size:
                x_even[last_valid[-1]:] = x_even[last_valid[-1]]
        x_even = np.asarray(x_even, dtype=np.float64)
    else:
        x_even = np.interp(t_even, t, x)
    return t_even, x_even


def _hr_from_rr(rr_ms: np.ndarray) -> np.ndarray:
    """Instantaneous heart rate (bpm) per beat."""
    rr_ms = np.asarray(rr_ms, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 60000.0 / rr_ms


def _pct_time_below_hr(rr_ms: np.ndarray, threshold_bpm: float = 50.0, fs_for_interp: float = 1.0):
    """Percent time HR<threshold (bpm) + list of brady episodes (start_s, end_s)."""
    t, rr_even = _interp_even(rr_ms, fs=fs_for_interp)
    if t.size == 0:
        return np.nan, []
    hr_t = 60000.0 / rr_even
    below = hr_t < threshold_bpm
    pct = 100.0 * below.sum() / below.size
    episodes = []
    if below.any():
        idx = np.flatnonzero(np.diff(np.r_[False, below, False]))
        starts = idx[0::2]
        ends = idx[1::2] - 1
        for s, e in zip(starts, ends):
            episodes.append((float(t[s]), float(t[e])))
    return float(pct), episodes


def _hrv_triangular_index(rr_ms: np.ndarray, bin_ms: float = 7.8125) -> float:
    """HRV triangular index (Task Force, 1996)."""
    rr = np.asarray(rr_ms, dtype=np.float64)
    if rr.size < 2:
        return float("nan")
    hist, _ = np.histogram(rr, bins=np.arange(rr.min(), rr.max() + bin_ms, bin_ms))
    h = hist.max() if hist.size else np.nan
    return float(rr.size / h) if (np.isfinite(h) and h > 0) else float("nan")


# ---- small helper to avoid deprecation warnings across NumPy versions --------
def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Use np.trapezoid when available, else fallback to np.trapz."""
    try:
        return float(np.trapezoid(y, x))
    except Exception:
        return float(np.trapz(y, x))


# ---------------------- Output prefix (writability) helpers -------------------

def _sanitize_for_filename(s: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    safe = safe.strip("._-")
    return safe or "output"

def _resolve_out_prefix(out_prefix: str,
                        fallback_root: Optional[str] = None,
                        verbose: bool = True) -> Tuple[str, Optional[Dict]]:
    """
    Make sure directory for out_prefix is writable; if not, fall back to a safe local folder:
      %USERPROFILE%/HRV_Output/<stem>
    Returns (final_out_prefix, fallback_info_or_None).
    """
    dirpath = os.path.dirname(out_prefix)
    if not dirpath:
        dirpath = os.getcwd()
    stem = os.path.basename(out_prefix) or "HRV_Output"
    try:
        os.makedirs(dirpath, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".__hrv_writetest__", dir=dirpath)
        os.close(fd)
        os.unlink(tmp)
        return out_prefix, None
    except Exception as e:
        if fallback_root is None:
            fallback_root = os.path.join(os.path.expanduser("~"), "HRV_Output")
        try:
            os.makedirs(fallback_root, exist_ok=True)
        except Exception:
            fallback_root = os.getcwd()
        new_prefix = os.path.join(fallback_root, _sanitize_for_filename(stem))
        if verbose:
            print(f"[WARN] Cannot write to '{dirpath}' ({type(e).__name__}: {e}). "
                  f"Falling back to '{fallback_root}'.")
        return new_prefix, {"fell_back": True, "why": f"{type(e).__name__}: {e}",
                            "from_dir": dirpath, "to_dir": fallback_root}


# ----------------------------- Time-domain ------------------------------------

def time_domain_metrics(rr_ms: np.ndarray) -> Dict:
    rr_ms = _as_ms(rr_ms)
    mask = _valid_mask_rr(rr_ms)
    rr_nn = rr_ms[mask]
    if rr_nn.size < 3:
        return {}
    diff = np.diff(rr_nn)
    sdnn = float(np.std(rr_nn, ddof=1))
    rmssd = float(np.sqrt(np.mean(diff ** 2)))
    pnn50 = 100.0 * np.mean(np.abs(diff) > 50.0)
    pnn20 = 100.0 * np.mean(np.abs(diff) > 20.0)
    tri_idx = _hrv_triangular_index(rr_nn)
    hr_per_beat = _hr_from_rr(rr_nn)
    pct_below_50, episodes = _pct_time_below_hr(rr_nn, threshold_bpm=50.0, fs_for_interp=1.0)
    return {
        "hr_bpm_mean": float(np.mean(hr_per_beat)),
        "hr_bpm_median": float(np.median(hr_per_beat)),
        "hr_bpm_min": float(np.min(hr_per_beat)),
        "hr_bpm_max": float(np.max(hr_per_beat)),
        "pct_time_hr_lt_50": pct_below_50,
        "bradycardia_episodes": episodes,
        "sdnn_ms": sdnn,
        "rmssd_ms": rmssd,
        "pnn50_pct": float(pnn50),
        "pnn20_pct": float(pnn20),
        "triangular_index": float(tri_idx),
        "num_beats_used": int(rr_nn.size),
    }


# ----------------------------- PRSA (DC/AC) -----------------------------------

def prsa_dc_ac(rr_ms: np.ndarray, L: int = 30, T: int = 1, slope_eps_ms: float = 0.0) -> Dict:
    """PRSA with deceleration/acceleration anchors → DC/AC and PRSA curves."""
    rr = _as_ms(rr_ms)
    if rr.size < (2 * L + 5):
        return {}
    d = np.diff(rr)
    decel_idx = np.where(d > slope_eps_ms)[0] + 1
    accel_idx = np.where(d < -slope_eps_ms)[0] + 1

    def _avg_curve(anchors):
        segs = []
        for i in anchors:
            lo = i - L * T
            hi = i + L * T
            if lo < 0 or hi >= rr.size:
                continue
            seg = rr[lo:hi + 1:T]
            if seg.size == 2 * L + 1:
                segs.append(seg)
        if not segs:
            return np.array([]), 0
        segs = np.vstack(segs)
        return np.nanmean(segs, axis=0), segs.shape[0]

    prsa_decel, nD = _avg_curve(decel_idx)
    prsa_accel, nA = _avg_curve(accel_idx)
    k = np.arange(-L, L + 1, dtype=int)

    def _capacity(curve):
        if curve.size == 0:
            return float("nan")
        Lidx = np.where(k == -1)[0][0]
        Ridx = np.where(k == +1)[0][0]
        L2idx = np.where(k == -2)[0][0] if (-2 in k) else Lidx
        R2idx = np.where(k == +2)[0][0] if (+2 in k) else Ridx
        return float((curve[Ridx] + curve[R2idx] - curve[Lidx] - curve[L2idx]) / 4.0)

    return {
        "dc_ms": _capacity(prsa_decel),
        "ac_ms": _capacity(prsa_accel),
        "prsa_curve_decel": prsa_decel.tolist() if prsa_decel.size else [],
        "prsa_curve_accel": prsa_accel.tolist() if prsa_accel.size else [],
        "k_offsets": k.tolist(),
        "n_decel_anchors": int(nD),
        "n_accel_anchors": int(nA),
    }


# --------------------------- Frequency-domain ---------------------------------

def _welch_psd(rr_ms: np.ndarray, fs_interp: float = 4.0, nperseg: int = 256, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD of tachogram (RR in ms) → (f Hz, Pxx ms²/Hz)."""
    if sig is None:
        raise RuntimeError("scipy.signal is required for Welch PSD.")
    t, x = _interp_even(rr_ms, fs=fs_interp)
    if x.size < max(64, nperseg):
        nperseg = max(32, x.size // 4)
    if x.size < 32:
        return np.array([]), np.array([])
    x = sig.detrend(x, type="linear") if (detrend and x.size >= 3) else x
    f, Pxx = sig.welch(x, fs=fs_interp, nperseg=min(nperseg, x.size), window="hann", noverlap=None, detrend=False, scaling="density")
    return f, Pxx


def _burg_ar(x: np.ndarray, order: int = 16) -> Tuple[np.ndarray, float]:
    """Burg AR parameter estimation → (a[0..p], variance E)."""
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if N <= order + 1:
        raise ValueError("Time series too short for AR order.")
    ef = x[1:].copy()
    eb = x[:-1].copy()
    a = np.zeros(order + 1, dtype=np.float64); a[0] = 1.0
    E = float(np.dot(x, x)) / N
    for k in range(1, order + 1):
        num = -2.0 * np.dot(eb, ef)
        den = np.dot(ef, ef) + np.dot(eb, eb)
        if den <= 0: break
        gamma = num / den
        ef_new = ef + gamma * eb
        eb = eb + gamma * ef
        ef = ef_new[:-1]; eb = eb[1:]
        a_new = a.copy()
        a_new[1:k] = a[1:k] + gamma * a[k - 1:0:-1]
        a_new[k] = gamma
        a = a_new
        E *= (1.0 - gamma ** 2)
    return a, float(E)


def _ar_psd(rr_ms: np.ndarray, fs_interp: float = 4.0, order: int = 16, nfft: int = 2048, detrend: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Burg AR PSD of tachogram → (f Hz in [0, fs/2], Pxx ms²/Hz)."""
    if sig is None:
        raise RuntimeError("scipy.signal is required for AR PSD.")
    t, x = _interp_even(rr_ms, fs=fs_interp)
    if x.size < max(64, order * 4):
        return np.array([]), np.array([])
    x = sig.detrend(x, type="linear") if (detrend and x.size >= 3) else x
    a, E = _burg_ar(x, order=order)
    w = np.linspace(0.0, np.pi, num=nfft, endpoint=True)
    Aejw = np.dot(a, np.exp(-1j * np.outer(np.arange(0, a.size), w)))
    Pxx = (E / (np.abs(Aejw) ** 2)).real
    f = (w / (2.0 * np.pi)) * fs_interp
    return f, Pxx


def bandpowers_from_psd(f: np.ndarray, Pxx: np.ndarray) -> Dict:
    """Integrate VLF/LF/HF; compute nu, LF/HF, HF peak & resp rate."""
    if f.size == 0 or Pxx.size == 0:
        return {}
    vlf_lo, vlf_hi = 0.003, 0.04
    lf_lo, lf_hi = 0.04, 0.15
    hf_lo, hf_hi = 0.15, 0.40

    def _int(lo, hi):
        m = (f >= lo) & (f < hi)
        return _trapz(Pxx[m], f[m]) if m.any() else 0.0

    vlf = _int(vlf_lo, vlf_hi)
    lf  = _int(lf_lo, lf_hi)
    hf  = _int(hf_lo, hf_hi)
    total = _trapz(Pxx[(f >= vlf_lo) & (f < hf_hi)], f[(f >= vlf_lo) & (f < hf_hi)])
    denom = max(total - vlf, 1e-9)
    lf_nu = 100.0 * lf / denom
    hf_nu = 100.0 * hf / denom
    lf_hf = (lf / hf) if hf > 0 else float("inf")
    m_hf = (f >= hf_lo) & (f < hf_hi)
    hf_peak = float(f[m_hf][Pxx[m_hf].argmax()]) if m_hf.any() else float("nan")
    resp_rate_bpm = float(hf_peak * 60.0) if np.isfinite(hf_peak) else float("nan")
    return {
        "vlf_power_ms2": float(vlf),
        "lf_power_ms2": float(lf),
        "hf_power_ms2": float(hf),
        "total_power_ms2": float(total),
        "lf_nu": float(lf_nu),
        "hf_nu": float(hf_nu),
        "lf_hf_ratio": float(lf_hf),
        "hf_peak_hz": hf_peak,
        "resp_rate_bpm_from_hf_peak": resp_rate_bpm,
    }


def frequency_domain_metrics(rr_ms: np.ndarray, method: str = "welch", fs_interp: float = 4.0, ar_order: int = 16) -> Dict:
    """Compute PSD (Welch or AR) + band powers, nu, LF/HF, HF peak."""
    if method not in ("welch", "ar"):
        raise ValueError("method must be 'welch' or 'ar'")
    f, Pxx = _welch_psd(rr_ms, fs_interp=fs_interp) if method == "welch" else _ar_psd(rr_ms, fs_interp=fs_interp, order=ar_order)
    return {"psd_freqs_hz": f, "psd_ms2_per_hz": Pxx, **bandpowers_from_psd(f, Pxx)}


# -------------------------- Non-linear / Complexity ---------------------------

def poincare_metrics(rr_ms: np.ndarray) -> Dict:
    """Poincaré SD1/SD2 (+ratio)."""
    rr = _as_ms(rr_ms)
    if rr.size < 3:
        return {}
    sdnn = np.std(rr, ddof=1)
    rmssd = np.sqrt(np.mean(np.diff(rr) ** 2))
    sd1 = np.sqrt(0.5) * rmssd
    sd2_sq = 2 * (sdnn ** 2) - 0.5 * (rmssd ** 2)
    sd2 = np.sqrt(sd2_sq) if sd2_sq > 0 else float("nan")
    return {"sd1_ms": float(sd1), "sd2_ms": float(sd2), "sd1_sd2_ratio": float(sd1 / sd2) if np.isfinite(sd2) and sd2 > 0 else float("nan")}


def _phi_count(x: np.ndarray, m: int, r: float) -> float:
    """Proportion of matches (Chebyshev norm) for SampEn (exclude self)."""
    N = x.size
    if N <= m + 1:
        return 0.0
    x_m = _sliding_window_view(x, m)
    count = 0.0
    for i in range(x_m.shape[0] - 1):
        d = np.max(np.abs(x_m[i + 1:] - x_m[i]), axis=1)
        count += float(np.sum(d <= r))
    denom = x_m.shape[0] * (x_m.shape[0] - 1) / 2.0
    return count / max(denom, 1e-9)


def sample_entropy(rr_ms: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """SampEn(m=2, r=0.2·SD default)."""
    x = _as_ms(rr_ms)
    if x.size < (m + 2):
        return float("nan")
    r = 0.2 * np.std(x) if r is None else r
    A = _phi_count(x, m + 1, r)
    B = _phi_count(x, m, r)
    if B <= 0 or A <= 0:
        return float("inf")
    return float(-np.log(A / B))


def approximate_entropy(rr_ms: np.ndarray, m: int = 2, r: Optional[float] = None) -> float:
    """ApEn(m=2, r=0.2·SD default)."""
    x = _as_ms(rr_ms)
    if x.size < (m + 1):
        return float("nan")
    r = 0.2 * np.std(x) if r is None else r

    def _C(mm: int) -> float:
        x_m = _sliding_window_view(x, mm)
        if x_m.size == 0:
            return float("-inf")
        N = x_m.shape[0]
        Cm = np.zeros(N, dtype=np.float64)
        for i in range(N):
            d = np.max(np.abs(x_m - x_m[i]), axis=1)
            Cm[i] = np.mean(d <= r)
        return float(np.mean(np.log(Cm + 1e-12)))

    return float(_C(m) - _C(m + 1))


def dfa_alpha(rr_ms: np.ndarray, scale_min: int = 4, scale_max: int = 64, alpha1_range: Tuple[int, int] = (4, 16), alpha2_range: Tuple[int, int] = (16, 64)) -> Dict:
    """DFA α1/α2 with 2-sided boxing & log-log slope fits."""
    x = _as_ms(rr_ms)
    N = x.size
    if N < scale_max * 4:
        warnings.warn("RR series short for robust DFA; results may be unstable.")
    y = np.cumsum(x - np.mean(x))
    scales = np.arange(scale_min, scale_max + 1, dtype=int)
    Fn = np.full_like(scales, np.nan, dtype=np.float64)

    for idx, n in enumerate(scales):
        if n < 2:
            continue
        n_boxes = N // n
        if n_boxes < 2:
            continue
        # two non-overlapping coverings (front/back)
        y1 = y[:n_boxes * n].reshape(n_boxes, n)
        y2 = y[N - n_boxes * n:].reshape(n_boxes, n)

        F_vals = []
        for segment in (y1, y2):          # segment shape: (n_boxes, n)
            t = np.arange(n)
            # vectorized linear fit over each box: returns (2, n_boxes)
            coeffs = np.polyfit(t, segment.T, deg=1)
            slope = coeffs[0][:, None]    # (n_boxes, 1)
            intercept = coeffs[1][:, None]# (n_boxes, 1)
            trend = slope * t[None, :] + intercept  # (n_boxes, n)
            res = segment - trend
            # RMS per box, then average across boxes
            F_vals.append(np.sqrt(np.mean(res ** 2, axis=1)))  # (n_boxes,)

        # combine both coverings and average
        F_all = np.concatenate(F_vals)  # (2*n_boxes,)
        Fn[idx] = float(np.mean(F_all))

    def _fit_alpha(lo, hi):
        m = (scales >= lo) & (scales <= hi) & np.isfinite(Fn) & (Fn > 0)
        if m.sum() < 3:
            return float("nan")
        s = np.polyfit(np.log(scales[m]), np.log(Fn[m]), deg=1)[0]
        return float(s)

    return {
        "dfa_scales_beats": scales.tolist(),
        "dfa_Fn": [float(v) if np.isfinite(v) else float("nan") for v in Fn],
        "dfa_alpha1": _fit_alpha(*alpha1_range),
        "dfa_alpha2": _fit_alpha(*alpha2_range),
    }


def multiscale_entropy(rr_ms: np.ndarray, max_scale: int = 20, m: int = 2, r: Optional[float] = None) -> Dict:
    """MSE up to scale max_scale using SampEn with fixed r (0.15·SD default)."""
    x = _as_ms(rr_ms)
    if x.size < 10:
        return {"mse_scales": [], "mse_sampen": []}
    r = 0.15 * np.std(x) if r is None else r

    def _coarse(signal: np.ndarray, scale: int) -> np.ndarray:
        n = signal.size // scale
        return signal[:n * scale].reshape(n, scale).mean(axis=1) if n > 0 else np.array([])

    scales = list(range(1, int(max_scale) + 1))
    s_values = [sample_entropy(_coarse(x, s), m=m, r=r) for s in scales]
    return {"mse_scales": scales, "mse_sampen": [float(v) if np.isfinite(v) else float("nan") for v in s_values]}


# ------------------------------ Pipeline & API --------------------------------

def hrv_pipeline(rr_ms: np.ndarray, fs_interp: float = 4.0, psd_method: str = "welch", ar_order: int = 16, do_mse: bool = True) -> Dict:
    """Full HRV analysis from RR (ms)."""
    rr_ms = _as_ms(rr_ms)
    mask = _valid_mask_rr(rr_ms)
    rr_nn = rr_ms[mask]
    out = {}
    out["time_domain"] = time_domain_metrics(rr_nn)
    out["prsa"] = prsa_dc_ac(rr_nn)
    out["freq_domain"] = frequency_domain_metrics(rr_nn, method=psd_method, fs_interp=fs_interp, ar_order=ar_order)
    out["nonlinear"] = {
        **poincare_metrics(rr_nn),
        "sampen_m2_r02": float(sample_entropy(rr_nn, m=2, r=0.2 * np.std(rr_nn))) if rr_nn.size >= 20 else float("nan"),
        "apen_m2_r02": float(approximate_entropy(rr_nn, m=2, r=0.2 * np.std(rr_nn))) if rr_nn.size >= 20 else float("nan"),
        **dfa_alpha(rr_nn),
        **(multiscale_entropy(rr_nn, max_scale=20) if do_mse else {"mse_scales": [], "mse_sampen": []}),
    }
    out["rr_used_ms"] = rr_nn
    return out


def hdf5_list_datasets(h5_path: str) -> List[Tuple[str, Tuple[int, ...], str]]:
    """Return (path, shape, dtype) for all HDF5 datasets."""
    if h5 is None:
        raise RuntimeError("h5py is required to inspect HDF5 files.")
    out = []
    with h5.File(h5_path, "r") as f:
        def _visit(name, obj):
            if isinstance(obj, h5.Dataset):
                out.append((name, tuple(obj.shape), str(obj.dtype)))
        f.visititems(_visit)
    return out


def hdf5_guess_rr(h5_path: str) -> Dict:
    """
    Heuristics:
      1) datasets containing 'rr','ibi','interbeat'
      2) else diff R-peak arrays
      3) else derive RR from any '/hr/.../value' dataset (BedMaster HR stream)
    """
    info = {"rr_ms": None, "candidates": [], "from": None}
    if h5 is None:
        return info
    with h5.File(h5_path, "r") as f:
        candidates = []
        def _visit(name, obj):
            if isinstance(obj, h5.Dataset):
                lname = name.lower()
                if any(k in lname for k in ["rr", "ibi", "interbeat"]):
                    candidates.append(name)
        f.visititems(_visit)
        info["candidates"] = candidates

        for path in candidates:
            try:
                arr = np.asarray(f[path]).squeeze()
                if arr.ndim != 1 or arr.size < 5:
                    continue
                arr_ms = _as_ms(arr)
                if _valid_mask_rr(arr_ms).mean() > 0.6:
                    info["rr_ms"] = arr_ms; info["from"] = f"dataset:{path}"; return info
            except Exception:
                pass

        rpeak_paths = []
        def _visit_r(name, obj):
            if isinstance(obj, h5.Dataset):
                lname = name.lower()
                if "rpeak" in lname or "r_peaks" in lname or "rpeaks" in lname or lname.endswith("/peaks"):
                    rpeak_paths.append(name)
        f.visititems(_visit_r)
        for path in rpeak_paths:
            try:
                peaks = np.asarray(f[path]).squeeze()
                if peaks.ndim != 1 or peaks.size < 5:
                    continue
                fs = None
                parent = "/".join(path.split("/")[:-1])
                for cand in ["fs", "sampling_rate", "sample_rate", "frequency"]:
                    for k in [parent + "/" + cand, cand, parent + "/meta/" + cand]:
                        if k in f and isinstance(f[k], h5.Dataset):
                            val = np.array(f[k]).squeeze()
                            if np.isscalar(val) and 10 <= float(val) <= 5000:
                                fs = float(val); break
                    if fs is not None: break
                if fs is not None and np.all(np.diff(peaks) > 0):
                    if np.issubdtype(peaks.dtype, np.integer) or np.median(peaks) > 10:
                        t = peaks / fs
                    else:
                        t = peaks
                else:
                    t = peaks  # assume seconds
                rr_ms = np.diff(t.astype(np.float64)) * 1000.0
                if _valid_mask_rr(rr_ms).mean() > 0.6:
                    info["rr_ms"] = rr_ms; info["from"] = f"diff(rpeaks):{path}"; return info
            except Exception:
                pass

        # HR -> RR fallback (BedMaster)
        try:
            hr_paths = []
            def _visit_hr(name, obj):
                if isinstance(obj, h5.Dataset):
                    lname = name.lower()
                    if "/hr" in lname and lname.endswith("/value"):
                        hr_paths.append(name)
            f.visititems(_visit_hr)
            for path in hr_paths:
                hr = np.asarray(f[path]).squeeze().astype(np.float64)
                hr = hr[np.isfinite(hr)]
                if hr.size < 10: continue
                m = (hr >= 20) & (hr <= 240)
                if m.mean() < 0.8: continue
                rr_ms = 60000.0 / hr[m]
                if _valid_mask_rr(rr_ms).mean() > 0.6:
                    info["rr_ms"] = rr_ms; info["from"] = f"derived_from_hr:{path}"; return info
        except Exception:
            pass
    return info


def segment_metrics(rr_ms: np.ndarray, window_sec: int = 300, step_sec: int = 60, fs_interp: float = 4.0, psd_method: str = "welch") -> pd.DataFrame:
    """Rolling-window metrics (default 5-min window, 1-min step)."""
    rr_ms = _as_ms(rr_ms)
    t = _cumsums(rr_ms)
    if t.size == 0:
        return pd.DataFrame()
    rows = []
    start = 0.0
    end_total = t[-1]
    while start + window_sec <= end_total:
        end = start + window_sec
        m = (t >= start) & (t < end)
        idx = np.where(m)[0]
        if idx.size >= 30:
            rr_win = rr_ms[idx[0]: idx[-1] + 1]
            td = time_domain_metrics(rr_win)
            fd = frequency_domain_metrics(rr_win, method=psd_method, fs_interp=fs_interp)
            nl = {**poincare_metrics(rr_win),
                  "sampen_m2_r02": float(sample_entropy(rr_win, m=2, r=0.2 * np.std(rr_win))) if rr_win.size >= 100 else float("nan"),
                  **dfa_alpha(rr_win)}
            rows.append({
                "t_start_s": start, "t_end_s": end,
                **{k: v for k, v in (td or {}).items() if not isinstance(v, (list, np.ndarray))},
                **{k: v for k, v in (fd or {}).items() if not isinstance(v, (list, np.ndarray)) and not k.startswith("psd_")},
                **nl
            })
        start += step_sec
    return pd.DataFrame(rows)


def analyze_rr(rr_ms: np.ndarray, fs_interp: float = 4.0, psd_method: str = "welch", ar_order: int = 16, do_segments: bool = True, window_sec: int = 300, step_sec: int = 60) -> Tuple[Dict, pd.DataFrame]:
    """Full analysis from RR + optional rolling windows."""
    summary = hrv_pipeline(rr_ms, fs_interp=fs_interp, psd_method=psd_method, ar_order=ar_order, do_mse=True)
    segments = segment_metrics(rr_ms, window_sec=window_sec, step_sec=step_sec, fs_interp=fs_interp, psd_method=psd_method) if do_segments else pd.DataFrame()
    return summary, segments


def analyze_hdf5(h5_path: str, dataset_path: Optional[str] = None, **kwargs) -> Tuple[Dict, pd.DataFrame, Dict]:
    """Load RR via heuristics (or explicit dataset) and run analyze_rr()."""
    if dataset_path is None:
        info = hdf5_guess_rr(h5_path)
        rr_ms = info.get("rr_ms", None)
        source = info.get("from", None)
        candidates = info.get("candidates", [])
    else:
        if h5 is None:
            raise RuntimeError("h5py is required to read HDF5 files.")
        with h5.File(h5_path, "r") as f:
            rr = np.asarray(f[dataset_path]).squeeze()
        rr_ms = _as_ms(rr); source = f"dataset:{dataset_path}"; candidates = [dataset_path]
    if rr_ms is None or len(rr_ms) < 10:
        raise ValueError("Failed to obtain RR intervals from HDF5. Use --list to inspect datasets and provide --dataset explicitly.")
    summary, segments = analyze_rr(rr_ms, **kwargs)
    meta = {"rr_source": source, "candidates": candidates, "n_beats": int(len(rr_ms))}
    return summary, segments, meta


# ------------------------------ Export helpers --------------------------------

def summary_to_dataframe(summary_dict: Dict) -> pd.DataFrame:
    """Flatten nested summary dict into a single-row DataFrame."""
    flat: Dict[str, float] = {}
    def _flatten(prefix, d):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict): _flatten(key, v)
            elif isinstance(v, (list, np.ndarray)): continue
            else: flat[key] = v
    _flatten("", summary_dict)
    return pd.DataFrame([flat])


def export_results(summary: Dict, segments_df: pd.DataFrame, out_prefix: str) -> Dict[str, Optional[str]]:
    """Write summary.csv, segments.csv, psd.csv, mse.csv with fallback if needed."""
    # helper to write with local fallback
    def _write_csv_with_fallback(df: pd.DataFrame, path: str, tag: str) -> Tuple[Optional[str], Optional[Dict]]:
        try:
            df.to_csv(path, index=False)
            return path, None
        except Exception as e:
            # second safety net: fallback to ~/HRV_Output
            fallback_dir = os.path.join(os.path.expanduser("~"), "HRV_Output")
            os.makedirs(fallback_dir, exist_ok=True)
            new_path = os.path.join(fallback_dir, os.path.basename(path))
            try:
                df.to_csv(new_path, index=False)
                print(f"[WARN] Could not write {tag} to '{path}' ({type(e).__name__}: {e}); wrote to '{new_path}' instead.")
                return new_path, {"fell_back": True, "why": f"{type(e).__name__}: {e}", "from": path, "to": new_path}
            except Exception as e2:
                print(f"[ERROR] Failed to write {tag} to both '{path}' and '{new_path}'. Error: {e2}")
                return None, {"fell_back": True, "why": f"{type(e2).__name__}: {e2}", "from": path, "to": new_path}

    out_paths: Dict[str, Optional[str]] = {"summary_csv": None, "segments_csv": None, "psd_csv": None, "mse_csv": None}

    df_summary = summary_to_dataframe(summary)
    summary_path = f"{out_prefix}_summary.csv"
    out_paths["summary_csv"], _ = _write_csv_with_fallback(df_summary, summary_path, "summary")

    if segments_df is not None and not segments_df.empty:
        segments_path = f"{out_prefix}_segments.csv"
        out_paths["segments_csv"], _ = _write_csv_with_fallback(segments_df, segments_path, "segments")

    fd = summary.get("freq_domain", {})
    f = fd.get("psd_freqs_hz", np.array([])); P = fd.get("psd_ms2_per_hz", np.array([]))
    if f.size and P.size:
        df_psd = pd.DataFrame({"freq_hz": f, "psd_ms2_per_hz": P})
        psd_path = f"{out_prefix}_psd.csv"
        out_paths["psd_csv"], _ = _write_csv_with_fallback(df_psd, psd_path, "psd")

    nl = summary.get("nonlinear", {})
    scales = nl.get("mse_scales", []); mse = nl.get("mse_sampen", [])
    if scales and mse:
        df_mse = pd.DataFrame({"scale": scales, "sampen": mse})
        mse_path = f"{out_prefix}_mse.csv"
        out_paths["mse_csv"], _ = _write_csv_with_fallback(df_mse, mse_path, "mse")

    return out_paths


# ----------------------------------- CLI --------------------------------------

def run_with_params(h5_path: str = DEFAULT_H5,
                    dataset: Optional[str] = DEFAULT_DATASET,
                    out_prefix: str = DEFAULT_OUT_PREFIX,
                    fs_interp: float = 4.0,
                    psd_method: str = "welch",
                    ar_order: int = 16,
                    window_sec: int = 300,
                    step_sec: int = 60,
                    do_segments: bool = True,
                    list_only: bool = False,
                    emit_json: bool = False):
    """Entry-point callable from code or CLI."""

    # Optional: list datasets
    try:
        ds = hdf5_list_datasets(h5_path)
        print("=== HDF5 datasets (first 80) ===")
        for pth, shape, dt in ds[:80]:
            print(f"{pth:70s} | shape={shape!s:16s} | dtype={dt}")
        if len(ds) > 80:
            print(f"... ({len(ds) - 80} more)")
        if list_only:
            return {"listed": True, "h5": h5_path}
    except Exception as e:
        print(f"(Could not list datasets: {e})")

    # Analyze + export
    summary, segments, meta = analyze_hdf5(
        h5_path,
        dataset_path=dataset,
        fs_interp=fs_interp,
        psd_method=psd_method,
        ar_order=ar_order,
        do_segments=do_segments,
        window_sec=window_sec,
        step_sec=step_sec
    )

    # Ensure output prefix is writable; fall back if necessary
    out_prefix_final, fbinfo = _resolve_out_prefix(out_prefix, verbose=True)

    paths = export_results(summary, segments, out_prefix_final)

    print("\n=== Source meta ==="); print(json.dumps(meta, indent=2))
    if fbinfo:
        print("\n=== Output fallback info ==="); print(json.dumps(fbinfo, indent=2))
    print("\n=== CSV outputs ==="); print(json.dumps(paths, indent=2))

    if emit_json:
        out_json = {
            "meta": meta,
            "exports": paths,
            "time_domain": summary.get("time_domain", {}),
            "freq_domain": {k: v for k, v in summary.get("freq_domain", {}).items()
                            if not (isinstance(v, (list, np.ndarray)) and k.startswith("psd_"))},
            "nonlinear": {k: v for k, v in summary.get("nonlinear", {}).items()
                          if k not in ("mse_scales", "mse_sampen", "dfa_Fn", "dfa_scales_beats")},
            "prsa": {k: v for k, v in summary.get("prsa", {}).items() if not isinstance(v, list)},
            "output_fallback": fbinfo or {}
        }
        print("\n=== JSON (key results) ===")
        print(json.dumps(out_json, indent=2))

    return {"meta": meta, "exports": paths, "fallback": fbinfo}


def main():
    p = argparse.ArgumentParser(description="HRV analysis (all-in-one). If no args are provided, defaults are used.")
    p.add_argument("--h5", default=DEFAULT_H5, help=f"HDF5 path [default: {DEFAULT_H5}]")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Explicit RR/IBI dataset path inside HDF5")
    p.add_argument("--out", default=DEFAULT_OUT_PREFIX, help=f"Output prefix [default: {DEFAULT_OUT_PREFIX}]")
    p.add_argument("--fs_interp", type=float, default=4.0, help="Interpolation rate for PSD (Hz)")
    p.add_argument("--psd_method", choices=["welch", "ar"], default="welch", help="PSD method")
    p.add_argument("--ar_order", type=int, default=16, help="AR order if psd_method='ar'")
    p.add_argument("--window_sec", type=int, default=300, help="Segment window length (sec)")
    p.add_argument("--step_sec", type=int, default=60, help="Segment step (sec)")
    p.add_argument("--no_segments", action="store_true", help="Disable segmented (rolling) metrics")
    p.add_argument("--list", dest="list_only", action="store_true", help="List datasets and exit")
    p.add_argument("--json", dest="emit_json", action="store_true", help="Print JSON summary of outputs")
    args = p.parse_args()

    # Run with given or default params
    run_with_params(
        h5_path=args.h5,
        dataset=args.dataset,
        out_prefix=args.out,
        fs_interp=args.fs_interp,
        psd_method=args.psd_method,
        ar_order=args.ar_order,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        do_segments=(not args.no_segments),
        list_only=args.list_only,
        emit_json=args.emit_json
    )


if __name__ == "__main__":
    main()
