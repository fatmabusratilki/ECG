#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECG Morphology & Repolarization Analysis — Bedmaster-friendly, single-file script with plots & QC
------------------------------------------------------------------------------------------------
Computes (per spec):
  • R-peak amplitude (lead-specific) & variability; R/S ratio; QRS slope/area/energy.
  • ST: J-point and +60/+80 ms amplitudes; lead-wise & max deviation (uses stv*/sti* if available).
  • T-wave amplitude & frontal T-axis (I & aVF), T-wave alternans (time-domain & spectral).
  • QRS fragmentation (notches), morphology similarity (beat↔median correlation),
    wavelet energies across scales.

Also:
  • Stronger dataset discovery for Bedmaster HDF5: prefers waveforms, avoids vitals/time/value/event series.
  • Attempts to read fs from sibling 'sample_freq' dataset when attrs are missing.
  • Prints the EXACT dataset path chosen; can list all items with --list-all.
  • Saves CSVs + plots (PNG + combined PDF) + a small QC text summary.

NOT a medical device. Research use only.
"""

import os
import sys
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import h5py
from scipy import signal

# Non-interactive backend for file output
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import pywt
    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

# -----------------------------
# Defaults (your UNC path here)
# -----------------------------
DEFAULT_INPUT = "/Users/barko/Desktop/BLUESENSE/OPIOID_OVERDOSE/C/raw/3254689660.hd5"
DEFAULT_OUTPUT_PREFIX = "/Users/barko/Desktop/BLUESENSE/OPIOID_OVERDOSE/C/"
DEFAULT_MAX_DURATION_SEC = 600.0
DEFAULT_MAX_BEATS = 600
DEFAULT_FS_IF_MISSING = None  # set e.g. 500.0 if your file lacks fs

# -----------------------------
# Utilities
# -----------------------------
def safe_trapezoid(y, dx: float) -> float:
    """Compat wrapper to avoid deprecation warnings for np.trapz."""
    try:
        return float(np.trapezoid(y, dx=dx))  # numpy >= 2.0
    except Exception:
        return float(np.trapz(y, dx=dx))      # fallback

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def downsample_for_plot(x: np.ndarray, t: np.ndarray, max_points: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= max_points:
        return x, t
    step = max(1, n // max_points)
    return x[::step], t[::step]

def choose_twa_lead(lead_names: List[str], primary_name: str) -> str:
    pref = ["V5", "V4", "V6", "II", "V2", "I", "aVF"]
    names_up = [n.upper() for n in lead_names]
    for p in pref:
        if p in names_up:
            return lead_names[names_up.index(p)]
    return primary_name

# -----------------------------
# HDF5 discovery helpers
# -----------------------------
def collect_h5_structure(group, prefix="") -> List[Tuple[str, str, tuple, str]]:
    items = []
    for k in group.keys():
        obj = group[k]
        path = f"{prefix}/{k}".replace("//", "/")
        if isinstance(obj, h5py.Dataset):
            items.append((path, str(obj.dtype), obj.shape, "dataset"))
        elif isinstance(obj, h5py.Group):
            items.append((path, "NA", (), "group"))
            items.extend(collect_h5_structure(obj, prefix=path))
    return items

def _bytes_to_str_list(b) -> List[str]:
    if isinstance(b, (list, tuple, np.ndarray)):
        out = []
        for x in b:
            if isinstance(x, (bytes, bytearray, np.bytes_)):
                out.append(x.decode("utf-8", errors="ignore"))
            else:
                out.append(str(x))
        return out
    if isinstance(b, (bytes, bytearray, np.bytes_)):
        return [b.decode("utf-8", errors="ignore")]
    return [str(b)]

def get_channel_names(ds: h5py.Dataset) -> Optional[List[str]]:
    for k in ["labels","label","names","channels","channel_names","lead_names","leads","columns","col_names"]:
        if k in ds.attrs:
            try:
                return _bytes_to_str_list(ds.attrs[k])
            except Exception:
                pass
    return None

# --- Sampling rate helpers ---
def find_fs_in_attrs(h5: h5py.File, ds: Optional[h5py.Dataset]) -> Optional[float]:
    keys = ["fs","Fs","FS","sampling_rate","sample_rate","SampleRate","SampleRateHz","SamplingRateHz","frequency","Hz"]
    if ds is not None:
        for k in keys:
            if k in ds.attrs:
                try:
                    fs = float(ds.attrs[k])
                    if fs > 0:
                        return fs
                except Exception:
                    pass
    for k in keys:
        if k in h5.attrs:
            try:
                fs = float(h5.attrs[k])
                if fs > 0:
                    return fs
            except Exception:
                pass
    return None

def find_fs_from_sibling_sample_freq(ds: h5py.Dataset) -> Optional[float]:
    """
    Many Bedmaster exports include a sibling dataset named 'sample_freq' within the same group.
    It can be a scalar, a 1x1, or a structured record (e.g., [('f0','<f8'),('f1','<i8')]).
    """
    try:
        grp = ds.parent
        if grp is None:
            return None
        if "sample_freq" in grp:
            sf = grp["sample_freq"][()]
            # Try common shapes/dtypes
            if isinstance(sf, np.ndarray):
                if sf.shape == (1,):
                    sf0 = sf[0]
                    try:
                        return float(sf0) if np.isscalar(sf0) else float(np.array(sf0).astype(float)[0])
                    except Exception:
                        pass
                if sf.shape == () and np.isscalar(sf):
                    return float(sf)
                # Structured 1-element record
                if sf.dtype.fields and sf.shape == (1,):
                    # Pick the first float-like field
                    for name, (dt, _) in sf.dtype.fields.items():
                        try:
                            val = sf[name][0]
                            v = float(val)
                            if v > 0:
                                return v
                        except Exception:
                            continue
    except Exception:
        pass
    return None

def looks_like_waveform_candidate(path: str, dtype: str, shape: tuple) -> bool:
    """
    Prefer continuous waveforms:
      - float dtype (or at least not int-only vitals)
      - very long vectors or long 2-D arrays
      - avoid typical vitals/time/event/samples datasets
    """
    p = path.lower()
    # Explicitly exclude metadata/time/event
    if any(ex in p for ex in ["/vitals/", "/time", "/event", "samples_per_ts", "sample_freq"]):
        return False
    
    # If it's a 'value' dataset, it MUST be in a waveforms group to be valid
    if "/value" in p and "waveforms" not in p:
        return False

    if "bedmaster" in p and any(k in p for k in ["wave", "full_disclosure", "ecg"]):
        pass  # favorable
    # Dtype
    score = 0
    if any(s in dtype for s in ["float32", "float64", "float", "f4", "f8"]):
        score += 3
    elif "int" in dtype:
        score -= 2
    # Length/shape
    if len(shape) == 1 and shape[0] >= 200000:
        score += 3
    if len(shape) == 2 and (max(shape) >= 200000 or min(shape) in (1, 2, 3, 8, 12)):
        score += 3
    return score >= 3

def discover_ecg_dataset(h5: h5py.File, dataset_override: Optional[str] = None, list_all: bool = False):
    """
    Return (dataset, channel_names, fs, structure_list, chosen_path).
    Discovery prefers waveform-like datasets and avoids vitals/time/value/event series.
    """
    all_items = collect_h5_structure(h5, prefix="")
    if list_all:
        print("---- FULL HDF5 STRUCTURE ----")
        for row in all_items:
            print(row)
        print(f"({len(all_items)} total items)")

    if dataset_override is not None:
        if dataset_override in h5:
            ds = h5[dataset_override]
            ch = get_channel_names(ds)
            fs = find_fs_in_attrs(h5, ds) or find_fs_from_sibling_sample_freq(ds)
            return ds, ch, fs, all_items, dataset_override
        raise KeyError(f"Dataset '{dataset_override}' not found in file.")

    # Rank candidates (Adayları puanla)
    candidates = []
    for path, dtype, shape, kind in all_items:
        if kind != "dataset":
            continue
        try:
            if looks_like_waveform_candidate(path, dtype, shape):
                candidates.append((path, dtype, shape))
        except Exception:
            continue

    # Fallback: Hiçbir şey bulunamazsa filtreleri gevşet
    if not candidates:
        for path, dtype, shape, kind in all_items:
            if kind != "dataset":
                continue
            p = path.lower()
            # Yine de bariz metadata dosyalarını ele
            if any(ex in p for ex in ["/vitals/", "/time", "/event", "samples_per_ts", "sample_freq"]):
                continue
            candidates.append((path, dtype, shape))

    if not candidates:
        return None, None, None, all_items, None

    # Score + sort (Puanla ve sırala)
    def score_item(path, dtype, shape):
        s = 0
        p = path.lower()
        if any(w in p for w in ["wave", "full_disclosure", "/ecg"]):
            s += 3
        if any(t in dtype for t in ["float32", "float64", "float", "f4", "f8"]):
            s += 3
        if len(shape) == 1 and shape[0] >= 200000:
            s += 3
        if len(shape) == 2 and (max(shape) >= 200000 or min(shape) in (1, 2, 3, 8, 12)):
            s += 3
        return s

    candidates.sort(key=lambda it: score_item(*it), reverse=True)

    # Adayları sırayla dene
    for path, _, _ in candidates:
        try:
            ds = h5[path]
            ch = get_channel_names(ds)
            fs = find_fs_in_attrs(h5, ds) or find_fs_from_sibling_sample_freq(ds)
            return ds, ch, fs, all_items, path
        except Exception:
            continue

    return None, None, None, all_items, None

# -----------------------------
# Filtering & preprocessing
# -----------------------------
def butter_bandpass(low, high, fs, order=4):
    return signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")

def highpass(fc, fs, order=2):
    return signal.butter(order, fc, btype="highpass", fs=fs, output="sos")

def lowpass(fc, fs, order=4):
    return signal.butter(order, fc, btype="lowpass", fs=fs, output="sos")

def zero_phase_filter(x, sos):
    return signal.sosfiltfilt(sos, x)

def preprocess_lead(x, fs):
    """High-pass 0.5 Hz + Low-pass 45 Hz to remove baseline & noise."""
    try:
        x = zero_phase_filter(x, highpass(0.5, fs, order=2))
        x = zero_phase_filter(x, lowpass(45, fs, order=4))
    except Exception:
        pass
    return x

# -----------------------------
# R-peak detection (Pan–Tompkins-like)
# -----------------------------
def detect_r_peaks(primary_lead: np.ndarray, fs: float) -> np.ndarray:
    """Derivative→square→moving-window integration + adaptive threshold, then R refinement."""
    try:
        qrs_sos = butter_bandpass(5, 15, fs, order=2)
        y = zero_phase_filter(primary_lead, qrs_sos)
    except Exception:
        y = primary_lead.copy()
    dy = np.diff(y, prepend=y[0])
    sq = dy**2
    win = max(1, int(0.150 * fs))
    integ = signal.convolve(sq, np.ones(win)/win, mode="same")
    thr = np.median(integ) + 3.5 * np.median(np.abs(integ - np.median(integ)))
    distance = int(0.25 * fs)  # 250 ms refractory
    peaks, _ = signal.find_peaks(integ, height=thr, distance=distance)
    base = preprocess_lead(primary_lead, fs)
    refined = []
    halfw = int(0.100 * fs)
    for p in peaks:
        a = max(0, p - halfw)
        b = min(len(base), p + halfw + 1)
        if a < b:
            refined.append(np.argmax(base[a:b]) + a)
    return np.array(sorted(set(refined)), dtype=int)

# -----------------------------
# Beat delineation
# -----------------------------
@dataclass
class BeatWindows:
    qrs_on: int
    r_idx: int
    qrs_off: int
    j_idx: int
    t_start: int
    t_peak: int
    t_end: int
    baseline_idx: int

def delineate_single_beat(x: np.ndarray, r_idx: int, fs: float) -> Optional[BeatWindows]:
    """QRS onset/offset (J) via slope energy plateaus; T-window via 100–500 ms post-J, T-peak via |max|."""
    n = len(x)
    a = max(0, r_idx - int(0.120 * fs))
    b = min(n, r_idx + int(0.160 * fs))
    seg = x[a:b]
    if len(seg) < 10:
        return None

    d = np.diff(seg, prepend=seg[0])
    e = signal.convolve(
        d**2,
        np.ones(int(max(1, 0.015 * fs))) / max(1, int(0.015 * fs)),
        mode="same"
    )
    e = e / (np.max(e) + 1e-12)

    rel_r = r_idx - a
    low_span = int(0.010 * fs)

    # QRS onset: backward to near-zero slope energy
    qrs_on_rel = None
    for i in range(rel_r, max(0, rel_r - int(0.120*fs)), -1):
        j0 = max(0, i - low_span)
        if np.all(e[j0:i+1] < 0.05):
            qrs_on_rel = i
    if qrs_on_rel is None:
        qrs_on_rel = max(0, rel_r - int(0.060 * fs))

    # QRS offset: forward to near-zero slope energy
    qrs_off_rel = None
    for i in range(rel_r, min(len(e)-1, rel_r + int(0.160*fs))):
        j1 = min(len(e)-1, i + low_span)
        if np.all(e[i:j1] < 0.05):
            qrs_off_rel = i
            break
    if qrs_off_rel is None:
        qrs_off_rel = min(len(seg)-1, rel_r + int(0.060 * fs))

    qrs_on = a + qrs_on_rel
    qrs_off = a + qrs_off_rel
    j_idx = qrs_off

    # Baseline: median region [-250, -50] ms pre-R
    preA = max(0, r_idx - int(0.250*fs))
    preB = max(0, r_idx - int(0.050*fs))
    if preB <= preA + 5:
        baseline_idx = preA
    else:
        preSeg = x[preA:preB]
        baseline_idx = preA + int(np.argmin(np.abs(preSeg - np.median(preSeg))))

    # T window: 100–500 ms post-J (fallback 80–400)
    tA = min(n-1, j_idx + int(0.100 * fs))
    tB = min(n,   j_idx + int(0.500 * fs))
    if tB - tA < max(5, int(0.080*fs)):
        tA = min(n-1, j_idx + int(0.080 * fs))
        tB = min(n,   j_idx + int(0.400 * fs))

    wlen = max(3, int(0.010 * fs) | 1)
    try:
        smooth = signal.savgol_filter(
            x[tA:tB],
            window_length=min(wlen, ((tB - tA - 1) | 1)),
            polyorder=2
        )
    except Exception:
        smooth = x[tA:tB]
    if len(smooth) == 0:
        return None

    t_peak = tA + int(np.argmax(np.abs(smooth)))

    td = np.diff(smooth, prepend=smooth[0])
    te = signal.convolve(
        td**2,
        np.ones(int(max(1, 0.015 * fs))) / max(1, int(0.015 * fs)),
        mode="same"
    )
    te = te / (np.max(te) + 1e-12)

    t_end_rel = None
    for i in range(int(np.argmax(np.abs(smooth))), len(te)):
        j1 = min(len(te)-1, i + int(0.015 * fs))
        if np.all(te[i:j1] < 0.03):
            t_end_rel = i
            break
    if t_end_rel is None:
        t_end_rel = min(len(smooth)-1, int(np.argmax(np.abs(smooth))) + int(0.150 * fs))
    t_end = tA + t_end_rel

    return BeatWindows(qrs_on=qrs_on, r_idx=r_idx, qrs_off=qrs_off, j_idx=j_idx,
                       t_start=tA, t_peak=t_peak, t_end=t_end, baseline_idx=baseline_idx)

# -----------------------------
# Feature computations
# -----------------------------
def qrs_metrics(x: np.ndarray, win: BeatWindows, fs: float) -> Dict[str, float]:
    seg = x[win.qrs_on:win.qrs_off+1]
    if len(seg) < 3:
        return {k: np.nan for k in ["r_amp","s_amp","rs_ratio","qrs_max_slope","qrs_area","qrs_energy"]}

    r_amp = x[win.r_idx] - x[win.baseline_idx]
    post = x[win.r_idx:win.qrs_off+1]
    s_amp = np.min(post) - x[win.baseline_idx] if len(post)>0 else np.nan

    d = np.diff(seg, prepend=seg[0]) * fs
    qrs_max_slope = float(np.max(np.abs(d)))

    seg_rel = seg - x[win.baseline_idx]
    qrs_area = safe_trapezoid(np.abs(seg_rel), dx=1/fs)
    qrs_energy = float(np.sum(seg_rel**2))
    rs_ratio = float((r_amp)/abs(s_amp)) if (s_amp is not None and s_amp != 0) else np.nan
    return dict(r_amp=float(r_amp), s_amp=float(s_amp), rs_ratio=rs_ratio,
                qrs_max_slope=qrs_max_slope, qrs_area=qrs_area, qrs_energy=qrs_energy)

def st_metrics(x: np.ndarray, win: BeatWindows, fs: float) -> Dict[str, float]:
    base = x[win.baseline_idx]
    st_j = float(x[win.j_idx] - base)
    j60 = float(x[min(len(x)-1, win.j_idx + int(0.060*fs))] - base)
    j80 = float(x[min(len(x)-1, win.j_idx + int(0.080*fs))] - base)
    return dict(st_j=st_j, st_60ms=j60, st_80ms=j80)

def t_metrics(x: np.ndarray, win: BeatWindows, fs: float) -> Dict[str, float]:
    base = x[win.baseline_idx]
    t_amp = float(x[win.t_peak] - base)
    t_win = x[win.t_start:win.t_end+1] - base
    t_area = safe_trapezoid(t_win, dx=1/fs) if len(t_win) else np.nan
    return dict(t_peak_amp=t_amp, t_area=t_area)

def fragmentation_metrics(x: np.ndarray, win: BeatWindows, fs: float) -> Dict[str, float]:
    seg = x[win.qrs_on:win.qrs_off+1]
    if len(seg) < 4:
        return dict(qrs_notches=np.nan, qrs_is_fragmented=np.nan)
    d = np.diff(seg, prepend=seg[0])
    zc = np.where(np.diff(np.sign(d)) != 0)[0]
    r_amp = abs(x[win.r_idx] - x[win.baseline_idx])
    thr = 0.05 * r_amp + 1e-9
    notches, last_idx = 0, -1
    for i in zc:
        if last_idx >= 0 and (i - last_idx) < int(0.008 * fs):
            continue
        a = max(0, i-1); b = min(len(seg)-1, i+2)
        if (np.max(seg[a:b]) - np.min(seg[a:b])) > thr:
            notches += 1; last_idx = i
    is_frag = 1.0 if notches >= 2 else 0.0
    return dict(qrs_notches=float(notches), qrs_is_fragmented=is_frag)

def beat_vector(all_leads: np.ndarray, r_idx: int, fs: float,
                w_pre=0.150, w_post=0.400) -> Optional[np.ndarray]:
    """Concatenate baseline-relative multi-lead waveforms in a fixed window around R."""
    n = all_leads.shape[1]
    a = int(r_idx - w_pre*fs); b = int(r_idx + w_post*fs)
    if a < 0 or b > n: return None
    arrs = []
    for li in range(all_leads.shape[0]):
        x = all_leads[li, a:b].copy()
        preA = max(0, int(r_idx - 0.250*fs))
        preB = max(0, int(r_idx - 0.050*fs))
        base = np.median(all_leads[li, preA:preB]) if preB > preA else 0.0
        arrs.append(x - base)
    return np.concatenate(arrs, axis=0)

def morphology_correlation(all_leads: np.ndarray, r_locs: np.ndarray, fs: float):
    """Beat-to-median correlation (across leads). Returns (corrs, median_template)."""
    vecs, idx_map = [], []
    for i, r in enumerate(r_locs):
        v = beat_vector(all_leads, r, fs)
        if v is not None:
            vecs.append(v); idx_map.append(i)
    if len(vecs) < 3:
        return np.full(len(r_locs), np.nan, dtype=float), np.array([])
    M = np.vstack(vecs)
    med = np.median(M, axis=0)
    corrs = []
    for i in range(M.shape[0]):
        a = M[i]
        corrs.append(np.corrcoef(a, med)[0,1] if (np.std(a)>0 and np.std(med)>0) else np.nan)
    out = np.full(len(r_locs), np.nan, dtype=float)
    for j, i in enumerate(idx_map):
        out[i] = corrs[j]
    return out, med

def wavelet_energies(x: np.ndarray, win: BeatWindows, fs: float,
                     levels=5, wavelet='db4') -> Dict[str, float]:
    if not HAVE_PYWT:
        return {f"waveE_L{k}": np.nan for k in range(1, levels+1)}
    a = max(0, win.r_idx - int(0.100 * fs))
    b = min(len(x), win.r_idx + int(0.300 * fs))
    sig = x[a:b] - x[win.baseline_idx]
    if len(sig) < 16:
        return {f"waveE_L{k}": np.nan for k in range(1, levels+1)}
    try:
        w = pywt.Wavelet(wavelet)
        max_level = pywt.dwt_max_level(len(sig), w.dec_len)
        L = max(1, min(levels, max_level))
        coeffs = pywt.wavedec(sig, wavelet=wavelet, level=L)
    except Exception:
        return {f"waveE_L{k}": np.nan for k in range(1, levels+1)}
    dets = coeffs[1:][::-1]  # L1..L{L}
    out = {}
    for idx in range(1, levels+1):
        if idx <= len(dets):
            c = np.asarray(dets[idx-1])
            out[f"waveE_L{idx}"] = float(np.sum(c**2))
        else:
            out[f"waveE_L{idx}"] = np.nan
    return out

# -----------------------------
# STV / STI channel extraction
# -----------------------------
def extract_st_channels(all_leads_by_name: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for name, arr in all_leads_by_name.items():
        ln = name.lower()
        if ln.startswith("stv") or ln.startswith("sti"):
            out[name] = arr
    return out

# -----------------------------
# Plotting helpers
# -----------------------------
def plot_and_save(fig, path_png: str, pdf: Optional[PdfPages]):
    fig.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight")
    if pdf is not None:
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_overview(primary: np.ndarray, fs: float, r_locs: np.ndarray,
                  wins: List[BeatWindows], out_dir: str, base: str, pdf: Optional[PdfPages]):
    t = np.arange(len(primary))/fs
    x, tt = downsample_for_plot(primary, t, max_points=50000)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tt, x, linewidth=0.7)
    if len(r_locs) > 0:
        idx = np.linspace(0, len(r_locs)-1, min(400, len(r_locs))).astype(int)
        ax.scatter(r_locs[idx]/fs, primary[r_locs[idx]], s=10, marker="o")
    for w in wins[:min(12, len(wins))]:
        ax.axvspan(w.qrs_on/fs, w.qrs_off/fs, alpha=0.15)
        ax.axvspan(w.t_start/fs, w.t_end/fs,  alpha=0.10)
    ax.set_title("Primary lead overview with R-peaks & windows")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Signal (units)")
    plot_and_save(fig, os.path.join(out_dir, f"{base}_overview_primary.png"), pdf)

def plot_st_trends(beats_df: pd.DataFrame, out_dir: str, base: str, pdf: Optional[PdfPages]):
    if "ST60_max_abs" not in beats_df.columns and "ST80_max_abs" not in beats_df.columns:
        return
    t = beats_df.get("beat_time_sec", pd.Series(range(len(beats_df))))
    if "ST60_max_abs" in beats_df.columns:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, beats_df["ST60_max_abs"], linewidth=1.0)
        ax.set_title("ST max |deviation| @ +60 ms (per beat)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("|ST60| (units)")
        plot_and_save(fig, os.path.join(out_dir, f"{base}_st60_max_trend.png"), pdf)
    if "ST80_max_abs" in beats_df.columns:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(t, beats_df["ST80_max_abs"], linewidth=1.0)
        ax.set_title("ST max |deviation| @ +80 ms (per beat)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("|ST80| (units)")
        plot_and_save(fig, os.path.join(out_dir, f"{base}_st80_max_trend.png"), pdf)

def compute_t_amp_series(lead: np.ndarray, wins: List[BeatWindows], fs: float) -> np.ndarray:
    vals = []
    for w in wins:
        base = lead[w.baseline_idx]
        vals.append(lead[w.t_peak] - base)
    return np.asarray(vals, dtype=float)

def plot_twa_series(series: np.ndarray, out_dir: str, base: str, lead_name: str, pdf: Optional[PdfPages],
                    td_amp: Optional[float]=None, spec_amp: Optional[float]=None, kscore: Optional[float]=None):
    if len(series) == 0: return
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(np.arange(len(series)), series, linewidth=1.0)
    ax.set_title(f"T-peak amplitude by beat — lead {lead_name}")
    ax.set_xlabel("Beat index"); ax.set_ylabel("T-peak amp (units)")
    if td_amp is not None or spec_amp is not None or kscore is not None:
        txt = []
        if td_amp is not None and np.isfinite(td_amp): txt.append(f"TD alternans ≈ {td_amp:.4g}")
        if spec_amp is not None and np.isfinite(spec_amp): txt.append(f"S(0.5 c/b) ≈ {spec_amp:.4g}")
        if kscore is not None and np.isfinite(kscore): txt.append(f"K-score ≈ {kscore:.2f}")
        ax.text(0.01, 0.95, " | ".join(txt), transform=ax.transAxes, va="top")
    plot_and_save(fig, os.path.join(out_dir, f"{base}_twa_series_{lead_name}.png"), pdf)

def plot_twa_spectrum(series: np.ndarray, out_dir: str, base: str, lead_name: str, pdf: Optional[PdfPages]):
    s = series.copy()
    s = s[~np.isnan(s)]
    if len(s) < 16: return
    s = signal.detrend(s, type="linear")
    N = len(s)
    V = np.fft.rfft(s)
    freqs = np.fft.rfftfreq(N, d=1.0)  # cycles/beat
    amp = (2.0/N) * np.abs(V)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(freqs, amp, linewidth=1.0)
    ax.axvline(0.5, linestyle="--", linewidth=1.0)
    ax.set_xlim(0, 1.0)
    ax.set_title(f"TWA spectrum (cycles/beat) — lead {lead_name}")
    ax.set_xlabel("Frequency (cycles/beat)"); ax.set_ylabel("Amplitude (units)")
    plot_and_save(fig, os.path.join(out_dir, f"{base}_twa_spectrum_{lead_name}.png"), pdf)

def plot_morph_corr(beats_df: pd.DataFrame, out_dir: str, base: str, pdf: Optional[PdfPages]):
    if "morph_corr_allleads" not in beats_df.columns: return
    y = beats_df["morph_corr_allleads"].astype(float).values
    if not np.isfinite(y).any(): return
    t = beats_df.get("beat_time_sec", pd.Series(range(len(beats_df))))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t, y, linewidth=1.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Beat ↔ median morphology correlation (multi-lead)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Correlation (r)")
    plot_and_save(fig, os.path.join(out_dir, f"{base}_morph_corr.png"), pdf)

def plot_wavelet_summary(summary_df: pd.DataFrame, out_dir: str, base: str, pdf: Optional[PdfPages]):
    cols = [c for c in summary_df.columns if c.startswith("waveE_L") and c.endswith("_median")]
    if len(cols) == 0: return
    vals = []
    for c in cols:
        try:
            vv = summary_df[c].astype(float).values
            vals.append(float(np.nanmedian(vv)))
        except Exception:
            vals.append(np.nan)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, len(cols)+1), vals)
    ax.set_xticks(range(1, len(cols)+1))
    ax.set_xticklabels([f"L{k}" for k in range(1, len(cols)+1)])
    ax.set_title("Wavelet energy medians across leads")
    ax.set_xlabel("Detail level"); ax.set_ylabel("Energy (a.u.)")
    plot_and_save(fig, os.path.join(out_dir, f"{base}_wavelet_energy.png"), pdf)

def plot_fragmentation(summary_df: pd.DataFrame, out_dir: str, base: str, pdf: Optional[PdfPages]):
    if "qrs_fragmentation_rate" not in summary_df.columns: return
    df = summary_df[["lead","qrs_fragmentation_rate"]].copy()
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(df["lead"].astype(str), df["qrs_fragmentation_rate"].astype(float))
    ax.set_title("QRS fragmentation rate by lead")
    ax.set_xlabel("Lead"); ax.set_ylabel("Fragmentation rate (0..1)")
    ax.tick_params(axis='x', rotation=45)
    plot_and_save(fig, os.path.join(out_dir, f"{base}_fragmentation_rate.png"), pdf)

def plot_st_medians(summary_df: pd.DataFrame, out_dir: str, base: str, pdf: Optional[PdfPages]):
    cols = ["st_j_median", "st_60ms_median", "st_80ms_median"]
    if not all(c in summary_df.columns for c in cols): return
    df = summary_df[["lead"] + cols].copy()
    for c in cols:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(df["lead"].astype(str), df[c].astype(float))
        ax.set_title(f"{c.replace('_',' ').upper()} by lead (median)")
        ax.set_xlabel("Lead"); ax.set_ylabel("Amplitude (units)")
        ax.tick_params(axis='x', rotation=45)
        plot_and_save(fig, os.path.join(out_dir, f"{base}_{c}.png"), pdf)

def t_axis_frontal(all_leads_by_name: Dict[str, np.ndarray], wins: List[BeatWindows], fs: float) -> np.ndarray:
    # 1. Gerekli kanalları bul (Büyük/küçük harf duyarsız)
    lead_I = None
    lead_II = None
    lead_aVF = None

    for name, data in all_leads_by_name.items():
        n = name.upper()
        if n == "I" or n == "LEADI":
            lead_I = data
        elif n == "II" or n == "LEADII":
            lead_II = data
        elif n == "AVF" or n == "LEADAVF":
            lead_aVF = data

    # 2. Lead I şart, yoksa hesaplayamayız
    if lead_I is None:
        return np.full(len(wins), np.nan)

    # 3. aVF yoksa, II ve I'den türet: aVF = II - 0.5 * I
    if lead_aVF is None:
        if lead_II is not None:
            lead_aVF = lead_II - 0.5 * lead_I
        else:
            # Ne aVF var ne de II, hesaplayamayız
            return np.full(len(wins), np.nan)

    # 4. Her atım için T-peak genliğini ölç ve açıyı hesapla
    axes = []
    for w in wins:
        # Baseline düzeltmesi yapılmış genlikleri al
        # (Baseline_idx'teki değeri çıkarıyoruz)
        amp_I = lead_I[w.t_peak] - lead_I[w.baseline_idx]
        amp_aVF = lead_aVF[w.t_peak] - lead_aVF[w.baseline_idx]

        # Genlikler çok küçükse (gürültü) açı hesaplama
        if abs(amp_I) < 1e-6 and abs(amp_aVF) < 1e-6:
            axes.append(np.nan)
        else:
            # atan2(y, x) -> radyan cinsinden açı
            # Dereceye çevir: degrees(atan2(aVF, I))
            angle = math.degrees(math.atan2(amp_aVF, amp_I))
            axes.append(angle)

    return np.array(axes)

# -----------------------------
# Main analysis
# -----------------------------
def analyze(input_path: str,
            dataset_path: Optional[str],
            max_duration_sec: float,
            max_beats: int,
            output_prefix: str,
            default_fs: Optional[float],
            make_plots: bool = True,
            plots_dir: Optional[str] = None,
            twa_lead_override: Optional[str] = None,
            list_all: bool = False):

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    with h5py.File(input_path, "r") as h5:
        ds, ch_names, fs, struct, chosen_path = discover_ecg_dataset(
            h5, dataset_override=dataset_path, list_all=list_all
        )

        # Show a concise structure header (always)
        print("---- HDF5 STRUCTURE (first 50 entries) ----")
        for row in struct[:50]:
            print(row)
        if len(struct) > 50:
            print(f"... ({len(struct)} total items)")
        # Show chosen dataset
        print(f"Chosen dataset path: {chosen_path}")

        if ds is None:
            raise RuntimeError("Could not identify an ECG waveform dataset automatically. "
                               "Use --dataset to specify the correct path.")

        if fs is None:
            fs = find_fs_in_attrs(h5, ds) or find_fs_from_sibling_sample_freq(ds) \
                 or (default_fs if (default_fs is not None and default_fs > 0) else 500.0)
            print(f"Sampling rate resolved to fs={fs} Hz (attrs/sibling/default).")

        data = ds[()]
        # Orient to (channels, samples)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif data.ndim == 2:
            # If it's (N, ch), transpose; if it's (ch, N), keep
            if data.shape[0] <= 32 and data.shape[1] > data.shape[0]:
                pass  # already (ch, N)
            else:
                data = data.T  # (N, ch) -> (ch, N)
        else:
            # Flatten higher dims to (ch, N) best-effort
            data = np.reshape(data, (-1, data.shape[-1]))

        n_ch, n = data.shape
        if ch_names is None or len(ch_names) != n_ch:
            ch_names = [f"ch{idx}" for idx in range(n_ch)]

        # Limit duration
        max_samples = int(max_duration_sec * fs)
        if n > max_samples:
            data = data[:, :max_samples]
            n = data.shape[1]
            print(f"Truncated to first {max_duration_sec} s ({n} samples).")

        # Preprocess leads
        proc = np.vstack([preprocess_lead(data[i], fs) for i in range(n_ch)])

        # Primary lead for detection: max peak-to-peak
        p2p = np.ptp(proc, axis=1)
        primary_idx = int(np.argmax(p2p))
        primary = proc[primary_idx]
        primary_name = ch_names[primary_idx]
        print(f"Primary lead for detection: {primary_name}")

        # R peaks
        r_locs = detect_r_peaks(primary, fs)
        if len(r_locs) < 5:
            print("Warning: few R peaks detected; results may be unreliable.")
        if len(r_locs) > max_beats:
            r_locs = r_locs[:max_beats]
            print(f"Capped to first {max_beats} beats.")

        if len(r_locs) < 2:
            raise RuntimeError("Insufficient beats detected for analysis.")

        times = r_locs / fs
        rr = np.diff(times)
        hr = 60.0 / np.median(rr) if len(rr) > 0 else np.nan
        dur_sec = n / fs
        print(f"Detected {len(r_locs)} beats over ~{dur_sec:.1f} s. Median HR ≈ {hr:.1f} bpm.")

        # QC sanity flags
        if (len(r_locs) / max(1.0, dur_sec)) * 60.0 < 20 or hr > 220:
            print("QC flag: Detected beat rate is implausible for a typical adult ICU tracing. "
                  "Verify dataset path is a true ECG waveform and fs is correct.")

        # Delineation windows using primary lead
        wins: List[BeatWindows] = []
        for r in r_locs:
            w = delineate_single_beat(primary, int(r), fs)
            if w is not None:
                wins.append(w)
        if len(wins) == 0:
            raise RuntimeError("Failed to delineate beats (QRS/T).")

        # Build name->lead map
        all_leads_by_name: Dict[str, np.ndarray] = {ch_names[i]: proc[i] for i in range(n_ch)}

        # T-axis (if I & aVF)
        t_axis = t_axis_frontal(all_leads_by_name, wins, fs)
        if t_axis is not None and len(t_axis) > 0:
            median_axis = np.nanmedian(t_axis)
            print(f"T-axis (median): {median_axis:.1f}°")
        else:
            print("T-axis not available (need I & aVF).")

        # ST-derived channels if present (stv*/sti*)
        st_deriv = extract_st_channels(all_leads_by_name)
        if len(st_deriv) > 0:
            print(f"Found derived ST channels: {list(st_deriv.keys())}")

        # Per-beat features
        rows = []
        for b_idx, w in enumerate(wins):
            row = {"beat_index": b_idx, "beat_time_sec": r_locs[b_idx] / fs}
            for name, x in all_leads_by_name.items():
                qm = qrs_metrics(x, w, fs)
                sm = st_metrics(x, w, fs)
                tm = t_metrics(x, w, fs)
                frag = fragmentation_metrics(x, w, fs)
                wavE = wavelet_energies(x, w, fs, levels=5)
                for k, v in {**qm, **sm, **tm, **frag, **wavE}.items():
                    row[f"{name}.{k}"] = v
            # ST raw channels (stv*/sti*) if present
            if len(st_deriv) > 0:
                for name, x in st_deriv.items():
                    row[f"{name}.rawJ"]   = float(x[w.j_idx])
                    row[f"{name}.rawJ60"] = float(x[min(len(x)-1, w.j_idx + int(0.060*fs))])
                    row[f"{name}.rawJ80"] = float(x[min(len(x)-1, w.j_idx + int(0.080*fs))])
            rows.append(row)

        beats_df = pd.DataFrame(rows)

        # Morphology similarity (beat↔median, across leads) + median template
        corrs, med_template = morphology_correlation(proc, r_locs, fs)
        beats_df["morph_corr_allleads"] = corrs

        # Per-lead summary
        summary_rows = []
        def summarize(col: str):
            if col in beats_df.columns:
                vals = beats_df[col].astype(float).values
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    return np.nan, np.nan, np.nan, np.nan
                return (float(np.nanmean(vals)),
                        float(np.nanmedian(vals)),
                        float(np.nanstd(vals)),
                        float(np.nanmedian(np.abs(vals - np.nanmedian(vals)))))
            return np.nan, np.nan, np.nan, np.nan

        for name, x in all_leads_by_name.items():
            r_mean, r_med, r_sd, _ = summarize(f"{name}.r_amp")
            _, s_med, _, _ = summarize(f"{name}.s_amp")
            _, rsr_med, _, _ = summarize(f"{name}.rs_ratio")
            _, slope_med, _, _ = summarize(f"{name}.qrs_max_slope")
            _, area_med, _, _  = summarize(f"{name}.qrs_area")
            _, energy_med, _, _ = summarize(f"{name}.qrs_energy")
            _, stj_med, _, _    = summarize(f"{name}.st_j")
            _, st60_med, _, _   = summarize(f"{name}.st_60ms")
            _, st80_med, _, _   = summarize(f"{name}.st_80ms")
            _, tpk_med, _, _    = summarize(f"{name}.t_peak_amp")

            # Fragmentation
            frag_rate = np.nan
            frag_notches_med = np.nan
            if f"{name}.qrs_is_fragmented" in beats_df.columns:
                fvals = beats_df[f"{name}.qrs_is_fragmented"].astype(float).values
                if np.isfinite(fvals).any():
                    frag_rate = float(np.nanmean(fvals))
            if f"{name}.qrs_notches" in beats_df.columns:
                frag_notches_med = float(np.nanmedian(beats_df[f"{name}.qrs_notches"].astype(float).values))

            # TWA (from T-peak amplitudes)
            tpeaks = compute_t_amp_series(x, wins, fs)
            def alternans_time_domain(vals: np.ndarray) -> float:
                v = np.asarray(vals, dtype=float); v = v[~np.isnan(v)]
                if len(v) < 6: return np.nan
                ev, od = v[0::2], v[1::2]
                if len(ev)==0 or len(od)==0: return np.nan
                return float(abs(np.nanmean(ev) - np.nanmean(od)) / 2.0)
            def alternans_spectral(vals: np.ndarray) -> Tuple[float, float]:
                v = np.asarray(vals, dtype=float); v = v[~np.isnan(v)]
                if len(v) < 16: return np.nan, np.nan
                v = signal.detrend(v, type='linear')
                N = len(v)
                V = np.fft.rfft(v)
                freqs = np.fft.rfftfreq(N, d=1.0)   # cycles/beat
                k = int(np.argmin(np.abs(freqs - 0.5)))
                A_half = 2.0/N * np.abs(V[k])
                idx_noise = [i for i in range(len(freqs)) if i not in (0, k)]
                noise_mag = np.abs(V[idx_noise])
                if len(noise_mag) < 3: return float(A_half), np.nan
                mu, sd = np.mean(noise_mag), np.std(noise_mag) + 1e-12
                K = (np.abs(V[k]) - mu) / sd
                return float(A_half), float(K)
            twa_td = alternans_time_domain(tpeaks)
            twa_spec, kscore = alternans_spectral(tpeaks)

            # Wavelet energies (medians)
            waveE = {}
            for k in range(1,6):
                _, med, _, _ = summarize(f"{name}.waveE_L{k}")
                waveE[f"waveE_L{k}_median"] = med

            summary_rows.append({
                "lead": name,
                "n_beats": len(wins),
                "r_amp_mean": r_mean,
                "r_amp_sd": r_sd,
                "r_amp_cv": (r_sd / (abs(r_mean)+1e-12)) if np.isfinite(r_sd) and abs(r_mean) > 1e-12 else np.nan,
                "r_amp_median": r_med,
                "s_amp_median": s_med,
                "rs_ratio_median": rsr_med,
                "qrs_max_slope_median": slope_med,
                "qrs_area_median": area_med,
                "qrs_energy_median": energy_med,
                "st_j_median": stj_med,
                "st_60ms_median": st60_med,
                "st_80ms_median": st80_med,
                "t_peak_amp_median": tpk_med,
                "qrs_fragmentation_rate": frag_rate,
                "qrs_notches_median": frag_notches_med,
                "twa_td_amp": twa_td,
                "twa_spec_amp": twa_spec,
                "twa_kscore": kscore,
                **waveE
            })

        summary_df = pd.DataFrame(summary_rows)

        # Lead-wise max ST deviation per beat
        def _max_st(abs_vals: np.ndarray, cols: List[str]):
            max_abs = np.nanmax(abs_vals, axis=1)
            max_lead = []
            for i in range(abs_vals.shape[0]):
                row = abs_vals[i]
                if not np.isfinite(row).any():
                    max_lead.append(np.nan)
                else:
                    j = np.nanargmax(row)
                    max_lead.append(cols[j])
            return max_abs, max_lead

        st60_cols = [c for c in beats_df.columns if c.endswith(".st_60ms")]
        st80_cols = [c for c in beats_df.columns if c.endswith(".st_80ms")]
        if len(st60_cols) > 0:
            abs_ = np.abs(beats_df[st60_cols].values.astype(float))
            beats_df["ST60_max_abs"], beats_df["ST60_max_lead"] = _max_st(abs_, st60_cols)
        if len(st80_cols) > 0:
            abs_ = np.abs(beats_df[st80_cols].values.astype(float))
            beats_df["ST80_max_abs"], beats_df["ST80_max_lead"] = _max_st(abs_, st80_cols)

        # Global morphology stability & T axis
        if "morph_corr_allleads" in beats_df.columns and np.isfinite(beats_df["morph_corr_allleads"].astype(float).values).any():
            summary_df["morph_corr_median"] = float(np.nanmedian(beats_df["morph_corr_allleads"].values))
        else:
            summary_df["morph_corr_median"] = np.nan
        if t_axis is not None and len(t_axis) > 0:
            summary_df["t_axis_deg_global"] = float(np.nanmedian(t_axis))
        else:
            summary_df["t_axis_deg_global"] = np.nan

        # Save CSVs
        beats_csv = f"{output_prefix}_beats.csv"
        summary_csv = f"{output_prefix}_summary.csv"
        template_npy = f"{output_prefix}_median_template.npy"
        beats_df.to_csv(beats_csv, index=False)
        summary_df.to_csv(summary_csv, index=False)
        if len(med_template) > 0:
            np.save(template_npy, med_template)

        print("\n--- DONE (tables) ---")
        print(f"Saved per-beat CSV:     {beats_csv}")
        print(f"Saved per-lead summary: {summary_csv}")
        if len(med_template) > 0:
            print(f"Saved median template:  {template_npy}")

        # -----------------------------
        # Plots
        # -----------------------------
        if make_plots:
            out_dir = plots_dir or f"{output_prefix}_plots"
            ensure_dir(out_dir)
            pdf_path = os.path.join(out_dir, f"{os.path.basename(output_prefix)}_plots.pdf")
            with PdfPages(pdf_path) as pdf:
                # 1) Overview with R-peaks & windows
                plot_overview(primary, fs, r_locs, wins, out_dir, os.path.basename(output_prefix), pdf)
                # 2) ST trends (max abs @ 60/80 ms)
                plot_st_trends(beats_df, out_dir, os.path.basename(output_prefix), pdf)
                # 3) TWA series & spectrum (chosen lead)
                twa_lead = twa_lead_override or choose_twa_lead(list(all_leads_by_name.keys()), primary_name)
                t_series = compute_t_amp_series(all_leads_by_name[twa_lead], wins, fs)
                row = summary_df[summary_df["lead"] == twa_lead]
                td, spec, ks = (row["twa_td_amp"].values[0] if not row.empty else np.nan,
                                row["twa_spec_amp"].values[0] if not row.empty else np.nan,
                                row["twa_kscore"].values[0] if not row.empty else np.nan)
                plot_twa_series(t_series, out_dir, os.path.basename(output_prefix), twa_lead, pdf, td, spec, ks)
                plot_twa_spectrum(t_series, out_dir, os.path.basename(output_prefix), twa_lead, pdf)
                # 4) Morphology correlation
                plot_morph_corr(beats_df, out_dir, os.path.basename(output_prefix), pdf)
                # 5) Wavelet energy (L1-L5) medians across leads
                plot_wavelet_summary(summary_df, out_dir, os.path.basename(output_prefix), pdf)
                # 6) Fragmentation rate per lead
                plot_fragmentation(summary_df, out_dir, os.path.basename(output_prefix), pdf)
                # 7) ST medians per lead
                plot_st_medians(summary_df, out_dir, os.path.basename(output_prefix), pdf)

            print(f"Saved plots to: {out_dir}")
            print(f"Combined PDF:  {pdf_path}")

        # Basic QC text
        try:
            qc_lines = []
            rr = np.diff(beats_df["beat_time_sec"].values.astype(float)) if "beat_time_sec" in beats_df.columns else np.array([])
            rr = rr[np.isfinite(rr)]
            if rr.size > 0:
                qc_lines.append(f"Median RR={np.median(rr):.3f} s -> HR≈{60.0/np.median(rr):.1f} bpm")
            if "ST60_max_abs" in beats_df.columns:
                v = beats_df["ST60_max_abs"].astype(float).values; v = v[np.isfinite(v)]
                if v.size > 0:
                    qc_lines.append(f"ST60_max_abs: median={np.median(v):.4g}, 95th={np.percentile(v,95):.4g}, max={np.max(v):.4g}")
            if "ST80_max_abs" in beats_df.columns:
                v = beats_df["ST80_max_abs"].astype(float).values; v = v[np.isfinite(v)]
                if v.size > 0:
                    qc_lines.append(f"ST80_max_abs: median={np.median(v):.4g}, 95th={np.percentile(v,95):.4g}, max={np.max(v):.4g}")
            qc_path = f"{output_prefix}_qc.txt"
            with open(qc_path, "w") as f:
                f.write("\n".join(qc_lines))
            print(f"QC summary: {qc_path}")
        except Exception:
            pass

        return beats_csv, summary_csv, template_npy, chosen_path, fs

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="ECG Morphology & Repolarization Analysis (Bedmaster-friendly, with plots)")
    ap.add_argument("--input", default=None, help="Path to HDF5 ECG file (defaults to UNC path in script)")
    ap.add_argument("--dataset", default=None, help="Dataset path inside HDF5 (overrides auto-discovery)")
    ap.add_argument("--max-duration-sec", type=float, default=DEFAULT_MAX_DURATION_SEC, help="Analyze at most this many seconds")
    ap.add_argument("--max-beats", type=int, default=DEFAULT_MAX_BEATS, help="Max number of beats to analyze")
    ap.add_argument("--output-prefix", type=str, default=DEFAULT_OUTPUT_PREFIX, help="Prefix for outputs")
    ap.add_argument("--fs", type=float, default=DEFAULT_FS_IF_MISSING, help="Default fs (Hz) if not present in the file")
    ap.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    ap.add_argument("--plots-dir", default=None, help="Directory to save plots (default: <prefix>_plots)")
    ap.add_argument("--twa-lead", default=None, help="Lead name to use for TWA plots (default: primary or V5/II/etc.)")
    ap.add_argument("--list-all", action="store_true", help="Print ALL HDF5 items to help pick the correct dataset")
    args = ap.parse_args()

    input_path = args.input or DEFAULT_INPUT
    return analyze(
        input_path=input_path,
        dataset_path=args.dataset,
        max_duration_sec=args.max_duration_sec,
        max_beats=args.max_beats,
        output_prefix=args.output_prefix,
        default_fs=args.fs,
        make_plots=(not args.no_plots),
        plots_dir=args.plots_dir,
        twa_lead_override=args.twa_lead,
        list_all=args.list_all
    )

if __name__ == "__main__":
    main()
