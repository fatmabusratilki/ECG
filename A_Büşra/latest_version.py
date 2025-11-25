"""
ECG_all_in_one.py
-----------------
Self-contained ECG beat analysis (no external imports required).
Targets Bedmaster files and computes per-beat conduction metrics:

- RR interval (ms): mean/median/SD/MAD; SDSD; % successive-change outliers >20%
- PR interval (ms): P-onset -> QRS-onset (best-effort; NaN if not visible)
- QRS/"QRC" (ms): onset->offset; variability & % beats >120 ms
- RTpeak / RTend (ms): R->Tpeak / R->Tend; mean/SD/trend (ms/hr)
- QT (ms) and QTc (Bazett/Fridericia); % beats with QTc > 480 / > 500 ms
- Tpeak–Tend (ms)

Bedmaster auto-detect:
- If dataset like /bedmaster/waveforms/v/value is given, fs is taken from sibling /sample_freq (mode).

Outputs:
- per_beat_metrics.csv
- summary.json
- windowed_rr.csv
- annotated_preview.png  (first 10 seconds with annotations)
"""

from __future__ import annotations

import os
import sys
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

# Optional heavy deps
try:
    import scipy.signal as sps
except Exception:
    sps = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import h5py
except Exception:
    h5py = None


# =========================
# Data structures
# =========================

@dataclass
class Beat:
    index: int
    r_index: int
    time_s: float
    time_dt: Optional[datetime]
    rr_ms_prev: float | None
    rr_ms_next: float | None
    qrs_onset: Optional[int]
    qrs_offset: Optional[int]
    qrs_ms: Optional[float]
    p_onset: Optional[int]
    p_ms: Optional[float]
    t_peak: Optional[int]
    t_end: Optional[int]
    rt_peak_ms: Optional[float]
    rt_end_ms: Optional[float]
    qt_ms: Optional[float]
    qtc_bazett_ms: Optional[float]
    qtc_fridericia_ms: Optional[float]
    tpte_ms: Optional[float]


# =========================
# HDF5 I/O (Bedmaster-aware)
# =========================

def _safe_attr_value(v):
    try:
        if hasattr(v, "item"):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist() if v.size <= 10 else f"<ndarray shape={v.shape} dtype={v.dtype}>"
        return v
    except Exception:
        return str(v)

def list_hd5_datasets(file_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """List dataset paths (+ shape/dtype/attrs) and root attributes."""
    if h5py is None:
        raise RuntimeError("h5py is required to read .hd5 files.")
    datasets, root_attrs = [], {}
    with h5py.File(file_path, "r") as f:
        root_attrs = {k: _safe_attr_value(f.attrs.get(k)) for k in f.attrs.keys()}
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                attrs = {k: _safe_attr_value(obj.attrs.get(k)) for k in obj.attrs.keys()}
                datasets.append({
                    "path": "/" + name if not name.startswith("/") else name,
                    "shape": tuple(obj.shape),
                    "dtype": str(obj.dtype),
                    "attrs": attrs
                })
        f.visititems(lambda n, o: visitor(n, o))
    return datasets, root_attrs

def guess_ecg_dataset(datasets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Heuristically pick a likely ECG dataset."""
    def is_numeric(dt: str) -> bool:
        dt = dt.lower()
        return any(tag in dt for tag in ["int", "float"])

    cands = []
    for d in datasets:
        if not is_numeric(d["dtype"]): 
            continue
        name = d["path"].lower()
        shape = d["shape"]
        score = 0
        if "waveforms" in name and name.endswith("/value"): score += 4  # Bedmaster waveform
        if "ecg" in name: score += 3
        if any(k in name for k in ["signal", "data", "lead", "channel", "/v/"]): score += 1
        if len(shape) == 1 and shape[0] >= 10_000: score += 3
        elif len(shape) == 2 and min(shape) <= 16 and max(shape) >= 10_000: score += 3
        if "float" in d["dtype"].lower(): score += 1
        cands.append((score, d))
    if not cands: 
        return None
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1]

def _extract_fs_from_attrs(attrs: Dict[str, Any]) -> Optional[float]:
    if not attrs: return None
    keys = [
        "fs", "Fs", "FS", "sps", "hz", "Hz", "frequency", "sample_rate", "sampling_rate",
        "SamplingRate", "SamplingFrequency", "sr", "SR", "SRate", "sampleFrequency"
    ]
    for k in keys:
        if k in attrs:
            v = attrs[k]
            try: return float(v)
            except Exception:
                try: return float(v.decode("utf-8"))
                except Exception: pass
    for k, v in attrs.items():
        if isinstance(v, (bytes, str)):
            s = v.decode("utf-8") if isinstance(v, bytes) else v
            if "hz" in s.lower():
                num = "".join(ch for ch in s if ch.isdigit() or ch == ".")
                if num:
                    try: return float(num)
                    except Exception: pass
    return None

def _extract_bedmaster_fs(f: "h5py.File", dataset_path: str) -> Optional[float]:
    """For /bedmaster/waveforms/<lead>/value → read sibling /sample_freq and return mode."""
    if not dataset_path.lower().endswith("/value"): 
        return None
    group = dataset_path.rsplit("/", 1)[0]
    sf_path = group + "/sample_freq"
    if sf_path not in f: 
        return None
    ds = f[sf_path]
    data = ds[()]
    fs_vals = []
    if isinstance(data, np.ndarray) and data.dtype.names:
        # structured array (e.g., [('f0','<f8'), ('f1','<i8')])
        for name in data.dtype.names:
            try:
                col = np.array(data[name]).astype(float)
                fs_vals.extend(col[np.isfinite(col) & (col > 0)].tolist())
                break
            except Exception:
                continue
    else:
        try:
            arr = np.array(data).astype(float)
            fs_vals.extend(arr[np.isfinite(arr) & (arr > 0)].tolist())
        except Exception:
            pass
    if not fs_vals: 
        return None
    vals, counts = np.unique(np.round(fs_vals, 6), return_counts=True)
    return float(vals[np.argmax(counts)])

def read_ecg_from_hd5(file_path: str, dataset_path: Optional[str] = None, channel: int = 0
    ) -> Tuple[np.ndarray, Optional[float], Dict[str, Any]]:
    """Return (ecg 1D float array, fs (Hz) or None, metadata)."""
    if h5py is None:
        raise RuntimeError("h5py is required to read .hd5 files.")
    datasets, root_attrs = list_hd5_datasets(file_path)
    if dataset_path:
        chosen = next((d for d in datasets if d["path"] == dataset_path), None)
        if chosen is None:
            raise ValueError(f"Dataset {dataset_path} not found.")
    else:
        chosen = guess_ecg_dataset(datasets)
        if chosen is None:
            raise RuntimeError("No suitable ECG-like dataset found; pass dataset_path.")
    with h5py.File(file_path, "r") as f:
        ds = f[chosen["path"]]
        arr = ds[()]
        fs = _extract_bedmaster_fs(f, chosen["path"])
        if fs is None:
            fs = _extract_fs_from_attrs(ds.attrs) or _extract_fs_from_attrs(f.attrs)

    if arr.ndim == 1:
        ecg = arr.astype(float)
    elif arr.ndim == 2:
        axis_ch = int(np.argmin(arr.shape))
        if arr.shape[axis_ch] <= channel:
            raise ValueError(f"Requested channel {channel} but dataset has {arr.shape[axis_ch]} channels.")
        ecg = (arr[channel, :] if axis_ch == 0 else arr[:, channel]).astype(float)
    else:
        flat = arr.reshape(-1).astype(float)
        if flat.size < 1000:
            raise RuntimeError(f"Unsupported shape {arr.shape} for ECG.")
        ecg = flat

    meta = dict(file_root_attrs=root_attrs, dataset_attrs=chosen["attrs"],
                dataset_path=chosen["path"], dtype=chosen["dtype"],
                shape=chosen["shape"], detected_fs=fs)
    return ecg, fs, meta


# =========================
# Filtering helpers
# =========================

def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1: return x.astype(float).copy()
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = (c[win:] - c[:-win]) / float(win)
    left = np.full((win//2,), out[0])
    right = np.full((len(x) - len(out) - len(left),), out[-1])
    return np.concatenate([left, out, right])

def butter_lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 3) -> np.ndarray:
    if sps is None:
        win = max(1, int(fs / max(cutoff, 0.1)))
        return _moving_average(x, win)
    nyq = 0.5 * fs
    wc = min(0.999, cutoff / nyq)
    b, a = sps.butter(order, wc, btype="low")
    return sps.filtfilt(b, a, x)

def butter_bandpass(x: np.ndarray, fs: float, low: float, high: float, order: int = 3) -> np.ndarray:
    if sps is None:
        win_lo = max(1, int(fs * 0.5))   # ~1s MA (remove baseline)
        x_hp = x - _moving_average(x, win_lo)
        win_hi = max(1, int(fs * 0.02))  # ~20ms MA (smooth HF)
        return _moving_average(x_hp, win_hi)
    nyq = 0.5 * fs
    lowc = max(0.001, low / nyq)
    highc = min(0.999, high / nyq)
    b, a = sps.butter(order, [lowc, highc], btype="band")
    return sps.filtfilt(b, a, x)

def notch_filter(x: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0) -> np.ndarray:
    if sps is None:
        return x
    b, a = sps.iirnotch(w0=freq/(fs/2.0), Q=q)
    return sps.filtfilt(b, a, x)


# =========================
# Peak detectors
# =========================

def _simple_find_peaks(x: np.ndarray, thr: float, distance: int) -> np.ndarray:
    peaks, last = [], -distance
    for i in range(1, len(x)-1):
        if x[i] > thr and x[i] > x[i-1] and x[i] >= x[i+1]:
            if i - last >= distance:
                peaks.append(i); last = i
    return np.array(peaks, dtype=int)

def detect_r_peaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """Pan–Tompkins‑like: derivative → square → MWI → adaptive threshold + refractory; refine on |ECG|."""
    qrs_band = butter_bandpass(ecg, fs, 5.0, 18.0, order=2)
    der = np.diff(qrs_band, prepend=qrs_band[0]); sq = der * der
    win = max(1, int(0.150 * fs))   # 150 ms MWI
    mwi = _moving_average(sq, win)
    med = np.median(mwi); mad = np.median(np.abs(mwi - med)) + 1e-12
    # Lowered to 2.5 to catch low-voltage beats (prevents missed beats)
    thr = med + 2.5 * mad 

    # Increased to 500ms to ignore T-waves (prevents double counting). Max HR: 120 BPM.
    min_dist = int(0.50 * fs)      # Changed 0.25 to 0.50  
    if sps is not None:
        peaks, _ = sps.find_peaks(mwi, height=thr, distance=min_dist)
    else:
        peaks = _simple_find_peaks(mwi, thr, min_dist)
    r_peaks = []
    search = max(1, int(0.06 * fs))
    for p in peaks:
        a = max(0, p - search); b = min(len(ecg)-1, p + search)
        r = a + np.argmax(np.abs(ecg[a:b+1]))
        r_peaks.append(r)
    return np.unique(np.array(r_peaks, dtype=int))

def estimate_qrs_onset_offset(ecg: np.ndarray, fs: float, r_peaks: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Energy-based on QRS-band signal; calm segments around R define onset/offset."""
    qrs_sig = butter_bandpass(ecg, fs, 5.0, 25.0, order=2)
    der = np.diff(qrs_sig, prepend=qrs_sig[0]); energy = der * der
    left_win = int(0.12 * fs); right_win = int(0.12 * fs); calm = max(1, int(0.010 * fs))

    # Added 
    # Skip 15ms from the peak to avoid the "zero-slope trap" (prevent early cut-off).
    skip_r = max(1, int(0.015 * fs))

    onsets = np.full(len(r_peaks), -1, dtype=int); offsets = np.full(len(r_peaks), -1, dtype=int)

    for i, rp in enumerate(r_peaks):
        a = max(0, rp - left_win); b = min(len(ecg)-1, rp + right_win)
        loc_en = energy[a:b+1]; thr = 0.08 * (np.max(loc_en) + 1e-12)

        # --- QRS onset ---
        on = a
        # Updated
        # Start backward search from (R - 15ms) to avoid the "zero-energy" point at the peak.
        j = rp - skip_r
     
        while j > a + calm:
            if np.all(energy[j-calm:j] < thr):
                on = j - calm
                while on > a + calm and np.all(energy[on-calm:on] < thr): on -= calm
                break
            j -= 1

        # --- QRS offset ---
        off = b
        # Updated
        # Start forward search from (R + 15ms) to avoid the peak's zero derivative.
        j = rp + skip_r

        while j < b - calm:
            if np.all(energy[j:j+calm] < thr):
                off = j + calm
                while off < b - calm and np.all(energy[off:off+calm] < thr): off += calm
                break
            j += 1
        onsets[i], offsets[i] = max(a, on), min(b, off)
    return onsets, offsets

def detect_tpeaks_tends(ecg: np.ndarray, fs: float, r_peaks: np.ndarray, qrs_offsets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
    """T-peak: |lowpass 8 Hz| extremum in [80, 600] ms after R/QRS.
       T-end: first sustained near-baseline & low-slope region after T-peak."""
    
    # Lowpass filter at 15 Hz (increased from 8 Hz) to preserve T-wave morphology and peak position.
    t_sig = butter_lowpass(ecg, fs, 15.0, order=3)
    t_peaks = np.full(len(r_peaks), -1, dtype=int); t_ends = np.full(len(r_peaks), -1, dtype=int)

    # Added 
    # --- SEARCH WINDOW PARAMETERS ---
    min_delay_ms = 40       # Minimum delay after QRS offset (skip ST segment)
    abs_min_from_r_ms = 150 # Absolute minimum distance from R-peak
    max_search_ms = 600     # Maximum duration to search for T-wave


    for i, rp in enumerate(r_peaks):
        # Determine Start Point:
        # Ideally start after QRS offset + delay. If QRS offset is missing, use R + 150ms.
        # Added 
        if i< len(qrs_offsets) and qrs_offsets[i]>=rp:
            start = qrs_offsets[i] + int((min_delay_ms / 1000.0) * fs)
        else:
            start = rp + int((abs_min_from_r_ms / 1000.0) * fs)

        # Safety check: Ensure we are far enough from R-peak
        start = max(start, rp + int((abs_min_from_r_ms / 1000.0) * fs))
        end = min(len(ecg)-1, rp + int((max_search_ms / 1000.0) * fs))

        if start >= end: continue
        seg = t_sig[start:end+1]; 
        if seg.size < 3: continue

        # --- Find T-peak (Absolute Amplitude)---
        local_max_idx = np.argmax(np.abs(seg))
        tpk = start + int(local_max_idx)

        # --- Find T-end ---
        # Baseline Definition: Median of the first 10ms of the search window (ST segment start).
        # This is more stable than using the window mean.

        base_win = max(1, int(0.01 * fs))  
        baseline = np.median(t_sig[start:start + base_win]) if (start + base_win) <= len(t_sig) else t_sig[start]

        t_amp = abs(float(t_sig[tpk] - baseline))

        # Amplitude Threshold: Signal must drop below 15% of peak height relative to baseline.
        amp_thr = max(0.15 * t_amp, 1e-6)

        # Slope Calculation
        slope = np.diff(t_sig, prepend=t_sig[0])

        # Slope Threshold: Must be less than 8% of the maximum slope in the segment.
        seg_slope_max = np.max(np.abs(slope[start:end+1]))
        slope_thr = max(0.08 * seg_slope_max, 1e-6)

        calm = max(1, int(0.015 * fs)) # Sustain condition: 15ms
        tend = -1
        j = tpk
        
        # Search forward from T-Peak
        while j < end - calm:
            a_win = j
            b_win = j + calm

            # Condition 1: Amplitude is close to baseline
            amp_condition = np.all(np.abs(t_sig[a_win:b_win] - baseline) < amp_thr)
            # Condition 2: Slope is flat
            slope_condition = np.all(np.abs(slope[a_win:b_win]) < slope_thr)

            if amp_condition and slope_condition:
                tend = b_win # Mark the end of the 'calm' period as T-end
                break
            j += 1

        t_peaks[i] = tpk
        t_ends[i] = tend if tend >= 0 else -1

    return t_peaks, t_ends

def detect_p_onsets(ecg: np.ndarray, fs: float, qrs_onsets: np.ndarray) -> np.ndarray:
    """Best-effort P-onset: in [-250,-50] ms pre-QRS; gate by amplitude."""
    p_sig = butter_lowpass(ecg, fs, 12.0, order=3)
    p_onsets = np.full(len(qrs_onsets), -1, dtype=int)
    for i, qon in enumerate(qrs_onsets):
        if qon < 0: continue
        a = max(0, qon - int(0.25 * fs)); b = max(0, qon - int(0.05 * fs))
        if b <= a: continue
        seg = p_sig[a:b]; 
        if seg.size < 3: continue
        ppk = a + int(np.argmax(np.abs(seg - np.median(seg))))
        rwin = p_sig[qon:qon + int(0.10 * fs)] if qon + int(0.10 * fs) < len(p_sig) else p_sig[qon:]
        r_amp = np.max(np.abs(rwin - np.median(rwin))) if rwin.size else np.max(np.abs(seg - np.median(seg)))
        if r_amp <= 0: continue
        if np.max(np.abs(seg - np.median(seg))) < 0.05 * r_amp:  # gate
            continue
        baseline = float(np.median(seg))
        j = ppk
        while j > a + 1 and abs(p_sig[j] - baseline) > 0.02 * r_amp:
            j -= 1
        p_onsets[i] = j
    return p_onsets


# =========================
# Metrics & summaries
# =========================

def beats_to_dataframe(beats: List[Beat], fs: float):
    if pd is None:
        raise RuntimeError("pandas is required to build the per-beat table.")
    rows = []
    for b in beats:
        rows.append({
            "beat_index": b.index, "t_s": b.time_s, "t_dt":b.time_dt, "r_index": b.r_index,
            "RR_prev_ms": b.rr_ms_prev, "RR_next_ms": b.rr_ms_next,
            "P_onset_idx": b.p_onset, "QRS_onset_idx": b.qrs_onset, "QRS_offset_idx": b.qrs_offset,
            "QRS_ms": b.qrs_ms, "PR_ms": b.p_ms,
            "T_peak_idx": b.t_peak, "T_end_idx": b.t_end,
            "RT_peak_ms": b.rt_peak_ms, "RT_end_ms": b.rt_end_ms,
            "QT_ms": b.qt_ms, "QTc_Bazett_ms": b.qtc_bazett_ms, "QTc_Fridericia_ms": b.qtc_fridericia_ms,
            "Tpeak_Tend_ms": b.tpte_ms,
        })
    return pd.DataFrame(rows)

def compute_summary(per_beat_df, fs: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rr = per_beat_df["RR_prev_ms"].dropna().values
    if rr.size:
        out["RR_ms_mean"] = float(np.mean(rr))
        out["RR_ms_median"] = float(np.median(rr))
        out["RR_ms_sd"] = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
        out["RR_ms_mad"] = float(np.median(np.abs(rr - np.median(rr))))
        d = np.diff(rr)
        out["RR_SDSD_ms"] = float(np.std(d, ddof=1)) if d.size > 1 else float("nan")
        rel = np.abs(d) / np.maximum(rr[:-1], 1e-9)
        out["RR_pct_outliers_gt_20pct"] = float(100.0 * np.mean(rel > 0.20)) if rel.size else float("nan")
    else:
        out.update({"RR_ms_mean": float("nan"), "RR_ms_median": float("nan"),
                    "RR_ms_sd": float("nan"), "RR_ms_mad": float("nan"),
                    "RR_SDSD_ms": float("nan"), "RR_pct_outliers_gt_20pct": float("nan")})
    qrs = per_beat_df["QRS_ms"].dropna().values
    out["QRS_ms_mean"] = float(np.mean(qrs)) if qrs.size else float("nan")
    out["QRS_ms_sd"] = float(np.std(qrs, ddof=1)) if qrs.size > 1 else (0.0 if qrs.size == 1 else float("nan"))
    out["QRS_pct_gt_120ms"] = float(100.0 * np.mean(qrs > 120.0)) if qrs.size else float("nan")
    for col in ["RT_peak_ms", "RT_end_ms"]:
        arr = per_beat_df[col].dropna().values
        if arr.size:
            out[f"{col}_mean"] = float(np.mean(arr))
            out[f"{col}_sd"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            t = per_beat_df.loc[per_beat_df[col].notna(), "t_s"].values
            x = t / 3600.0
            if x.size >= 2:
                xm, ym = x.mean(), arr.mean()
                num = np.sum((x - xm) * (arr - ym)); den = np.sum((x - xm)**2) + 1e-12
                out[f"{col}_trend_ms_per_hr"] = float(num / den)
            else:
                out[f"{col}_trend_ms_per_hr"] = float("nan")
        else:
            out[f"{col}_mean"] = out[f"{col}_sd"] = out[f"{col}_trend_ms_per_hr"] = float("nan")
    qt = per_beat_df["QT_ms"].dropna().values
    out["QT_ms_mean"] = float(np.mean(qt)) if qt.size else float("nan")
    out["QT_ms_sd"] = float(np.std(qt, ddof=1)) if qt.size > 1 else (0.0 if qt.size == 1 else float("nan"))
    for cor in ["QTc_Bazett_ms", "QTc_Fridericia_ms"]:
        arr = per_beat_df[cor].dropna().values
        if arr.size:
            out[f"{cor}_mean"] = float(np.mean(arr))
            out[f"{cor}_sd"] = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            out[f"{cor}_pct_gt_480ms"] = float(100.0 * np.mean(arr > 480.0))
            out[f"{cor}_pct_gt_500ms"] = float(100.0 * np.mean(arr > 500.0))
        else:
            out[f"{cor}_mean"] = out[f"{cor}_sd"] = out[f"{cor}_pct_gt_480ms"] = out[f"{cor}_pct_gt_500ms"] = float("nan")
    tpte = per_beat_df["Tpeak_Tend_ms"].dropna().values
    out["Tpeak_Tend_ms_mean"] = float(np.mean(tpte)) if tpte.size else float("nan")
    out["Tpeak_Tend_ms_sd"] = float(np.std(tpte, ddof=1)) if tpte.size > 1 else (0.0 if tpte.size == 1 else float("nan"))
    return out

def windowed_rr_stats(per_beat_df, windows_sec=(30, 60, 120, 300, 600)):
    """Sliding-window RR stats (step=5s)."""
    if pd is None: 
        raise RuntimeError("pandas is required for windowed summaries.")
    df = per_beat_df.sort_values("t_s").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["window_s","t_s_center","RR_ms_mean","RR_ms_median","RR_ms_sd","RR_ms_mad"])
    times = df["t_s"].values; rr = df["RR_prev_ms"].values
    rows = []
    for W in windows_sec:
        half = W / 2.0
        centers = np.arange(times[0] + half, times[-1] - half, 5.0)
        for c in centers:
            a, b = c-half, c+half
            vals = rr[(times >= a) & (times <= b)]
            vals = vals[~np.isnan(vals)]
            if not vals.size: continue
            rows.append({
                "window_s": W, "t_s_center": float(c),
                "RR_ms_mean": float(np.mean(vals)),
                "RR_ms_median": float(np.median(vals)),
                "RR_ms_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "RR_ms_mad": float(np.median(np.abs(vals - np.median(vals))))
            })
    return pd.DataFrame(rows)


# =========================
# Full pipeline
# =========================

def analyze_ecg(ecg: np.ndarray, fs: float, enable_notch: bool = True, mains_hz: float = 60.0, start_time_s: Optional[float] = None
    ) -> Tuple["pd.DataFrame", Dict[str, Any]]:
    if fs is None or fs <= 0:
        raise ValueError("Sampling rate fs must be provided and > 0.")
    
    # Absolute start time calculation
    if start_time_s is not None:
        start_time_dt = datetime.fromtimestamp(start_time_s)
    else:
        start_time_dt = None

    x = butter_bandpass(ecg.astype(float), fs, 0.5, 40.0, order=3)
    if enable_notch and sps is not None and mains_hz > 0:
        try: x = notch_filter(x, fs, freq=mains_hz, q=30.0)
        except Exception: pass
    r = detect_r_peaks(x, fs)
    if r.size < 3:
        raise RuntimeError("Too few R-peaks detected; check fs/filtering.")
    q_on, q_off = estimate_qrs_onset_offset(x, fs, r)
    t_pk, t_en = detect_tpeaks_tends(x, fs, r, q_off)
    p_on = detect_p_onsets(x, fs, q_on)
    beats: List[Beat] = []
    for i, rp in enumerate(r):
        t = rp / fs

        if start_time_dt is not None:
            time_dt = start_time_dt + timedelta(seconds=t)
        else:
            time_dt = None


        rr_prev = (rp - r[i-1]) / fs * 1000.0 if i > 0 else np.nan
        rr_next = (r[i+1] - rp) / fs * 1000.0 if i < len(r)-1 else np.nan
        qon = q_on[i] if q_on[i] >= 0 else None
        qoff = q_off[i] if q_off[i] >= 0 else None
        qrs_ms = ((qoff - qon) / fs * 1000.0) if (qon is not None and qoff is not None and qoff > qon) else np.nan
        pon = p_on[i] if p_on[i] >= 0 else None
        p_ms = ((qon - pon) / fs * 1000.0) if (pon is not None and qon is not None and qon > pon) else np.nan
        tpk = t_pk[i] if t_pk[i] >= 0 else None
        ten = t_en[i] if t_en[i] >= 0 else None
        rtpk_ms = ((tpk - rp) / fs * 1000.0) if (tpk is not None) else np.nan
        rtend_ms = ((ten - rp) / fs * 1000.0) if (ten is not None) else np.nan
        qt_ms = ((ten - qon) / fs * 1000.0) if (ten is not None and qon is not None and ten > qon) else np.nan
        rr_s = rr_prev / 1000.0 if not np.isnan(rr_prev) else np.nan
        qt_s = qt_ms / 1000.0 if not np.isnan(qt_ms) else np.nan
        if not (np.isnan(qt_s) or np.isnan(rr_s) or rr_s <= 0):
            qtc_b = qt_s / np.sqrt(rr_s) * 1000.0
            qtc_f = qt_s / (rr_s ** (1.0/3.0)) * 1000.0
        else:
            qtc_b = qtc_f = np.nan
        tpte_ms = ((ten - tpk) / fs * 1000.0) if (ten is not None and tpk is not None and ten > tpk) else np.nan
        beats.append(Beat(
            index=i, r_index=int(rp), time_s=float(t),
            time_dt=time_dt,
            rr_ms_prev=float(rr_prev) if not np.isnan(rr_prev) else None,
            rr_ms_next=float(rr_next) if not np.isnan(rr_next) else None,
            qrs_onset=int(qon) if qon is not None else None,
            qrs_offset=int(qoff) if qoff is not None else None,
            qrs_ms=float(qrs_ms) if not np.isnan(qrs_ms) else None,
            p_onset=int(pon) if pon is not None else None,
            p_ms=float(p_ms) if not np.isnan(p_ms) else None,
            t_peak=int(tpk) if tpk is not None else None,
            t_end=int(ten) if ten is not None else None,
            rt_peak_ms=float(rtpk_ms) if not np.isnan(rtpk_ms) else None,
            rt_end_ms=float(rtend_ms) if not np.isnan(rtend_ms) else None,
            qt_ms=float(qt_ms) if not np.isnan(qt_ms) else None,
            qtc_bazett_ms=float(qtc_b) if not np.isnan(qtc_b) else None,
            qtc_fridericia_ms=float(qtc_f) if not np.isnan(qtc_f) else None,
            tpte_ms=float(tpte_ms) if not np.isnan(tpte_ms) else None
        ))
    df = beats_to_dataframe(beats, fs)
    summary = compute_summary(df, fs)
    return df, summary


# =========================
# Runner
# =========================

def _save_outputs(per_beat_df, summary: Dict[str, Any], outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    perbeat_csv = outdir / "per_beat_metrics.csv"
    summary_json = outdir / "summary.json"
    window_csv = outdir / "windowed_rr.csv"

    if pd is None:
        # Minimal CSV if pandas missing
        cols = list(per_beat_df.columns)
        with open(perbeat_csv, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for _, row in per_beat_df.iterrows():
                f.write(",".join(str(row[c]) for c in cols) + "\n")
    else:
        per_beat_df.to_csv(perbeat_csv, index=False)
        windowed_rr_stats(per_beat_df, windows_sec=(30,60,120,300,600)).to_csv(window_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote:")
    print(f" - {perbeat_csv}")
    print(f" - {summary_json}")
    if pd is not None:
        print(f" - {window_csv}")
# =========================
# Plotting
# =========================

def plot_annotated_segment(
    ecg: np.ndarray, fs: float, r_peaks: np.ndarray, q_on: np.ndarray, 
    q_off: np.ndarray, t_pk: np.ndarray, t_en: np.ndarray, 
    start_s: float, end_s: float
) -> "plt.Axes":
    """Plots a segment of ECG with annotations (R, QRS, P, T-peak, T-end)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[error] matplotlib is required for plotting.")
        raise

    # 1. Defining Segment Boundaries
    start_idx = max(0, int(start_s * fs))
    end_idx = min(len(ecg), int(end_s * fs))
    
    if end_idx <= start_idx:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.set_title("ECG Segment: Indices are invalid.")
        return ax

    segment = ecg[start_idx:end_idx]
    # Create relative time axis (relative to the start of the recording)
    time_s = np.arange(start_idx, end_idx) / fs
    
    # 2. Create Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.plot(time_s, segment, color='k', linewidth=1)
    
    # 3. Add Annotations
    # We filter indices to ensure we only plot points that exist within the current view segment.
    # a) R-peak (Red Circle)
    r_in_seg = r_peaks[(r_peaks >= start_idx) & (r_peaks < end_idx)]
    ax.plot(r_in_seg / fs, ecg[r_in_seg], 'o', color='r', label='R-peak', markersize=4)

    # b) QRS Onsets/Offsets (Blue & Green Triangles)
    # QRS Onset (Blue Up-Triangle)
    q_on_idx = q_on[(q_on >= start_idx) & (q_on < end_idx) & (q_on >= 0)]
    ax.plot(q_on_idx / fs, ecg[q_on_idx], '^', color='b', label='QRS Onset', markersize=3)
    
    # QRS Offset (Green Up-Triangle)
    q_off_idx = q_off[(q_off >= start_idx) & (q_off < end_idx) & (q_off >= 0)]
    ax.plot(q_off_idx / fs, ecg[q_off_idx], '^', color='g', label='QRS Offset', markersize=3)
    
    # c) T-peaks/T-ends (Magenta & Cyan Down-Triangles)
    # T-peak (Magenta Down-Triangle)
    t_pk_idx = t_pk[(t_pk >= start_idx) & (t_pk < end_idx) & (t_pk >= 0)]
    ax.plot(t_pk_idx / fs, ecg[t_pk_idx], 'v', color='m', label='T-peak', markersize=3)
    
    # T-end (Cyan Down-Triangle)
    t_en_idx = t_en[(t_en >= start_idx) & (t_en < end_idx) & (t_en >= 0)]
    ax.plot(t_en_idx / fs, ecg[t_en_idx], 'v', color='c', label='T-end', markersize=3)

    # 4. Axis Settings
    ax.set_title(f"Annotated ECG Segment ({start_s:.2f}s to {end_s:.2f}s)")
    ax.set_xlabel("Time (s) [Relative to Start of Recording]")
    ax.set_ylabel("Amplitude")
    ax.legend(loc='upper right', fontsize='small') 
    ax.grid(True, linestyle=':', alpha=0.6)
    
    return ax

def _plot_preview(ecg: np.ndarray, fs: float, outdir: Path, mains_hz: float = 60.0):
    try:
        import matplotlib.pyplot as plt
        x = butter_bandpass(ecg, fs, 0.5, 40.0, order=3)
        if sps is not None and mains_hz > 0:
            try: x = notch_filter(x, fs, freq=mains_hz, q=30.0)
            except Exception: pass
        r = detect_r_peaks(x, fs)
        q_on, q_off = estimate_qrs_onset_offset(x, fs, r)
        t_pk, t_en = detect_tpeaks_tends(x, fs, r, q_off)
        dur = min(len(x)/fs, 10.0)
        ax = plot_annotated_segment(x, fs, r, q_on, q_off, t_pk, t_en, 0.0, dur)
        fig = ax.get_figure()
        png = outdir / "annotated_preview.png"
        fig.savefig(png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved preview plot:", png)
    except Exception as e:
        print("[warn] Plotting failed:", e)

def get_args_or_defaults():
    """
    Accept Spyder's --wdir, and fall back to defaults if --file/--dataset are not provided.
    """
    import argparse
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--file", default=None, help="Path to .hd5 (UNC or local)")
    p.add_argument("--dataset", default=None, help="Dataset path (default Bedmaster: /bedmaster/waveforms/v/value)")
    p.add_argument("--channel", type=int, default=0, help="Channel index if 2D dataset")
    p.add_argument("--fs", type=float, default=None, help="Override sampling rate (Hz)")
    p.add_argument("--mains_hz", type=float, default=60.0, help="Notch freq (50 or 60)")
    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--limit_seconds", type=float, default=None, help="Analyze only first N seconds")
    p.add_argument("--wdir", default=None, help="(Ignored) for Spyder compatibility")
    args, _ = p.parse_known_args()

    # Defaults tailored to your file:
    if args.file is None:
        args.file = r"3254689660.hd5"
    if args.dataset is None:
        args.dataset = "/bedmaster/waveforms/v/value"
    if args.outdir is None:
        args.outdir = os.path.join(os.getcwd(), "ecg_output_3254689660")
    return args

def main():
    args = get_args_or_defaults()

    # Introspection (helpful for troubleshooting)
    try:
        datasets, root_attrs = list_hd5_datasets(args.file)
        print(f"Found {len(datasets)} dataset(s) in file.")
        # Print just a couple to avoid huge logs:
        for d in datasets[:5]:
            print(f" - {d['path']} shape={d['shape']} dtype={d['dtype']}")
        if len(datasets) > 5:
            print(" ... (truncated)")
    except Exception as e:
        print("[warn] Could not list datasets:", e)

    # Load ECG
    print("\nLoading ECG ...")
    try:
        ecg, fs_guess, meta = read_ecg_from_hd5(args.file, dataset_path=args.dataset, channel=args.channel)
    except ValueError as e:
        # Dataset not found → fallback to guess
        print(f"[warn] {e}  Falling back to automatic dataset guess.")
        ecg, fs_guess, meta = read_ecg_from_hd5(args.file, dataset_path=None, channel=args.channel)

    fs = args.fs if args.fs is not None else fs_guess
    if fs is None or fs <= 0:
        fs = 240.0  # reasonable Bedmaster default; adjust per site if needed
        print(f"[warn] Could not auto-detect fs. Falling back to fs={fs} Hz")

    print(f"Using fs={fs} Hz  dataset={meta['dataset_path']}  len={len(ecg)} samples (~{len(ecg)/fs/3600:.2f} h)")


    # --- TIMESTAMP DETECTION ---
    # Attempt to read absolute start time from a sibling '/time' dataset.
    start_time_s = None
    if h5py is not None:
        try:
            # Construct the path to the time dataset by replacing the suffix
            # Example: /bedmaster/waveforms/v/value -> /bedmaster/waveforms/v/time
            time_path = meta['dataset_path'].rsplit('/', 1)[0] + '/time'
            
            with h5py.File(args.file, "r") as f:
                if time_path in f:
                    start_timestamp_data = f[time_path][()]
                    
                    if isinstance(start_timestamp_data, np.ndarray) and start_timestamp_data.size > 0:
                        if start_timestamp_data.dtype.names:
                            # Structured array (common in Bedmaster vitals)
                            start_time_s = start_timestamp_data[0][0]
                        else:
                            # Standard flat array
                            start_time_s = start_timestamp_data[0]
                    elif np.isscalar(start_timestamp_data):
                        start_time_s = start_timestamp_data

                    # Heuristic: Convert Unix milliseconds to seconds
                    # If timestamp > 1e12 (approx year 33658 in seconds), it must be milliseconds.
                    if start_time_s is not None and start_time_s > 1e12:
                        start_time_s /= 1000.0
                    
                    if start_time_s is not None:
                        meta['start_time_s'] = float(start_time_s)
                        print(f"Start Timestamp Read from Dataset: {time_path}")
                        print(f" -> Absolute Start Date: {datetime.fromtimestamp(start_time_s).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    
                else:
                    print(f"[warn] No timestamp dataset found under path '{time_path}'. Relative time will be used.")
        except Exception as e:
            print(f"[ERROR] Error reading time dataset: {e}")
            start_time_s = None 


    # Optional: shorten for quick test
    if args.limit_seconds and args.limit_seconds > 0:
        n = int(args.limit_seconds * fs)
        ecg = ecg[:n]
        print(f"Trimmed to first {args.limit_seconds} s ({n} samples).")

    # Run analysis
    print("\nAnalyzing ...")
    per_beat_df, summary = analyze_ecg(ecg, fs, enable_notch=True, mains_hz=args.mains_hz, start_time_s=start_time_s)

    # Save outputs
    outdir = Path(args.outdir)
    _save_outputs(per_beat_df, summary, outdir)

    # Quick annotated preview
    _plot_preview(ecg, fs, outdir, mains_hz=args.mains_hz)

if __name__ == "__main__":
    main()
