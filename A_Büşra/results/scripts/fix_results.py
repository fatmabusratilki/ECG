import pandas as pd
import numpy as np
import json
import sys
import os
from pathlib import Path

# =================================================================
# BÖLÜM 1: ORIGINAL_A.PY'NİN BEYNİ (HİÇ DOKUNULMADI)
# =================================================================
# Bu fonksiyonlar original_a.py dosyasından birebir kopyalanmıştır.
# Böylece Summary ve Windowed sonuçları %100 orijinal formatta olur.

def compute_summary(per_beat_df) -> dict:
    out = {}
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
    df = per_beat_df.sort_values("t_s").reset_index(drop=True)
    if df.empty:
        return pd.DataFrame(columns=["window_s","t_s_center","RR_ms_mean","RR_ms_median","RR_ms_sd","RR_ms_mad"])
    times = df["t_s"].values; rr = df["RR_prev_ms"].values
    rows = []
    
    print(f"   -> Windowed analiz başlıyor ({len(df)} atış)...")
    
    for W in windows_sec:
        half = W / 2.0
        centers = np.arange(times[0] + half, times[-1] - half, 5.0)
        
        # Hızlandırma: Numpy searchsorted
        start_indices = np.searchsorted(times, centers - half, side='left')
        end_indices = np.searchsorted(times, centers + half, side='right')
        
        for i, (a_idx, b_idx) in enumerate(zip(start_indices, end_indices)):
            vals = rr[a_idx:b_idx]
            vals = vals[~np.isnan(vals)]
            if not vals.size: continue
            rows.append({
                "window_s": W, "t_s_center": float(centers[i]),
                "RR_ms_mean": float(np.mean(vals)),
                "RR_ms_median": float(np.median(vals)),
                "RR_ms_sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                "RR_ms_mad": float(np.median(np.abs(vals - np.median(vals))))
            })
    return pd.DataFrame(rows)

# =================================================================
# BÖLÜM 2: AMELİYATHANE (GİRDİ/ÇIKTI)
# =================================================================

def main():
    # 1. DOSYA YOLLARI (Otomatik Bulma)
    # Script: .../src/fix_results.py
    SCRIPT_DIR = Path(__file__).resolve().parent 
    PROJECT_ROOT = SCRIPT_DIR.parent
    
    # GİRDİ: Kirli ama full olan Lead II CSV'si
    INPUT_PATH = PROJECT_ROOT / "data/processed/3254689660/lead_II_FULL/per_beat_metrics.csv"
    
    # ÇIKTI: Results klasörü altında "lead_II_FINAL"
    OUTPUT_DIR = PROJECT_ROOT / "results/lead_II_FINAL"
    
    # TEMİZLİK EŞİĞİ (Senin Jupyter'de yaptığın ayar)
    RR_MIN = 300
    RR_MAX = 2000

    print("--- BLUESENSE: FINAL RESULT FIXER ---")
    print(f"[INPUT] {INPUT_PATH}")
    
    if not INPUT_PATH.exists():
        print("[HATA] Dosya bulunamadı!")
        sys.exit(1)

    if not OUTPUT_DIR.exists():
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. YÜKLEME
    print("[1/4] CSV Yükleniyor...")
    df_dirty = pd.read_csv(INPUT_PATH)
    original_len = len(df_dirty)
    
    # 3. FİLTRELEME (Thresholding)
    print(f"[2/4] Temizleniyor (Threshold: {RR_MIN}-{RR_MAX} ms)...")
    df_clean = df_dirty[
        (df_dirty["RR_prev_ms"] >= RR_MIN) & 
        (df_dirty["RR_prev_ms"] <= RR_MAX)
    ].copy()
    
    removed = original_len - len(df_clean)
    print(f"   -> {removed} adet 'çöp' veri silindi.")
    print(f"   -> {len(df_clean)} adet temiz atış kaldı.")

    # 4. HESAPLAMA (Orijinal Fonksiyonlarla)
    print("[3/4] İstatistikler orijinal mantıkla yeniden hesaplanıyor...")
    
    # Summary JSON
    summary_clean = compute_summary(df_clean)
    
    # Windowed RR
    windowed_clean = windowed_rr_stats(df_clean)

    # 5. KAYDETME
    print(f"[4/4] Kaydediliyor: {OUTPUT_DIR}")
    
    # A) Temiz CSV (B Dosyası için)
    df_clean.to_csv(OUTPUT_DIR / "cleaned_rr.csv", index=False)
    
    # B) Temiz Summary (Rapor için)
    with open(OUTPUT_DIR / "summary_clean.json", "w") as f:
        json.dump(summary_clean, f, indent=2)
        
    # C) Temiz Windowed (Grafik için)
    windowed_clean.to_csv(OUTPUT_DIR / "windowed_rr.csv", index=False)
    
    print("\n✅ MÜKEMMEL! Artık elinde:")
    print("   1. cleaned_rr.csv (HRV için ham madde)")
    print("   2. summary_clean.json (Eksiksiz, orijinal formatta rapor)")
    print("   3. windowed_rr.csv (Temizlenmiş zaman serisi)")

if __name__ == "__main__":
    main()