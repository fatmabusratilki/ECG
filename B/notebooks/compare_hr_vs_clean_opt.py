import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py

# Add B/src to path to import b_hrv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import b_hrv

# Configure plotting
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 6]

# Paths
H5_PATH = "/Users/barko/Desktop/BLUESENSE/OPIOID_OVERDOSE/B/data/raw/3254689660.hd5"
CSV_PATH = "/Users/barko/Desktop/BLUESENSE/OPIOID_OVERDOSE/A/results/lead_II_FINAL/cleaned_rr.csv"
OUTPUT_DIR = os.path.dirname(__file__)

print("--- 1. Loading Dirty Data (BedMaster HR) [OPTIMIZED] ---")
try:
    with h5py.File(H5_PATH, 'r') as f:
        # Load ONLY first 10,000 samples to be fast
        hr_dirty = f['bedmaster/vitals/hr/value'][:10000]
        print(f"Loaded HR data (subset): {hr_dirty.shape} samples")
        
        hr_dirty = hr_dirty.astype(float)
        hr_dirty[hr_dirty <= 0] = np.nan
        
        rr_dirty = 60000.0 / hr_dirty
        rr_dirty = rr_dirty[~np.isnan(rr_dirty)]
        print(f"Converted to RR intervals: {len(rr_dirty)} beats")
        
    summary_dirty, _ = b_hrv.analyze_rr(rr_dirty)
    print("Dirty Analysis Complete.")
except Exception as e:
    print(f"Failed to load/analyze HDF5: {e}")
    summary_dirty = None

print("\n--- 2. Loading Clean Data (Lead II) ---")
if os.path.exists(CSV_PATH):
    df = pd.read_csv(CSV_PATH)
    # Take a subset of clean data too for fair comparison if needed, but full is fine
    rr_clean = df['RR_next_ms'].dropna().values
    print(f"Loaded Clean RR data: {len(rr_clean)} beats")
    
    summary_clean, _ = b_hrv.analyze_rr(rr_clean)
    print("Clean Analysis Complete.")
else:
    print("CSV file not found!")
    summary_clean = None

print("\n--- 3. Generating Comparison Plots ---")
if summary_dirty and summary_clean:
    # Metrics
    metrics = ['rmssd_ms', 'sdnn_ms', 'pnn50_pct', 'lf_hf_ratio']
    data = []
    for m in metrics:
        val_dirty = summary_dirty['time_domain'].get(m, summary_dirty['freq_domain'].get(m, 0))
        val_clean = summary_clean['time_domain'].get(m, summary_clean['freq_domain'].get(m, 0))
        data.append({'Metric': m, 'Value': val_dirty, 'Source': 'Dirty (BedMaster)'})
        data.append({'Metric': m, 'Value': val_clean, 'Source': 'Clean (Lead II)'})
    
    df_comp = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_comp, x='Metric', y='Value', hue='Source', palette=['#e74c3c', '#2ecc71'])
    plt.title('Impact of Data Cleaning: Dirty vs Clean')
    plt.ylabel('Value')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_comparison_metrics_opt.png'))
    print("Saved final_comparison_metrics_opt.png")
    
    # Poincaré
    def plot_poincare(rr, title, ax, color):
        rr_n = rr[:-1]
        rr_n1 = rr[1:]
        ax.scatter(rr_n, rr_n1, alpha=0.3, s=5, c=color)
        ax.set_xlabel('$RR_n$ (ms)')
        ax.set_ylabel('$RR_{n+1}$ (ms)')
        ax.set_title(title)
        ax.set_xlim(0, 2000)
        ax.set_ylim(0, 2000)
        ax.plot([0, 2000], [0, 2000], 'k--', alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    rr_d = summary_dirty.get('rr_used_ms', rr_dirty)
    plot_poincare(rr_d, 'Dirty Data (BedMaster Monitor)', axes[0], '#e74c3c')
    
    rr_c = summary_clean.get('rr_used_ms', rr_clean)
    plot_poincare(rr_c, 'Clean Data (Lead II Algorithm)', axes[1], '#2ecc71')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_comparison_poincare_opt.png'))
    print("Saved final_comparison_poincare_opt.png")

else:
    print("Skipping plots due to missing data.")
