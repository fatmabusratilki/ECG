# BLUESENSE ECG Analysis Pipeline 🫀

Automated ECG signal processing and feature extraction pipeline designed for **Opioid Overdose Detection** research. This project processes raw Bedmaster (`.hd5`) files to extract high-fidelity cardiac metrics, focusing on QT prolongation and HRV (Heart Rate Variability).

## 📂 Data Structure & Input
* **Input Format:** `.hd5` (Bedmaster files).
* **Sampling Rate (fs):** Auto-detected from file metadata (Defaults to **240Hz** if unavailable).
* **Target Leads:** Pipeline validates and selects the optimal lead (typically **Lead II**) for rhythm analysis.

## ⚙️ Signal Processing Methodology

Raw ECG data is inherently noisy due to patient movement (respiration) and powerline interference. The pipeline applies a robust filtering chain:

### 1. Filtering 🧹
* **Bandpass Filter:** Removes low-frequency baseline wander (respiration) and high-frequency artifacts (muscle tremors). Isolates the cardiac frequency band.
* **Notch Filter:** Eliminates 50Hz/60Hz powerline interference ("humming" noise).

### 2. Beat Detection (R-Peak) 📉
Uses a derivative-based logic (Pan-Tompkins approach):
* **Method:** Derivative → Squaring → Moving Window Integration → Adaptive Thresholding.
* **Logic:** "Sudden, sharp rise in amplitude indicates a beat."
* **Safety:** Implements a **refractory period** to avoid false positives (e.g., confusing T-waves with R-peaks).

### 3. Waveform Segmentation (QRS, P, T)
* **QRS Complex:** Measures ventricular depolarization (contraction).
    * *Q:* First downward deflection.
    * *R:* Peak amplitude.
    * *S:* Downward deflection following R.
    * *Metric:* **QRS_ms** (Prolongation may indicate conduction slowing/toxicity).
* **T-Wave:** Represents ventricular repolarization (recharging).
    * Crucial for QT calculation.
    * Algorithm detects **T-peak** and **T-end** (using tangent/slope method).
* **P-Wave:** Atrial depolarization. Detected via backward search from QRS onset.

## ⚠️ Clinical Relevance: Opioid Overdose Detection

This pipeline focuses on two critical biomarkers associated with opioid toxicity:

### 1. QT Interval & Opioids
The time from the start of the Q wave to the end of the T wave.
* **Mechanism:** Many opioids (e.g., methadone) and associated drugs delay cardiac repolarization.
* **Risk:** This leads to **Long QT Syndrome (LQTS)**. Excessive prolongation (>500ms) can trigger fatal arrhythmias (Torsades de Pointes).
* **Pipeline Role:** Accurately measures QTc (Corrected QT) to detect risk early.

### 2. Heart Rate Variability (HRV)
The variation in time intervals between consecutive heartbeats (RR intervals).
* **Mechanism:** Opioids suppress the respiratory center in the brainstem, causing respiratory depression.
* **Impact:** Reduced respiration disrupts the natural "Respiratory Sinus Arrhythmia" (RSA), leading to distinct changes in HRV patterns (e.g., reduced High-Frequency power).

## 📊 Outputs

The pipeline generates the following artifacts for each processed file:

| File | Description |
| :--- | :--- |
| **`per_beat_metrics.csv`** | Granular data for every single heartbeat. Contains timing for P, Q, R, S, T, and QT intervals. |
| **`summary.json`** | High-level report containing aggregated stats (Mean HR, Mean QTc, Signal Quality Score). |
| **`annotated_preview.png`** | Visual validation plot showing the first 10 seconds with marked R-peaks (🔴), QRS boundaries (mV), and T-waves. |
| **`windowed_rr.csv`** | Time-series data of RR intervals, ready for HRV spectral analysis. |

---
*Project maintained by the BLUESENSE AI Team.*
