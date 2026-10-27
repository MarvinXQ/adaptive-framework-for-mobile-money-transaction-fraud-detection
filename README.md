# Adaptive Framework for Mobile Money Fraud Detection

A production-grade machine-learning pipeline that detects fraudulent transactions in mobile-money platforms (e.g., Airtel Money, M-Pesa). The framework uses **LightGBM** with class-imbalance handling, **SMOTEENN** oversampling, and continuously adapts to changing fraud patterns through **ADWIN drift detection** and **scheduled retraining**.

**Demonstrated on PaySim**: F2 = **0.987**, Recall = **0.985**, Precision = **0.996**, PR-AUC = **0.998**.

---

## Overview

Mobile-money services process millions of transactions daily. Fraud patterns evolve over time, so a static model degrades within weeks. This project implements an adaptive pipeline that:

- Extracts customer-centric features **without look-ahead bias**
- Handles severe class imbalance (fraud ≈ 0.13 % of transactions) via **`scale_pos_weight`** + **SMOTEENN**
- Uses **ADWIN** to detect concept drift in prediction errors
- Retrains automatically on drift, and every 4 weeks on a schedule
- Tunes the classification threshold to **maximise F2** subject to a **recall floor ≥ 0.85** — prioritising catching fraud over minimising false positives

---

## Dataset

**Source**: PaySim — a synthetic mobile-money simulator that mimics real M-Pesa logs.
**Location in Kaggle**: `/kaggle/input/datasets/moonknightmarvel/paysim/`
**Size**: ~6.36 M transactions
**Fraud rate**: 0.1291 %

| Column | Description |
|---|---|
| `step` | 1 step = 1 hour of simulation |
| `type` | CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER |
| `amount` | Transaction amount |
| `nameOrig` / `nameDest` | Anonymised customer / merchant IDs |
| `oldbalanceOrg` / `newbalanceOrig` | Originator balance before / after |
| `oldbalanceDest` / `newbalanceDest` | Destination balance before / after |
| `isFraud` | 1 = fraud, 0 = legitimate (target) |
| `isFlaggedFraud` | Legacy flag (unused) |

> ⚠️ **PaySim is synthetic and has near-deterministic fraud rules.** Fraud only occurs on `TRANSFER` / `CASH_OUT` transactions that fully drain the originator's balance. That makes balance-error features (`error_orig`, `error_dest`) almost perfect predictors. The 0.987 F2 shown here is honest *on PaySim* but should **not** be interpreted as a real-world fraud-detection rate. See *Feature Ablation* below.

---

## Methodology

### Feature Engineering

- **Temporal**: `hour`, `day_of_week`, `is_night`, `is_weekend`
- **Amount**: log-transformed `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`
- **Balance consistency**: `error_orig`, `error_dest` and their absolute values — the strongest signals on PaySim
- **Zero-balance flags**: `orig_zero_before`, `dest_zero_before`
- **Customer behaviour** (expanding window, **no look-ahead**):
  - Transaction count, cumulative amount, average amount, max amount for both `nameOrig` and `nameDest`
  - `orig_time_since_last` — hours since the same originator's previous transaction
- **No-leakage target encoding**: `nameOrig_te`, `nameDest_te` computed with a **shift-by-one** expanding mean, so each row only sees *prior* labels, never its own

All expanding features depend on a single stable `sort_values(…, kind="mergesort")` sort by `timestamp`.

### Model

- **LightGBM** binary classifier
- **`scale_pos_weight = neg / (pos + 1)`** handles residual imbalance after resampling
- **Hyperparameters**:
  - `learning_rate=0.03`, `num_leaves=63`, `min_data_in_leaf=20`
  - `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`
  - `lambda_l1=0.1`, `lambda_l2=0.1`
  - `metric=["auc", "average_precision"]`
  - `num_boost_round=1200` with `early_stopping(100)` on PR-AUC
- **Threshold tuning**: picks the operating point that maximises **F2 (β=2)** subject to **recall ≥ 0.85** — prevents the model from landing on a high-F2/low-recall threshold that would be useless in fraud operations

### Class-Imbalance Handling

Retraining data passes through **SMOTEENN** (SMOTE oversampling + Edited-Nearest-Neighbour cleaning). To keep memory and runtime bounded:

- Majority class down-sampled to at most **20:1** vs. minority (or `SMOTE_MAX_MAJORITY=200 000` rows, whichever is smaller)
- SMOTEENN skipped automatically when a retraining slice has fewer than 50 fraud samples

### Adaptive Mechanism

1. **Bootstrap**: train the initial model on the first 7 days of data
2. **Weekly loop** over subsequent weeks:
   - Predict fraud on the new batch
   - Feed **binary classification errors** to **ADWIN** (`delta=0.002`)
   - On **drift signal** or every **4 weeks**: retrain on all available history (80/20 train/val split by date)
   - After retraining, re-tune the F2 threshold with the same recall floor

---

## Results

### Aggregated test performance (weeks 2–31)

| Metric | Value |
|---|---:|
| **F2** | **0.9870** |
| **F1** | 0.9904 |
| **Precision** | 0.9960 |
| **Recall** | 0.9848 |
| **MCC** | 0.9904 |
| **ROC-AUC** | 0.9997 |
| **PR-AUC** | 0.9976 |

Confusion matrix (aggregated over all test weeks):

| | Predicted Legit | Predicted Fraud |
|---|---:|---:|
| **Actual Legit** | ~4,429,858 | ~95 |
| **Actual Fraud** | ~96 | ~6,215 |

The notebook also produces:

- Weekly F2 / precision / recall / ROC-AUC / PR-AUC / MCC / volume / fraud-count plot
- Feature-importance plot for the final model
- `weekly_performance.csv` with per-week metrics
- `adaptive_fraud_model.pkl` containing the final model, threshold, and feature list

### Feature Ablation (recommended when reporting)

To expose performance **without the PaySim shortcut signals**, re-run the pipeline with the following features removed before training:

- `error_orig`, `error_orig_abs`
- `error_dest`, `error_dest_abs`
- `nameOrig_te`, `nameDest_te`

On PaySim this typically drops F2 from ~0.99 to **~0.85–0.90** — the number that is closer to a realistic mobile-money fraud benchmark. Always report **both** runs in papers or portfolios, with the caveat that PaySim is synthetic.

---

## How to Run

### On Kaggle (recommended)

The notebook is designed as **three cells**. The order matters.

1. **Create a Kaggle notebook** from the PaySim dataset page:
   `https://www.kaggle.com/datasets/moonknightmarvel/paysim` → **New Notebook**
   This auto-attaches the dataset.

2. **Paste the three cells** into three code cells (in order).

3. **Run Cell 1** — environment setup:
   - Installs **SciPy 1.14.1** with `--no-deps --target /kaggle/working/numpy_fix`
   - Skips install entirely if SciPy 1.14.1 is already present
   - No `pip` dependency-conflict warnings, no post-install `ERROR:` noise
   - Output is 3 clean lines on success

4. **Run Cell 2** — full pipeline:
   - Prepends `/kaggle/working/numpy_fix` to `sys.path` so the compatible SciPy is loaded **before** the system one
   - Performs a **functional** environment check (`from scipy.stats import gaussian_kde`), not a version-number assertion
   - Trains, adapts, and saves all artefacts to `/kaggle/working/`

5. **Run Cell 3** — final evaluation and plots.

> ✅ No kernel restart is required between cells.
> ⚠️ If you see `AttributeError: _blas_supports_fpe`, it means an ABI mismatch slipped through — run `Run → Restart & Clear Cell Outputs`, then re-run Cell 1 → Cell 2.

### Locally

1. Create a virtual environment (Python ≥ 3.10):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate

   pip install "numpy>=1.24,<2.3" "scipy>=1.11" \
            "scikit-learn>=1.3" "lightgbm>=4.0" \
            "river>=0.21" "imbalanced-learn>=0.12" \
            "pandas>=2.0" "matplotlib>=3.7" "seaborn>=0.13" "joblib"
