# Adaptive Framework for Mobile Money Fraud Detection

A production‑ready machine learning pipeline that detects fraudulent transactions in mobile money platforms (e.g., Airtel Money, M‑Pesa). The framework uses **LightGBM** with class‑imbalance handling and continuously adapts to changing fraud patterns through **sliding‑window retraining** and **drift detection**.

---

## Overview
Mobile money services process millions of transactions daily. Fraud patterns evolve over time, requiring models that can adapt. This project implements an adaptive fraud detection system that:

- Extracts customer‑centric features without look‑ahead bias
- Handles severe class imbalance (fraud ~0.13% of transactions)
- Uses **ADWIN** to detect concept drift in prediction errors
- Automatically retrains when drift is detected or on a schedule
- Tunes the classification threshold to maximise **F2‑score** (prioritising recall)

---

## Dataset
**Source**: PaySim – a synthetic mobile money simulator that mimics real M‑Pesa logs.  
**Location in Kaggle**: `../input/datasets/moonknightmarvel/paysim`  
**Size**: ~6.3 million transactions  
**Columns**: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

| Column              | Description                                                   |
|---------------------|---------------------------------------------------------------|
| `step`              | 1 step = 1 hour of simulation                                 |
| `type`              | CASH‑IN, CASH‑OUT, DEBIT, PAYMENT, TRANSFER                   |
| `amount`            | Transaction amount                                            |
| `nameOrig` / `nameDest` | Anonymised customer/merchant IDs                         |
| `oldbalanceOrg` / `newbalanceOrig` | Originator balance before and after transaction   |
| `oldbalanceDest` / `newbalanceDest` | Destination balance before and after transaction   |
| `isFraud`           | 1 = fraud, 0 = legitimate (target)                            |

---

## Methodology

### Feature Engineering
- **Temporal features**: hour, day of week, night flag, weekend flag
- **Transaction patterns**: log‑transformed amounts, balance errors (discrepancies that often indicate fraud)
- **Customer behaviour**: expanding‑window aggregations (transaction count, cumulative/average/max amount, time since last transaction) for both originator and destination – computed without future information
- **Target encoding**: smoothed expanding‑mean encoding of `nameOrig` and `nameDest` to capture account risk profiles

### Model
- **LightGBM** binary classifier with `scale_pos_weight` for imbalance
- **Hyperparameters**: `learning_rate=0.05`, `num_leaves=31`, early stopping with 50 rounds
- **Threshold tuning**: optimised on validation set to maximise F2‑score (β=2)

### Adaptive Mechanism
1. Train initial model on the **first 7 days** of data
2. Process subsequent weeks as batches:
   - Predict fraud on new transactions
   - Feed misclassification errors to **ADWIN** drift detector
   - If ADWIN signals drift **or** every 4 weeks → retrain on all historical data (can be configured for a sliding window)
3. After retraining, re‑tune the threshold on recent validation data

---

## Results
*(Performance metrics will be displayed after running the notebook.)*

The framework outputs:
- Weekly F2‑score, AUC‑ROC, precision, and recall
- Overall classification report and aggregated metrics
- Feature importance plot
- Drift alerts and retraining log

---

## How to Run
### On Kaggle
1. Open a new Kaggle notebook with **GPU disabled** (CPU only is sufficient)
2. Add the dataset: **moonknightmarvel/paysim** (via Kaggle’s “Add Data” button)
3. Copy the full code from the notebook into a single cell and run it

### Locally
1. Install dependencies:
   ```bash
   pip install lightgbm river scikit-learn pandas numpy matplotlib seaborn joblib
