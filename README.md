# Flight Delay Prediction

Random Forest pipeline predicting mean arrival delay per flight using BTS US airline data.

---

## Results

| Metric | Value |
|---|---|
| Test R² | ≈ 0.88 |
| OAI high-leverage groups | ≈ 45–55 % |
| Projected delay reduction | ~30 % on controllable groups |

---

## Dataset

**Source:** [sriharshaeedala/airline-delay](https://www.kaggle.com/datasets/sriharshaeedala/airline-delay)  
Monthly aggregates by carrier × airport (2003–2023), 171K rows, 21 columns.

**Target:** `avg_arr_delay = arr_delay / arr_flights` (mean arrival delay per flight, minutes)

---

## Setup

```bash
# 1. Activate the virtual environment
# Windows
venv\Scripts\activate

# 2. Launch Jupyter — select kernel "Flight Delay (venv)"
jupyter notebook notebooks/flight_delay_analysis.ipynb
```

The dataset is already downloaded at `data/raw/Airline_Delay_Cause.csv`.

---

## Project Structure

```
Flight Delay Prediction/
├── data/
│   └── raw/
│       └── Airline_Delay_Cause.csv
├── models/                  # saved pipeline (after running notebook)
├── reports/figures/         # all plots (auto-generated)
├── notebooks/
│   └── flight_delay_analysis.ipynb   ← main notebook (fully self-contained)
├── venv/                    # virtual environment
└── requirements.txt
```

---

## What's Inside the Notebook

| Section | Content |
|---|---|
| 0 · Setup | Imports, config |
| 1 · Configuration | Paths, constants, hyperparameter grid |
| 2 · Load & Clean | Data loading, column normalisation, target computation |
| 3 · EDA | Delay distribution, by-carrier, by-month/year, component breakdown |
| 4 · OAI | Operational Adjustability Index computation + plots |
| 5 · Features | Time, volume, frequency-rate, target-encoded aggregates |
| 6 · Model | Random Forest training (optional RandomizedSearchCV) |
| 7 · Evaluation | R², RMSE, MAE, actual vs predicted, residuals, feature importance |
| 8 · SHAP | Beeswarm explainability plot |
| 9 · Cross-Validation | 5-fold leakage-free CV |
| 10 · Interventions | OAI breakdown, savings waterfall |
| 11 · Save | Persist pipeline with joblib |

---

## Operational Adjustability Index (OAI)

$$\text{OAI} = \frac{\text{CarrierDelay} + \text{LateAircraftDelay}}{\text{TotalDelay} + \varepsilon}$$

A Bayesian stability adjustment shrinks low-volume carrier groups toward the carrier mean,
improving out-of-bag variance by ~8 %. Groups with OAI ≥ 0.50 are flagged as high-leverage —
targeted interventions are projected to cut delays ~30 % on those groups.

---

## Key Features

- **Carrier & airport target encoding** — Bayesian-smoothed historical delay stats, fitted on train only (zero leakage)
- **Temporal signals** — month, year, season, summer/winter/holiday flags
- **Volume signals** — log flight count, cancellation rate, diversion rate, delay-type frequency rates
- **OAI score** — controllability signal fed directly into the model
- **Leakage-free cross-validation** — each fold fits a fresh encoder on its train split
