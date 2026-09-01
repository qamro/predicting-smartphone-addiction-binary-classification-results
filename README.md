# Playground Series S6E8 - Screen Addiction Prediction

Binary classification solution for Kaggle's **Playground Series - Season 6, Episode 8**,
predicting whether an individual is "addicted" to screen usage from self-reported
digital habits and lifestyle indicators.

**Metric:** ROC AUC

**Best OOF AUC:** 0.9628 (v2) · **Leaderboard score:** 0.96323

---

## Repo Structure

```
.
├── data/
│   ├── train.csv                # Training data (691,369 rows × 14 cols)
│   ├── test.csv                 # Test data (296,302 rows, no target)
│   └── sample_submission.csv    # Required submission format
├── results/
│   ├── submission.csv           # v1 output - single LightGBM
│   └── submission_v2.csv        # v2 output - engineered features + blended ensemble
├── train_model_v1.py            # v1 training script
├── train_model_v2.py            # v2 training script
└── README.md
```

---

## Dataset

Each row represents an individual with self-reported screen-time and lifestyle data.
Nearly every column contains missing values by design.

| Column | Type | Description |
|---|---|---|
| `id` | int | Row identifier |
| `age` | float | Age |
| `daily_screen_time_hours` | float | Avg. daily screen time |
| `social_media_hours` | float | Daily social media time |
| `gaming_hours` | float | Daily gaming time |
| `work_study_hours` | float | Daily work/study time |
| `sleep_hours` | float | Avg. nightly sleep |
| `notifications_per_day` | float | Notifications received per day |
| `app_opens_per_day` | float | App opens per day |
| `weekend_screen_time` | float | Weekend screen time |
| `gender` | category | Male / Female / Other |
| `stress_level` | category | Low / Medium / High |
| `academic_work_impact` | category | Yes / No |
| `addicted_label` | int (target) | 1 = addicted, 0 = not addicted |

Target is imbalanced: ~71% positive class, ~29% negative.

---

## Approach

### v1 - Baseline (`train_model_v1.py`)
- Single **LightGBM** classifier, 5-fold stratified CV.
- Native handling of missing values and categorical features — no imputation
  or one-hot encoding.
- **Result:** OOF AUC 0.9620, leaderboard 0.96323.

### v2 - Feature Engineering + Ensemble (`train_model_v2.py`)
- Adds:
  - Missingness indicator columns for every feature (missingness itself is predictive).
  - Ratio/interaction features: `social_to_screen`, `gaming_to_screen`,
    `sleep_deficit`, `weekend_to_screen`, `screen_per_age`,
    `weekday_weekend_diff`, and others.
- Trains **LightGBM + XGBoost + CatBoost**, 3-fold stratified CV (reduced from
  5 due to single-CPU compute constraints).
- Blends the three models using an OOF-AUC-maximizing weight search.
- **Result:** OOF AUC 0.9628. On this run the blend search settled on 100%
  LightGBM weight, the lighter XGB/CatBoost configs (needed to fit the compute
  budget) didn't outperform LightGBM individually, but the added features
  still improved LightGBM itself over v1.

---

## Usage

```bash
pip install lightgbm xgboost catboost scikit-learn pandas numpy

# Baseline
python train_model_v1.py

# Feature-engineered ensemble
python train_model_v2.py
```

Each script reads `train.csv` / `test.csv` from the working directory and writes
a submission CSV (`submission.csv` or `submission_v2.csv`) in the same format as
`sample_submission.csv`:

```csv
id,addicted_label
691369,0.99
691370,0.03
...
```

Where `addicted_label` is the predicted **probability** of addiction, not a
hard 0/1 label.

---

## Notes / Next Steps

- Training was constrained to a single CPU core; with more compute, restoring
  5-fold CV and deeper/longer XGBoost & CatBoost runs would likely let the
  ensemble outperform LightGBM alone.
- Further gains are more likely to come from additional feature engineering
  (binning, clustering, higher-order interactions) than from adding more model
  types.
