# Data

Place `fraud_oracle.csv` in this directory before running the notebook.

## Source

The dataset is publicly available on Kaggle:
👉 [Vehicle Claim Fraud Detection — Kaggle](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)

## Dataset Summary

| Property | Value |
|---|---|
| File | `fraud_oracle.csv` |
| Rows | ~15,000+ |
| Features | 33 columns |
| Target | `FraudFound_P` (0 = Legitimate, 1 = Fraud) |
| Class Imbalance | Yes — fraud is the minority class |

## Important Notes

- `fraud_oracle.csv` is **not committed to this repository** (see `.gitignore`) due to file size
- Download from Kaggle and place it here as `data/fraud_oracle.csv`
- The notebook expects the file at this exact path: `../data/fraud_oracle.csv` (relative to `notebooks/`)
