<div align="center">

# 🚗 Vehicle Insurance Claim Fraud Detection

[![Python](https://img.shields.io/badge/Python-3.9+-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-%23FA0F00.svg?style=for-the-badge&logo=Jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)

**An end-to-end machine learning project that uses historical vehicle policy, claim, and customer data to detect fraudulent insurance claims — comparing 12+ models across traditional ML, ensemble methods, and deep learning.**

*Texas Tech University — MS Data Science Capstone Project*

</div>

---

## 🎯 Problem Statement

Insurance fraud costs the industry billions annually. This project builds an **automated fraud-detection system** that flags high-risk claims early while balancing:
- **Recall** — catching as many fraudulent claims as possible
- **Precision** — minimizing false alarms and unnecessary investigations
- **F1-Score & ROC-AUC** — the primary evaluation metrics given severe class imbalance

---

## 🏗️ Project Pipeline

```
┌─────────────────┐    ┌──────────────────────┐    ┌───────────────────────┐
│  Data Loading   │───▶│  EDA & Visualization │───▶│  Data Preprocessing   │
│  fraud_oracle   │    │                      │    │                       │
│  .csv           │    │  Class imbalance     │    │  Drop irrelevant cols │
│                 │    │  Distribution plots  │    │  Impute Age = 0       │
│                 │    │  Correlation heatmap │    │  Ordinal encoding     │
└─────────────────┘    └──────────────────────┘    │  One-Hot encoding     │
                                                    └───────────┬───────────┘
                                                                │
                       ┌────────────────────────────────────────▼──────────────────────┐
                       │                    Feature Selection                           │
                       │         Chi-Square Test (p-value < 0.05)                      │
                       │  Drop: DayOfWeek, WitnessPresent, WeekOfMonthClaimed,         │
                       │         DayOfWeekClaimed, DriverRating, WeekOfMonth,          │
                       │         NumberOfCars, RepNumber, Days_Policy_Claim            │
                       └───────────────────────────┬────────────────────────────────────┘
                                                   │
              ┌────────────────────────────────────▼────────────────────────────────────┐
              │                     Model Training & Evaluation                          │
              │  XGBoost · LightGBM · Decision Tree · Random Forest · AdaBoost          │
              │  Gradient Boosting · MLP · SVM · KNN · Naive Bayes · Ensemble           │
              │  Neural Network (Keras) · GNN                                           │
              └─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
vehicle-fraud-detection/
├── notebooks/
│   └── Final_project_Raman81.ipynb    # Full end-to-end analysis notebook
├── data/
│   └── fraud_oracle.csv               # Source dataset (place here)
├── src/
│   ├── preprocessing.py               # Data cleaning & encoding utilities
│   ├── feature_selection.py           # Chi-Square test feature selection
│   └── models.py                      # Model training & evaluation functions
├── reports/
│   └── Vehicle_Insurance_Fraud_Detection_Presentation.pptx
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📊 Dataset — `fraud_oracle.csv`

The dataset contains historical vehicle insurance policy and claim records with the following key features:

| Feature Category | Columns |
|---|---|
| **Policy Info** | PolicyNumber, PolicyType, BasePolicy, Deductible |
| **Vehicle Info** | Make, VehicleCategory, AgeOfVehicle, VehiclePrice |
| **Claim Info** | Month, MonthClaimed, WeekOfMonth, DayOfWeek, AccidentArea |
| **Customer Info** | Age, Sex, MaritalStatus, AgeOfPolicyHolder, DriverRating |
| **Incident Info** | Fault, PoliceReportFiled, WitnessPresent, NumberOfCars |
| **Target** | `FraudFound_P` (0 = Legitimate, 1 = Fraudulent) |

> ⚠️ **Class Imbalance:** The dataset is significantly imbalanced — fraudulent claims are a minority class. F1-Score and ROC-AUC are therefore the primary evaluation metrics.

---

## 🔧 Preprocessing Steps

1. **Drop `PolicyNumber`** — purely incremental row ID, no predictive value
2. **Drop `PolicyType`** — redundant (concatenation of `VehicleCategory` + `BasePolicy`)
3. **Remove records** where `DayOfWeekClaimed` or `MonthClaimed` = `'0'`
4. **Impute `Age = 0`** → `16.5` (cross-referenced with `AgeOfPolicyHolder` values of 16–17)
5. **Ordinal Encoding** for binary/ordered categorical features (AccidentArea, Sex, Fault, etc.)
6. **One-Hot Encoding** for nominal features (Make, MaritalStatus, VehicleCategory, BasePolicy)

---

## 🔬 Feature Selection — Chi-Square Test

Chi-Square tests (α = 0.05) were run between every feature and the target (`FraudFound_P`). Features with p-value > 0.05 (fail to reject H₀) were dropped:

| Dropped Feature | Reason |
|---|---|
| `Days_Policy_Claim` | Not statistically significant |
| `DayOfWeek` | Not statistically significant |
| `WitnessPresent` | Not statistically significant |
| `WeekOfMonthClaimed` | Not statistically significant |
| `DayOfWeekClaimed` | Not statistically significant |
| `DriverRating` | Not statistically significant |
| `WeekOfMonth` | Not statistically significant |
| `NumberOfCars` | Not statistically significant |
| `RepNumber` | Not statistically significant |

**Top predictors retained:** `BasePolicy_Liability`, `VehicleCategory`, `Fault`, `AccidentArea`, `Make`

---

## 🤖 Models Implemented

| # | Model | Type |
|---|---|---|
| 1 | XGBoost | Gradient Boosted Trees |
| 2 | LightGBM | Gradient Boosted Trees |
| 3 | Decision Tree | Tree-based |
| 4 | Random Forest | Ensemble (Bagging) |
| 5 | AdaBoost | Ensemble (Boosting) |
| 6 | Gradient Boosting | Ensemble (Boosting) |
| 7 | MLP (Multi-Layer Perceptron) | Neural Network |
| 8 | SVM (Support Vector Machine) | Kernel-based |
| 9 | KNN (K-Nearest Neighbors) | Instance-based |
| 10 | Naive Bayes | Probabilistic |
| 11 | Ensemble (Voting) | Meta-model |
| 12 | Neural Network (Keras) | Deep Learning |
| 13 | GNN (Graph Neural Network) | Deep Learning |

**Train/Test Split:** 80/20 with `stratify=y` to preserve class distribution

---

## 📈 Key Results

### Best Performing Models

| Model | Accuracy | Recall | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Naive Bayes** | — | **0.88** | Low | — | — |
| **Gradient Boosting** | High | Balanced | High | — | Strong |
| **XGBoost** | High | Balanced | High | — | Strong |
| **Decision Tree (FS)** | — | — | — | **0.246** | — |
| **Neural Network (Keras)** | Competitive | — | — | — | Competitive |

### Key Insights

- **Naive Bayes** showed the highest recall (0.88) — best for catching fraud at the cost of more false alarms
- **Gradient Boosting and XGBoost** balanced accuracy and precision well — best for production deployment
- **Decision Tree with feature selection** achieved the strongest F1-score (0.246)
- **Neural Networks and GNNs** captured complex non-linear patterns with competitive AUC scores
- **Feature selection** reduced model complexity without significant performance loss
- **Class imbalance** makes F1 and ROC-AUC far more informative than raw accuracy

### Before vs. After Feature Selection

Feature selection (dropping 9 statistically insignificant columns) reduced model complexity while maintaining or improving performance — confirming that simpler models with clean features can match complex overparameterized ones.

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ramanbarnwalgit81/vehicle-fraud-detection.git
cd vehicle-fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the Dataset

Place `fraud_oracle.csv` in the `data/` folder.

> The dataset is available on Kaggle: [Vehicle Insurance Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)

### 4. Run the Notebook

```bash
jupyter notebook notebooks/Final_project_Raman81.ipynb
```

---

## 📦 Requirements

See `requirements.txt` for full dependencies. Core packages:

```
pandas
numpy
matplotlib
seaborn
scipy
scikit-learn
xgboost
lightgbm
tensorflow
category_encoders
imbalanced-learn
jupyter
```

---

## 🔍 Notebook Walkthrough

| Section | Description |
|---|---|
| **1. Data Loading** | Load `fraud_oracle.csv`, inspect shape, dtypes, unique values |
| **2. EDA** | Class distribution, fraud by month/vehicle/area, correlation heatmap |
| **3. Preprocessing** | Drop columns, impute age, ordinal + one-hot encoding |
| **4. Feature Selection** | Chi-Square test, p-value analysis, feature dropping |
| **5. Baseline Models** | Train 13 models, collect accuracy/recall/precision/F1/AUC |
| **6. Feature-Selected Models** | Retrain with reduced feature set, compare metrics |
| **7. Model Comparison** | Side-by-side confusion matrices and ROC curves |
| **8. Neural Networks** | Keras MLP and GNN implementations |
| **9. Insights** | Final recommendations for production deployment |

---

## 💡 Recommendations for Production

1. **Primary model:** Gradient Boosting or XGBoost for balanced precision-recall
2. **High-recall scenario:** Naive Bayes when catching all fraud is the priority (e.g. low-cost investigation workflows)
3. **Address class imbalance** with SMOTE or class-weight adjustments before production deployment
4. **Monitor model drift** — fraud patterns evolve; retrain quarterly with new claim data
5. **Feature pipeline:** Automate the ordinal + one-hot encoding pipeline using `sklearn.Pipeline` for production serving

---

## 📄 License

This project is for academic and educational purposes — Texas Tech University, MS Data Science.

---

<div align="center">

*Built with scikit-learn · XGBoost · LightGBM · TensorFlow/Keras*

*Texas Tech University — MS Data Science*

</div>
