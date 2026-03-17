"""
models.py
---------
Model training and evaluation utilities for Vehicle Insurance Fraud Detection.
Covers: traditional ML, ensemble methods, and deep learning (Keras).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve
)
from sklearn.base import clone
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ─────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────

def get_models() -> dict:
    """Return dictionary of all models to evaluate."""
    return {
        'XGBoost': XGBClassifier(
            use_label_encoder=False,
            objective='binary:logistic',
            eval_metric='aucpr',
            random_state=42
        ),
        'LightGBM': LGBMClassifier(random_state=42, verbose=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'MLP': MLPClassifier(max_iter=300, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
    }


# ─────────────────────────────────────────────
# Training & Metrics Collection
# ─────────────────────────────────────────────

def fit_and_evaluate(models: dict, X_train, X_test, y_train, y_test,
                     label_suffix: str = "") -> pd.DataFrame:
    """
    Train each model, collect metrics, return results DataFrame.

    Parameters:
        models       : dict of {name: model}
        X_train/test : Feature sets
        y_train/test : Target arrays
        label_suffix : Append to model name (e.g. ' (FS)' for feature-selected)

    Returns:
        DataFrame with columns [model, accuracy, recall, precision, f1, roc_auc]
    """
    results = []

    for name, model in models.items():
        print(f"Training: {name}{label_suffix}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        row = {
            'model':     name + label_suffix,
            'accuracy':  accuracy_score(y_test, y_pred),
            'recall':    recall_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'f1':        f1_score(y_test, y_pred, zero_division=0),
            'roc_auc':   roc_auc_score(y_test, y_prob) if y_prob is not None else None,
        }
        results.append(row)
        print(f"  → F1: {row['f1']:.4f} | ROC-AUC: {row['roc_auc']:.4f if row['roc_auc'] else 'N/A'}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def plot_confusion_matrices(models: dict, X_train, X_test, y_train, y_test,
                             cols: int = 3, figsize=(18, 12)):
    """Plot confusion matrices for all models in a grid."""
    n = len(models)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for idx, (name, model) in enumerate(models.items()):
        m = clone(model)
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=axes[idx], colorbar=False)
        axes[idx].set_title(name, fontsize=11)

    # Hide unused axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models: dict, X_train, X_test, y_train, y_test, figsize=(12, 8)):
    """Plot ROC curves for all models on a single chart."""
    plt.figure(figsize=figsize)

    for name, model in models.items():
        m = clone(model)
        m.fit(X_train, y_train)
        if hasattr(m, 'predict_proba'):
            y_prob = m.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — All Models')
    plt.legend(loc='lower right', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(df_results: pd.DataFrame, figsize=(14, 6)):
    """Bar chart comparing all models across key metrics."""
    metrics = ['accuracy', 'recall', 'precision', 'f1', 'roc_auc']
    df_plot = df_results.set_index('model')[metrics]

    df_plot.plot(kind='bar', figsize=figsize, colormap='tab10')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# Keras Neural Network
# ─────────────────────────────────────────────

def build_keras_model(input_dim: int):
    """
    Build a simple feedforward neural network for binary fraud classification.

    Architecture:
        Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.2) → Dense(1, sigmoid)
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


if __name__ == "__main__":
    print("Available models:")
    for name in get_models():
        print(f"  - {name}")
