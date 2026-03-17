"""
feature_selection.py
--------------------
Chi-Square test based feature selection for Vehicle Insurance Fraud Detection.
"""

import pandas as pd
import numpy as np
import scipy.stats as ss
from itertools import product


# Columns identified as statistically insignificant (p-value > 0.05)
# Drop these before final model training
COLS_TO_DROP = [
    'Days_Policy_Claim',
    'DayOfWeek',
    'WitnessPresent',
    'WeekOfMonthClaimed',
    'DayOfWeekClaimed',
    'DriverRating',
    'WeekOfMonth',
    'NumberOfCars',
    'RepNumber',
]


def run_chi_square_test(df: pd.DataFrame, target: str = 'FraudFound_P') -> pd.DataFrame:
    """
    Run Chi-Square tests between all feature pairs and return sorted results.

    Parameters:
        df     : Encoded DataFrame
        target : Target column name

    Returns:
        DataFrame with columns [var1, var2, coeff, result]
    """
    cat_var_prod = list(product(df.columns, df.columns))

    results = []
    for var1, var2 in cat_var_prod:
        if var1 != var2:
            try:
                chi2, p_val, _, _ = ss.chi2_contingency(
                    pd.crosstab(df[var1], df[var2])
                )
                results.append((var1, var2, p_val))
            except Exception:
                pass

    chi_df = pd.DataFrame(results, columns=['var1', 'var2', 'coeff'])
    chi_df['result'] = chi_df['coeff'].apply(lambda x: 'Reject H0' if x <= 0.05 else 'Accept H0')
    return chi_df


def get_target_significance(chi_df: pd.DataFrame, target: str = 'FraudFound_P') -> pd.DataFrame:
    """
    Filter Chi-Square results to show only target variable relationships,
    sorted by p-value.
    """
    result = (
        chi_df[chi_df['var1'] == target]
        .sort_values('coeff')
        .reset_index(drop=True)
    )
    return result


def drop_insignificant_features(X_train: pd.DataFrame, X_test: pd.DataFrame,
                                 cols_to_drop: list = None):
    """
    Drop statistically insignificant features from train and test sets.

    Parameters:
        X_train     : Training feature set
        X_test      : Test feature set
        cols_to_drop: List of columns to drop (defaults to COLS_TO_DROP)

    Returns:
        (X_train_fs, X_test_fs) with dropped columns
    """
    if cols_to_drop is None:
        cols_to_drop = COLS_TO_DROP

    # Only drop columns that actually exist in the DataFrame
    existing = [c for c in cols_to_drop if c in X_train.columns]
    dropped = [c for c in cols_to_drop if c not in X_train.columns]

    if dropped:
        print(f"Warning: columns not found (skipped): {dropped}")

    X_train_fs = X_train.drop(columns=existing)
    X_test_fs = X_test.drop(columns=existing)

    print(f"Dropped {len(existing)} features: {existing}")
    print(f"Feature-selected shape: {X_train_fs.shape}")
    return X_train_fs, X_test_fs


if __name__ == "__main__":
    print("Columns to drop after Chi-Square feature selection:")
    for col in COLS_TO_DROP:
        print(f"  - {col}")
