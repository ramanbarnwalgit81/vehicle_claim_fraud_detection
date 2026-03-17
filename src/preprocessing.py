"""
preprocessing.py
----------------
Data cleaning and encoding utilities for Vehicle Insurance Fraud Detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce


def load_data(filepath: str) -> pd.DataFrame:
    """Load the fraud_oracle dataset."""
    df = pd.read_csv(filepath)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all preprocessing steps:
    1. Drop PolicyNumber (row ID only)
    2. Drop PolicyType (redundant — concat of VehicleCategory + BasePolicy)
    3. Remove records where MonthClaimed == '0'
    4. Impute Age == 0 with 16.5
    """
    df = df.copy()

    # Drop irrelevant columns
    df = df.drop(columns=['PolicyNumber', 'PolicyType'], errors='ignore')

    # Remove invalid records
    df = df[~(df['MonthClaimed'] == '0')]

    # Impute Age = 0 -> 16.5 (cross-referenced with AgeOfPolicyHolder 16-17)
    df['Age'] = df['Age'].replace({0: 16.5})

    print(f"After cleaning: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def encode_features(df: pd.DataFrame):
    """
    Encode categorical features:
    - Ordinal encoding for binary/ordered features
    - One-Hot encoding for nominal features
    Returns encoded DataFrame.
    """
    df = df.copy()

    # Ordinal / binary mappings
    col_ordering = [
        {'col': 'AccidentArea',      'mapping': {'Urban': 1, 'Rural': 0}},
        {'col': 'Sex',               'mapping': {'Female': 1, 'Male': 0}},
        {'col': 'Fault',             'mapping': {'Policy Holder': 1, 'Third Party': 0}},
        {'col': 'PoliceReportFiled', 'mapping': {'Yes': 1, 'No': 0}},
        {'col': 'WitnessPresent',    'mapping': {'Yes': 1, 'No': 0}},
        {'col': 'AgentType',         'mapping': {'External': 1, 'Internal': 0}},
    ]

    ord_encoder = ce.OrdinalEncoder(mapping=col_ordering, return_df=True)
    df_encoded = ord_encoder.fit_transform(df)

    # One-Hot encoding for nominal features
    ohe = ce.OneHotEncoder(
        cols=['Make', 'MaritalStatus', 'VehicleCategory', 'BasePolicy'],
        use_cat_names=True,
        return_df=True
    )
    df_encoded = ohe.fit_transform(df_encoded)

    print(f"After encoding: {df_encoded.shape[1]} columns")
    return df_encoded, ord_encoder, ohe


def get_features_target(df: pd.DataFrame):
    """Split into features X and target y."""
    X = df.drop(columns='FraudFound_P')
    y = df['FraudFound_P']
    return X, y


if __name__ == "__main__":
    df = load_data("../data/fraud_oracle.csv")
    df = clean_data(df)
    df_encoded, _, _ = encode_features(df)
    X, y = get_features_target(df_encoded)
    print(f"\nFinal X shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
