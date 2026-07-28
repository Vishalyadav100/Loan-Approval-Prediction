# ==========================================================
# Loan Approval Prediction Dashboard
# preprocessing.py (Part A)
# ==========================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

# ==========================================================
# Dataset Summary
# ==========================================================

def dataset_summary(df):
    """
    Returns basic dataset information
    """

    return {

        "rows": df.shape[0],

        "columns": df.shape[1],

        "features": df.shape[1] - 1,

        "duplicates": df.duplicated().sum(),

        "missing": int(df.isnull().sum().sum())

    }


# ==========================================================
# Missing Value Information
# ==========================================================

def missing_values(df):

    missing = pd.DataFrame({

        "Column": df.columns,

        "Missing Values": df.isnull().sum(),

        "Percentage": round(
            (df.isnull().sum()/len(df))*100,
            2
        )

    })

    return missing


# ==========================================================
# Fill Missing Values
# ==========================================================

def fill_missing_values(df):

    df = df.copy()

    numeric_columns = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    # Fill Numerical Columns
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].median())

    # Fill Categorical Columns
    for col in categorical_columns:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna("Unknown")

    return df

# ==========================================================
# Remove Duplicate Rows
# ==========================================================

def remove_duplicates(df):

    return df.drop_duplicates()


# ==========================================================
# Remove Unwanted Columns
# ==========================================================

def drop_columns(df):

    df = df.copy()

    if "Loan_ID" in df.columns:

        df.drop(
            columns=["Loan_ID"],
            inplace=True
        )

    return df

# ==========================================================
# Label Encoding
# ==========================================================

def encode_data(df):

    df = df.copy()

    encoder = LabelEncoder()

    categorical_columns = df.select_dtypes(
        include=["object"]
    ).columns

    for col in categorical_columns:

        df[col] = encoder.fit_transform(df[col])

    return df


# ==========================================================
# Split Features & Target
# ==========================================================

def split_features_target(df, target):

    X = df.drop(columns=[target])

    y = df[target]

    return X, y


# ==========================================================
# Train Test Split
# ==========================================================

def split_data(
        X,
        y,
        test_size,
        random_state
):

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=test_size,

        random_state=random_state,

        stratify=y

    )

    return X_train, X_test, y_train, y_test


# ==========================================================
# Feature Scaling
# ==========================================================

def scale_data(
        X_train,
        X_test
):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_test = scaler.transform(X_test)

    return X_train, X_test


# ==========================================================
# Complete Preprocessing Pipeline
# ==========================================================

def preprocess_pipeline(
        df,
        target,
        test_size,
        random_state
):

    # Copy Dataset
    processed_df = df.copy()

    # Missing Values
    processed_df = fill_missing_values(
        processed_df
    )

    

    print("=" * 50)
    print("Missing values after fill_missing_values():")
    print(processed_df.isnull().sum())
    print("Total Missing Values:", processed_df.isnull().sum().sum())
    print("=" * 50)

    # Duplicate Rows
    processed_df = remove_duplicates(
        processed_df
    )

    # Drop Loan_ID
    processed_df = drop_columns(
        processed_df
    )

    # Encoding
    processed_df = encode_data(
        processed_df
    )

    # Feature Target Split
    X, y = split_features_target(
        processed_df,
        target
    )

    # Train Test Split
    X_train, X_test, y_train, y_test = split_data(

        X,

        y,

        test_size,

        random_state

    )

    # Scaling
    X_train, X_test = scale_data(

        X_train,

        X_test

    )

    return (

        processed_df,

        X_train,

        X_test,

        y_train,

        y_test,

        X.columns

    )