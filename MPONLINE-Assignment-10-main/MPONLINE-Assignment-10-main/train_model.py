"""
AI-ML Assignment - 10
Train Model Script
Topic: Heart Disease Prediction - Model Training & Serialization

NOTE: This script expects 'heart.csv' from the Kaggle dataset:
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
Download it and place it in this same folder before running.
If not found, a small synthetic dataset with the same schema is
generated so the full pipeline (training + serialization) can still be
demonstrated end-to-end.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DATA_PATH = "heart.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

# The standard Kaggle "heart-disease-dataset" (johnsmith88) columns
FEATURE_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
TARGET_COLUMN = "target"


def load_data(path=DATA_PATH):
    if os.path.exists(path):
        print(f"Loading dataset from '{path}'")
        return pd.read_csv(path)

    print(f"WARNING: '{path}' not found in this folder.")
    print("Download it from: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset")
    print("Generating a small synthetic dataset (same schema) for demonstration.\n")

    rng = np.random.default_rng(42)
    n = 800
    df = pd.DataFrame({
        "age": rng.integers(29, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 200, n),
        "chol": rng.integers(126, 564, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 2, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.uniform(0, 6.2, n), 1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 5, n),
        "thal": rng.integers(0, 4, n),
    })
    # Synthetic target loosely correlated with a few risk factors
    risk_score = (
        (df["age"] > 54).astype(int)
        + (df["chol"] > 240).astype(int)
        + (df["trestbps"] > 140).astype(int)
        + (df["exang"] == 1).astype(int)
        + rng.integers(0, 2, n)
    )
    df[TARGET_COLUMN] = (risk_score >= 3).astype(int)
    return df


def main():
    # -----------------------------------------------------------------------
    # Task 1: Data Understanding and Preprocessing (2 Marks)
    # -----------------------------------------------------------------------
    print("=" * 72)
    print("TASK 1: DATA UNDERSTANDING AND PREPROCESSING")
    print("=" * 72)

    df = load_data()

    print("\nFirst five records:")
    print(df.head())

    numerical_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    print(f"\nNumerical/clinical features: {numerical_features}")
    print(f"Target variable: '{TARGET_COLUMN}' (1 = heart disease present, 0 = absent)")

    print("\nMissing values per column:")
    print(df.isnull().sum())
    df = df.dropna().reset_index(drop=True)

    X = df[numerical_features]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples : {X_test.shape[0]}")

    # -----------------------------------------------------------------------
    # Task 2: Model Development (2 Marks)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("TASK 2: MODEL DEVELOPMENT")
    print("=" * 72)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel: Logistic Regression")
    print(f"Test Accuracy: {accuracy:.4f}")

    # Save trained model and scaler with Joblib
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(numerical_features, "feature_columns.pkl")
    print(f"\nModel saved to '{MODEL_PATH}'")
    print(f"Scaler saved to '{SCALER_PATH}'")
    print(f"Feature column order saved to 'feature_columns.pkl'")


if __name__ == "__main__":
    main()
