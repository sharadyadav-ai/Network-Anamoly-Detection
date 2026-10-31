"""
Train Network Anomaly Detection models on UNSW-NB15 training CSV.

Run:
    python train_model.py --data "UNSW_NB15_training-set(in).csv"

The script trains:
1. Random Forest: supervised normal-vs-attack classifier.
2. Isolation Forest: unsupervised anomaly detector fitted only on normal training rows.
"""
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


def make_preprocessor(X):
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipe, numeric_cols),
        ("categorical", categorical_pipe, categorical_cols)
    ])


def evaluate(name, y_true, y_pred):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["Normal", "Attack"],
            output_dict=True, zero_division=0
        )
    }
    print(f"\n===== {name} =====")
    print("Accuracy :", round(result["accuracy"], 4))
    print("Precision:", round(result["precision"], 4))
    print("Recall   :", round(result["recall"], 4))
    print("F1-score :", round(result["f1_score"], 4))
    print("Confusion matrix:\n", np.array(result["confusion_matrix"]))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="UNSW_NB15_training-set(in).csv",
                        help="Path to the UNSW-NB15 training CSV")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Could not find {args.data}. Put the CSV beside train_model.py "
            "or pass its path using --data."
        )

    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(args.data, low_memory=False)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    # Remove rows without a target label.
    df = df.dropna(subset=["label"]).copy()
    y = pd.to_numeric(df["label"], errors="coerce")
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(int)

    # Exclude ID and target/leakage columns from model input.
    drop_cols = [c for c in ["id", "label", "attack_cat"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    feature_columns = X.columns.tolist()

    # Replace infinite values; imputers will handle missing values.
    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Rows: {len(df):,}")
    print(f"Training rows: {len(X_train):,}; test rows: {len(X_test):,}")
    print("Training Random Forest...")

    rf_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(X_train)),
        ("model", RandomForestClassifier(
            n_estimators=250,
            max_depth=24,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_predictions = rf_pipeline.predict(X_test)
    rf_results = evaluate("Random Forest", y_test, rf_predictions)
    joblib.dump(rf_pipeline, "models/random_forest_pipeline.joblib")

    print("\nTraining Isolation Forest on NORMAL training rows only...")
    normal_train = X_train.loc[y_train == 0]
    if_pipeline = Pipeline([
        ("preprocessor", make_preprocessor(normal_train)),
        ("model", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
            n_jobs=-1
        ))
    ])
    if_pipeline.fit(normal_train)

    # Isolation Forest returns 1 for inliers and -1 for outliers.
    if_raw = if_pipeline.predict(X_test)
    if_predictions = np.where(if_raw == -1, 1, 0)
    if_results = evaluate("Isolation Forest", y_test, if_predictions)
    joblib.dump(if_pipeline, "models/isolation_forest_pipeline.joblib")

    with open("models/feature_columns.json", "w") as f:
        json.dump(feature_columns, f, indent=2)

    summary = {
        "dataset": os.path.basename(args.data),
        "rows_used": int(len(df)),
        "features_used": feature_columns,
        "random_forest": rf_results,
        "isolation_forest": if_results
    }
    with open("reports/evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nTraining complete.")
    print("Models saved in the models/ folder.")
    print("Evaluation saved to reports/evaluation.json")


if __name__ == "__main__":
    main()
