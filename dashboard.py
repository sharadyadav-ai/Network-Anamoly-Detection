"""
Streamlit dashboard for the trained UNSW-NB15 anomaly detection models.

Run after training:
    streamlit run dashboard.py
Upload a CSV with the same feature columns as the training dataset.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Network Anomaly Detection", page_icon="🛡️",
                   layout="wide")

st.title("🛡️ Network Anomaly Detection System")
st.caption("UNSW-NB15 | Random Forest + Isolation Forest")

MODEL_DIR = "models"
required_files = [
    "random_forest_pipeline.joblib",
    "isolation_forest_pipeline.joblib",
    "feature_columns.json"
]
if not all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in required_files):
    st.error("Models not found. Run `python train_model.py` first.")
    st.stop()

@st.cache_resource
def load_models():
    rf = joblib.load(os.path.join(MODEL_DIR, "random_forest_pipeline.joblib"))
    iso = joblib.load(os.path.join(MODEL_DIR, "isolation_forest_pipeline.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_columns.json")) as f:
        columns = json.load(f)
    return rf, iso, columns

rf_model, iso_model, feature_columns = load_models()

uploaded = st.file_uploader("Upload network traffic CSV", type=["csv"])
st.info("Use a CSV containing the same network feature columns as the UNSW-NB15 training data. "
        "The `id`, `label`, and `attack_cat` columns are optional and are not used as model inputs.")

if uploaded is not None:
    data = pd.read_csv(uploaded, low_memory=False)
    actual_labels = data["label"].copy() if "label" in data.columns else None

    # Remove labels and identifiers before prediction.
    X = data.drop(columns=[c for c in ["id", "label", "attack_cat"] if c in data.columns])
    missing = [c for c in feature_columns if c not in X.columns]
    if missing:
        st.error("This file is missing required feature columns: " + ", ".join(missing[:15]))
        st.stop()

    X = X[feature_columns].replace([np.inf, -np.inf], np.nan)

    with st.spinner("Analyzing network traffic..."):
        rf_pred = rf_model.predict(X).astype(int)
        rf_prob = rf_model.predict_proba(X)[:, 1]
        iso_raw = iso_model.predict(X)
        iso_pred = (iso_raw == -1).astype(int)

    results = data.copy()
    results["RF_prediction"] = np.where(rf_pred == 1, "Attack", "Normal")
    results["RF_attack_probability"] = np.round(rf_prob, 4)
    results["IF_prediction"] = np.where(iso_pred == 1, "Anomaly", "Normal")

    total = len(results)
    attacks = int(rf_pred.sum())
    anomalies = int(iso_pred.sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analyzed", f"{total:,}")
    c2.metric("RF attack predictions", f"{attacks:,}")
    c3.metric("IF anomalies", f"{anomalies:,}")
    c4.metric("RF predicted normal", f"{total-attacks:,}")

    st.subheader("Detection overview")
    chart_data = pd.DataFrame({
        "Detection": ["RF: Attack", "RF: Normal", "IF: Anomaly", "IF: Normal"],
        "Count": [attacks, total-attacks, anomalies, total-anomalies]
    })
    st.plotly_chart(px.bar(chart_data, x="Detection", y="Count",
                           title="Model predictions"), use_container_width=True)

    st.subheader("Traffic prediction results")
    filter_choice = st.selectbox("Filter results", ["All", "RF Attack", "IF Anomaly"])
    shown = results
    if filter_choice == "RF Attack":
        shown = results[results["RF_prediction"] == "Attack"]
    elif filter_choice == "IF Anomaly":
        shown = results[results["IF_prediction"] == "Anomaly"]
    st.dataframe(shown.head(1000), use_container_width=True)

    if actual_labels is not None:
        st.subheader("Evaluation against supplied labels")
        actual = pd.to_numeric(actual_labels, errors="coerce")
        valid = actual.notna()
        if valid.any():
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            y_true = actual[valid].astype(int)
            y_hat = pd.Series(rf_pred, index=data.index)[valid]
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy", f"{accuracy_score(y_true, y_hat):.3f}")
            e2.metric("Precision", f"{precision_score(y_true, y_hat, zero_division=0):.3f}")
            e3.metric("Recall", f"{recall_score(y_true, y_hat, zero_division=0):.3f}")
            e4.metric("F1-score", f"{f1_score(y_true, y_hat, zero_division=0):.3f}")
        else:
            st.warning("No valid numeric labels found.")

    st.download_button(
        "Download detection results",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="network_anomaly_results.csv",
        mime="text/csv"
    )
else:
    st.write("Upload a compatible CSV to view model predictions.")
    st.subheader("What this dashboard does")
    st.markdown(
        "- Random Forest predicts **Normal** or **Attack** using labeled training data.\n"
        "- Isolation Forest flags traffic that differs from the normal training profile.\n"
        "- Both outputs are shown separately because an anomaly is not automatically an attack."
    )
