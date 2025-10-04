import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from io import BytesIO

st.set_page_config(page_title="Fraud Detection demo", layout="wide")

@st.cache_data
def load_artifacts():
    preprocessor = joblib.load("Fraud_Detection/artifacts/preprocessor.joblib")
    model = joblib.load("Fraud_Detection//artifacts/model.joblib")
    with open("Fraud_Detection/artifacts/metrics.json", "r") as f:
        metrics = json.load(f)
    return preprocessor, model, metrics

preprocessor, model, metrics = load_artifacts()

st.title("🚨 Fraud Detection (Streamlit deployment)")
st.write("This demo loads a trained model and preprocessor. Upload transactions CSV or enter a single transaction.")

# show model metrics
st.subheader("Model test metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC-AUC", f"{metrics.get('roc_auc', None):.4f}")
col2.metric("F1", f"{metrics.get('f1', None):.4f}")
col3.metric("Precision", f"{metrics.get('precision', None):.4f}")
col4.metric("Recall", f"{metrics.get('recall', None):.4f}")

st.markdown("---")

# SECTION 1: Batch CSV upload
st.header("Batch prediction: Upload CSV")
st.write("CSV must contain the same feature columns the model expects (V1..V28, Amount, Time).")
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])
if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Data preview:", data.head())
        # select expected columns
        expected_cols = [c for c in data.columns if c in [f"V{i}" for i in range(1,29)] + ["Amount","Time"]]
        if len(expected_cols) < 2:
            st.error("Uploaded file doesn't contain expected columns (V1..V28, Amount, Time).")
        else:
            X = data[expected_cols].copy()
            X_proc = preprocessor.transform(X)
            preds = model.predict(X_proc)
            probs = model.predict_proba(X_proc)[:,1]
            result = data.copy()
            result["predicted_class"] = preds
            result["fraud_probability"] = probs
            st.write("Prediction sample:", result.head())

            # Download results
            to_download = result.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", to_download, file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Error processing file: {e}")

st.markdown("---")

# SECTION 2: Single transaction form
st.header("Single Transaction Prediction")
st.write("Fill the fields below to predict whether the transaction is fraudulent.")

cols = st.columns(4)
# build default values (zeros) for V1..V28
v_vals = {}
for i in range(1,29):
    v = f"V{i}"
    v_vals[v] = cols[(i-1) % 4].number_input(v, value=0.0, format="%.6f", key=f"v_{i}")

amount = st.number_input("Amount", value=1.0, step=0.5, format="%.2f")
time_val = st.number_input("Time (seconds since first txn)", value=0.0, format="%.2f")

# prepare input DF
single_df = pd.DataFrame([{**v_vals, "Amount": amount, "Time": time_val}])
if st.button("Predict single transaction"):
    try:
        Xs = single_df[[*v_vals.keys(), "Amount", "Time"]]
        Xs_proc = preprocessor.transform(Xs)
        pred = model.predict(Xs_proc)[0]
        prob = model.predict_proba(Xs_proc)[0,1]
        st.write("Prediction:", "Fraud" if pred==1 else "Legit")
        st.write(f"Fraud probability: {prob:.4f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.write("Notes:")
st.write("- This app expects the same preprocessing used during training.")
st.write("- For production: add authentication, logging, monitoring, and input validation.")
