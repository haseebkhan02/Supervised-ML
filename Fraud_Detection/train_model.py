# train_model.py
"""
Train a Fraud Detection model and save artifacts for deployment.
- Input: creditcard.csv (Kaggle dataset)
- Output: artifacts/model.joblib, artifacts/preprocessor.joblib, artifacts/metrics.json
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from imblearn.over_sampling import SMOTE

# -------------------------
# Config
# -------------------------
DATA_PATH = "../Data/creditcard.csv"      # put your CSV in the same folder or change this path
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -------------------------
# Load data
# -------------------------
print("Loading dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
# Expect columns: Time, V1..V28, Amount, Class
if "Class" not in df.columns:
    raise ValueError("Expected `Class` column in dataset (0 = legit, 1 = fraud).")

# -------------------------
# Feature / target
# -------------------------
FEATURE_COLS = [c for c in df.columns if c not in ["Class"]]
TARGET_COL = "Class"

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# -------------------------
# Preprocessing pipeline
# -------------------------
# For this dataset, V1..V28 are already scaled (PCA), scale Amount and Time
num_features = [c for c in X.columns if c.startswith("V")] + ["Amount", "Time"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features)
    ],
    remainder="drop",  # drop other unexpected columns
)

# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Preprocess train set
print("Fitting preprocessor..")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

# -------------------------
# Handle imbalance (SMOTE) on training set
# -------------------------
print("Applying SMOTE to training set..")
sm = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = sm.fit_resample(X_train_proc, y_train)

print("Train balanced shape:", X_train_bal.shape, y_train_bal.shape)
print("Class distribution after SMOTE:", np.bincount(y_train_bal))

# -------------------------
# Train model
# -------------------------
print("Training RandomForestClassifier..")
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
clf.fit(X_train_bal, y_train_bal)

# -------------------------
# Evaluate
# -------------------------
y_pred = clf.predict(X_test_proc)
y_proba = clf.predict_proba(X_test_proc)[:, 1]

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "test_size": int(len(y_test)),
}

print("Test metrics:", metrics)

# -------------------------
# Save artifacts
# -------------------------
joblib.dump(preprocessor, ARTIFACT_DIR / "preprocessor.joblib")
joblib.dump(clf, ARTIFACT_DIR / "model.joblib")

with open(ARTIFACT_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved artifacts to", ARTIFACT_DIR)