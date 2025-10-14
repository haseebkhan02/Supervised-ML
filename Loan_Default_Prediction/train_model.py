import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# 1. Load Dataset
# Dataset link https://www.kaggle.com/datasets/nikhil1e9/loan-default
df = pd.read_csv("../Data/Loan_default.csv")  # Replace with your dataset path
print("Loaded Dataset!")

# Drop LoanID if exists
if "LoanID" in df.columns:
    df = df.drop("LoanID", axis=1)

# 2. Feature Selection for Deployment
selected_features = [
    "Age", "Income", "LoanAmount", "CreditScore", 
    "MonthsEmployed", "NumCreditLines", "InterestRate", 
    "LoanTerm", "DTIRatio", "EmploymentType", "Education",
    "HasCoSigner", "HasDependents", "HasMortgage"
]

X = df[selected_features].copy()
y = df["Default"]

# 3. Encode categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le
print("Encoded Categorical Columns")

# 4. Scale numeric columns safely
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
X[num_cols] = X[num_cols].astype(float)
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
print("Scaled Numeric Columns")

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 6. Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("SMOTE applied. New training shape:", X_train_res.shape)

# 7. Train Models with Class Weight
# Logistic Regression
print("Training Logistic Regression")
log_reg = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'  # Helps with minority class
)
log_reg.fit(X_train_res, y_train_res)

# Random Forest
print("Training Random Forest")
rf = RandomForestClassifier(
    n_estimators=300,          # increased estimators
    max_depth=12,              # limit depth to reduce overfitting
    min_samples_leaf=10,       # prevent small leaves
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'    # handle imbalance
)
rf.fit(X_train_res, y_train_res)

# 8. Evaluate Models
models = {"Logistic Regression": log_reg, "Random Forest": rf}
for name, model in models.items():
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\nModel: {name}")
    print(classification_report(y_test, y_pred, digits=4))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 9. Save Models, Scaler, Encoders
joblib.dump(log_reg, "logistic_regression_model.pkl")
joblib.dump(rf, "random_forest_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
print("\nModels, scaler, and encoders saved successfully!")
