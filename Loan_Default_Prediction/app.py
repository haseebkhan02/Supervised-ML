import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load Models, Scaler, and Encoders (with error handling)
try:
    log_reg = joblib.load("Loan_Default_Prediction/logistic_regression_model.pkl")
    rf = joblib.load("Loan_Default_Prediction/random_forest_model.pkl")
    scaler = joblib.load("Loan_Default_Prediction/scaler.pkl")
    le_dict = joblib.load("Loan_Default_Prediction/label_encoders.pkl")
except Exception as e:
    st.error(f"Error loading model or preprocessors: {e}")
    st.stop()
# Streamlit Title and Description
st.title("Loan Default Prediction")
st.write("""
This application predicts whether a loan applicant will default or successfully repay their loan. 
Adjust the sidebar to examine model performance and select your preferred model for prediction.
""")

with st.expander("ℹ️ What does 'Default' mean?"):
    st.write("A 'Default' means the applicant fails to fully repay the loan based on the given information.")

# Sidebar - Model Selection & Performance
st.sidebar.header("Select Model for Prediction")
model_choice = st.sidebar.radio("Choose the model", ["Random Forest", "Logistic Regression"])

st.sidebar.header("Model Performance Metrics")
with st.sidebar.expander("About These Metrics"):
    st.write("""
    - **Accuracy:** Overall correct predictions<br>
    - **Precision:** Correct 'Default' predictions out of all predicted 'Default'<br>
    - **Recall:** Correct 'Default' predictions out of all actual 'Default'<br>
    - **F1-score:** Harmonic mean of Precision & Recall<br>
    - **ROC-AUC:** Model's ability to distinguish defaults from non-defaults.
    """, unsafe_allow_html=True)

if model_choice == "Random Forest":
    st.sidebar.markdown("""
<small>
Model: Random Forest  <br>
Accuracy: 0.7765  <br>
Precision: Default → 0.2605, Non-Default → 0.9256  <br>
Recall: Default → 0.5026, Non-Default → 0.8125  <br>
F1-score: Default → 0.3431, Non-Default → 0.8654  <br>
ROC-AUC: 0.7371  <br>
</small>
""", unsafe_allow_html=True)
elif model_choice == "Logistic Regression":
    st.sidebar.markdown("""
<small>
Model: Logistic Regression  <br>
Accuracy: 0.6795  <br>
Precision: Default → 0.2209, Non-Default → 0.9443  <br>
Recall: Default → 0.6962, Non-Default → 0.6773  <br>
F1-score: Default → 0.3354, Non-Default → 0.7889  <br>
ROC-AUC: 0.7496  <br>
</small>
""", unsafe_allow_html=True)

# User Input Function (with tooltips/explanations)
def user_input_features():
    st.markdown("### Fill Applicant Information")
    # Numeric Inputs
    Age = st.number_input("Age", 21, 100, 30, help="Applicant's age in years.")
    Income = st.number_input("Income", 10000, 500000, 50000, step=1000, help="Annual income (INR).")
    LoanAmount = st.number_input("Loan Amount", 500, 100000, 10000, step=500, help="Requested loan amount (INR).")
    CreditScore = st.number_input("Credit Score", 300, 850, 650, help="Typical range: 300 (poor) to 850 (excellent).")
    MonthsEmployed = st.number_input("Months Employed", 0, 600, 36, step=1, help="How many months in current job.")
    NumCreditLines = st.number_input("Number of Credit Lines", 0, 20, 3, step=1, help="Total open credit lines.")
    InterestRate = st.number_input("Interest Rate (%)", 1.0, 25.0, 10.0, step=0.1, help="Proposed interest rate (percent).")
    LoanTerm = st.number_input("Loan Term (Months)", 12, 60, 36, step=1, help="Duration of loan in months.")
    DTIRatio = st.number_input("Debt-to-Income Ratio", 0.0, 1.0, 0.3, step=0.01, help="Debt divided by income (0.3 means 30%).")
    # Categorical Inputs - use known classes from training LabelEncoders
    Education = st.selectbox("Education", le_dict["Education"].classes_, help="Applicant's highest education level.")
    EmploymentType = st.selectbox("Employment Type", le_dict["EmploymentType"].classes_, help="Employment status.")
    HasCoSigner = st.selectbox("Has Co-signer?", [0, 1], format_func=lambda x: "Yes" if x else "No", help="Is there a co-signer or guarantor?")
    HasDependents = st.number_input("Number of Dependents", 0, 10, 0, step=1, format="%d", help="How many dependents (children/others) does the applicant have?")
    HasMortgage = st.selectbox("Has Mortgage?", [0, 1], format_func=lambda x: "Yes" if x else "No", help="Active mortgage on a property?")

    # Create dataframe
    data = {
        "Age": Age,
        "Income": Income,
        "LoanAmount": LoanAmount,
        "CreditScore": CreditScore,
        "MonthsEmployed": MonthsEmployed,
        "NumCreditLines": NumCreditLines,
        "InterestRate": InterestRate,
        "LoanTerm": LoanTerm,
        "DTIRatio": DTIRatio,
        "EmploymentType": EmploymentType,
        "Education": Education,
        "HasCoSigner": int(HasCoSigner),
        "HasDependents": int(HasDependents),
        "HasMortgage": int(HasMortgage)
    }
    df_input = pd.DataFrame(data, index=[0])

    # Encode categorical columns using saved label encoders
    for col in ["Education", "EmploymentType"]:
        le = le_dict[col]
        df_input[col] = le.transform(df_input[col].astype(str))

    return df_input

# Get user input
input_df = user_input_features()

# Reorder columns to match training
feature_order = [
    "Age", "Income", "LoanAmount", "CreditScore", "MonthsEmployed", "NumCreditLines",
    "InterestRate", "LoanTerm", "DTIRatio", "EmploymentType", "Education",
    "HasCoSigner", "HasDependents", "HasMortgage"
]

missing_cols = set(feature_order) - set(input_df.columns)
if missing_cols:
    st.error(f"Missing required input columns: {missing_cols}")
    st.stop()
input_df = input_df[feature_order]

# Scale all input features
input_df = pd.DataFrame(
    scaler.transform(input_df),
    columns=input_df.columns
)

# Show entered values before prediction
with st.expander("See summary of your inputs"):
    st.dataframe(input_df)

# Prediction Button & Display
if st.button("Predict"):
    st.subheader("Prediction Result")
    if model_choice == "Logistic Regression":
        pred = log_reg.predict(input_df)[0]
        proba = log_reg.predict_proba(input_df)[0, 1]
    elif model_choice == "Random Forest":
        pred = rf.predict(input_df)[0]
        proba = rf.predict_proba(input_df)[0, 1]
    result_label = "Default" if pred == 1 else "No Default"
    st.write(f"### Model Used: {model_choice}")
    st.success(f"Prediction: {result_label}")
    st.write("Probability of Default: {:.2f}%".format(proba * 100))
    st.progress(int(proba * 100))  # Progress bar visualization

    if proba > 0.7:
        st.warning("High likelihood of default—use caution for loan approval.")
    elif proba < 0.3:
        st.info("Low chance of default—applicant appears reliable.")

