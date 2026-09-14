import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load and prepare dataset
# -----------------------------
df = pd.read_csv("loan.csv")

# Fill missing values
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(exclude="object").columns:
    df[col] = df[col].fillna(df[col].median())

# Convert Dependents
df["Dependents"] = df["Dependents"].replace("3+", "3")

# Remove Loan_ID
df = df.drop("Loan_ID", axis=1)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Features and target
X = df.drop("Loan_Status_Y", axis=1)
y = df["Loan_Status_Y"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit Application
# -----------------------------
st.title("🏦 Loan Approval Prediction")
st.write("Enter applicant information to predict loan approval.")

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

applicant_income = st.number_input(
    "Applicant Income", min_value=0.0, value=5000.0
)

coapplicant_income = st.number_input(
    "Coapplicant Income", min_value=0.0, value=2000.0
)

loan_amount = st.number_input(
    "Loan Amount", min_value=0.0, value=150.0
)

loan_term = st.number_input(
    "Loan Term", min_value=0, value=360
)

credit_history = st.selectbox(
    "Credit History", [1, 0]
)

property_area = st.selectbox(
    "Property Area", ["Urban", "Semiurban", "Rural"]
)

if st.button("Predict Loan Approval"):

    input_data = pd.DataFrame([{
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Dependents_1": 1 if dependents == "1" else 0,
        "Dependents_2": 1 if dependents == "2" else 0,
        "Dependents_3": 1 if dependents == "3" else 0,
        "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0
    }])

    # Make columns exactly match training data
    input_data = input_data.reindex(
        columns=X.columns,
        fill_value=0
    )

    probability = model.predict_proba(input_data)[0][1]
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    st.write(
        f"**Approval Probability: {probability * 100:.2f}%**"
    )

    if prediction == 1:
        st.success("✅ LOAN APPROVED")
    else:
        st.error("❌ LOAN REJECTED")