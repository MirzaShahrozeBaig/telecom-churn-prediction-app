import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Telecom Churn Prediction")

st.title("📡 Telecom Customer Churn Prediction System")
st.markdown("Predict customer churn using Machine Learning")

st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])

tenure = st.sidebar.slider("Tenure", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 20, 150, 70)
total = st.sidebar.slider("Total Charges", 0, 9000, 1500)

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

def encode():
    return [
        1 if gender == "Male" else 0,
        1 if senior == "Yes" else 0,
        1 if partner == "Yes" else 0,
        1 if dependents == "Yes" else 0,
        tenure,
        monthly,
        total,
        1 if internet == "Fiber optic" else 0,
        1 if contract == "Two year" else 0,
        1 if payment == "Electronic check" else 0
    ]

if st.button("Predict"):
    data = np.array(encode()).reshape(1, -1)
    data = scaler.transform(data)

    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]

    if pred == 1:
        st.error(f"❌ Customer WILL CHURN (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Customer WILL STAY (Probability: {(1-prob):.2%})")
