import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

st.title("📡 Telecom Customer Churn Prediction System")
st.write("Predict customer churn using Machine Learning")

@st.cache_data
def load_data():
    df = pd.read_csv("telecom_customer_churn.csv")
    df = pd.get_dummies(df, drop_first=True)
    return df

df = load_data()

X = df.drop("Churn_Yes", axis=1)
y = df["Churn_Yes"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_scaled, y)

st.sidebar.header("Customer Details")

input_data = {}

for col in X.columns:
    input_data[col] = st.sidebar.number_input(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))

input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"⚠️ Customer will churn (Risk: {prob*100:.2f}%)")
    else:
        st.success(f"✅ Customer will NOT churn (Confidence: {(1-prob)*100:.2f}%)")
