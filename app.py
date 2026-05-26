import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("indian_gold_model.pkl")

# Title
st.title("Indian Gold Price Prediction")

st.write("Enter the details below:")

# User Inputs
usd_inr = st.number_input("USD to INR")
silver_price = st.number_input("Silver Price")
oil_price = st.number_input("Oil Price")
inflation = st.number_input("Inflation")

# Prediction Button
if st.button("Predict Gold Price"):

    # Input Data
    data = np.array([[usd_inr, silver_price, oil_price, inflation]])

    # Prediction
    prediction = model.predict(data)

    # Output
    st.success(f"Predicted Gold Price: ₹ {prediction[0]:,.2f} per 10 gram")