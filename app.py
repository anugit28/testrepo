import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model
with open("best_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the scaler (if needed)
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Streamlit App Title
st.title("🩺 Disease Risk Prediction App")
st.write("Enter your health parameters to predict the risk of disease.")

# Input fields for user data
Pregnancies = st.number_input("Pregnancies", min_value=1, max_value=120, value=30)
glucose = st.number_input("glucose", min_value=50, max_value=300, value=100)
gender = st.selectbox("gender", ["Male", "Female"])
BloodPressure = st.number_input("Blood Pressure", min_value=50, max_value=200, value=120)
SkinThickness = st.number_input("SkinThickness", max_value=400, value=200)
Insulin = st.number_input("Insulin", min_value=50, max_value=300, value=100)
BMI = st.number_input("BMI ", min_value=10.0, max_value=50.0, value=25.0)
Age = st.number_input("Age", min_value=1, max_value=120, value=30)

family_history = st.selectbox("Family History of Disease", ["No", "Yes"])

# Convert categorical inputs to numerical values
gender = 1 if gender == "Male" else 0


family_history = 1 if family_history == "Yes" else 0

# Convert inputs to a NumPy array
input_data = np.array([[Pregnancies,glucose,BloodPressure,SkinThickness,Insulin,BMI,Age,family_history]])

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Predict when the user clicks the button
if st.button("🔍 Predict"):
    prediction = model.predict(input_data_scaled)
    
    if prediction[0] == 1:
        st.error("⚠️ High risk of disease detected. Please consult a doctor.")
    else:
        st.success("✅ No significant disease risk detected. Keep maintaining a healthy lifestyle!")


