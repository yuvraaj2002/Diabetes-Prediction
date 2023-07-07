import streamlit as st
import pandas as pd
from src.components.predict_pipe import Predict_Pipeline


# Create a title and a header
st.title("Diabetes Prediction Model")

# Show the user input
st.sidebar.write(
    "This application based on your input data will help predict whether a person has diabetes or not."
)


# Function to get user input
def get_user_input():
    pregnancies = st.sidebar.slider("Pregnancies", 0, 17, 3)
    glucose = st.sidebar.slider("Glucose", 0, 200, 100)
    blood_pressure = st.sidebar.slider("Blood Pressure", 0, 122, 70)
    skin_thickness = st.sidebar.slider("Skin Thickness", 0, 99, 20)
    insulin = st.sidebar.slider("Insulin", 0, 846, 79)
    bmi = st.sidebar.slider("BMI", 0.0, 67.1, 31.9)
    diabetes_pedigree = st.sidebar.slider(
        "Diabetes Pedigree Function", 0.078, 2.42, 0.3725
    )
    age = st.sidebar.slider("Age", 21, 81, 29)

    # Create a dictionary from the user input
    user_data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }

    return user_data


# Get user input
user_input_dict = get_user_input()
prediction_obj = Predict_Pipeline()
user_input_df = prediction_obj.build_dataframe(user_input_dict)

# Show the user input
st.subheader("User Input:")
st.write(user_input_df)

# Add a "Predict" button
if st.button("Predict"):

    # Let's get the predictions
    output = prediction_obj.predict(user_input_df)

    # Convert the output value to "No" or "Yes"
    st.subheader("Output:")
    if output == 0:
        st.write("No diabetes")
    else:
        st.write("Diabetic")
