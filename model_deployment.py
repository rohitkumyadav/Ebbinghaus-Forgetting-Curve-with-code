import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the trained model
model = joblib.load("memory_retention_model_V.1.pkl")

# ---  title
st.title("Memory Retention Predictor")
st.write("Predict memory retention based on time,memory_strength, stress, sleep, and study method and more.")

# --- Sidebar inputs
time_days = st.slider("Time Since Learning (days)", 0.0, 30.0, 1.0, 0.5)
strength_of_memory = st.slider("Strength of Memory", 0.5, 5.0, 2.0, 0.1)

learning_method = st.selectbox("Learning Method", ['Listening', 'Watching', 'Doing'])
presentation_type = st.selectbox("Presentation Type", ['Text', 'Visual', 'Audio'])

material_complexity = st.selectbox("Material Complexity", ['Low', 'Medium', 'High'])
information_relevance = st.selectbox("Information Relevance", ['Low', 'Medium', 'High'])

sleep_quality = st.selectbox("Sleep Quality", ['Poor', 'Average', 'Good'])
stress_level = st.selectbox("Stress Level", ['Low', 'Moderate', 'High'])

# --- Predict button
if st.button("Predict Retention"):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'time_days': time_days,
        'strength_of_memory': strength_of_memory,
        'learning_method': learning_method,
        'material_complexity': material_complexity,
        'information_relevance': information_relevance,
        'presentation_type': presentation_type,
        'sleep_quality': sleep_quality,
        'stress_level': stress_level
    }])

    prediction = model.predict(input_data)[0]  # ---  Make prediction

    # --- Output
    st.success(f"Predicted Memory Retention: **{prediction:.2f}%**")


'''
made by : Rohit Kumar Yadav
'''
