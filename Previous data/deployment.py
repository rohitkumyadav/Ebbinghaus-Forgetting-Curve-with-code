import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load pipeline model
model = joblib.load("forgetfulness_prediction_pipeline.pkl")

st.set_page_config(page_title="🧠 Forgetfulness Predictor", layout="centered")
st.title("📉 Predict Forgetfulness Probability")

# === Inputs ===
subject = st.selectbox("Subject / Topic", ["Arithmetic", "Algebra", "Geometry", "Calculus", "Physics"])
complexity = st.slider("Complexity (1–10)", 1, 10, 5)
relevance = st.slider("Relevance (1–10)", 1, 10, 5)
study_time = st.number_input("Study Time (mins)", 1, 300, 60)
learning_method = st.selectbox("Learning Method", ["Text", "Video", "Practical"])
stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)
sleep_hours = st.number_input("Sleep Hours", 0.0, 24.0, 7.0)
recall_1hr = st.slider("Recall Score After 1 Hour", 0.0, 10.0, 9.0)
recall_1day = st.slider("Recall Score After 1 Day", 0.0, 10.0, 8.0)
recall_1week = st.slider("Recall Score After 1 Week", 0.0, 10.0, 7.0)
days_since = st.number_input("Days Since Learning", 0, 100, 10)
memory_strength = st.slider("Memory Strength", 1, 10, 5)
predicted_retention = st.slider("Predicted Retention (%)", 0.0, 100.0, 50.0)

# === Subject frequency map (used instead of subject) ===
subject_freq_map = {
    "Arithmetic": 0.25,
    "Algebra": 0.20,
    "Geometry": 0.15,
    "Calculus": 0.25,
    "Physics": 0.15
}
subject_topic_freq = subject_freq_map.get(subject, 0.05)

# === Feature Engineering ===
recall_decay_1hr_to_1day = recall_1hr - recall_1day
recall_decay_1day_to_1week = recall_1day - recall_1week
study_efficiency = recall_1hr / study_time
complexity_study_interaction = complexity * study_time
stress_to_sleep_ratio = stress_level / sleep_hours
stress_sleep_interaction = stress_level * sleep_hours
inverse_recall_1week = 10 - recall_1week
zero_forget_flag = 0  # Default

# === Manual OneHotEncoding for learning_method ===
learning_method_text = 1 if learning_method == "Text" else 0
learning_method_video = 1 if learning_method == "Video" else 0
learning_method_practical = 1 if learning_method == "Practical" else 0

# === Final input dataframe ===
input_data = pd.DataFrame([{
    "complexity_1-10": complexity,
    "relevance_1-10": relevance,
    "study_time_mins": study_time,
    "stress_level": stress_level,
    "sleep_hours": sleep_hours,
    "recall_score_1hr": recall_1hr,
    "recall_score_1day": recall_1day,
    "recall_score_1week": recall_1week,
    "days_since_learning": days_since,
    "memory_strength": memory_strength,
    "predicted_retention_%": predicted_retention,
    "recall_decay_1hr_to_1day": recall_decay_1hr_to_1day,
    "recall_decay_1day_to_1week": recall_decay_1day_to_1week,
    "study_efficiency": study_efficiency,
    "complexity_study_interaction": complexity_study_interaction,
    "stress_to_sleep_ratio": stress_to_sleep_ratio,
    "stress_sleep_interaction": stress_sleep_interaction,
    "inverse_recall_1week": inverse_recall_1week,
    "zero_forget_flag": zero_forget_flag,
    "subject_topic_freq": subject_topic_freq,
    "learning_method_Text": learning_method_text,
    "learning_method_Video": learning_method_video,
    "learning_method_Practical": learning_method_practical
}])

# === Predict ===
if st.button("🔮 Predict Forgetfulness"):
    try:
        prediction_log = model.predict(input_data)[0]
        prediction = np.expm1(prediction_log)
        st.success(f"📉 Predicted Forgetfulness Probability: **{prediction:.2f}%**")
    except Exception as e:
        st.error(f"❌ Error during prediction:\n{e}")
