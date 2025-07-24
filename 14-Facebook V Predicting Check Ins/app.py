import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Info box – App purpose
st.info("""
This application predicts the most likely 3 place IDs (Top-3) based on the user's input: location and time-related features.
It is useful for WiFi/GPS-based systems with low-resolution data to infer where a user might be located.
""")

# Load model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("model_45_41.pkl")
    le = joblib.load("le_45_41.pkl")
    return model, le

model, le = load_model()

st.title("📍 Predict Place ID (Grid: 45_41)")
st.markdown("Enter location features to predict the most likely place_id (Top-3)")


# Input features
x = st.number_input("X Coordinate", min_value=0.0, max_value=10.0, value=1.5)
y = st.number_input("Y Coordinate", min_value=0.0, max_value=10.0, value=3.5)
accuracy = st.number_input("Accuracy", min_value=1, max_value=1000, value=65)
hour = st.slider("Hour", 0, 23, 12)
weekday = st.slider("Weekday (0=Mon)", 0, 6, 3)

# Predict button
if st.button("Predict"):
    features = np.array([[x, y, accuracy, hour, weekday]])
    proba = model.predict_proba(features)
    top_3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]
    top_3_place_ids = le.inverse_transform(top_3[0])
    
    st.success(f"🏁 Top 3 Predicted Place IDs: {', '.join(map(str, top_3_place_ids))}")
