import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import OneHotEncoder

# Set Streamlit page config at the top
st.set_page_config(page_title="NFL EPA Prediction App", layout="centered")

# Load model
@st.cache_resource
def load_model():
    with open("epa_regression_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# UI Header
st.title("🏈 NFL EPA Prediction App")
st.markdown("""
### What we offer:
- Upload your own play data
- Predict Expected Points Added (EPA)
- Evaluate player or team performance with data-driven insights
""")

# 📁 File uploader
st.subheader("📂 Upload Your CSV File")
uploaded_file = st.file_uploader("Upload a CSV file with the same format as training data", type=["csv"])

# Sample CSV Downloads
st.markdown("### 📌 Sample CSV Files")
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("Download Sample 1", open("sample_plays_fast.csv", "rb"), "sample_plays_fast.csv")
with col2:
    st.download_button("Download Sample 2", open("sample_plays_balanced.csv", "rb"), "sample_plays_medium.csv")
with col3:
    st.download_button("Download Sample 3", open("sample_plays_directional.csv", "rb"), "sample_plays_slow.csv")

# Main prediction logic
if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    # Optional: display preview
    st.subheader("📊 Preview of Uploaded Data")
    st.dataframe(input_df.head())

    # Preprocessing (assumes one-hot encoding of position)
    if 'ball_carrier_position' in input_df.columns:
        input_df = pd.get_dummies(input_df, columns=['ball_carrier_position'])

    # Align columns with training data
    expected_features = model.feature_names_in_
    for col in expected_features:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_features]

    # Predict
    predictions = model.predict(input_df)

    # Display Results
    st.subheader("📈 Predicted EPA Values")
    input_df['Predicted_EPA'] = predictions
    st.dataframe(input_df[['Predicted_EPA']].head(10))

    # Allow user to download results
    csv = input_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions as CSV", data=csv, file_name="epa_predictions.csv")
