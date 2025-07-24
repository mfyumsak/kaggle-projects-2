import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Load model
@st.cache_resource
def load_model():
    with open("lgbm_optuna_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

# Top 50 features (update this with actual top 50)
TOP_50_FEATURES = [
    'f87', 'f39', 'f10', 'f27', 'f94', 'f101', 'f14', 'f52', 'f48', 'f110',
    'f24', 'f34', 'f108', 'f102', 'f29', 'f46', 'f23', 'f83', 'f65', 'f15',
    'f25', 'f118', 'f97', 'f36', 'f61', 'f3', 'f53', 'f71', 'f16', 'f74',
    'f18', 'f33', 'f21', 'f38', 'f12', 'f50', 'f43', 'f37', 'f92', 'f13',
    'f35', 'f86', 'f42', 'f17', 'f70', 'f100', 'f19', 'f31', 'f85', 'f63'
]

# Sample options
sample_files = {
    "Example 1": "sample1.csv",
    "Example 2": "sample2.csv",
    "Example 3": "sample3.csv"
}

st.title("Tabular Playground September 2021 – Claim Prediction App")

st.markdown("""
Upload your test file or choose from sample datasets to predict the probability of a claim (`claim = 1`).
""")

# User can choose from samples or upload own file
option = st.radio("Choose input method:", ["Use sample data", "Upload your own CSV"])

df = None

if option == "Use sample data":
    sample_choice = st.selectbox("Select a sample dataset", list(sample_files.keys()))
    file_path = sample_files[sample_choice]

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        st.error(f"Sample file {file_path} not found. Please check your setup.")
elif option == "Upload your own CSV":
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)

if df is not None:
    if 'id' not in df.columns:
        st.error("❌ 'id' column is missing.")
    elif not all(feat in df.columns for feat in TOP_50_FEATURES):
        st.error("❌ Some required features are missing.")
    else:
        model = load_model()
        X = df[TOP_50_FEATURES]
        preds = model.predict_proba(X)[:, 1]

        submission = pd.DataFrame({
            'id': df['id'],
            'claim': preds
        })

        st.success("✅ Prediction completed. Preview:")
        st.dataframe(submission.head())

        csv = submission.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Submission CSV",
            data=csv,
            file_name="submission.csv",
            mime='text/csv'
        )
