import streamlit as st
import pandas as pd
import joblib
import lightgbm as lgb
import base64
import os

# ✅ MUST BE FIRST
st.set_page_config(page_title="American Express Default Predictor", layout="centered")

# 🎯 Title
st.title("💳 American Express Default Prediction")
st.write("🔍 This app predicts whether a customer is likely to default based on their profile.")

# 📁 Load trained model
@st.cache_resource
def load_model():
    model = joblib.load("lgbm_default_model.pkl")
    return model

model = load_model()

# 📁 Sample dataset options
sample_files = {
    "🟢 Low Risk Customer": "test_customer_low.csv",
    "🔴 High Risk Customer": "test_customer_high.csv",
    "🟡 Average Risk Customer": "test_customer_avg.csv",
    "⚙️ Noisy/Random Features": "test_customer_noise.csv",
    "🧪 Edge Case (Extreme Values)": "test_customer_edge.csv"
}

# ✅ Dataset selection
selected_option = st.selectbox("📂 Select a sample test customer:", list(sample_files.keys()))

# 📥 Load selected CSV
df = pd.read_csv(sample_files[selected_option])

# 👁️ Show inputs
st.subheader("🔎 Input Features")
st.write(df)

# ✅ Predict
X_input = df.drop(columns=["customer_ID"])
prediction_proba = model.predict(X_input)[0]
prediction_percent = round(prediction_proba * 100, 2)

# 🎯 Show result
st.subheader("📈 Prediction")
st.metric(label="📊 Probability of Default", value=f"{prediction_percent}%")

# ⬇️ Allow download
def get_table_download_link(df, filename="selected_data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">📥 Download this dataset</a>'
    return href

st.markdown(get_table_download_link(df, filename=sample_files[selected_option].split("/")[-1]), unsafe_allow_html=True)

# ℹ️ Footer
st.markdown("---")
st.markdown("Created for educational purposes. Model: LightGBM | Competition: American Express Default Prediction")
