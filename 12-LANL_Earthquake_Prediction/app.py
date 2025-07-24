import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Title
st.title("🌍 Earthquake Time to Failure Prediction")
st.markdown("Select a preloaded acoustic segment to predict the **time to next failure**. You can also download the segment used for prediction.")

# Load model
model = joblib.load("best_random_forest_model.pkl")

# --- Option: Use preloaded test segments ---
st.sidebar.header("📁 Select a Test Segment")

test_segments = {
    "Test Segment 1": "test_segment_1.csv",
    "Test Segment 2": "test_segment_2.csv",
    "Test Segment 3": "test_segment_3.csv",
    "Test Segment 4": "test_segment_4.csv",
    "Test Segment 5": "test_segment_5.csv",
}

selected_segment = st.sidebar.selectbox("Select a segment:", list(test_segments.keys()))

if st.sidebar.button("Predict Selected Segment"):
    try:
        df = pd.read_csv(test_segments[selected_segment])
        x = df['acoustic_data'].values

        features = {
            "mean": np.mean(x),
            "std": np.std(x),
            "min": np.min(x),
            "max": np.max(x),
            "q01": np.quantile(x, 0.01),
            "q05": np.quantile(x, 0.05),
            "q95": np.quantile(x, 0.95),
            "q99": np.quantile(x, 0.99),
        }

        input_df = pd.DataFrame([features])
        prediction = model.predict(input_df)[0]
        st.success(f"⏱️ Predicted Time to Failure: **{prediction:.5f} seconds**")

        # Provide CSV download button for the selected segment
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download This Segment",
            data=csv,
            file_name=selected_segment.replace(" ", "_").lower(),
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"❌ Error processing test segment: {e}")
