import streamlit as st
import pandas as pd
import pickle

# 🎯 Load trained model
with open("logreg_model.pkl", "rb") as f:
    model = pickle.load(f)

# 🏷️ Encoding maps (match training)
sex_map = {'male': 1, 'female': 0, 'unknown': 2}
anatomy_map = {
    'head/neck': 3,
    'lower extremity': 4,
    'torso': 6,
    'upper extremity': 7,
    'palms/soles': 5,
    'oral/genital': 2,
    'unknown': 8
}

# 🧾 Title
st.title("Melanoma Risk Prediction App")
st.write("Enter patient information to estimate melanoma risk.")

# 🧍 User Inputs
age = st.number_input("Age Approx", min_value=0, max_value=100, value=45)

sex = st.selectbox("Sex", options=['male', 'female', 'unknown'])
anatomy = st.selectbox("Anatomical Site", options=list(anatomy_map.keys()))

# 📊 Predict button
if st.button("Predict"):
    sex_encoded = sex_map[sex]
    anatomy_encoded = anatomy_map[anatomy]

    input_data = pd.DataFrame([[age, sex_encoded, anatomy_encoded]],
                              columns=['age_approx', 'sex_encoded', 'anatom_site_encoded'])
    
    proba = model.predict_proba(input_data)[0][1]
    st.success(f"🧠 Melanoma Probability: {proba:.2%}")

    if proba > 0.5:
        st.warning("⚠️ High Risk of Melanoma")
    else:
        st.info("✅ Low Risk of Melanoma")
