import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Load the trained model
@st.cache_resource
def load_model():
    with open("model_v2.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# Label Encoders (same mapping as training)
country_map = {'Canada': 0, 'Finland': 1, 'Italy': 2, 'Kenya': 3, 'Norway': 4, 'Singapore': 5}
store_map = {'Discount Stickers': 0, 'Kaggle Stickers': 1, 'Stickers for Less': 2}
product_map = {
    'Holographic Goose': 0,
    'Kaggle': 1,
    'Kaggle Tiers': 2,
    'Kerneler': 3,
    'Kerneler Dark Mode': 4
}

# UI
st.title("📦 Forecasting Sticker Sales")
st.markdown("Predict daily sticker sales based on date, country, store and product.")

# Inputs
date_input = st.date_input("Select a Date", value=datetime(2017, 1, 1))
country = st.selectbox("Country", list(country_map.keys()))
store = st.selectbox("Store", list(store_map.keys()))
product = st.selectbox("Product", list(product_map.keys()))

# Feature Engineering
def make_features(date, country, store, product):
    year = date.year
    month = date.month
    day = date.day
    weekday = date.weekday()
    weekofyear = date.isocalendar()[1]
    is_weekend = int(weekday in [5, 6])
    day_of_year = date.timetuple().tm_yday
    sin_doy = np.sin(2 * np.pi * day_of_year / 365)
    cos_doy = np.cos(2 * np.pi * day_of_year / 365)
    is_holiday = int(month == 1 and day == 1)

    features = pd.DataFrame([{
        "year": year,
        "month": month,
        "day": day,
        "weekday": weekday,
        "weekofyear": weekofyear,
        "is_weekend": is_weekend,
        "country_enc": country_map[country],
        "store_enc": store_map[store],
        "product_enc": product_map[product],
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
        "is_holiday": is_holiday
    }])

    return features

# Prediction
if st.button("Predict"):
    features = make_features(date_input, country, store, product)
    log_pred = model.predict(features)[0]
    prediction = np.expm1(log_pred)
    st.success(f"📈 Predicted Sales: **{int(round(prediction))} stickers**")
