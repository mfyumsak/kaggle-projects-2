import streamlit as st
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
lgb_model = lgb.Booster(model_file='final_lgbm_model.txt')
ridge_model = joblib.load('ridge_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Title
st.title("📦 Avito Demand Prediction App")

# Info box
with st.expander("ℹ️ Purpose of this App"):
    st.write("""
    🎯 **Purpose**: Based on the details of your product listing, this app predicts how likely your product is to be sold on the Avito platform (**deal_probability**).
    
    The prediction is made using an ensemble of two machine learning models:
    - LightGBM (trained on numeric and categorical features)
    - Ridge Regression (trained on text features using TF-IDF)
    
    You can adjust your listing and immediately see how the prediction score changes.
    """)

# User Inputs
st.subheader("📝 Listing Information")

title = st.text_input("Title", "Super stylish kids' bed")
description = st.text_area("Description", "Selling a children's bed with mattress in excellent condition.")
price = st.number_input("Product Price (₽)", value=5000)

param_1 = st.selectbox("Product Type", ['missing', 'Beds', 'Clothing', 'Phones'])
param_2 = st.selectbox("Material or Feature", ['missing', 'Wood', 'Leather'])
param_3 = st.selectbox("Condition", ['missing', 'New', 'Used'])

# Feature Engineering
log_price = np.log1p(price)
title_len = len(title)
description_len = len(description.split())
text = title + ' ' + description
text_vector = tfidf.transform([text])
ridge_pred = ridge_model.predict(text_vector)[0]

# LightGBM input simulation (simplified version)
lgb_features = pd.DataFrame([{
    'log_price': log_price,
    'param_1': 0,
    'param_2': 0,
    'param_3': 0,
    'title_len': title_len,
    'description_len': description_len,
    'region': 0,
    'city': 0,
    'parent_category_name': 0,
    'category_name': 0,
    'user_type': 0,
    'item_seq_number': 123,
    'image_top_1': -1,
    'activation_weekday': 2,
    'activation_week': 13,
    'activation_day': 20,
    'activation_month': 3
}])

lgb_pred = lgb_model.predict(lgb_features)[0]
final_score = 0.6 * lgb_pred + 0.4 * ridge_pred

# Output
st.subheader("📊 Prediction Result")
st.metric(label="Estimated deal_probability", value=f"{final_score * 100:.2f}%")

