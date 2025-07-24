# 🏠 Zillow Home Value - Streamlit Minimal UI

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

# Model yükle
model = joblib.load("linear_model.joblib")

# Tüm özelliklerin sabit listesi
model_features = [
    'airconditioningtypeid', 'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
    'calculatedbathnbr', 'finishedfloor1squarefeet', 'calculatedfinishedsquarefeet',
    'finishedsquarefeet12', 'finishedsquarefeet50', 'fips', 'fireplacecnt',
    'fullbathcnt', 'garagecarcnt', 'garagetotalsqft', 'heatingorsystemtypeid',
    'latitude', 'longitude', 'lotsizesquarefeet', 'poolcnt', 'pooltypeid7',
    'propertylandusetypeid', 'rawcensustractandblock', 'regionidcity', 'regionidcounty',
    'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr', 'unitcnt',
    'yearbuilt', 'numberofstories', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt',
    'assessmentyear', 'landtaxvaluedollarcnt', 'taxamount', 'censustractandblock'
]

# 🎯 Sadece temel input'lar
st.title("🏡 Zillow’s Home Value Prediction")
st.sidebar.header("📋 Basic Property Info")

input_data = {}

# Kullanıcıdan alınan temel inputlar
input_data['calculatedfinishedsquarefeet'] = st.sidebar.number_input("Living Area (sq ft)", 500, 10000, 1500)
input_data['bathroomcnt'] = st.sidebar.slider("Number of Bathrooms", 0.0, 10.0, 2.0)
input_data['bedroomcnt'] = st.sidebar.slider("Number of Bedrooms", 0.0, 10.0, 3.0)
input_data['lotsizesquarefeet'] = st.sidebar.number_input("Lot Size (sq ft)", 0, 50000, 7000)
input_data['yearbuilt'] = st.sidebar.number_input("Year Built", 1900, 2023, 2005)
input_data['regionidzip'] = st.sidebar.number_input("ZIP Code", 90001, 99999, 95825)
input_data['structuretaxvaluedollarcnt'] = st.sidebar.number_input("Structure Tax Value", 0, 2000000, 300000)
input_data['taxvaluedollarcnt'] = st.sidebar.number_input("Total Tax Value", 0, 3000000, 450000)
input_data['taxamount'] = st.sidebar.number_input("Annual Tax Amount ($)", 0, 30000, 6000)
input_data['assessmentyear'] = st.sidebar.number_input("Assessment Year", 2000, 2025, 2015)

# Geri kalan inputları otomatik olarak 0 ile doldur
for feature in model_features:
    if feature not in input_data:
        input_data[feature] = 0

# DataFrame oluştur
input_df = pd.DataFrame([input_data])
input_df = input_df[model_features]

# Tahmin butonu
if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Log Error: {prediction:.5f}")
