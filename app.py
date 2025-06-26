# app.py
import streamlit as st
import pandas as pd
import joblib

# Load model dan scaler
model = joblib.load('model_raisin_nb.pkl')
scaler = joblib.load('scaler_raisin.pkl')

st.title("Prediksi Klasifikasi Raisin")

# Form input pengguna
with st.form("input_form"):
    # Contoh fitur, sesuaikan dengan dataset Raisin
    area = st.number_input("Area")
    perimeter = st.number_input("Perimeter")
    major_axis = st.number_input("Major Axis Length")
    ecc = st.number_input("Eccentricity")
    # ... sesuaikan jumlah dan nama fitur
    
    submit = st.form_submit_button("Prediksi")

if submit:
    # Buat DataFrame input
    user_input = pd.DataFrame([[area, perimeter, major_axis, ecc]], 
                              columns=["Area", "Perimeter", "MajorAxisLength", "Eccentricity"])
    # Normalisasi
    user_input_scaled = scaler.transform(user_input)
    # Prediksi
    prediction = model.predict(user_input_scaled)[0]
    st.success(f"Prediksi hasil klasifikasi: {prediction}")
