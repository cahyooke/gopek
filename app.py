import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
model = joblib.load("model_knn.pkl")

# --- Hanya fitur yang digunakan saat training (7 fitur) ---
selected_features = [
    'Area', 'Perimeter', 'Eccentricity', 'Solidity',
    'Extent', 'Compactness', 'Roundness'
]

st.title("Prediksi Jenis Kismis (7 Fitur) - Model KNN")

st.markdown("Masukkan nilai fitur berikut:")

# Form input manual
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01)

# Prediksi saat tombol diklik
if st.button("Prediksi"):
    try:
        input_df = pd.DataFrame([user_input])
        prediction = model.predict(input_df)[0]
        st.success(f"Jenis kismis diprediksi sebagai: **{prediction}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
