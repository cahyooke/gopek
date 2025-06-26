import streamlit as st
import pandas as pd
import joblib

# --- Load model ---
model = joblib.load("model_knn.pkl")  # Hanya model, tanpa scaler

# --- Daftar fitur yang digunakan saat training ---
selected_features = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'Eccentricity', 'ConvexArea', 'Extent', 'Solidity',
    'Roundness', 'Compactness'
]

# --- Judul Aplikasi ---
st.title("Prediksi Jenis Kismis (Raisin) - KNN Classifier")

st.markdown("Silakan isi data karakteristik kismis untuk diprediksi.")

# --- Input dari pengguna ---
user_input = {}
for feature in selected_features:
    user_input[feature] = st.number_input(f"{feature}", min_value=0.0, step=0.01)

# --- Tombol Prediksi ---
if st.button("Prediksi"):
    try:
        # Buat DataFrame dari input
        input_df = pd.DataFrame([user_input])

        # Lakukan prediksi
        prediction = model.predict(input_df)[0]

        st.success(f"Hasil Prediksi: Jenis kismis adalah **{prediction}**")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input: {e}")
