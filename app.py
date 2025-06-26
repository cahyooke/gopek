import streamlit as st
import pandas as pd
import joblib

# --- Load model (tanpa scaler) ---
model = joblib.load("model_knn.pkl")

# --- Daftar fitur yang digunakan (10 fitur utama Raisin) ---
selected_features = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'Eccentricity', 'ConvexArea', 'Extent', 'Solidity',
    'Roundness', 'Compactness'
]

# --- Judul Aplikasi ---
st.title("Prediksi Jenis Kismis (Raisin) - KNN Classifier (Tanpa Scaler)")

# --- Upload file CSV ---
uploaded_file = st.file_uploader("Upload file CSV untuk prediksi (10 fitur Raisin)", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca file CSV
        df = pd.read_csv(uploaded_file)

        st.subheader("Data Input:")
        st.dataframe(df)

        # Validasi kolom
        if all(feature in df.columns for feature in selected_features):
            # Ambil hanya kolom yang sesuai urutan training
            df_input = df[selected_features]

            # Prediksi langsung (tanpa scaler)
            predictions = model.predict(df_input)
            df['Prediksi'] = predictions

            st.success("Prediksi berhasil dilakukan!")
            st.subheader("Hasil Prediksi:")
            st.dataframe(df)

            # Download hasil
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil Prediksi", data=csv, file_name="hasil_prediksi_raisin.csv", mime="text/csv")
        else:
            missing = set(selected_features) - set(df.columns)
            st.error(f"Kolom berikut tidak ditemukan di file CSV: {', '.join(missing)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.info("Silakan upload file CSV dengan 10 fitur raisin untuk memulai.")
