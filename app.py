import streamlit as st
import pandas as pd
import joblib

# --- Load model dan scaler ---
model = joblib.load("model_knn.pkl")
scaler = joblib.load("scaler_knn.pkl")  # pastikan ini diupload juga

# --- Daftar fitur yang digunakan saat training ---
selected_features = [
    'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength',
    'Eccentricity', 'ConvexArea', 'Extent', 'Solidity',
    'Roundness', 'Compactness'
]

# --- Judul App ---
st.title("Prediksi Jenis Kismis (Raisin) - KNN Classifier")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload file CSV untuk prediksi (hanya 10 fitur utama)", type=["csv"])

if uploaded_file is not None:
    try:
        # Baca data
        df = pd.read_csv(uploaded_file)

        # Validasi kolom
        if all(feature in df.columns for feature in selected_features):
            # Ambil hanya kolom yang dibutuhkan dan urutkan
            df_input = df[selected_features]

            # Skala data
            df_scaled = scaler.transform(df_input)

            # Prediksi
            predictions = model.predict(df_scaled)
            df['Prediksi'] = predictions

            st.success("Prediksi berhasil!")
            st.dataframe(df)

            # Tombol download hasil
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Hasil Prediksi", data=csv, file_name="hasil_prediksi_raisin.csv", mime="text/csv")

        else:
            missing = set(selected_features) - set(df.columns)
            st.error(f"Kolom berikut tidak ditemukan di file CSV: {', '.join(missing)}")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

else:
    st.info("Silakan upload file CSV untuk memulai prediksi.")
