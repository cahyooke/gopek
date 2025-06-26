# app.py
import streamlit as st
import pandas as pd
import joblib

# Simpan model yang sudah dilatih
joblib.dump(model, 'model_raisin_nb.pkl')

# Simpan scaler
joblib.dump(scaler, 'scaler_raisin.pkl')


# Judul aplikasi
st.title("Prediksi Jenis Kismis (Raisin)")

# Input fitur dari pengguna
st.write("Masukkan nilai fitur untuk prediksi:")
area = st.number_input("Area")
perimeter = st.number_input("Perimeter")
major_axis = st.number_input("Major Axis Length")
minor_axis = st.number_input("Minor Axis Length")
eccentricity = st.number_input("Eccentricity")
convex_area = st.number_input("Convex Area")
extent = st.number_input("Extent")
solidity = st.number_input("Solidity")
roundness = st.number_input("Roundness")
compactness = st.number_input("Compactness")
shape_factor_1 = st.number_input("ShapeFactor1")
shape_factor_2 = st.number_input("ShapeFactor2")
shape_factor_3 = st.number_input("ShapeFactor3")
shape_factor_4 = st.number_input("ShapeFactor4")

# Tombol prediksi
if st.button("Prediksi"):
    # Buat dataframe input pengguna
    input_df = pd.DataFrame([[
        area, perimeter, major_axis, minor_axis, eccentricity, convex_area,
        extent, solidity, roundness, compactness,
        shape_factor_1, shape_factor_2, shape_factor_3, shape_factor_4
    ]], columns=[
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'ConvexArea', 'Extent', 'Solidity', 'Roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ])

    # Skala input
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]

    st.success(f"Prediksi jenis kismis: {prediction}")
