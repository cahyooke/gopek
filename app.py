import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('model_knn.pkl')

st.title("Prediksi Jenis Kismis (Raisin) - Model KNN")

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

# Prediksi saat tombol ditekan
if st.button("Prediksi"):
    # Susun input ke DataFrame
    input_df = pd.DataFrame([[
        area, perimeter, major_axis, minor_axis, eccentricity, convex_area,
        extent, solidity, roundness, compactness,
        shape_factor_1, shape_factor_2, shape_factor_3, shape_factor_4
    ]], columns=[
        'Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',
        'ConvexArea', 'Extent', 'Solidity', 'Roundness', 'Compactness',
        'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4'
    ])

    # Prediksi
    prediction = model.predict(input_df)[0]
    st.success(f"Prediksi jenis kismis: {prediction}")
