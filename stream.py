import streamlit as st
import pandas as pd
import joblib
from pycaret.regression import load_model, predict_model
# Modeli yükle
model = load_model('AutoMLbest_model')
# Label Encoder ve Standard Scaler'ı yükle
le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')
# Eğitimde kullanılan özellik isimleri
feature_names = ['CS', 'Density', 'WC', 'pH', 'EC', 'F', 'G', 'Pollen_analysis', 'Viscosity', 'Purity']
# Streamlit başlık ve açıklama
st.title("Honey Price Prediction")
st.write("Bu uygulama, balın fiyatını tahmin etmek için oluşturulmuştur.")
# Kullanıcı girişi
cs = st.number_input("CS", value=2.81)
density = st.number_input("Density", value=1.75)
wc = st.number_input("WC", value=23.04)
ph = st.number_input("pH", value=6.29)
ec = st.number_input("EC", value=0.76)
f = st.number_input("F", value=39.02)
g = st.number_input("G", value=33.63)
pollen_analysis = st.selectbox("Pollen Analysis", le.classes_)
viscosity = st.number_input("Viscosity", value=4844.5)
purity = st.number_input("Purity", value=0.68)

# Kullanıcıdan alınan verileri bir DataFrame'e koy
input_data = pd.DataFrame({
    'CS': [cs],
    'Density': [density],
    'WC': [wc],
    'pH': [ph],
    'EC': [ec],
    'F': [f],
    'G': [g],
    'Pollen_analysis': [pollen_analysis],
    'Viscosity': [viscosity],
    'Purity': [purity]
})

# Kategorik sütunu işle
input_data['Pollen_analysis'] = le.transform(input_data['Pollen_analysis'])

# Veriyi ölçeklendir
input_data_scaled = scaler.transform(input_data)
input_data_df = pd.DataFrame(input_data_scaled, columns=feature_names)

# Tahmin yap
if st.button("Tahmin Et"):
    prediction = predict_model(model, data=input_data_df)
    predicted_price = prediction['prediction_label'][0]
    st.write(f"Tahmin Edilen Fiyat: {predicted_price} $")
