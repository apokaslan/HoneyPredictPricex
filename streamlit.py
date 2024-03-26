import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Veri setini yükle
veri_seti = pd.read_csv("honey_purity_dataset.csv")

# Bağımsız değişkenler ve hedef değişken arasında bölme yapalım
X = veri_seti[['CS', 'pH', 'EC', 'WC', 'Viscosity', 'Density', 'G', 'F']]
y = veri_seti['Purity']

# Eğitim ve test veri kümelerini oluşturalım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verileri ölçeklendir
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Yapay sinir ağı modelini tanımla
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=100, random_state=42)

# Modeli eğit
model.fit(X_train_scaled, y_train)

# Modelden tahminler yap
y_pred = model.predict(X_test_scaled)

# Hata metriklerini hesapla
mse = mean_squared_error(y_test, y_pred)

# Tahmin fonksiyonu
def calculate_price(cs, ph, ec, wc, viscosity, density, g, f):
    scaled_values = scaler.transform([[cs, ph, ec, wc, viscosity, density, g, f]])
    predicted_purity = model.predict(scaled_values)
    return predicted_purity[0]

### arayüz 

st.set_page_config(
    page_title="HoneyMoney",
    page_icon=""
)
# Streamlit arayüzünü oluşturma
st.title("Honey Price Estimation with Neural Network")

# Değerleri al
cs = st.number_input("CS (Color Score)", value=5.5, step=0.1)
ph = st.number_input("pH", value=4.0, step=0.1)
ec = st.number_input("EC (Electrical Conductivity)", value=3.0, step=0.1)
wc = st.number_input("WC (Water Content)", value=15.0, step=0.1)
viscosity = st.number_input("Viscosity", value=5000, step=100)
density = st.number_input("Density", value=1.3, step=0.01)
g = st.number_input("G (Glucose Level)", value=30, step=1)
f = st.number_input("F (Fructose Level)", value=25, step=1)

# Butona basıldığında hesaplama yap ve sonucu göster
if st.button("Calculate"):
    predicted_purity = calculate_price(cs, ph, ec, wc, viscosity, density, g, f)
    st.write("Estimated Purity:", predicted_purity)
    st.write("Mean Squared Error:", mse)

st.write(predicted_purity*100)