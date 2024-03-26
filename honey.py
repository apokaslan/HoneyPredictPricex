# Gerekli kütüphaneleri yükleyelim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def train_and_predict_price(cs, ph, ec, wc, viscosity, density, g, f, purity, price_multiplier):
    # Veri setini yükleyelim
    veri_seti = pd.read_csv("honey_purity_dataset.csv")

    # Bağımsız değişkenler ve hedef değişken arasında bölme yapalım
    X = veri_seti[['CS', 'pH', 'EC', 'WC', 'Viscosity', 'Density', 'G', 'F']]
    y = veri_seti['Purity']

    # Eğitim ve test veri kümelerini oluşturalım
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ridge Regresyon modelini eğitelim
    ridge_model = Ridge(alpha=1.0) # Alpha parametresi düzenleme parametresidir
    ridge_model.fit(X_train, y_train)

    # Purity tahmini yapalım
    predicted_purity = ridge_model.predict([[cs, ph, ec, wc, viscosity, density, g, f]])
    predicted_purity = [predicted_purity]  # Tek bir örnek olduğu için listeye dönüştür


    # Tahminler ile gerçek değerler arasındaki hata miktarını değerlendirelim
    mse = mean_squared_error(y_test, predicted_purity)
    print("Mean Squared Error:", mse)

    # Fiyatı hesaplayalım
    price = predicted_purity * price_multiplier

    return predicted_purity, price