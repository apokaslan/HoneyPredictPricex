import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 

# Veri setini yükle
df = pd.read_csv("honey_purity_dataset.csv")

# Sayısal sütunları belirle
num_columns = ['CS', 'pH', 'EC', 'WC', 'Viscosity', 'Density', 'G', 'F', "Purity","Price"]

st.title("Correlation Matrix for HONEY Dataset")

# Streamlit sütunlarını oluştur
col1 , col2 = st.columns ( [0.25,0.75])

# Korelasyon matrisini oluştur
correlation_matrix = df[num_columns].corr()

# Korelasyon matrisini görselleştir
with col2:
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig)

# İlgili korelasyon değerlerini yazdır
with col1:
    st.write("Correlation Matrix:")
    st.dataframe(correlation_matrix)
