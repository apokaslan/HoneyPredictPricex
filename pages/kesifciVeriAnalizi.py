import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Veriyi oku
df = pd.read_csv("honey_purity_dataset.csv")

# DataFrame'i göster
st.write("Veri Seti Başlangıcı:")
st.write(df.head())

# DataFrame'in sonunu göster
st.write("Veri Seti Sonu:")
st.write(df.tail())

# DataFrame'in boyutunu göster
st.write("Veri Seti Boyutu:")
st.write(df.shape)

# DataFrame'in bilgilerini göster
st.write("Veri Seti Bilgileri:")
st.write(df.info())

# DataFrame'in istatistiklerini göster
st.write("Veri Seti İstatistikleri:")
st.write(df.describe())

# DataFrame'in sütunlarını göster
st.write("Veri Seti Sütunları:")
st.write(df.columns)

# Eksik değerlerin toplamını göster
st.write("Eksik Değerlerin Toplamı:")
st.write(df.isnull().sum())




