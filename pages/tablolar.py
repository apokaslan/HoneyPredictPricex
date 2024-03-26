import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd 

# Veri setini yükle
df = pd.read_csv("honey_purity_dataset.csv")

# Sayısal ve kategorik sütunları belirle
num_columns = ['CS', 'pH', 'EC', 'WC', 'Viscosity', 'Density', 'G', 'F', "Purity","Price"]
cat_column = "Pollen_analysis"

st.title("Scatter plot for HONEY Dataset")

# Streamlit sütunlarını oluştur
col1 , col2 = st.columns ( [0.25,0.75])

# X, Y ve Renk eksenlerini seçim kutuları ile belirle
with col1:
    x_axis = st.selectbox("X-Axis:", num_columns, index=0)
    y_axis = st.selectbox("Y-Axis:", num_columns, index=1)
    color_palette = sns.color_palette("husl", len(df[cat_column].unique()))  # Kategori sayısı kadar renk oluştur
    c_axis = st.selectbox("Colour:", df[cat_column].unique(), index=0)
    
with col2:
    fig, ax = plt.subplots()
    sns.scatterplot(x=x_axis, y=y_axis, hue=cat_column, palette=color_palette,
                    data=df[df[cat_column] == c_axis], legend='full')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title=cat_column)  # Renk paletinin gösterilmesi
    st.pyplot(fig)
