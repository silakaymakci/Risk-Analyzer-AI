import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Chaos & Coherence: Dinamik Risk Analizörü")

# 1. Veri Çekme
hisseler = ["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=F"]
data = yf.download(hisseler, start="2023-01-01")['Close']

# 2. Günlük Getiri ve Korelasyon (Matematik kısmı)
returns = data.pct_change()
corr = returns.corr()

# 3. Görselleştirme
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write("Matematikçinin Notu: Portföyünüzdeki varlıkların birbiriyle olan ilişkisini yukarıdaki tabloda görebilirsiniz.")
