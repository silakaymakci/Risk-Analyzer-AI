import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px

# --- KRİTİK: BU KOMUT HER ŞEYDEN ÖNCE GELMELİ ---
st.set_page_config(page_title="Chaos & Coherence | Risk Analizörü", layout="wide")

# --- TASARIM (Dark Academia) ---
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: Dinamik Risk Analizörü")
st.subheader("Matematiksel Korelasyon ve Yapay Zeka Destekli Portföy Yönetimi")

# --- SIDEBAR ---
st.sidebar.header("Portföy Seçimi")
default_hisseler = ["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=F"]
secilenler = st.sidebar.multiselect("Analiz edilecek varlıkları seçin:", 
                                     ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "GC=F", "AAPL", "MSFT"],
                                     default=default_hisseler)

gun_sayisi = st.sidebar.slider("Geçmiş veri gün sayısı", 30, 365, 180)

# --- ANALİZ ---
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Piyasa verileri çekiliyor...'):
        # Veri çekme (Hata payını azaltmak için sütunları netleştiriyoruz)
        raw_data = yf.download(secilenler, period=f"{gun_sayisi}d")
        data = raw_data['Close']
        returns = data.pct_change().dropna()

    st.write("### 📊 Varlıklar Arası İlişki (Pearson Korelasyon Matrisi)")
    corr = returns.corr()
    
    fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                         color_continuous_scale='RdBu_r', 
                         title="Hangi varlıklar beraber hareket ediyor?")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.info(f"**Matematikçi Notu:** Şu an seçtiğiniz {len(secilenler)} varlık için korelasyon katsayıları hesaplandı.")
else:
    st.info("Lütfen sol menüden varlıkları seçip 'Analizi Başlat' butonuna tıklayın.")
