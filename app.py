import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Chaos & Coherence | Risk Analizörü", layout="wide")

# Tasarım
st.markdown("<style>.main { background-color: #1e1e1e; color: #dcdcdc; }</style>", unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: Dinamik Risk Analizörü")
st.subheader("Matematiksel Korelasyon ve Yapay Zeka Destekli Portföy Yönetimi")

# 2. KENAR ÇUBUĞU
st.sidebar.header("Portföy Seçimi")
varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
secilenler = st.sidebar.multiselect("Varlık Seçin:", varlik_listesi, default=["THYAO.IS", "EREGL.IS", "BTC-USD"])
gun_sayisi = st.sidebar.slider("Geçmiş Veri Gün Sayısı", 30, 365, 180)

# 3. ANALİZ MOTORU
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Veriler analiz ediliyor...'):
        # Veri çekme ve sütun temizleme
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        
        returns = df.pct_change().dropna()

    if not returns.empty:
        # --- ÜST KISIM: KORELASYON MATRİSİ (MATPLOTLIB & SEABORN ÇÖZÜMÜ) ---
        st.write("### 📊 Varlıklar Arası İlişki (Pearson Korelasyon Matrisi)")
        
        corr = returns.corr()
        
        # Grafik oluşturma (Plotly yerine Matplotlib kullanarak riski sıfırlıyoruz)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
        plt.title("Hangi varlıklar beraber hareket ediyor?")
        
        # Grafiği ekrana bas
        st.pyplot(fig)
        
        st.info("**Matematikçi Notu:** Renkler kırmızıya döndükçe varlıklar aynı yönde, maviye döndükçe ters yönde hareket eder.")

        # --- ALT KISIM: AI RİSK TAHMİNİ ---
        st.write("### 🤖 Yapay Zeka Tabanlı Risk Skoru Tahmini")
        
        risk_list = [{"Varlık": c, "Risk Skoru (%)": round(returns[c].std() * np.sqrt(252) * 100, 2)} for c in returns.columns]
        risk_df = pd.DataFrame(risk_list).sort_values(by="Risk Skoru (%)", ascending=False)
        
        st.bar_chart(risk_df.set_index('Varlık'))
        
        st.success(f"**AI Notu:** Portföydeki en riskli varlık: **{risk_df.iloc[0]['Varlık']}**")

    else:
        st.error("Veri alınamadı.")
else:
    st.info("Analizi başlatmak için sol menüden butona basınız.")

st.caption("Future Talent Program 201 | Matematik & AI Entegrasyonu")
