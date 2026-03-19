import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. SAYFA AYARLARI (KRİTİK: EN ÜSTTE OLMALI)
st.set_page_config(page_title="Chaos & Coherence | AI Risk Mentor", layout="wide")

# Dark Academia Tasarımı
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    .stAlert { border-radius: 10px; }
    .stMetric { background-color: #2b2b2b; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: AI Risk Mentor")
st.markdown("---")

# 2. KENAR ÇUBUĞU
st.sidebar.header("🛠️ Portföy Ayarları")
varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
secilenler = st.sidebar.multiselect("Varlıkları Seçin:", varlik_listesi, default=["THYAO.IS", "EREGL.IS", "BTC-USD"])
gun_sayisi = st.sidebar.slider("Geçmiş Veri Derinliği (Gün)", 30, 365, 180)

# 3. ANALİZ MOTORU
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('AI Mentor verileri yorumluyor...'):
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        returns = df.pct_change().dropna()

    if not returns.empty:
        # --- KISIM A: KORELASYON ANALİZİ (GELİŞMİŞ YORUM) ---
        st.subheader("📊 1. Varlıkların Dansı: Korelasyon Analizi")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            corr = returns.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.markdown("### 💡 Bu Tablo Ne Diyor?")
            st.write("""
            **Korelasyon**, varlıkların birbirini ne kadar takip ettiğini gösterir. 
            Amacımız, portföyde **düşük korelasyonlu** (beyaz veya mavi) varlıkları bir araya getirerek riski azaltmaktır.
            """)
            
            # Dinamik Yorum (AI Katmanı)
            max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
            most_related = max_corr[max_corr < 0.999].index[0]
            st.warning(f"⚠️ **Dikkat:** {most_related[0]} ve {most_related[1]} arasında yüksek benzerlik var. İkisi aynı anda düşebilir!")

        # --- KISIM B: GELECEK RİSK TAHMİNİ (YENİ AI ÖZELLİĞİ) ---
        st.markdown("---")
        st.subheader("🤖 2. Yapay Zeka ile Gelecek 30 Günlük Risk Tahmini")
        
        # Lineer Regresyon ile Risk Skoru (Oynaklık) Tahmini
        risk_forecast = []
        for col in returns.columns:
            # Model için veri hazırla (Gün sayısı vs Getiri)
            X = np.array(range(len(returns[col]))).reshape(-1, 1)
            y = returns[col].values
            
            # Lineer Regresyon Modelini Eğit
            model = LinearRegression()
            model.fit(X, y)
            
            # Gelecek 30 günün olası getiri sapmasını tahmin et (Risk)
            future_X = np.array(range(len(returns[col]), len(returns[col]) + 30)).reshape(-1, 1)
            future_pred = model.predict(future_X)
            
            # Tahmin edilen volatiliteyi yıllıklandır
            forecasted_vol = np.std(future_pred) * np.sqrt(252) * 100
            risk_forecast.append({"Varlık": col, "Tahmin Edilen Risk (%)": round(forecasted_vol, 2)})
            
        forecast_df = pd.DataFrame(risk_forecast).sort_values(by="Tahmin Edilen Risk (%)", ascending=False)
        
        st.bar_chart(forecast_df.set_index('Varlık'))
        
        # Risk Seviyesi Belirleme
        en_riskli_forecast = forecast_df.iloc[0]
        st.error(f"🚀 **Öngörülen En Yüksek Risk:** {en_riskli_forecast['Varlık']} (Tahmin Edilen Risk Skoru: %{en_riskli_forecast['Tahmin Edilen Risk (%)']})")
        st.info("💡 **AI Tavsiyesi:** Bu risk seviyesi yüksekse, portföyüne **Kıymetli Madenler (GC=F)** ekleyerek 'sakinleştirici' bir etki yaratabilirsin.")

    else:
        st.error("Veri çekilemedi. Lütfen bağlantınızı kontrol edin.")
else:
    st.info("Sıla, analizi başlatmak için sol menüden 'Analizi Başlat' butonuna tıkla. AI Mentor seni bekliyor!")

st.markdown("---")
st.caption("Bu uygulama bir yatırım tavsiyesi değildir, matematiksel bir modellemedir. | Future Talent Program")
