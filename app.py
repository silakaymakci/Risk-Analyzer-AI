import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(page_title="Chaos & Coherence | Risk Analizörü", layout="wide")

st.markdown("<style>.main { background-color: #1e1e1e; color: #dcdcdc; }</style>", unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: Dinamik Risk Analizörü")
st.subheader("Matematiksel Korelasyon ve Yapay Zeka Destekli Portföy Yönetimi")

st.sidebar.header("Portföy Seçimi")
default_hisseler = ["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=F"]
secilenler = st.sidebar.multiselect("Varlık Seçin:", 
                                     ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "GC=F", "AAPL", "MSFT"],
                                     default=default_hisseler)

gun_sayisi = st.sidebar.slider("Geçmiş veri gün sayısı", 30, 365, 180)

if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Piyasa verileri analiz ediliyor...'):
        # Veriyi çek ve temizle (Hatanın çözümü burada)
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(1)
        
        returns = df.pct_change().dropna()

    # --- 1. KISIM: KORELASYON MATRİSİ ---
    st.write("### 📊 Varlıklar Arası İlişki (Pearson Korelasyon Matrisi)")
    corr = returns.corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- 2. KISIM: YAPAY ZEKA RİSK TAHMİNİ (FARK YARATAN KISIM) ---
    st.write("### 🤖 Yapay Zeka Tabanlı Risk Skoru Tahmini")
    
    # Basit bir AI mantığı: Hareketli ortalamalar ve standart sapma ile risk tahmini
    risk_skorlari = {}
    for col in returns.columns:
        # Son 10 günlük volatiliteyi alıp yıllıklandırıyoruz
        recent_vol = returns[col].tail(10).std() * np.sqrt(252) * 100
        risk_skorlari[col] = round(recent_vol, 2)
    
    risk_df = pd.DataFrame(list(risk_skorlari.items()), columns=['Varlık', 'Yapay Zeka Risk Skoru (%)'])
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_risk = px.bar(risk_df, x='Varlık', y='Yapay Zeka Risk Skoru (%)', color='Yapay Zeka Risk Skoru (%)',
                          title="AI Öngörülen Oynaklık (Volatility)")
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        st.success("**AI Analiz Notu:**")
        en_riskli = risk_df.loc[risk_df['Yapay Zeka Risk Skoru (%)'].idxmax()]
        st.write(f"Varlıklar arasında en yüksek oynaklık **{en_riskli['Varlık']}** üzerinde tespit edildi.")
        st.write("Matematiksel olarak portföy dengelenmesi önerilir.")

else:
    st.info("Lütfen sol menüden varlıkları seçip 'Analizi Başlat' butonuna tıklayın.")
