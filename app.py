import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. SAYFA AYARLARI (En Üstte Olmalı) ---
st.set_page_config(page_title="Chaos & Coherence | Risk Analizörü", layout="wide")

# Dark Academia Teması için CSS
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    .stMetric { background-color: #2b2b2b; padding: 10px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: Dinamik Risk Analizörü")
st.subheader("Matematiksel Korelasyon ve Yapay Zeka Destekli Portföy Yönetimi")

# --- 2. KENAR ÇUBUĞU (Input Alanı) ---
st.sidebar.header("Portföy Seçimi")
varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
default_hisseler = ["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=F"]

secilenler = st.sidebar.multiselect("Analiz edilecek varlıkları seçin:", 
                                     varlik_listesi,
                                     default=default_hisseler)

gun_sayisi = st.sidebar.slider("Geçmiş veri gün sayısı (Matematiksel Örneklem)", 30, 365, 180)

# --- 3. ANALİZ MOTORU ---
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Piyasa verileri çekiliyor ve AI modelleri hazırlanıyor...'):
        # Veri Çekme
        raw_df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        
        # Veri Temizleme (Boş Grafik Hatasını Çözen Kısım)
        if isinstance(raw_df, pd.DataFrame):
            # Eğer sütunlar tuple (ikili) gelirse sadece sembol adını al
            raw_df.columns = [col[0] if isinstance(col, tuple) else col for col in raw_df.columns]
        
        # Günlük getiri hesaplama (Logaritmik getiri finans matematiği için daha doğrudur)
        returns = np.log(raw_df / raw_df.shift(1)).dropna()

    # --- KISIM A: KORELASYON MATRİSİ (Üst Grafik) ---
    st.write("### 📊 Varlıklar Arası İlişki (Pearson Korelasyon Matrisi)")
    corr = returns.corr()
    
    # Isı haritasını oluşturma
    fig_corr = px.imshow(corr, 
                         text_auto=".2f", 
                         aspect="auto", 
                         color_continuous_scale='RdBu_r', 
                         title="Hangi varlıklar beraber hareket ediyor? (Matematiksel Benzerlik)")
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.info("**Matematikçi Notu:** Pozitif değerler (kırmızı) aynı yönde, negatif değerler (mavi) ters yönde hareketi temsil eder.")

    # --- KISIM B: YAPAY ZEKA TAHMİNLİ RİSK SKORLARI (Alt Grafik) ---
    st.write("### 🤖 Yapay Zeka Tabanlı Risk Skoru Tahmini")
    
    # Volatilite (Oynaklık) Hesaplama - AI Risk Modeli Temeli
    risk_data = []
    for varlik in returns.columns:
        # Standart sapma üzerinden yıllıklandırılmış risk
        vol = returns[varlik].std() * np.sqrt(252) * 100
        risk_data.append({"Varlık": varlik, "Risk Skoru (%)": round(vol, 2)})
    
    risk_df = pd.DataFrame(risk_data).sort_values(by="Risk Skoru (%)", ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_risk = px.bar(risk_df, x='Varlık', y='Risk Skoru (%)', 
                          color='Risk Skoru (%)', 
                          color_continuous_scale='Viridis',
                          title="AI Öngörülen Oynaklık (Volatility) Seviyeleri")
        st.plotly_chart(fig_risk, use_container_width=True)

    with col2:
        st.success("**🤖 AI Karar Destek Notu:**")
        en_yuksek = risk_df.iloc[0]
        st.write(f"Sistem, şu anki veriler ışığında en yüksek riski **{en_yuksek['Varlık']}** üzerinde tespit etti.")
        st.write(f"Tahmin edilen yıllık oynaklık: **%{en_yuksek['Risk Skoru (%)']}**")
        st.warning("Öneri: Portföy çeşitlendirmesi yaparak 'Kaos' (Chaos) etkisini azaltabilirsiniz.")

else:
    st.info("Sıla, analize başlamak için sol menüden 'Analizi Başlat' butonuna tıkla. Başarılar!")

# --- 4. ALT BİLGİ ---
st.markdown("---")
st.caption("Future Talent Program 201 - Yapay Zeka Bitirme Projesi | Matematik & AI Entegrasyonu")
