import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Chaos & Coherence | AI Risk Mentor", layout="wide")

# Dark Academia Tasarımı
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    .stAlert { border-radius: 10px; }
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
        # --- ÜST KISIM: KORELASYON ANALİZİ ---
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
            **Korelasyon**, varlıkların birbirini ne kadar takip ettiğini gösterir:
            - **+1.0 (Koyu Kırmızı):** Ayrılmaz ikililer. Biri artarsa diğeri de artar.
            - **0.0 (Beyaz):** Birbirinden habersizler. En iyi çeşitlendirme burada olur.
            - **-1.0 (Koyu Mavi):** Zıt kardeşler. Biri düşerken diğeri yükselir (Risk kalkanı!).
            """)
            
            # Dinamik Yorum (AI Katmanı)
            max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
            most_related = max_corr[max_corr < 0.999].index[0]
            st.warning(f"⚠️ **Dikkat:** {most_related[0]} ve {most_related[1]} arasında yüksek benzerlik var. İkisi aynı anda düşebilir!")

        # --- ALT KISIM: RİSK TAHMİNİ ---
        st.markdown("---")
        st.subheader("🤖 2. Yapay Zeka Risk Karnesi")
        
        risk_list = [{"Varlık": c, "Risk": round(returns[c].std() * np.sqrt(252) * 100, 2)} for c in returns.columns]
        risk_df = pd.DataFrame(risk_list).sort_values(by="Risk", ascending=False)
        
        st.bar_chart(risk_df.set_index('Varlık'))
        
        # Risk Seviyesi Belirleme
        en_riskli = risk_df.iloc[0]
        st.error(f"🚀 **En Hareketli Varlık:** {en_riskli['Varlık']} (Risk Skoru: %{en_riskli['Risk']})")
        st.info("💡 **Öneri:** Eğer bu risk sana fazlaysa, portföyüne korelasyonu düşük olan **Altın (GC=F)** eklemeyi düşünebilirsin.")

    else:
        st.error("Veri çekilemedi. Lütfen bağlantınızı kontrol edin.")
else:
    st.info("Hoş geldin Sıla! Portföyünü analiz etmek ve AI tavsiyelerini görmek için 'Analizi Başlat' butonuna tıkla.")

st.markdown("---")
st.caption("Bu uygulama bir yatırım tavsiyesi değildir, matematiksel bir modellemedir. | Future Talent Program")
