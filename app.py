import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import numpy as np

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Chaos & Coherence | Risk Analizörü", layout="wide")

# Dark Academia Teması
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: Dinamik Risk Analizörü")
st.subheader("Matematiksel Korelasyon ve Yapay Zeka Destekli Portföy Yönetimi")

# 2. KENAR ÇUBUĞU
st.sidebar.header("Portföy Seçimi")
varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
secilenler = st.sidebar.multiselect("Varlık Seçin:", varlik_listesi, default=["THYAO.IS", "EREGL.IS", "BTC-USD", "GC=F"])
gun_sayisi = st.sidebar.slider("Geçmiş Veri Gün Sayısı", 30, 365, 180)

# 3. ANALİZ BUTONU VE İŞLEMLER
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('Veriler işleniyor...'):
        # Veriyi çek
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        
        # --- KRİTİK VERİ TEMİZLEME (BOŞ GRAFİK ÇÖZÜMÜ) ---
        # Eğer yfinance MultiIndex (karmaşık başlık) gönderirse temizle
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Eğer hala bir karışıklık varsa sütun isimlerini string'e zorla
        df.columns = [str(c) for c in df.columns]
        
        # Günlük Getiriler
        returns = df.pct_change().dropna()

    # --- ÜST KISIM: KORELASYON MATRİSİ ---
    st.write("### 📊 Varlıklar Arası İlişki (Pearson Korelasyon Matrisi)")
    
    if not returns.empty:
        corr = returns.corr()
        # Matrisin boş kalmaması için Plotly ayarlarını sabitledik
        fig_corr = px.imshow(
            corr, 
            text_auto=".2f", 
            color_continuous_scale='RdBu_r',
            labels=dict(color="Korelasyon"),
            x=corr.columns,
            y=corr.columns
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.info("**Matematikçi Notu:** Tablodaki değerler -1 ile +1 arasındadır. 0'a yakın değerler varlıkların birbirinden bağımsız hareket ettiğini gösterir.")

        # --- ALT KISIM: AI RİSK TAHMİNİ ---
        st.write("### 🤖 Yapay Zeka Tabanlı Risk Skoru Tahmini")
        
        risk_list = []
        for col in returns.columns:
            vol = returns[col].std() * np.sqrt(252) * 100
            risk_list.append({"Varlık": col, "Risk Skoru (%)": round(vol, 2)})
        
        risk_df = pd.DataFrame(risk_list).sort_values(by="Risk Skoru (%)", ascending=False)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_risk = px.bar(risk_df, x='Varlık', y='Risk Skoru (%)', color='Risk Skoru (%)', color_continuous_scale='Viridis')
            st.plotly_chart(fig_risk, use_container_width=True)
        with c2:
            st.success("**AI Analiz Notu:**")
            st.write(f"En yüksek volatilite: **{risk_df.iloc[0]['Varlık']}**")
            st.write(f"Risk Oranı: **%{risk_df.iloc[0]['Risk Skoru (%)']}**")
    else:
        st.error("Veri alınamadı, lütfen farklı varlıklar veya gün sayısı seçin.")

else:
    st.info("Sıla, analizi başlatmak için butona basman yeterli. Tüm sistem hazır!")
