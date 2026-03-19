import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. SAYFA AYARLARI (KRİTİK)
st.set_page_config(page_title="Chaos & Coherence | AI Mentor", layout="wide")

# Dark Academia Tasarımı ve Sol Menü Özelleştirme
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    /* Sol menü için şık bir dokunuş */
    section[data-testid="stSidebar"] {
        background-color: #262626;
        border-right: 1px solid #444;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        background-color: #3e3e3e;
        color: white;
        font-weight: bold;
        border: 1px solid #666;
    }
    .stButton>button:hover {
        background-color: #555;
        border: 1px solid #fff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: AI Risk Mentor")
st.write("Anlaşılır ve Matematiksel Portföy Analizi")
st.markdown("---")

# --- 2. GELİŞTİRİLMİŞ SOL MENÜ (SIDEBAR) ---
with st.sidebar:
    st.markdown("### 🛠️ Kontrol Paneli")
    st.info("Portföy ayarlarını buradan yapabilirsin.")
    
    st.markdown("---")
    st.subheader("🔍 Varlık Seçimi")
    varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
    secilenler = st.multiselect("Analiz Edilecek Varlıklar:", varlik_listesi, default=["THYAO.IS", "EREGL.IS", "BTC-USD", "GC=F"])

    st.markdown("---")
    st.subheader("📅 Veri Derinliği")
    gun_sayisi = st.select_slider("Geçmiş Veri Gün Sayısı:", options=[30, 60, 90, 180, 365], value=180)
    
    st.markdown("---")
    st.write(f"✅ **Durum:** {len(secilenler)} varlık hazır.")
    baslat = st.button("ANALİZİ BAŞLAT")

# --- 3. ANALİZ MOTORU VE SAMİMİ AÇIKLAMALAR ---
if baslat:
    with st.spinner('AI Mentor verileri sadeleştiriyor...'):
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        returns = df.pct_change().dropna()

    if not returns.empty:
        # --- KISIM 1: KORELASYON (SAMİMİ DİL) ---
        st.subheader("📊 1. Varlıkların Birbirine Uyumu (Korelasyon)")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            corr = returns.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            # Grafiği dark tema için biraz daha şık yapalım
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            st.pyplot(fig)
        
        with c2:
            st.info("### 🧐 Bu Tabloyu Nasıl Okumalıyım?")
            st.write("""
            Bu tablo, seçtiğin varlıkların 'beraber hareket etme' huyunu ölçer:
            - **Koyu Kırmızı (1'e yakın):** Bu varlıklar **'kanka'** gibidir. Biri düşerse diğeri de muhtemelen düşer. Sepetinde çok kırmızı olması **tehlikelidir**.
            - **Beyaz/Mavi (0 ve altı):** Bu varlıklar **'zıt karakterlidir'**. Biri sarsılırken diğeri seni koruyabilir. 
            - **Hedef:** Güvenli bir sepet için her şeyin kırmızı olmamasını sağlamaktır.
            """)
            
            # Dinamik Uyarı
            max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
            most_related = max_corr[max_corr < 0.999].index[0]
            if max_corr[most_related] > 0.6:
                st.warning(f"⚠️ **Dikkat:** {most_related[0]} ve {most_related[1]} çok benzer hareket ediyor. İkisi aynı anda düşebilir!")

        # --- KISIM 2: GELECEK TAHMİNİ ---
        st.markdown("---")
        st.subheader("🤖 2. Yapay Zeka ile Gelecek 30 Günlük Risk Öngörüsü")
        
        forecast_data = []
        for col in returns.columns:
            X = np.array(range(len(returns[col]))).reshape(-1, 1)
            y = returns[col].values
            model = LinearRegression().fit(X, y)
            future_X = np.array(range(len(returns[col]), len(returns[col]) + 30)).reshape(-1, 1)
            vol = np.std(model.predict(future_X)) * np.sqrt(252) * 100
            forecast_data.append({"Varlık": col, "Gelecek Risk Skoru": round(vol, 2)})
        
        f_df = pd.DataFrame(forecast_data).sort_values(by="Gelecek Risk Skoru", ascending=False)
        st.bar_chart(f_df.set_index('Varlık'))
        
        st.success(f"📈 **AI Özet:** Seçtiğin varlıklar arasında önümüzdeki 30 gün boyunca en **'heyecanlı' (hareketli)** olması beklenen varlık: **{f_df.iloc[0]['Varlık']}**.")

        # --- KISIM 3: SÖZLÜK ---
        st.markdown("---")
        with st.expander("📚 Anlamayanlar İçin Terimler Sözlüğü"):
            st.write("""
            - **Korelasyon:** Varlıkların birbirini ne kadar takip ettiğidir. 1 ise ikiz gibidirler, 0 ise tamamen bağımsızlar.
            - **Volatilite:** Fiyatlardaki 'huzursuzluk' seviyesidir. Yüksek puan, yüksek dalgalanma demektir.
            """)

    else:
        st.error("Veri çekilemedi.")
else:
    st.info("Sıla, analizi başlatmak için sol taraftaki 'Kontrol Paneli'nden butona tıkla. Her şeyi senin için yorumlayacağım!")

st.caption("Future Talent Program 201 | Matematik & AI Entegrasyonu")
