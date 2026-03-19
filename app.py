import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. SAYFA AYARLARI (KRİTİK: EN ÜSTTE OLMALI)
st.set_page_config(page_title="Chaos & Coherence | AI Mentor", layout="wide")

# Dark Academia Tasarımı
st.markdown("<style>.main { background-color: #1e1e1e; color: #dcdcdc; }</style>", unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: AI Risk Mentor")
st.write("Anlaşılır ve Matematiksel Portföy Analizi")
st.markdown("---")

# 2. KENAR ÇUBUĞU (ORİJİNAL SADE HALİ)
st.sidebar.header("🛠️ Ayarlar")
varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
secilenler = st.sidebar.multiselect("Varlıkları Seçin:", varlik_listesi, default=["THYAO.IS", "EREGL.IS", "BTC-USD", "GC=F"])
gun_sayisi = st.sidebar.slider("Geçmiş Veri Gün Sayısı", 30, 365, 180)

# 3. ANALİZ MOTORU VE GÖRSELLEŞTİRME
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('AI Mentor verileri sadeleştiriyor...'):
        # Veri çekme ve sütun temizleme
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        returns = df.pct_change().dropna()

    if not returns.empty:
        # --- KISIM 1: KORELASYON ANALİZİ ---
        st.subheader("📊 1. Varlıkların Birbirine Uyumu (Korelasyon)")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            corr = returns.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.info("### 🧐 Bu Tabloyu Nasıl Okumalıyım?")
            st.write("""
            Bu tablo, seçtiğin varlıkların 'beraber hareket etme' huyunu ölçer:
            - **Koyu Kırmızı (1'e yakın):** Bu varlıklar 'kanka' gibidir. Biri düşerse diğeri de muhtemelen düşer.
            - **Beyaz/Mavi (0 ve altı):** Bu varlıklar 'zıt karakterlidir'. Biri sarsılırken diğeri seni koruyabilir.
            - **Hedef:** Güvenli bir sepet için her şeyin kırmızı olmamasını sağlamaktır.
            """)
            
            # Dinamik AI Uyarı Sistemi
            max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
            most_related = max_corr[max_corr < 0.999].index[0]
            if max_corr[most_related] > 0.6:
                st.warning(f"⚠️ **Risk Uyarısı:** {most_related[0]} ve {most_related[1]} çok benzer hareket ediyor. İkisi aynı anda düşebilir!")

        # --- KISIM 2: YAPAY ZEKA GELECEK ÖNGÖRÜSÜ ---
        st.markdown("---")
        st.subheader("🤖 2. Yapay Zeka ile Gelecek 30 Günlük Risk Öngörüsü")
        
        forecast_data = []
        for col in returns.columns:
            # Lineer Regresyon Modeli
            X = np.array(range(len(returns[col]))).reshape(-1, 1)
            y = returns[col].values
            model = LinearRegression().fit(X, y)
            
            # Gelecek tahmini ve volatilite hesabı
            future_X = np.array(range(len(returns[col]), len(returns[col]) + 30)).reshape(-1, 1)
            vol = np.std(model.predict(future_X)) * np.sqrt(252) * 100
            forecast_data.append({"Varlık": col, "Gelecek Risk Skoru": round(vol, 2)})
        
        f_df = pd.DataFrame(forecast_data).sort_values(by="Gelecek Risk Skoru", ascending=False)
        
        st.bar_chart(f_df.set_index('Varlık'))
        
        # Akıllı Özet
        en_riskli = f_df.iloc[0]
        st.success(f"📈 **AI Özet:** Seçtiğin varlıklar arasında önümüzdeki 30 gün boyunca en 'heyecanlı' (hareketli) olması beklenen varlık: **{en_riskli['Varlık']}**.")

        # --- KISIM 3: EĞİTİCİ SÖZLÜK ---
        st.markdown("---")
        with st.expander("📚 Terimler Sözlüğü (Neyi, Neden Ölçüyoruz?)"):
            st.write("""
            - **Korelasyon:** Varlıkların beraber hareket etme huyudur. 1 ise ikiz gibidirler, 0 ise birbirini tanımazlar.
            - **Volatilite (Oynaklık):** Bir varlığın fiyatının ne kadar 'huzursuz' olduğudur.
            - **Lineer Regresyon (AI):** Geçmiş trendleri bulup geleceğe uzatan yapay zeka yöntemidir.
            """)

    else:
        st.error("Veri çekilemedi.")
else:
    st.info("Sıla, analizi başlatmak için butona tıkla. Senin için tüm bu sayıları anlamlı bir hikayeye dönüştüreceğim!")

st.caption("Future Talent Program 201 | Matematik & AI Entegrasyonu")
