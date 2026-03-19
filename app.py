import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. SAYFA AYARLARI
st.set_page_config(page_title="Chaos & Coherence | AI Mentor", layout="wide")

# Dark Academia Tasarımı
st.markdown("<style>.main { background-color: #1e1e1e; color: #dcdcdc; }</style>", unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: AI Risk Mentor")
st.write("Anlaşılır ve Matematiksel Portföy Analizi")

# 2. KENAR ÇUBUĞU
st.sidebar.header("🛠️ Ayarlar")
varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
secilenler = st.sidebar.multiselect("Varlıkları Seçin:", varlik_listesi, default=["THYAO.IS", "EREGL.IS", "BTC-USD"])
gun_sayisi = st.sidebar.slider("Geçmiş Veri Gün Sayısı", 30, 365, 180)

# 3. ANALİZ MOTORU
if st.sidebar.button("Analizi Başlat"):
    with st.spinner('AI Mentor verileri sadeleştiriyor...'):
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        returns = df.pct_change().dropna()

    if not returns.empty:
        # --- KISIM 1: KORELASYON (ANLAŞILIR DİL) ---
        st.subheader("📊 1. Varlıkların Birbirine Uyumu")
        
        c1, c2 = st.columns([2, 1])
        with c1:
            corr = returns.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            st.pyplot(fig)
        
        with c2:
            st.info("### 🧐 Bu Tabloyu Nasıl Okumalıyım?")
            st.write("""
            - **Koyu Kırmızı (1'e yakın):** Bu varlıklar 'kanka' gibidir. Biri düşerse diğeri de muhtemelen düşer. Sepetinde çok fazla kırmızı olması **tehlikelidir**.
            - **Beyaz/Mavi (0 ve altı):** Bu varlıklar 'zıt karakterlidir'. Biri düşerken diğeri seni korur. **Güvenli bir sepet** için mavi/beyaz kutucuklar ararız.
            """)
            
            # Dinamik AI Uyarı Sistemi
            max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
            most_related = max_corr[max_corr < 0.999].index[0]
            if max_corr[most_related] > 0.6:
                st.warning(f"⚠️ **Risk Uyarısı:** {most_related[0]} ve {most_related[1]} çok benzer hareket ediyor. Birindeki sarsıntı diğerini de yakabilir!")

        # --- KISIM 2: GELECEK TAHMİNİ (BASİTLEŞTİRİLMİŞ) ---
        st.markdown("---")
        st.subheader("🤖 2. Yapay Zeka Gelecek Öngörüsü")
        
        # Lineer Regresyon ile Gelecek Tahmini
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
        
        # Özet Karnesi
        st.success(f"📈 **AI Özet:** Seçtiğin varlıklar arasında önümüzdeki 30 gün boyunca en 'heyecanlı' (hareketli) olması beklenen varlık: **{f_df.iloc[0]['Varlık']}**. Yatırım yaparken bu oynaklığı göz önünde bulundurmalısın.")

    else:
        st.error("Veri çekilemedi.")
else:
    st.info("Sıla, analizi başlatmak için butona tıkla. Senin için tüm bu sayıları anlamlı bir hikayeye dönüştüreceğim!")
