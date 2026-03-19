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
st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    .stSidebar { background-color: #262626; }
    .stButton>button { width: 100%; border-radius: 20px; background-color: #4B0082; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: AI Risk Mentor")

# --- 2. GELİŞTİRİLMİŞ KENAR ÇUBUĞU (SIDEBAR) ---
with st.sidebar:
    st.image("https://img.icons8.com/wired/64/ffffff/artificial-intelligence.png", width=50)
    st.header("🛠️ Kontrol Paneli")
    st.markdown("---")
    
    # Hızlı Seçim Modları
    st.subheader("🚀 Hızlı Stratejiler")
    strateji = st.radio("Bir analiz modu seçin:", ["Özel Seçim", "Güvenli Liman", "Kripto Ağırlıklı"])
    
    # Stratejiye göre default değerleri değiştirme
    if strateji == "Güvenli Liman":
        varsayilan = ["GC=F", "SISE.IS", "EREGL.IS"]
    elif strateji == "Kripto Ağırlıklı":
        varsayilan = ["BTC-USD", "ETH-USD", "THYAO.IS"]
    else:
        varsayilan = ["THYAO.IS", "EREGL.IS", "BTC-USD", "GC=F"]

    st.markdown("---")
    st.subheader("🔍 Varlık Listesi")
    varlik_listesi = ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "ETH-USD", "GC=F", "AAPL", "MSFT"]
    secilenler = st.multiselect("Analiz edilecekleri belirle:", varlik_listesi, default=varsayilan)

    gun_sayisi = st.select_slider("📅 Veri Derinliği (Gün)", options=[30, 60, 90, 180, 365], value=180)
    
    st.markdown("---")
    # Kullanıcıya anlık bilgi
    st.write(f"✅ **Durum:** {len(secilenler)} varlık seçildi.")
    if len(secilenler) < 2:
        st.warning("⚠️ Analiz için en az 2 varlık seçmelisin.")
    
    baslat = st.button("🔥 ANALİZİ BAŞLAT")

# --- 3. ANALİZ MOTORU --- (Aynı şekilde devam ediyor)
if baslat and len(secilenler) >= 2:
    with st.spinner('AI Mentor verileri sadeleştiriyor...'):
        df = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c) for c in df.columns]
        returns = df.pct_change().dropna()

    if not returns.empty:
        # Korelasyon Kısmı
        st.subheader("📊 1. Varlıkların Dansı: Korelasyon Analizi")
        c1, c2 = st.columns([2, 1])
        with c1:
            corr = returns.corr()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.heatmap(corr, annot=True, cmap='RdBu_r', center=0, ax=ax)
            st.pyplot(fig)
        with c2:
            st.info("### 🧐 Bu Tabloyu Nasıl Okumalıyım?")
            st.write("Bu tablo, seçtiğin varlıkların 'kanka' mı yoksa 'zıt karakterli' mi olduğunu söyler.")
            max_corr = corr.unstack().sort_values(ascending=False).drop_duplicates()
            most_related = max_corr[max_corr < 0.999].index[0]
            st.warning(f"⚠️ **Risk:** {most_related[0]} ve {most_related[1]} çok benzer.")

        # AI Öngörü Kısmı
        st.markdown("---")
        st.subheader("🤖 2. Yapay Zeka Gelecek Öngörüsü")
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
        st.success(f"📈 **AI Özet:** Önümüzdeki 30 günün en 'hareketli' adayı: **{f_df.iloc[0]['Varlık']}**")

        with st.expander("📚 Terimler Sözlüğü"):
            st.write("- **Korelasyon:** Varlıkların beraber hareket etme huyudur.\n- **Volatilite:** Fiyatlardaki 'huzursuzluk' seviyesidir.")
    else:
        st.error("Veri çekilemedi.")
elif baslat:
    st.error("Lütfen en az 2 varlık seçiniz.")
else:
    st.info("Analizi başlatmak için soldaki 'Fark Yaratacak' butonuna tıkla!")
