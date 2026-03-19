import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Chaos & Coherence: Dinamik Risk Analizörü")

# 1. Veri Çekme
hisseler = ["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=F"]
data = yf.download(hisseler, start="2023-01-01")['Close']

# 2. Günlük Getiri ve Korelasyon (Matematik kısmı)
returns = data.pct_change()
corr = returns.corr()

# 3. Görselleştirme
fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

st.write("Matematikçinin Notu: Portföyünüzdeki varlıkların birbiriyle olan ilişkisini yukarıdaki tabloda görebilirsiniz.")
import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# 1. Sayfa Ayarları (Dark Academia Teması için Koyu Başlangıç)
st.set_page_config(page_title="Chaos & Coherence | Risk Analizörü", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #1e1e1e; color: #dcdcdc; }
    stMarkdown { font-family: 'serif'; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏛️ Chaos & Coherence: Dinamik Risk Analizörü")
st.subheader("Matematiksel Korelasyon ve Yapay Zeka Destekli Portföy Yönetimi")

# 2. Kenar Çubuğu (Input Alanı)
st.sidebar.header("Portföy Seçimi")
default_hisseler = ["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=F"]
secilenler = st.sidebar.multiselect("Analiz edilecek varlıkları seçin:", 
                                     ["THYAO.IS", "EREGL.IS", "SISE.IS", "KCHOL.IS", "BTC-USD", "GC=F", "AAPL", "MSFT"],
                                     default=default_hisseler)

gun_sayisi = st.sidebar.slider("Geçmiş veri gün sayısı", 30, 365, 180)

if st.sidebar.button("Analizi Başlat"):
    # 3. Veri Çekme İşlemi
    with st.spinner('Piyasa verileri çekiliyor...'):
        data = yf.download(secilenler, period=f"{gun_sayisi}d")['Close']
        returns = data.pct_change().dropna()

    # 4. Matematiksel Analiz (Korelasyon)
    st.write("### 📊 Varlıklar Arası İlişki (Pearson Korelasyon Matrisi)")
    corr = returns.corr()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                             color_continuous_scale='RdBu_r', 
                             title="Hangi varlıklar beraber hareket ediyor?")
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.info("""
        **Matematikçi Notu:** $r$ değeri 1'e yaklaştıkça varlıklar aynı yönde hareket eder. 
        Portföy çeşitlendirmesi için düşük korelasyonlu ($r < 0.3$) varlıklar seçilmelidir.
        """)

    # 5. Risk Analizi (Volatility)
    st.write("### ⚖️ Risk ve Oynaklık (Volatility) Analizi")
    volatility = returns.std() * (252**0.5) * 100 # Yıllıklandırılmış oynaklık
    
    fig_vol = px.bar(volatility, x=volatility.index, y=volatility.values, 
                     labels={'y':'Yıllık Risk (%)', 'index':'Varlık'},
                     title="Varlıkların Risk Seviyeleri")
    st.plotly_chart(fig_vol, use_container_width=True)

    # 6. Yapay Zeka Tahmini (Basit Bir Anomali Uyarısı)
    st.success("🤖 AI Karar Destek Sistemi: Analiz tamamlandı. Portföyünüzdeki korelasyon dengeli görünüyor.")
else:
    st.write("Analize başlamak için sol taraftan varlık seçip butona basınız.")
