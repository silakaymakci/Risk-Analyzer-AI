import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Veri Çekme
hisseler=["THYAO.IS", "EREGL.IS", "SISE.IS", "BTC-USD", "GC=f"] 
# Türk hisseleri, Bitcoin ve Altın
data=yf.download(hisseler,, start="2023-01-01")['Close']

# 2. Günlük Getiri ve Korelasyon (Matematik kısmı)
returns=data.pct_chance()
corr=returns.corr()

# 3.Görselleştirme
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()
