"""
Tai ty gia USD/VND theo ngay tu Yahoo Finance (ticker VND=X).
Lich su dai (~2003 -> nay), mien phi.
Yeu cau: pip install yfinance
"""

import yfinance as yf

OUT_FILE = "usdvnd.csv"

# auto_adjust=False de giu nguyen OHLC goc; start de xa cho chac
df = yf.download("VND=X", start="2000-01-01", interval="1d", auto_adjust=False)

df.to_csv(OUT_FILE)
print(f"[OK] Da tai {len(df)} dong -> {OUT_FILE}")
print(f"[INFO] Tu {df.index.min().date()} den {df.index.max().date()}")
print(df.tail())