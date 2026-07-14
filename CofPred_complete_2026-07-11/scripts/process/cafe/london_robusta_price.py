"""
Lay cot Date va Price tu London Robusta Coffee Futures Historical Data.csv
- Chuyen dinh dang ngay: MM/DD/YYYY -> YYYY-MM-DD
- Chuyen Price: "3,470.00" -> 3470.0 (float)
- Sap xep theo ngay tang dan
- Xuat ra CSV
"""

import pandas as pd
import os

INPUT  = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Raw\coffe\London Robusta Coffee Futures Historical Data.csv"
OUTPUT = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Processed\coffe\london_robusta_price.csv"

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

# Doc file
df = pd.read_csv(INPUT)

# Chi lay 2 cot Date va Price
df = df[["Date", "Price"]].copy()

# Chuyen dinh dang ngay MM/DD/YYYY -> datetime -> YYYY-MM-DD
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")

# Chuyen Price: bo dau phay nghin, chuyen ve float
df["Price"] = df["Price"].str.replace(",", "", regex=False).astype(float)

# Sap xep theo ngay tang dan
df = df.sort_values("Date").reset_index(drop=True)

# Thong ke nhanh
print(f"Tong so dong: {len(df)}")
print(f"Tu ngay: {df['Date'].min().date()}")
print(f"Den ngay: {df['Date'].max().date()}")
print(f"Gia min: {df['Price'].min()}")
print(f"Gia max: {df['Price'].max()}")
print(f"\n5 dong dau:")
print(df.head().to_string(index=False))
print(f"\n5 dong cuoi:")
print(df.tail().to_string(index=False))

# Xuat CSV
df.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
print(f"\n[OK] Da luu -> {OUTPUT}")
