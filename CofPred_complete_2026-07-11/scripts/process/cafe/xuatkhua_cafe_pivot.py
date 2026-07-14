"""
xuatkhua_cafe.xlsx:
    - Gop Nam + Thang thanh cot Ngay (YYYY-MM-01)
    - Giu nguyen Luong (nghin tan) va Kim_Ngach (trieu USD)
    - Sort tang dan theo ngay
    - Xuat ra CSV
"""

import pandas as pd
import os

INPUT_FILE = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Raw\coffe\xuatkhua_cafe.xlsx"
OUTPUT_DIR  = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Processed\coffe"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "xuatkhua_cafe.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Doc file, bo hang header
df = pd.read_excel(INPUT_FILE, header=None).iloc[1:].copy()
df.columns = [0, 1, 2, 3] + list(df.columns[4:])
df = df.rename(columns={0: "Nam", 1: "Thang", 2: "Luong_nghin_tan", 3: "Kim_Ngach_trieu_USD"})
df = df[["Nam", "Thang", "Luong_nghin_tan", "Kim_Ngach_trieu_USD"]].copy()

# Chuyen kieu
df["Nam"]                = pd.to_numeric(df["Nam"],                errors="coerce").astype("Int64")
df["Thang"]              = pd.to_numeric(df["Thang"],              errors="coerce").astype("Int64")
df["Luong_nghin_tan"]    = pd.to_numeric(df["Luong_nghin_tan"],    errors="coerce")
df["Kim_Ngach_trieu_USD"]= pd.to_numeric(df["Kim_Ngach_trieu_USD"],errors="coerce")

# Bo dong thieu nam/thang
df = df.dropna(subset=["Nam", "Thang"])

# Gop thanh cot Ngay dang YYYY-MM
df["Ngay"] = df["Nam"].astype(str) + "-" + df["Thang"].astype(str).str.zfill(2)

# Chi giu 3 cot, sort tang dan
df = df[["Ngay", "Luong_nghin_tan", "Kim_Ngach_trieu_USD"]].sort_values("Ngay").reset_index(drop=True)

print(f"So dong: {len(df)}")
print(f"Tu: {df['Ngay'].min()} -> Den: {df['Ngay'].max()}")
print(df.head())
print(df.tail())

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\n[OK] Da luu -> {OUTPUT_CSV}")
