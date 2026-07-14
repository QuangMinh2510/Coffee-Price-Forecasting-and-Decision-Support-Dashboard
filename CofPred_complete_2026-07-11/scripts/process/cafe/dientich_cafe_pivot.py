"""
Chuyyen doi du lieu dien tich ca phe:
    - Doc file dientich_cafe.xlsx (dang long: Nam | Dia phuong | Gia tri | DVT)
    - Pivot: Nam la index (hang), moi Dia phuong la mot cot
    - Xuat ra file CSV/Excel da duoc pivot
"""

import pandas as pd
import os
import sys

# ──────────────────────────────────────────────
# 1. DOC FILE GOC
# ──────────────────────────────────────────────
INPUT_FILE = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Raw\coffe\dientich_cafe.xlsx"
OUTPUT_DIR  = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Processed\coffe"
LOG_FILE   = os.path.join(OUTPUT_DIR, "dientich_cafe_pivot_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Redirect stdout to log file (UTF-8)
log = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = log

try:
    # Doc file, bo dong header goc (row 0 chua ten cot)
    df_raw = pd.read_excel(INPUT_FILE, header=None)

    # Lay ten cot tu dong 0
    df = df_raw.iloc[1:].copy()
    df.columns = range(df.shape[1])

    # Dat ten ro rang cho cac cot can dung
    df = df.rename(columns={0: "Nam", 1: "Dia_phuong", 2: "Gia_tri", 3: "DVT"})

    # Chi giu 3 cot can thiet
    df = df[["Nam", "Dia_phuong", "Gia_tri"]].copy()

    # ──────────────────────────────────────────────
    # 2. LAM SACH DU LIEU
    # ──────────────────────────────────────────────
    # Xu ly gia tri "So bo nam 2024" → chuan hoa thanh so nam
    df["Nam"] = (
        df["Nam"]
        .astype(str)
        .str.extract(r"(\d{4})")   # lay 4 chu so nam
        [0]
        .astype(int)
    )

    # Ep gia tri dien tich ve so thuc
    df["Gia_tri"] = pd.to_numeric(df["Gia_tri"], errors="coerce")

    # Bo dong thieu du lieu
    df = df.dropna(subset=["Nam", "Dia_phuong", "Gia_tri"])

    # Chuan hoa ten dia phuong (trim khoang trang)
    df["Dia_phuong"] = df["Dia_phuong"].astype(str).str.strip()

    print(f"So dong sau lam sach: {len(df)}")
    print(f"Cac nam: {sorted(df['Nam'].unique())}")
    print(f"So tinh/thanh: {df['Dia_phuong'].nunique()}")
    print(f"Danh sach tinh/thanh:")
    for p in sorted(df['Dia_phuong'].unique()):
        print(f"  - {p}")

    # ──────────────────────────────────────────────
    # 3. PIVOT: Nam → hang, Dia phuong → cot
    # ──────────────────────────────────────────────
    # Neu co trung (Nam, Tinh) thi lay gia tri cuoi (keep last)
    df = df.drop_duplicates(subset=["Nam", "Dia_phuong"], keep="last")

    df_pivot = df.pivot(index="Nam", columns="Dia_phuong", values="Gia_tri")

    # Sap xep theo nam tang dan
    df_pivot = df_pivot.sort_index()

    # Dat ten index va cot
    df_pivot.index.name = "Nam"
    df_pivot.columns.name = None

    print(f"\nShape sau pivot: {df_pivot.shape}")
    print(df_pivot.head().to_string())

    # ──────────────────────────────────────────────
    # 4. XUAT KET QUA
    # ──────────────────────────────────────────────
    out_csv   = os.path.join(OUTPUT_DIR, "dientich_cafe.csv")
    df_pivot.to_csv(out_csv, encoding="utf-8-sig")


    print(f"\nDa luu CSV  : {out_csv}")
    print("DONE")

finally:
    log.close()
    sys.stdout = sys.__stdout__

print("Script completed. Check log file:", LOG_FILE)
