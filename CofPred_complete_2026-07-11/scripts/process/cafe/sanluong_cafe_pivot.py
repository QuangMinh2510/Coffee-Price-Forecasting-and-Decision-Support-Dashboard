"""
Pivot sanluong_cafe.xlsx:
    - Cot 0: Nam, Cot 1: Dia phuong, Cot 2: San luong (Tan)
    - Pivot: Nam la index (hang), Dia phuong la cot
    - Xu ly "So bo nam 2024" -> 2024
    - Xuat CSV
"""

import pandas as pd
import os
import sys

INPUT_FILE = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Raw\coffe\sanluong_cafe.xlsx"
OUTPUT_DIR  = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Processed\coffe"
LOG_FILE    = os.path.join(OUTPUT_DIR, "sanluong_cafe_pivot_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

log = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = log

try:
    # ──────────────────────────────────────────────
    # 1. DOC FILE
    # ──────────────────────────────────────────────
    df_raw = pd.read_excel(INPUT_FILE, header=None)
    print(f"Shape goc: {df_raw.shape}")

    # Bo header row (row 0)
    df = df_raw.iloc[1:].copy()
    df = df.rename(columns={0: "Nam", 1: "Dia_phuong", 2: "San_luong", 3: "DVT"})
    df = df[["Nam", "Dia_phuong", "San_luong"]].copy()

    # ──────────────────────────────────────────────
    # 2. LAM SACH
    # ──────────────────────────────────────────────
    # Chuan hoa nam: "So bo nam 2024" -> 2024
    df["Nam"] = (
        df["Nam"]
        .astype(str)
        .str.extract(r"(\d{4})")[0]
        .astype(int)
    )

    # Chuyen san luong ve so thuc
    df["San_luong"] = pd.to_numeric(df["San_luong"], errors="coerce")

    # Bo missing
    df = df.dropna(subset=["Nam", "Dia_phuong", "San_luong"])

    # Trim ten tinh
    df["Dia_phuong"] = df["Dia_phuong"].astype(str).str.strip()

    # ──────────────────────────────────────────────
    # CHUAN HOA TEN TINH (sua loi chinh ta / bien the)
    # ──────────────────────────────────────────────
    name_map = {
        "Hoà Bình"          : "Hòa Bình",          # dau khac nhau
        "Đắc Nông"          : "Đắk Nông",           # c -> k
        "Thanh Hoá"         : "Thanh Hóa",          # dau khac nhau
        "Thừa Thiên - Huế"  : "Thừa Thiên Huế",     # bo gach ngang
        "Thừa Thiên Hue"    : "Thừa Thiên Huế",
    }
    df["Dia_phuong"] = df["Dia_phuong"].replace(name_map)

    print(f"\nSau chuan hoa ten tinh: {df['Dia_phuong'].nunique()} tinh/thanh")
    print("\nDanh sach tinh/thanh sau chuan hoa:")
    for p in sorted(df["Dia_phuong"].unique()):
        print(f"  - {p}")

    print(f"Cac nam: {sorted(df['Nam'].unique())}")

    # ──────────────────────────────────────────────
    # 3. KIEM TRA TRUNG LAP
    # ──────────────────────────────────────────────
    dup = df[df.duplicated(subset=["Nam", "Dia_phuong"], keep=False)]
    if len(dup) > 0:
        print(f"\nCANH BAO: Co {len(dup)} dong trung (Nam, Dia_phuong):")
        print(dup.to_string())
    else:
        print("\nKhong co trung lap (Nam, Dia_phuong)")

    # Neu trung, giu dong cuoi
    df = df.drop_duplicates(subset=["Nam", "Dia_phuong"], keep="last")

    # ──────────────────────────────────────────────
    # 4. PIVOT
    # ──────────────────────────────────────────────
    df_pivot = df.pivot(index="Nam", columns="Dia_phuong", values="San_luong")
    df_pivot = df_pivot.sort_index()
    df_pivot.index.name = "Nam"
    df_pivot.columns.name = None

    print(f"\nShape sau pivot: {df_pivot.shape}")
    print(f"So nam: {len(df_pivot)}, So tinh: {len(df_pivot.columns)}")
    print("\n--- Preview (5 dong dau) ---")
    print(df_pivot.head().to_string())

    # So luong NaN theo tinh
    print("\n--- So nam co du lieu theo tung tinh ---")
    non_na = df_pivot.notna().sum().sort_values(ascending=False)
    print(non_na.to_string())

    # ──────────────────────────────────────────────
    # 5. XUAT
    # ──────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "sanluong_cafe.csv")
    df_pivot.to_csv(out_csv, encoding="utf-8-sig")
    print(f"\nDa luu CSV: {out_csv}")
    print("DONE")

finally:
    log.close()
    sys.stdout = sys.__stdout__

print("Script completed. Check:", LOG_FILE)
