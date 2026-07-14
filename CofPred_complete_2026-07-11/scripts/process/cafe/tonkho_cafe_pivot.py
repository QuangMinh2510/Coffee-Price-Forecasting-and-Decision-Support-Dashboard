"""
Pivot tonkho_cafe.xlsx:
    - Cot 0: Nien vu (YYYY/YYYY), Cot 1: Quoc gia, Cot 2: Ton kho (nghin bao)
    - Xu ly "Du bao 2025/2026" -> "2025/2026"
    - Pivot: Nien_vu la index (hang), Quoc_gia la cot
    - Xuat CSV
"""

import pandas as pd
import os
import sys
import re

INPUT_FILE = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Raw\coffe\tonkho_cafe.xlsx"
OUTPUT_DIR  = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Processed\coffe"
LOG_FILE    = os.path.join(OUTPUT_DIR, "tonkho_cafe_pivot_log.txt")
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
    df = df.rename(columns={0: "Nien_vu", 1: "Quoc_gia", 2: "Ton_kho", 3: "DVT"})
    df = df[["Nien_vu", "Quoc_gia", "Ton_kho"]].copy()

    # ──────────────────────────────────────────────
    # 2. LAM SACH
    # ──────────────────────────────────────────────
    # Chuan hoa nien vu: "Du bao 2025/2026" -> "2025/2026"
    def extract_nien_vu(val):
        val = str(val).strip()
        m = re.search(r"(\d{4}/\d{4})", val)
        return m.group(1) if m else None

    df["Nien_vu"] = df["Nien_vu"].apply(extract_nien_vu)

    # Danh dau nien vu du bao (goc la "Du bao ...")
    du_bao_raw = df_raw.iloc[1:][0].astype(str)
    df["La_du_bao"] = du_bao_raw.str.contains("báo|bao|Du bao|Dự báo", case=False, na=False).values

    # Chuyen ton kho ve so
    df["Ton_kho"] = pd.to_numeric(df["Ton_kho"], errors="coerce")

    # Trim ten quoc gia
    df["Quoc_gia"] = df["Quoc_gia"].astype(str).str.strip()

    # Bo missing
    df = df.dropna(subset=["Nien_vu", "Quoc_gia", "Ton_kho"])

    print(f"So dong sau lam sach: {len(df)}")
    print(f"\nCac nien vu:")
    for nv in sorted(df["Nien_vu"].unique()):
        is_db = df[df["Nien_vu"] == nv]["La_du_bao"].any()
        tag = " [Du bao]" if is_db else ""
        print(f"  - {nv}{tag}")

    print(f"\nSo quoc gia: {df['Quoc_gia'].nunique()}")
    print("\nDanh sach quoc gia:")
    for q in sorted(df["Quoc_gia"].unique()):
        print(f"  - {q}")

    # ──────────────────────────────────────────────
    # 3. KIEM TRA TRUNG LAP
    # ──────────────────────────────────────────────
    dup = df[df.duplicated(subset=["Nien_vu", "Quoc_gia"], keep=False)]
    if len(dup) > 0:
        print(f"\nCANH BAO: Co {len(dup)} dong trung (Nien_vu, Quoc_gia):")
        print(dup[["Nien_vu", "Quoc_gia", "Ton_kho"]].to_string())
    else:
        print(f"\nKhong co trung lap")

    df = df.drop_duplicates(subset=["Nien_vu", "Quoc_gia"], keep="last")

    # ──────────────────────────────────────────────
    # 4. PIVOT: Nien_vu -> hang, Quoc_gia -> cot
    # ──────────────────────────────────────────────
    df_pivot = df[["Nien_vu", "Quoc_gia", "Ton_kho"]].pivot(
        index="Nien_vu", columns="Quoc_gia", values="Ton_kho"
    )
    # Sap xep nien vu tang dan (YYYY/YYYY -> sort theo nam dau)
    df_pivot = df_pivot.loc[sorted(df_pivot.index)]
    df_pivot.index.name = "Nien_vu"
    df_pivot.columns.name = None

    print(f"\nShape sau pivot: {df_pivot.shape}")
    print(f"So nien vu: {len(df_pivot)}, So quoc gia: {len(df_pivot.columns)}")

    print("\n--- Preview (5 dong dau, 8 cot dau) ---")
    print(df_pivot.iloc[:5, :8].to_string())

    # So nien vu co du lieu theo quoc gia
    print("\n--- So nien vu co du lieu theo quoc gia (top 15) ---")
    non_na = df_pivot.notna().sum().sort_values(ascending=False)
    print(non_na.head(15).to_string())

    # Thong ke Vietnam rieng
    if "Vietnam" in df_pivot.columns:
        print("\n--- TON KHO VIETNAM (nghin bao 60kg) ---")
        vn = df_pivot[["Vietnam"]].dropna()
        print(vn.to_string())

    # ──────────────────────────────────────────────
    # 5. XUAT
    # ──────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "tonkho_cafe.csv")
    df_pivot.to_csv(out_csv, encoding="utf-8-sig")
    print(f"\nDa luu CSV: {out_csv}")
    print("DONE")

finally:
    log.close()
    sys.stdout = sys.__stdout__

print("Script completed. Check:", LOG_FILE)
