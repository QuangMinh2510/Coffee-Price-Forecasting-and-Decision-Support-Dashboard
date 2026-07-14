"""
Phan tich file gia_cafe.htm (UTF-16LE encoding):
    - Parse HTML table
    - Xuat thong ke tong quat
    - Ghi ket qua ra file log
"""

import pandas as pd
import os
import sys
from html.parser import HTMLParser

INPUT_FILE = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Raw\coffe\gia_cafe.htm"
OUTPUT_DIR  = r"e:\FPT\AI\SEM8_AI\DAP391m\project\CofPred\data\Processed\coffe"
LOG_FILE    = os.path.join(OUTPUT_DIR, "gia_cafe_analysis_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

log = open(LOG_FILE, "w", encoding="utf-8")
sys.stdout = log

try:
    # ──────────────────────────────────────────────
    # 1. DOC VA PARSE HTML (UTF-16LE)
    # ──────────────────────────────────────────────
    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.rows = []
            self._current_row = []
            self._current_cell = None
            self._in_cell = False

        def handle_starttag(self, tag, attrs):
            if tag in ("td", "th"):
                self._in_cell = True
                self._current_cell = ""

        def handle_endtag(self, tag):
            if tag in ("td", "th") and self._in_cell:
                self._current_row.append(self._current_cell.strip())
                self._in_cell = False
                self._current_cell = None
            elif tag == "tr":
                if self._current_row:
                    self.rows.append(self._current_row)
                    self._current_row = []

        def handle_data(self, data):
            if self._in_cell:
                self._current_cell += data

    # Thu cac encoding pho bien cho file HTM Windows
    for enc in ["utf-16", "utf-16-le", "utf-16-be", "utf-8-sig", "utf-8", "cp1252", "latin-1"]:
        try:
            with open(INPUT_FILE, "r", encoding=enc, errors="replace") as f:
                html_content = f.read()
            if "<table" in html_content.lower() or "<td" in html_content.lower():
                print(f"Doc thanh cong voi encoding: {enc}")
                print(f"  Do dai noi dung: {len(html_content)} ky tu")
                break
        except Exception as e:
            print(f"  Loi voi {enc}: {e}")
    else:
        print("Khong doc duoc file voi bat ky encoding nao!")
        sys.exit(1)

    parser = TableParser()
    parser.feed(html_content)

    rows = parser.rows
    print(f"  So dong tim thay trong HTML: {len(rows)}")

    if not rows:
        print("Khong tim thay du lieu trong bang HTML!")
        sys.exit(1)

    # Tim hang header (co nhieu cot nhat hoac chua cac tu khoa)
    header = rows[0]
    data   = rows[1:]
    print(f"  Header: {header}")
    print(f"  So dong data: {len(data)}")

    df = pd.DataFrame(data, columns=header)

    print("=" * 60)
    print("PHAN TICH FILE gia_cafe.htm")
    print("=" * 60)

    print(f"\n[1] TONG QUAN")
    print(f"  So dong (records): {len(df)}")
    print(f"  So cot           : {len(df.columns)}")
    print(f"\n  Ten cot goc:")
    for c in df.columns:
        print(f"    - {c}")

    # ──────────────────────────────────────────────
    # 2. RENAME COT & CHUAN HOA
    # ──────────────────────────────────────────────
    df.columns = ["Ten_mat_hang", "Thi_truong", "Loai_gia", "Don_vi", "Loai_tien", "Nguon", "Ngay", "Gia"]

    # Chuyen ngay ve datetime
    df["Ngay"] = pd.to_datetime(df["Ngay"], errors="coerce")

    # Chuyen gia ve so
    df["Gia"] = pd.to_numeric(df["Gia"], errors="coerce")

    print(f"\n[2] KHOANG THOI GIAN")
    print(f"  Tu: {df['Ngay'].min().date()}")
    print(f"  Den: {df['Ngay'].max().date()}")
    print(f"  So ngay: {(df['Ngay'].max() - df['Ngay'].min()).days} ngay")

    print(f"\n[3] CAC MAT HANG (Ten_mat_hang)")
    for item, count in df["Ten_mat_hang"].value_counts().items():
        print(f"  [{count:5d}] {item}")

    print(f"\n[4] CAC THI TRUONG")
    for mkt, count in df["Thi_truong"].value_counts().items():
        print(f"  [{count:5d}] {mkt}")

    print(f"\n[5] CAC LOAI GIA")
    for lt, count in df["Loai_gia"].value_counts().items():
        print(f"  [{count:5d}] {lt}")

    print(f"\n[6] CAC DON VI TINH")
    for dv, count in df["Don_vi"].value_counts().items():
        print(f"  [{count:5d}] {dv}")

    print(f"\n[7] THONG KE GIA (toan bo)")
    stats = df["Gia"].describe()
    print(f"  Count : {int(stats['count'])}")
    print(f"  Min   : {stats['min']:,.0f}")
    print(f"  Max   : {stats['max']:,.0f}")
    print(f"  Mean  : {stats['mean']:,.0f}")
    print(f"  Median: {df['Gia'].median():,.0f}")
    print(f"  Std   : {stats['std']:,.0f}")

    print(f"\n[8] THONG KE GIA THEO MAT HANG x THI TRUONG")
    grp = df.groupby(["Ten_mat_hang", "Thi_truong"])["Gia"].agg(["count", "min", "max", "mean"])
    grp.columns = ["count", "min", "max", "mean"]
    grp["mean"] = grp["mean"].round(0).astype(int)
    grp["min"]  = grp["min"].astype(int)
    grp["max"]  = grp["max"].astype(int)
    print(grp.to_string())

    print(f"\n[9] THONG KE GIA THEO MAT HANG x LOAI GIA")
    grp2 = df.groupby(["Ten_mat_hang", "Loai_gia"])["Gia"].agg(["count", "min", "max", "mean"])
    grp2.columns = ["count", "min", "max", "mean"]
    grp2["mean"] = grp2["mean"].round(0).astype(int)
    grp2["min"]  = grp2["min"].astype(int)
    grp2["max"]  = grp2["max"].astype(int)
    print(grp2.to_string())

    print(f"\n[10] MISSING VALUES")
    has_na = False
    for col in df.columns:
        na = df[col].isna().sum()
        if na > 0:
            print(f"  {col}: {na} missing")
            has_na = True
    if not has_na:
        print("  Khong co missing values")

    # ──────────────────────────────────────────────
    # 3. XUAT CSV
    # ──────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "gia_cafe.csv")
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n[11] Da luu CSV: {out_csv}")
    print("DONE")

finally:
    log.close()
    sys.stdout = sys.__stdout__

print("Script completed. Check log:", LOG_FILE)
