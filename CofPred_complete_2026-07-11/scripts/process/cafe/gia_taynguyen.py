#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
gia_taynguyen.py - BAN SUA (v2, 2026-06-11) - thay the ban cu bi LOOK-AHEAD LEAKAGE
================================================================================
VAN DE BAN CU:
  - Reindex lich NGAY DAY DU (ca T7/CN) roi:
        interpolate(method="time").ffill().bfill()
  - He qua: gia Thu 7 = T6 + (T2_tuan_sau - T6)/3  =>  DUNG GIA TUONG LAI.
    bfill() dau chuoi cung lay gia tuong lai. Cac doan noi suy thang tap
    lam mo hinh "du bao gioi gia tao" tren ~30% so dong (T7/CN).

BAN SUA (dung chuan chong ro ri cua du an):
  1) Loc dung loai gia 'Thuong lai thu mua' + 2 tinh anchor (Dak Lak, Lam Dong)
  2) Luoi NGAY LAM VIEC (business day) - khong co T7/CN
  3) Forward-fill toi da 3 ngay (chi dung qua khu); trong dai hon => de NaN
  4) Gia_target = trung binh 2 tinh (skipna)
  5) Co outlier Hampel (cua so 21, k=5) tren log-return - chi DANH DAU, khong xoa

CHAY (tu thu muc goc repo):
  python scripts/process/cafe/gia_taynguyen.py
OUTPUT: data/Processing/coffe/gia_taynguyen_processed.csv
  Cot: Ngay, Dak Lak, Lam Dong, Gia_target, so_tinh_co_data, is_ffilled,
       log_return, price_repeat, outlier_flag
================================================================================
"""
import argparse

import numpy as np
import pandas as pd

ANCHOR = ["\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng"]  # Dak Lak, Lam Dong
PRICE_TYPE = "Th\u01b0\u01a1ng l\u00e1i thu mua"  # Thuong lai thu mua


def log(msg):
    print(f"[gia_taynguyen] {msg}")


def load_and_normalize(path):
    df = pd.read_csv(path)
    df["Ngay"] = pd.to_datetime(df["Ngay"], errors="coerce")
    df["Gia"] = pd.to_numeric(df["Gia"], errors="coerce")
    for c in ["Thi_truong", "Loai_gia"]:
        df[c] = df[c].astype(str).str.strip()
    n0 = len(df)
    df = df.dropna(subset=["Ngay", "Gia"])
    if len(df) < n0:
        log(f"Bo {n0 - len(df)} dong loi (thieu Ngay/Gia).")
    return df


def build_target(df, max_ffill=3):
    # 1) Loc loai gia + 2 tinh anchor
    sub = df[(df["Loai_gia"] == PRICE_TYPE) & (df["Thi_truong"].isin(ANCHOR))].copy()
    log(f"Sau loc ('{PRICE_TYPE}' + 2 tinh anchor): {len(sub)} dong.")

    # 2) Khu trung (tinh, ngay) bang trung binh
    sub = sub.groupby(["Thi_truong", "Ngay"], as_index=False)["Gia"].mean()
    p = sub.pivot(index="Ngay", columns="Thi_truong", values="Gia").sort_index()

    # 3) Luoi NGAY LAM VIEC - KHONG T7/CN, KHONG interpolate, KHONG bfill
    bd = pd.bdate_range(p.index.min(), p.index.max())
    p = p.reindex(bd)
    p.index.name = "Ngay"
    log(f"Business-day grid: {len(p)} ngay ({p.index.min().date()} -> {p.index.max().date()})")

    # 4) Forward-fill toi da max_ffill ngay - CHI dung qua khu
    p_ff = p.ffill(limit=max_ffill)

    out = pd.DataFrame(index=p_ff.index)
    for m in ANCHOR:
        out[m] = p_ff[m]
    out["Gia_target"] = p_ff[ANCHOR].mean(axis=1, skipna=True)
    out["so_tinh_co_data"] = p_ff[ANCHOR].notna().sum(axis=1)
    out["is_ffilled"] = (p[ANCHOR].isna() & p_ff[ANCHOR].notna()).any(axis=1).astype(int)

    out["log_return"] = np.log(out["Gia_target"] / out["Gia_target"].shift(1))
    out["price_repeat"] = (out["Gia_target"].diff() == 0).astype("Int64")

    # 5) Co outlier Hampel (cua so 21, k=5) tren log-return - chi DANH DAU
    r = out["log_return"]
    med = r.rolling(21, center=True, min_periods=5).median()
    mad = (r - med).abs().rolling(21, center=True, min_periods=5).median()
    out["outlier_flag"] = ((r - med).abs() > 5 * 1.4826 * mad).astype("Int64")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/Processing/coffe/gia_cafe.csv")
    ap.add_argument("--output", default="data/Processing/coffe/gia_taynguyen_processed.csv")
    ap.add_argument("--max-ffill", type=int, default=3)
    args = ap.parse_args()

    df = load_and_normalize(args.input)
    out = build_target(df, args.max_ffill)

    log(f"Target thieu (gap dai, KHONG fill): {int(out['Gia_target'].isna().sum())} ngay")
    log(f"So ngay forward-fill: {int(out['is_ffilled'].sum())} | outlier flag: {int(out['outlier_flag'].sum())}")

    out_r = out.copy()
    for c in ANCHOR + ["Gia_target"]:
        out_r[c] = out_r[c].round(1)
    out_r.to_csv(args.output)
    log(f"[saved] {args.output} ({out_r.shape[0]} dong x {out_r.shape[1]} cot)")


if __name__ == "__main__":
    main()
