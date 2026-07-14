#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gop BANG TRAIN TONG cho du bao gia ca phe Robusta noi dia (du doan MUC gia).
Target: Gia_target (TB Dak Lak + Lam Dong, VND/kg).

Feature giu (chot voi user - 9 ngoai sinh + nhom gia target + lich):
  Gia target (tu hoi quy): target_lag1/2/3, target_ret_lag1, MA5, MA10, std5
  Lich: dayofweek, month
  London+ty gia: london_vnd_kg_lag1, usdvnd_lag1
  Dau: diesel, diesel_chg_1m, diesel_chg_3m
  Xuat khau: Luong_lag1m
  Thoi tiet+ONI: rain_90d, oni, waterbal_90d

Chong look-ahead:
  - Feature gia target deu .shift(1) (chi dung thong tin den hom truoc).
  - External da _lag1; diesel cong bo cung ngay.
  - Weather va Export ghep bang merge_asof theo 'available_from' (chi thay du lieu da cong bo).
  - Loc 2020-2025 (loai 2026 khoi train).
"""
import argparse
import numpy as np
import pandas as pd


def asof_backward(base, right, cols, right_key="Ngay"):
    r = right[[right_key] + cols].dropna(subset=[right_key]).sort_values(right_key)
    out = pd.merge_asof(base.sort_values("Ngay"), r,
                        left_on="Ngay", right_on=right_key, direction="backward")
    if right_key != "Ngay" and right_key in out.columns:
        out = out.drop(columns=[right_key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", default="data/Processing/coffe/gia_taynguyen_proxy_daily_filled.csv")
    ap.add_argument("--external", default="data/Processing/coffe/london_features.csv")
    ap.add_argument("--diesel", default="data/Processing/Fuel/diesel_daily.csv")
    ap.add_argument("--export", default="data/Processing/coffe/xuatkhau_clean.csv")
    ap.add_argument("--weather", default="data/Processing/weather/weather_oni_features.csv")
    ap.add_argument("--out", default="data/processed/gia_cafe_master.csv")
    ap.add_argument("--start_year", type=int, default=2020)
    ap.add_argument("--end_year", type=int, default=2025)
    a = ap.parse_args()

    # 1. Gia (target dung ngay, khong tao feature lag/MA)
    p = pd.read_csv(a.price)
    p["Ngay"] = pd.to_datetime(p["Ngay"])
    p = p.sort_values("Ngay").reset_index(drop=True)
    p = p.rename(columns={"Gia_TayNguyen_proxy": "Gia_target"})
    base = p[["Ngay", "Gia_target"]].copy()

    # 2. London + ty gia (da _lag1)
    ext = pd.read_csv(a.external); ext["Ngay"] = pd.to_datetime(ext["Ngay"])
    base = asof_backward(base, ext, ["london_vnd_kg_lag1", "usdvnd_lag1"])

    # 3. Dau (cong bo cung ngay)
    di = pd.read_csv(a.diesel); di["Ngay"] = pd.to_datetime(di["Ngay"])
    base = asof_backward(base, di, ["diesel", "diesel_chg_1m", "diesel_chg_3m"])

    # 4. Thoi tiet + ONI (theo available_from)
    w = pd.read_csv(a.weather); w["available_from"] = pd.to_datetime(w["available_from"])
    base = asof_backward(base, w, ["rain_90d", "oni", "waterbal_90d"], right_key="available_from")

    # 5. Xuat khau (theo available_from)
    ex = pd.read_csv(a.export); ex["available_from"] = pd.to_datetime(ex["available_from"])
    base = asof_backward(base, ex, ["Luong_lag1m"], right_key="available_from")

    # 6. Loc nam train
    base = base[(base["Ngay"].dt.year >= a.start_year) & (base["Ngay"].dt.year <= a.end_year)]
    base = base.sort_values("Ngay").reset_index(drop=True)
    base.to_csv(a.out, index=False)

    feats = [c for c in base.columns if c not in ("Ngay", "Gia_target", "outlier_flag")]
    print("Master:", base["Ngay"].min().date(), "->", base["Ngay"].max().date(),
          "| rows", len(base), "| so feature", len(feats))
    print("Feature:", feats)
    print("\nNaN per col:")
    nn = base.isna().sum(); nn = nn[nn > 0]
    print(nn.to_string() if len(nn) else "  (khong co NaN)")
    print("\n5 dong dau:")
    print(base.head(5).to_string(index=False))
    print("\n3 dong cuoi:")
    print(base.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
