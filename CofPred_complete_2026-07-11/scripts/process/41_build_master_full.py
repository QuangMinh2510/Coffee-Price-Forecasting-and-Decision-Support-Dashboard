#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gop BANG TRAIN TONG **DAY DU 24 FEATURE** cho du bao gia ca phe Robusta noi dia.
Target: Gia_target (TB Dak Lak + Lam Dong, VND/kg).

= Ban 40_build_master.py (18 feature) + nhom SUPPLY 6 feature (them o Vong 10-12):
  area_tn, prod_tn, yield_tn   : dien tich / san luong / nang suat 5 tinh Tay Nguyen (GSO, nam)
  tonkho_tan                   : ton kho cuoi nien vu USDA (tan)
  dongia_lag1m, dongia_ret_lag1m: don gia xuat khau USD/tan thang truoc + bien dong

Chong look-ahead (available_from):
  - GSO nam Y -> hieu luc tu 01/07 nam Y+1.
  - USDA nien vu ket thuc nam Y -> hieu luc tu 01/12 nam Y.
  - Xuat khau thang T -> hieu luc tu ngay 6 thang T+1 (da co san trong xuatkhau_clean.csv).
  - Cac feature gia target deu .shift(1); external da _lag1.

LUU Y ve NaN o dau chuoi (target_lag1/2/3, target_ret_lag1, MA5, MA10, std5):
  Day la WARM-UP co chu dich, KHONG phai loi. Du lieu gia bat dau 01/01/2020 nen
  khong ton tai lich su truoc do de tinh lag/MA/std cho cac dong dau. Du an chon
  de NaN (trung thuc) thay vi bia gia tri; cac dong nay bi loai (dropna) o buoc
  mo hinh hoa (~50 dong = 3,2%% tong the).
"""
import argparse
import numpy as np
import pandas as pd

# GSO 5 tinh Tay Nguyen cac nam TRUOC khi co file clean (dung cho cac dong 2020
# vi so lieu nam Y chi hieu luc tu 01/07 nam Y+1):
#   dien tich (nghin ha), san luong (tan) - tong Dak Lak + Lam Dong + Gia Lai + Dak Nong + Kon Tum
PRE_AREA = {2018: 203.1 + 175.6 + 89.3 + 129.5 + 20.5,
            2019: 208.1 + 175.2 + 96.3 + 129.2 + 21.6}
PRE_PROD = {2018: 478083.0 + 487411.4 + 222700.0 + 280974.0 + 42326.0,
            2019: 476424.0 + 515944.5 + 244318.6 + 300440.0 + 44087.0}


def asof_backward(base, right, cols, right_key="Ngay"):
    r = right[[right_key] + cols].dropna(subset=[right_key]).sort_values(right_key)
    out = pd.merge_asof(base.sort_values("Ngay"), r,
                        left_on="Ngay", right_on=right_key, direction="backward")
    if right_key != "Ngay" and right_key in out.columns:
        out = out.drop(columns=[right_key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price", default="data/Processing/coffe/gia_taynguyen_processed.csv")
    ap.add_argument("--target_col", default="Gia_target",
                    help="Ten cot gia target trong file price (se duoc doi ten thanh Gia_target trong output)")
    ap.add_argument("--business_days", type=int, default=1,
                    help="1 = chi giu ngay lam viec T2-T6 (chuan du an); 0 = giu nguyen luoi ngay cua file price")
    ap.add_argument("--external", default="data/Processing/coffe/london_features.csv")
    ap.add_argument("--diesel", default="data/Processing/Fuel/diesel_daily.csv")
    ap.add_argument("--export", default="data/Processing/coffe/xuatkhau_clean.csv")
    ap.add_argument("--weather", default="data/Processing/weather/weather_oni_features.csv")
    ap.add_argument("--area", default="data/Processing/coffe/dientich_taynguyen_clean.csv")
    ap.add_argument("--prod", default="data/Processing/coffe/sanluong_taynguyen_clean.csv")
    ap.add_argument("--tonkho", default="data/Processing/coffe/tonkho_clean.csv")
    ap.add_argument("--out", default="data/processed/gia_cafe_master_full.csv")
    ap.add_argument("--start_year", type=int, default=2020)
    ap.add_argument("--end_year", type=int, default=2025)
    a = ap.parse_args()

    # 1. Gia + feature tu hoi quy (shift(1) -> chi dung thong tin den hom truoc)
    p = pd.read_csv(a.price)
    p["Ngay"] = pd.to_datetime(p["Ngay"])
    if a.target_col != "Gia_target":
        if a.target_col not in p.columns:
            raise SystemExit(f"Khong tim thay cot '{a.target_col}' trong {a.price}. Cot hien co: {list(p.columns)}")
        p = p.rename(columns={a.target_col: "Gia_target"})
    if "Gia_target" not in p.columns:
        raise SystemExit(f"File {a.price} khong co cot Gia_target. Dung --target_col de chi dinh ten cot gia.")
    if a.business_days:
        n0 = len(p)
        p = p[p["Ngay"].dt.dayofweek < 5]
        if len(p) < n0:
            print(f"[business_days] loai {n0 - len(p)} dong T7/CN (luoi chuan cua du an la ngay lam viec)")
    p = p.sort_values("Ngay").reset_index(drop=True)
    g = p["Gia_target"]
    p["target_lag1"] = g.shift(1)
    p["target_lag2"] = g.shift(2)
    p["target_lag3"] = g.shift(3)
    p["target_ret_lag1"] = g.pct_change().shift(1)
    p["MA5"] = g.shift(1).rolling(5, min_periods=3).mean()
    p["MA10"] = g.shift(1).rolling(10, min_periods=5).mean()
    p["std5"] = g.shift(1).rolling(5, min_periods=3).std()
    p["dayofweek"] = p["Ngay"].dt.dayofweek
    p["month"] = p["Ngay"].dt.month
    keep = ["Ngay", "Gia_target", "outlier_flag", "target_lag1", "target_lag2",
            "target_lag3", "target_ret_lag1", "MA5", "MA10", "std5", "dayofweek", "month"]
    base = p[[c for c in keep if c in p.columns]].copy()

    # 2. London + ty gia (da _lag1)
    ext = pd.read_csv(a.external); ext["Ngay"] = pd.to_datetime(ext["Ngay"])
    base = asof_backward(base, ext, ["london_vnd_kg_lag1", "usdvnd_lag1"])

    # 3. Dau (cong bo cung ngay)
    di = pd.read_csv(a.diesel); di["Ngay"] = pd.to_datetime(di["Ngay"])
    base = asof_backward(base, di, ["diesel", "diesel_chg_1m", "diesel_chg_3m"])

    # 4. Thoi tiet + ONI (theo available_from)
    w = pd.read_csv(a.weather); w["available_from"] = pd.to_datetime(w["available_from"])
    base = asof_backward(base, w, ["rain_90d", "oni", "waterbal_90d"], right_key="available_from")

    # 5. Xuat khau: luong + don gia (theo available_from = ngay 6 thang sau)
    ex = pd.read_csv(a.export); ex["available_from"] = pd.to_datetime(ex["available_from"])
    ex = ex.rename(columns={"DonGia_lag1m": "dongia_lag1m",
                            "DonGia_ret_lag1m": "dongia_ret_lag1m"})
    base = asof_backward(base, ex, ["Luong_lag1m", "dongia_lag1m", "dongia_ret_lag1m"],
                         right_key="available_from")

    # 6. SUPPLY nam: dien tich + san luong + nang suat (GSO nam Y -> 01/07 nam Y+1)
    da = pd.read_csv(a.area).set_index("Nam")["Tong_TayNguyen"].to_dict()
    dp = pd.read_csv(a.prod).set_index("Nam")["Tong_TayNguyen"].to_dict()
    area = {**PRE_AREA, **{int(k): float(v) for k, v in da.items() if pd.notna(v)}}
    prod = {**PRE_PROD, **{int(k): float(v) for k, v in dp.items() if pd.notna(v)}}
    # nam chua co so lieu (vd 2025): keo so lieu nam gan nhat
    for y in range(min(area) + 1, a.end_year + 1):
        area.setdefault(y, area[y - 1]); prod.setdefault(y, prod[y - 1])
    ann = pd.DataFrame([{"available_from": pd.Timestamp(y + 1, 7, 1),
                         "area_tn": area[y], "prod_tn": prod[y],
                         "yield_tn": prod[y] / (area[y] * 1000.0)}
                        for y in sorted(area)])
    base = asof_backward(base, ann, ["area_tn", "prod_tn", "yield_tn"],
                         right_key="available_from")

    # 7. Ton kho USDA (nien vu ket thuc nam Y -> hieu luc 01/12 nam Y)
    tk = pd.read_csv(a.tonkho)
    tk["available_from"] = pd.to_datetime(tk["Nam_ket_thuc"].astype(int).astype(str) + "-12-01")
    tk = tk.rename(columns={"TonKho_tan": "tonkho_tan"})
    base = asof_backward(base, tk, ["tonkho_tan"], right_key="available_from")

    # 8. Loc nam train
    base = base[(base["Ngay"].dt.year >= a.start_year) & (base["Ngay"].dt.year <= a.end_year)]
    base = base.sort_values("Ngay").reset_index(drop=True)
    base.to_csv(a.out, index=False)

    feats = [c for c in base.columns if c not in ("Ngay", "Gia_target", "outlier_flag")]
    print("Master FULL:", base["Ngay"].min().date(), "->", base["Ngay"].max().date(),
          "| rows", len(base), "| so feature", len(feats))
    print("Feature:", feats)
    print("\nNaN per col (warm-up dau chuoi + dut bao gia >3 ngay la CHU DICH):")
    nn = base.isna().sum(); nn = nn[nn > 0]
    print(nn.to_string() if len(nn) else "  (khong co NaN)")
    print("\n5 dong dau:")
    print(base.head(5).to_string(index=False))
    print("\n3 dong cuoi:")
    print(base.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
