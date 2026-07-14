#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EDA thoi tiet (TB 5 tinh Tay Nguyen) + ONI.
5 bieu do:
  1. Tong luong mua theo thang (TB 5 tinh, bar chart)
  2. Nhiet do theo tinh (moi tinh 1 duong)
  3. Can bang nuoc (mua - ET0, tich luy 90 ngay)
  4. ONI (El Nino / La Nina)
  5. Overlay gia ca phe vs ONI
5 tinh: Dak Lak, Lam Dong, Dak Nong, Gia Lai, Kon Tum (bo Son La).
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TN5 = ["\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng", "\u0110\u1eafk N\u00f4ng", "Gia Lai", "Kon Tum"]

SEAS_MONTH = {"DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
              "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12}


def load_weather(path):
    d = pd.read_csv(path)
    d["Ngay"] = pd.to_datetime(d["date"].str[:10])
    d = d[d["province"].isin(TN5)].copy()
    num = ["temperature_2m_mean","precipitation_sum","rain_sum","et0_fao_evapotranspiration"]
    # Trung binh 5 tinh theo ngay (dung cho bieu do 3)
    g = d.groupby("Ngay")[num].mean().reset_index().sort_values("Ngay").reset_index(drop=True)
    # Du lieu tung tinh theo ngay (dung cho bieu do 2)
    per_prov = (d.groupby(["Ngay","province"])[num]
                .mean().reset_index().sort_values(["province","Ngay"]).reset_index(drop=True))
    return g, per_prov


def load_oni(path):
    o = pd.read_csv(path, sep=r"\s+")
    o.columns = [c.strip() for c in o.columns]
    o["month"] = o["SEAS"].map(SEAS_MONTH)
    o["center"] = pd.to_datetime(dict(year=o["YR"], month=o["month"], day=15))
    o = o.sort_values("center").reset_index(drop=True)
    o["ANOM"] = pd.to_numeric(o["ANOM"], errors="coerce")
    return o[["center","SEAS","YR","ANOM"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", default="data/Raw/weather/weather_all_regions.csv")
    ap.add_argument("--oni", default="data/Raw/weather/oni.txt")
    ap.add_argument("--price", default="data/Processing/coffe/gia_taynguyen_proxy_daily_filled.csv")
    ap.add_argument("--outdir", default="reports/figures/weather")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    w, per_prov = load_weather(a.weather)
    o = load_oni(a.oni)

    # Can bang nuoc tich luy 90 ngay (cho bieu do 3)
    w["waterbal_90d"] = (w["rain_sum"] - w["et0_fao_evapotranspiration"]).rolling(90, min_periods=30).sum()

    # Tong luong mua 5 tinh (cong don) theo thang (cho bieu do 1)
    # per_prov co precipitation_sum tung tinh -> sum theo (Ngay) -> resample thang
    rain_sum_daily = (per_prov.groupby("Ngay")["precipitation_sum"].sum()
                      .resample("ME").sum().reset_index())
    rain_sum_daily.columns = ["Thang", "rain_monthly"]
    rain_sum_daily["rain_12m"] = rain_sum_daily["rain_monthly"].rolling(12, min_periods=6).mean()

    fig, ax = plt.subplots(2, 3, figsize=(20, 11))

    # 1. Tong luong mua theo thang (cong tong 5 tinh)
    ax[0,0].bar(rain_sum_daily["Thang"], rain_sum_daily["rain_monthly"],
                width=28, color="#85c1e9", alpha=0.75, label="Tong mua/thang (5 tinh)")
    ax[0,0].plot(rain_sum_daily["Thang"], rain_sum_daily["rain_12m"],
                 color="#1f618d", lw=1.8, label="Xu huong 12 thang")
    ax[0,0].set_title("1. Tong luong mua theo thang (cong tong 5 tinh Tay Nguyen)")
    ax[0,0].set_ylabel("mm/thang"); ax[0,0].legend(fontsize=8)

    # 2. Nhiet do theo tung tinh (gia tri thuc theo ngay)
    colors_prov = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#8e44ad"]
    for i, prov in enumerate(TN5):
        pp = per_prov[per_prov["province"] == prov].sort_values("Ngay")
        ax[0,1].plot(pp["Ngay"], pp["temperature_2m_mean"],
                     color=colors_prov[i], lw=0.6, alpha=0.8, label=prov)
    ax[0,1].set_title("2. Nhiet do TB theo tung tinh (gia tri thuc theo ngay)")
    ax[0,1].set_ylabel("\u00b0C"); ax[0,1].legend(fontsize=7, ncol=2)

    # 3. Can bang nuoc
    ax[0,2].axhline(0, color="#7f8c8d", lw=0.8)
    ax[0,2].fill_between(w["Ngay"], w["waterbal_90d"], 0,
                         where=w["waterbal_90d"]>=0, color="#27ae60", alpha=0.5, label="Du nuoc (mua > bay hoi)")
    ax[0,2].fill_between(w["Ngay"], w["waterbal_90d"], 0,
                         where=w["waterbal_90d"]<0, color="#c0392b", alpha=0.5, label="Thieu nuoc / han (bay hoi > mua)")
    ax[0,2].set_title("3. Can bang nuoc (mua - ET0), tich luy 90 ngay")
    ax[0,2].set_ylabel("mm"); ax[0,2].legend(fontsize=8)

    # 4. ONI
    ax[1,0].axhline(0.5, color="#c0392b", ls="--", lw=0.8)
    ax[1,0].axhline(-0.5, color="#2980b9", ls="--", lw=0.8)
    ax[1,0].plot(o["center"], o["ANOM"], color="#34495e", lw=1.0)
    ax[1,0].fill_between(o["center"], o["ANOM"], 0.5, where=o["ANOM"]>=0.5, color="#e74c3c", alpha=0.6)
    ax[1,0].fill_between(o["center"], o["ANOM"], -0.5, where=o["ANOM"]<=-0.5, color="#3498db", alpha=0.6)
    ax[1,0].set_xlim(pd.Timestamp("2008-01-01"), o["center"].max())
    ax[1,0].set_title("4. ONI: El Nino (do) / La Nina (xanh)"); ax[1,0].set_ylabel("ONI")

    # 5. Overlay gia vs ONI
    ax5 = ax[1,1]
    try:
        p = pd.read_csv(a.price)
        pcol = "Gia_target" if "Gia_target" in p.columns else p.columns[1]
        p["Ngay"] = pd.to_datetime(p["Ngay"])
        ax5.plot(p["Ngay"], p[pcol], color="#16a085", lw=1.0, label="Gia ca phe")
        ax5.set_ylabel("Gia (VND/kg)", color="#16a085")
        ax5b = ax5.twinx()
        ax5b.plot(o["center"], o["ANOM"], color="#34495e", lw=1.2, label="ONI")
        ax5b.axhline(0.5, color="#c0392b", ls=":", lw=0.7)
        ax5b.axhline(-0.5, color="#2980b9", ls=":", lw=0.7)
        ax5b.set_ylabel("ONI", color="#34495e")
        ax5.set_xlim(pd.Timestamp("2018-01-01"), p["Ngay"].max())
        ax5.set_title("5. Overlay: gia ca phe vs ONI")
    except Exception as e:
        ax5.text(0.5, 0.5, "Khong load duoc gia:\n%s" % e, ha="center", va="center")
        ax5.set_title("5. Overlay gia (khong co du lieu)")

    ax[1,2].axis("off")  # khong dung o thu 6

    for r in range(2):
        for c in range(3):
            ax[r,c].tick_params(labelsize=8)

    plt.tight_layout()
    out = os.path.join(a.outdir, "weather_oni_eda.png")
    plt.savefig(out, dpi=110)
    print("Saved:", out)

    print("\n=== WEATHER (TB 5 tinh) ===")
    print("Date range:", w["Ngay"].min().date(), "->", w["Ngay"].max().date(), "| rows", len(w))
    print("Mua TB:", round(w["precipitation_sum"].mean(),2), "mm/ngay | Nhiet TB:",
          round(w["temperature_2m_mean"].mean(),2), "do C")
    print("\n=== ONI ===")
    elnino = (o["ANOM"]>=0.5).sum(); lanina = (o["ANOM"]<=-0.5).sum()
    print("Seasons:", len(o), "| El Nino:", elnino, "| La Nina:", lanina)
    print(o.tail(4).to_string(index=False))


if __name__ == "__main__":
    main()
