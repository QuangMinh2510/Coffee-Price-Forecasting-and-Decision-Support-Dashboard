#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xu ly thoi tiet (TB 5 tinh Tay Nguyen) + ONI -> feature ngay, san sang ghep model.
Nguyen tac:
- Gop TB 5 tinh (Dak Lak, Lam Dong, Dak Nong, Gia Lai, Kon Tum); bo Son La (Arabica mien Bac).
- Feature CHAM (cung): rolling 30/90 ngay; tinh tren TOAN chuoi roi loc 2020-2025 (tranh NaN bien).
- Climatology lay tu BASELINE truoc 2020 (2008-2019) -> tinh bat thuong, KHONG look-ahead.
- ONI: gan available_from = tre cong bo (~2 thang); forward-fill ra ngay bang merge_asof backward.
- Lich ngay lam viec ('B') 2020-2025.
Cot ra: Ngay, available_from, rain_30d, rain_90d, rain_90d_anom, waterbal_90d,
        tmax_30d, hot_days_30d, oni, oni_chg_3m, enso
  enso = +1 (El Nino) / -1 (La Nina) / 0 (trung tinh)
"""
import argparse, os
import numpy as np
import pandas as pd

TN5 = ["\u0110\u1eafk L\u1eafk", "L\u00e2m \u0110\u1ed3ng", "\u0110\u1eafk N\u00f4ng", "Gia Lai", "Kon Tum"]

SEAS_MONTH = {"DJF":1,"JFM":2,"FMA":3,"MAM":4,"AMJ":5,"MJJ":6,
              "JJA":7,"JAS":8,"ASO":9,"SON":10,"OND":11,"NDJ":12}


def load_weather(path):
    d = pd.read_csv(path)
    d["Ngay"] = pd.to_datetime(d["date"].str[:10])
    d = d[d["province"].isin(TN5)].copy()
    num = ["temperature_2m_mean","temperature_2m_max","precipitation_sum",
           "rain_sum","et0_fao_evapotranspiration"]
    g = d.groupby("Ngay")[num].mean().reset_index().sort_values("Ngay").reset_index(drop=True)
    return g


def load_oni(path, pub_lag_months=2):
    o = pd.read_csv(path, sep=r"\s+")
    o.columns = [c.strip() for c in o.columns]
    o["month"] = o["SEAS"].map(SEAS_MONTH)
    o["oni"] = pd.to_numeric(o["ANOM"], errors="coerce")
    o["center"] = pd.to_datetime(dict(year=o["YR"], month=o["month"], day=1))
    o = o.sort_values("center").reset_index(drop=True)
    o["oni_chg_3m"] = o["oni"].diff(3)
    o["enso"] = np.where(o["oni"]>=0.5, 1, np.where(o["oni"]<=-0.5, -1, 0))
    # ONI cong bo ~ dau thang sau khi ket thuc mua 3 thang -> tre pub_lag_months
    o["available_from"] = o["center"] + pd.DateOffset(months=pub_lag_months)
    return o[["available_from","oni","oni_chg_3m","enso"]].dropna(subset=["available_from"]).sort_values("available_from")


def process(w, hot_thr=33.0, base_end=2019):
    w = w.copy()
    # Rolling tren toan chuoi
    w["rain_30d"] = w["precipitation_sum"].rolling(30, min_periods=10).sum()
    w["rain_90d"] = w["precipitation_sum"].rolling(90, min_periods=30).sum()
    w["waterbal_90d"] = (w["rain_sum"] - w["et0_fao_evapotranspiration"]).rolling(90, min_periods=30).sum()
    w["tmax_30d"] = w["temperature_2m_max"].rolling(30, min_periods=10).mean()
    w["is_hot"] = (w["temperature_2m_max"] > hot_thr).astype(int)
    w["hot_days_30d"] = w["is_hot"].rolling(30, min_periods=10).sum()
    w["month"] = w["Ngay"].dt.month
    w["year"] = w["Ngay"].dt.year
    # Climatology rain_90d theo thang tu baseline (<= base_end), tranh look-ahead
    base = w[w["year"] <= base_end]
    clim = base.groupby("month")["rain_90d"].mean()
    w["rain_90d_anom"] = w["rain_90d"] - w["month"].map(clim)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", default="data/Raw/weather/weather_all_regions.csv")
    ap.add_argument("--oni", default="data/Raw/weather/oni.txt")
    ap.add_argument("--out", default="data/Processing/weather/weather_oni_features.csv")
    ap.add_argument("--start_year", type=int, default=2020)
    ap.add_argument("--end_year", type=int, default=2025)
    ap.add_argument("--hot_thr", type=float, default=33.0)
    ap.add_argument("--oni_pub_lag", type=int, default=2)
    ap.add_argument("--weather_buffer_days", type=int, default=0,
                    help="Do tre available_from (mac dinh 0: du lieu thoi tiet co ngay cung ngay)")
    ap.add_argument("--freq", default="B")
    a = ap.parse_args()

    w = load_weather(a.weather)
    feat = process(w, hot_thr=a.hot_thr)
    o = load_oni(a.oni, pub_lag_months=a.oni_pub_lag)

    wcols = ["Ngay","rain_30d","rain_90d","rain_90d_anom","waterbal_90d","tmax_30d","hot_days_30d"]
    fw = feat[wcols].dropna(subset=["rain_90d"]).sort_values("Ngay")

    # Lich ngay lam viec
    bdays = pd.DataFrame({"Ngay": pd.date_range("%d-01-01" % a.start_year,
                                                 "%d-12-31" % a.end_year, freq=a.freq)})
    # Map thoi tiet (backward = quan sat cung ngay/gan nhat)
    m = pd.merge_asof(bdays, fw, on="Ngay", direction="backward")
    # Map ONI theo available_from (chong look-ahead)
    m = pd.merge_asof(m.sort_values("Ngay"), o, left_on="Ngay", right_on="available_from", direction="backward")
    m = m.drop(columns=["available_from"])
    m["available_from"] = m["Ngay"] + pd.Timedelta(days=a.weather_buffer_days)
    m["enso"] = m["enso"].fillna(0).astype(int)

    cols = ["Ngay","available_from","rain_30d","rain_90d","rain_90d_anom",
            "waterbal_90d","tmax_30d","hot_days_30d","oni","oni_chg_3m","enso"]
    out = m[cols]
    out.to_csv(a.out, index=False)

    print("Daily (%s): %s -> %s | rows %d -> %s" % (a.freq, out["Ngay"].min().date(),
          out["Ngay"].max().date(), len(out), a.out))
    print("NaN per col:")
    print(out.isna().sum()[out.isna().sum()>0] if out.isna().sum().sum()>0 else "  (khong co NaN)")
    print("\nPhan bo ENSO (so ngay):")
    print(out["enso"].value_counts().sort_index())
    print("\n6 dong cuoi:")
    print(out.tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
