# -*- coding: utf-8 -*-
"""Ghép giá diesel DO 0,05S: webgia (2017->2020) + luatvietnam (2018-08->nay)."""
import pandas as pd

WEBGIA_DIESEL = "Điêzen 0,05S (DO 0.005S)"   # tên diesel 0,05S bên webgia
LV_DIESEL = "DO_0,05S"          # tên diesel 0,05S bên luatvietnam

# ---- 1) Đọc & lọc đúng dầu diesel 0,05S ----
wg = pd.read_csv("data/Raw/Fuel/webgia/fuel_webgia_long.csv")
lv = pd.read_csv("data/Raw/Fuel/luatvietnam/fuel_luatvietnam_long.csv")

wg_d = (wg[wg["mat_hang"] == WEBGIA_DIESEL][["ngay_ky", "vung1"]]
        .rename(columns={"vung1": "gia"}))
wg_d["nguon"] = "webgia"

lv_d = lv[lv["mat_hang"] == LV_DIESEL].copy()
# moi ngay_ky chi giu 1 lan dieu chinh MOI NHAT
# gio tre nhat = gia hieu luc; dong khong-gio (ky truoc tu trang ke sau) coi nhu cuoi ngay
if "gio" in lv_d.columns:
    lv_d["__t"] = lv_d["gio"].fillna("99:99").astype(str).replace("", "99:99")
    lv_d = lv_d.sort_values(["ngay_ky", "__t"]).drop_duplicates("ngay_ky", keep="last")
lv_d = lv_d[["ngay_ky", "gia"]].copy()
lv_d["nguon"] = "luatvietnam"

for df in (wg_d, lv_d):
    df["ngay_ky"] = pd.to_datetime(df["ngay_ky"])
    df["gia"] = pd.to_numeric(df["gia"], errors="coerce")
    df.dropna(subset=["gia"], inplace=True)

lv_start = lv_d["ngay_ky"].min()   # 2018-08-22

# ---- 2) Kiểm chứng chồng lấn: vung1(webgia) có == gia(luatvietnam)? ----
ovl = pd.merge(
    wg_d[wg_d["ngay_ky"] >= lv_start][["ngay_ky", "gia"]].rename(columns={"gia": "wg_vung1"}),
    lv_d[["ngay_ky", "gia"]].rename(columns={"gia": "lv_gia"}),
    on="ngay_ky", how="inner",
)
ovl["lech"] = ovl["wg_vung1"] - ovl["lv_gia"]
print(f"Khớp tuyệt đối: {(ovl['lech']==0).sum()}/{len(ovl)} kỳ | "
      f"lệch TB: {ovl['lech'].abs().mean():.1f}đ")

# ---- 3) Ghép: webgia (trước lv_start) + luatvietnam (từ lv_start) ----
wg_part = wg_d[wg_d["ngay_ky"] < lv_start]
merged = (pd.concat([wg_part, lv_d], ignore_index=True)
          .sort_values("ngay_ky")
          .drop_duplicates("ngay_ky", keep="last")
          .reset_index(drop=True))

# ---- 4) Bung ra lịch NGÀY + ffill (point-in-time) + đặc trưng ----
full = pd.date_range(merged["ngay_ky"].min(), merged["ngay_ky"].max(), freq="D")
daily = merged.set_index("ngay_ky").reindex(full)
daily[["gia", "nguon"]] = daily[["gia", "nguon"]].ffill()
daily.index.name = "ngay"
daily = daily.reset_index()

daily["pct_change"] = daily["gia"].pct_change()       # % thay đổi
daily["chenh_lech"] = daily["gia"].diff()             # thay đổi tuyệt đối
daily["is_doi_gia"] = (daily["chenh_lech"].fillna(0) != 0).astype(int)  # ngày có điều chỉnh

# ---- 5) Xuất file ----
merged.to_csv("data\processing\Fuel\diesel_merged_long.csv", index=False)     # theo kỳ điều chỉnh
daily.to_csv("data\processing\Fuel\diesel_2017_now_daily.csv", index=False)   # theo ngày (cho model)
print(f"Chuỗi: {daily['ngay'].min().date()} -> {daily['ngay'].max().date()} "
      f"| {len(daily)} ngày, {len(merged)} kỳ")