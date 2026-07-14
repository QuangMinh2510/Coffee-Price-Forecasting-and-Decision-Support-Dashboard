"""
Cào giá xăng dầu Petrolimex từ webgia.com (2017–2019, đầy đủ mặt hàng).
Output:
  - fuel_webgia_long.csv        : dạng dài (ngay_ky, mat_hang, don_vi, vung1, vung2)
  - fuel_all_daily_vung1.csv    : bảng rộng theo ngày, giá Vùng 1
  - fuel_all_daily_vung2.csv    : bảng rộng theo ngày, giá Vùng 2 (Tây Nguyên)
Cài đặt: pip install requests beautifulsoup4 pandas lxml
"""
import re, time, random, datetime as dt
import requests, pandas as pd
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "vi,en;q=0.9",
}
URL_TPL = "https://webgia.com/gia-xang-dau/petrolimex/{:04d}-{:02d}.html"

# ---------------------------------------------------------------- helpers ----
def make_session():
    s = requests.Session(); s.headers.update(HEADERS); return s

def get_html(session, url, retries=3, timeout=25):
    for a in range(retries):
        try:
            r = session.get(url, timeout=timeout); r.raise_for_status(); r.encoding = "utf-8"
            return r.text
        except requests.RequestException:
            if a == retries - 1: raise
            time.sleep(2 ** a + random.random())

def to_int(s):
    digits = re.sub(r"\D", "", (s or "").replace("\xa0", " "))   # '13.790' -> 13790
    return int(digits) if digits else None

def normalize_product(name):
    """Chuẩn hoá tên mặt hàng (gộp cách viết khác nhau giữa các năm)."""
    n = name.lower().replace(".", ",")
    if "0,001s" in n:                                  return "DO_0001S"   # diesel S thấp
    if "0,05s" in n or "0,005s" in n or "điêzen" in n: return "DO_005S"    # diesel chính
    if "e5" in n:                                      return "E5_RON92"
    if "95-iv" in n:                                   return "RON95_IV"
    if "ron 95" in n or "95-ii" in n or "95-iii" in n: return "RON95"
    if "ron 92" in n:                                  return "RON92"
    if "hỏa" in n or "hoa" in n:                       return "dau_hoa"
    if "no2b" in n and "3,0" in n:                     return "FO_30S"
    if "no2b" in n and "3,5" in n:                     return "FO_35S"
    if "no3" in n or "380" in n:                       return "FO_380"
    if "mazút" in n or "mazut" in n:                   return "FO_other"
    return re.sub(r"\s+", "_", name.strip())

# ---------------------------------------------------------------- parse ------
def parse_month(html):
    """Bóc tất cả (kỳ, mặt hàng) trong 1 trang tháng."""
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for tbl in soup.find_all("table"):
        date = None                              # ngày kỳ = heading gần nhất phía trên bảng
        for prev in tbl.find_all_previous(string=re.compile(r"xăng dầu ngày\s*\d{2}/\d{2}/\d{4}")):
            m = re.search(r"ngày\s*(\d{2}/\d{2}/\d{4})", prev)
            if m: date = m.group(1); break
        if not date:
            continue
        for tr in tbl.find_all("tr"):
            c = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
            if len(c) >= 4 and "Sản phẩm" not in c[0] and re.search(r"\d", c[2] or ""):
                out.append({"ngay_ky": dt.datetime.strptime(date, "%d/%m/%Y").date().isoformat(),
                            "mat_hang": c[0], "don_vi": c[1],
                            "vung1": to_int(c[2]), "vung2": to_int(c[3])})
    return out

# ---------------------------------------------------------------- crawl ------
def crawl_webgia(start=(2017, 1), end=(2019, 12), sleep=(1.0, 2.0),
                 out_csv="fuel_webgia_long.csv"):
    s = make_session()
    months, cur, last = [], dt.date(*start, 1), dt.date(*end, 1)
    while cur <= last:
        months.append((cur.year, cur.month))
        cur = (cur.replace(day=28) + dt.timedelta(days=7)).replace(day=1)
    rows = []
    for i, (yy, mm) in enumerate(months, 1):
        try:
            recs = parse_month(get_html(s, URL_TPL.format(yy, mm)))
            rows.extend(recs)
            print(f"[{i}/{len(months)}] {yy}-{mm:02d}: {len({r['ngay_ky'] for r in recs})} kỳ, {len(recs)} dòng")
        except Exception as e:
            print(f"[LỖI] {yy}-{mm:02d}: {e}")
        time.sleep(random.uniform(*sleep))
    if not rows:
        raise RuntimeError("Không thu được dữ liệu — kiểm tra lại headers/URL.")
    df = (pd.DataFrame(rows).drop_duplicates(["ngay_ky", "mat_hang"])
            .sort_values(["ngay_ky", "mat_hang"]))
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Long: {df['ngay_ky'].nunique()} kỳ, {len(df)} dòng -> {out_csv}")
    return df

# ---------------------------------------------------------------- wide -------
def build_all_daily(df_long, region="vung2", trading_calendar=None, out_csv=None):
    """Bảng RỘNG: mỗi mặt hàng 1 cột, giá theo ngày (point-in-time, ffill)."""
    out_csv = out_csv or f"fuel_all_daily_{region}.csv"
    d = df_long.copy()
    d["sp"] = d["mat_hang"].map(normalize_product)
    d["ngay_ky"] = pd.to_datetime(d["ngay_ky"])
    wide = (d.pivot_table(index="ngay_ky", columns="sp", values=region, aggfunc="last")
              .sort_index())
    idx = (pd.DatetimeIndex(pd.to_datetime(trading_calendar)).sort_values()
           if trading_calendar is not None
           else pd.date_range(wide.index.min(), wide.index.max(), freq="D"))
    daily = wide.reindex(wide.index.union(idx)).ffill().reindex(idx)
    daily.index.name = "date"
    daily.to_csv(out_csv, encoding="utf-8-sig")
    print(f"Wide {region}: {daily.shape[0]} ngày × {daily.shape[1]} mặt hàng -> {out_csv}")
    return daily

# ---------------------------------------------------------------- main -------
if __name__ == "__main__":
    df = crawl_webgia(start=(2017, 1), end=(2019, 12))     # đổi mốc nếu cần
    daily_v1 = build_all_daily(df, region="vung1")
    daily_v2 = build_all_daily(df, region="vung2")         # Vùng 2 = Tây Nguyên
    print("\nLong (head):"); print(df.head(10).to_string(index=False))
    print("\nWide Vùng 2 (tail):"); print(daily_v2.tail())