# -*- coding: utf-8 -*-
"""
fuel_luatvietnam.py  (BAN VIET LAI - chong sot ky)
=================================================
Cao gia ban le xang dau tu luatvietnam.vn -> CSV.

VI SAO VIET LAI (3 loi goc cua ban cu):
  1) Ban cu gan ngay_ky theo ngay DOC DUOC tu trang, khong so voi ngay yeu cau.
     Khi `?date=X` tra ve ky truoc do (vd 11/03 -> 07/03), no ghi trung ngay cu.
  2) Khu trung CHI theo ngay -> ky bi "rot ve ky truoc" bi xoa am tham.
  3) Chi parse 1 bang "ky hien tai", BO QUA bang "ky truoc do" + cac ky intraday.

CACH SUA (file nay):
  A) HARVEST MOI BANG tren moi trang: ky hien tai + ky truoc do + cac lan
     dieu chinh trong ngay (00:00 / 22:00 ...). Nho do moi ky luon duoc nhin
     thay it nhat 2 lan (tu trang cua chinh no VA tu trang ngay ke sau, noi no
     hien thi duoi dang "ky truoc do") -> khong bao gio mat ky.
  B) Khoa theo (NGAY + GIO + san pham), khong chi theo ngay.
  C) Build chuoi NGAY: moi ngay lay lan dieu chinh MOI NHAT (latest time).
  D) Tu kiem `chenh_lech`: neu gia[t]-gia[t-1] != chenh_lech[t] -> bao ky nghi thieu.
  E) Bo dau truoc khi match -> ben voi bien the font/encoding.

Chay that (co mang):   python fuel_luatvietnam.py
Chay self-test (offline): python fuel_luatvietnam.py selftest
"""
import re
import sys
import time
import random
import unicodedata
import datetime as dt

import pandas as pd

BASE = "https://luatvietnam.vn/bang-gia-xang-dau-hom-nay.html"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0 Safari/537.36"),
    "Accept-Language": "vi,en;q=0.9",
}

# ---------------------------------------------------------------------------
# 0) Bo dau tieng Viet -> ASCII (de regex don gian, ben encoding)
# ---------------------------------------------------------------------------
def _fold(s):
    s = s.replace("\u0111", "d").replace("\u0110", "D")     # d / D
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def _normalize(text):
    """Bo dau + gop '|' va khoang trang -> 1 space. Chay tot tren raw HTML lan markdown."""
    text = _fold(text).replace("|", " ")
    return re.sub(r"\s+", " ", text)


# ---------------------------------------------------------------------------
# 1) Mau san pham (canonical key -> regex tren text DA BO DAU)
#    Dat 0,005S TRUOC 0,05S de tranh nham.
# ---------------------------------------------------------------------------
PRODUCT_DEFS = [
    ("E5_RON92",  r"E5\s*RON\s*92"),
    ("RON95",     r"RON\s*95"),
    ("DO_0,005S", r"(?:DO|Diezen)\s*0[.,]005S|0[.,]001S"),
    ("DO_0,05S",  r"(?:DO|Diezen)\s*0[.,]05S"),
    ("KO",        r"Dau\s*(?:hoa|KO)"),
    ("FO",        r"FO\s*[\d.,]*%?\s*S|Mazut"),
]

# Tieu de moi BANG gia (da bo dau): "Gia dieu chinh [22:00 ]ngay DD/MM/YYYY"
DATE_HDR = re.compile(
    r"Gia\s*dieu\s*chinh\s*(?:(\d{1,2}:\d{2})\s*)?ngay\s*(\d{2}/\d{2}/\d{4})"
)
# Mot so token: "26.970", "- 3.880", "0"
NUM_TOK = re.compile(r"[-\u2013]?\s*\d[\d.]*")


def to_int(tok):
    """'26.970'->26970 ; '- 3.880'->-3880 ; '0'->0 ; rac->None"""
    if tok is None:
        return None
    s = tok.strip()
    neg = s.startswith("-") or s.startswith("\u2013")
    digits = re.sub(r"[^\d]", "", s)
    if digits == "":
        return None
    val = int(digits)
    return -val if neg else val


def parse_text(text):
    """
    Tra ve list bang tim thay tren trang:
      [{'date':'YYYY-MM-DD','time':'HH:MM'|None,'products':{key:(gia,chenh)}}]
    HARVEST MOI bang co tieu de ngay (ky hien tai + ky truoc + intraday).
    """
    t = _normalize(text)
    hdrs = list(DATE_HDR.finditer(t))
    tables = []
    for i, m in enumerate(hdrs):
        gio = m.group(1)                      # None neu khong co gio
        d, mo, y = m.group(2).split("/")
        date_iso = f"{y}-{mo}-{d}"
        block = t[m.end(): hdrs[i + 1].start() if i + 1 < len(hdrs) else len(t)]
        prods = {}
        for key, pat in PRODUCT_DEFS:
            pm = re.search(pat, block, re.IGNORECASE)
            if not pm:
                continue
            nums = [x for x in NUM_TOK.findall(block[pm.end():]) if re.search(r"\d", x)]
            if not nums:
                continue
            gia = to_int(nums[0])
            chenh = to_int(nums[1]) if len(nums) > 1 else None
            if gia is not None and gia > 1000:        # loc gia hop le (>1000 d/lit)
                prods[key] = (gia, chenh)
        if prods:
            tables.append({"date": date_iso, "time": gio, "products": prods})
    return tables


# ---------------------------------------------------------------------------
# 2) Mang (chi chay khi co internet, tren may cua ban)
# ---------------------------------------------------------------------------
def make_session():
    import requests
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def get_html(session, date_obj):
    """Lay HTML trang theo ngay (datetime.date)."""
    ds = date_obj.strftime("%d-%m-%Y")
    r = session.get(BASE, params={"date": ds}, timeout=30)
    r.raise_for_status()
    return r.text


def parse_html(html):
    """Boc text tu HTML roi parse (cau truc bang phang -> dung get_text)."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    return parse_text(soup.get_text(" "))


def crawl(start="2018-08-22", end=None, sleep=(1.0, 2.0), out_csv="fuel_luatvietnam_long.csv"):
    """
    Duyet LUI theo "ky truoc do": bat dau tu `end` (mac dinh hom nay), moi trang
    harvest het bang roi nhay ve ngay ky-truoc nho nhat < ngay hien tai.
    Cach nay KHONG phu thuoc dropdown va khong bo sot mat xich.
    """
    start_d = dt.date.fromisoformat(start)
    cur = dt.date.today() if end is None else dt.date.fromisoformat(end)
    session = make_session()

    store = {}            # (date_iso, time_or_'', product) -> (gia, chenh)
    visited = set()
    step = 0
    while cur >= start_d:
        if cur in visited:
            cur = cur - dt.timedelta(days=1)      # chong lap vo han
            continue
        visited.add(cur)
        step += 1
        try:
            tables = parse_html(get_html(session, cur))
        except Exception as e:
            print(f"  [!] loi {cur}: {e}")
            cur = cur - dt.timedelta(days=1)
            continue

        dates_seen = set()
        for tb in tables:
            dates_seen.add(tb["date"])
            for prod, (gia, chenh) in tb["products"].items():
                store[(tb["date"], tb["time"] or "", prod)] = (gia, chenh)
        print(f"[{step}] xin {cur} -> thay {sorted(dates_seen)} ({len(tables)} bang)")

        # nhay ve ngay ky-truoc gan nhat < cur
        earlier = [dt.date.fromisoformat(x) for x in dates_seen if dt.date.fromisoformat(x) < cur]
        cur = max(earlier) if earlier else (cur - dt.timedelta(days=1))
        time.sleep(random.uniform(*sleep))

    rows = [{"ngay_ky": k[0], "gio": k[1], "mat_hang": k[2], "gia": v[0], "chenh_lech": v[1]}
            for k, v in store.items()]
    df = pd.DataFrame(rows).sort_values(["ngay_ky", "gio", "mat_hang"]).reset_index(drop=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nDa luu {df['ngay_ky'].nunique()} ngay-ky, {len(df)} dong -> {out_csv}")
    return df


# ---------------------------------------------------------------------------
# 3) Build chuoi NGAY cho 1 san pham (mac dinh diesel DO 0,05S) + tu kiem
# ---------------------------------------------------------------------------
def build_daily(df_long, product="DO_0,05S", out_csv="diesel_0_05S_daily.csv"):
    d = df_long[df_long["mat_hang"] == product].copy()
    d["ngay_ky"] = pd.to_datetime(d["ngay_ky"])
    # moi NGAY lay lan dieu chinh MOI NHAT: khong-gio coi nhu hieu luc cuoi ngay ('99:99')
    d["sort_time"] = d["gio"].replace("", pd.NA).fillna("99:99")
    d = d.sort_values(["ngay_ky", "sort_time"])
    ky = d.groupby("ngay_ky", as_index=False).last()    # 1 dong/ngay-ky

    # --- TU KIEM chenh_lech ---
    ky = ky.sort_values("ngay_ky").reset_index(drop=True)
    ky["diff_thuc"] = ky["gia"].diff()
    gap = ky[(ky["diff_thuc"].notna()) &
             (ky["chenh_lech"].notna()) &
             ((ky["chenh_lech"] - ky["diff_thuc"]).abs() > 50)]
    if len(gap):
        print(f"[CANH BAO] {len(gap)} diem nghi thieu ky (chenh_lech khong khop):")
        print(gap[["ngay_ky", "gia", "chenh_lech", "diff_thuc"]].to_string(index=False))
    else:
        print("[OK] chenh_lech nhat quan, khong thieu ky.")

    # --- bung ra lich NGAY + ffill ---
    full = pd.date_range(ky["ngay_ky"].min(), ky["ngay_ky"].max(), freq="D")
    daily = ky.set_index("ngay_ky")[["gia", "chenh_lech"]].reindex(full)
    daily["gia"] = daily["gia"].ffill()
    daily.index.name = "ngay"
    daily = daily.reset_index()
    daily["pct_change"] = daily["gia"].pct_change()
    daily.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"Da luu {len(daily)} ngay -> {out_csv}")
    return daily


# ---------------------------------------------------------------------------
# 4) SELF-TEST offline: dung dung text trang that (12/03 + 11/03 + intraday)
# ---------------------------------------------------------------------------
_SAMPLE_12_03 = (
    "Gia dieu chinh ngay 12/03/2026 (Dong/lit) Chenh lech "
    "1 Xang RON 95-III 25.570 330 2 Xang E5 RON 92-II 22.500 - 451 "
    "3 Dau DO 0,05S-II 27.020 550 4 Dau KO 26.930 2.511 5 Dau FO 3,5%S 18.660 - 341 "
    "Gia dieu chinh ngay 11/03/2026 (Dong/lit) Chenh lech "
    "1 Xang RON 95-III 25.240 - 3.880 2 Xang E5 RON 92-II 22.951 - 3.619 "
    "3 Dau DO 0,05S-II 26.470 - 4.240 4 Dau KO 24.419 - 7.961 5 Dau FO 3,5%S 19.001 - 5.699"
)
# Trang cua chinh ngay 11/03: hai lan dieu chinh trong ngay (22:00 va 00:00)
_SAMPLE_11_03_INTRADAY = (
    "Gia dieu chinh 22:00 ngay 11/03/2026 (Dong/lit) Chenh lech "
    "3 Dau DO 0,05S-II 26.470 - 4.240 "
    "Gia dieu chinh 00:00 ngay 11/03/2026 (Dong/lit) Chenh lech "
    "3 Dau DO 0,05S-II 30.710 480"
)
# Mau co dau (giong trang that) de chac chan _fold hoat dong
_SAMPLE_ACCENT = (
    "Gi\u00e1 \u0111i\u1ec1u ch\u1ec9nh ng\u00e0y 28/05/2026 (\u0110\u1ed3ng/l\u00edt) "
    "3 D\u1ea7u DO 0,05S-II 27.650 - 1.110"
)


def selftest():
    ok = True

    # Test 0: bo dau hoat dong tren text co dau
    tabs0 = parse_text(_SAMPLE_ACCENT)
    print("Test0 (accent fold):", tabs0)
    ok &= len(tabs0) == 1 and tabs0[0]["products"]["DO_0,05S"] == (27650, -1110)

    # Test 1: harvest CA HAI bang tren 1 trang -> vot lai duoc 11/03
    tabs = parse_text(_SAMPLE_12_03)
    got = {t["date"]: t["products"].get("DO_0,05S") for t in tabs}
    print("Test1 (12/03 page):", got)
    ok &= got.get("2026-03-12") == (27020, 550)
    ok &= got.get("2026-03-11") == (26470, -4240)        # <-- ky truoc do duoc vot lai
    ok &= tabs[0]["products"]["RON95"] == (25570, 330)

    # Test 2: intraday -> giu ca 2 gio, build_daily lay lan MOI NHAT (22:00=26.470)
    tabs2 = parse_text(_SAMPLE_11_03_INTRADAY)
    times = sorted([t["time"] for t in tabs2])
    print("Test2 (intraday times):", times)
    ok &= times == ["00:00", "22:00"]
    rows = []
    for t in tabs2:
        g, c = t["products"]["DO_0,05S"]
        rows.append({"ngay_ky": t["date"], "gio": t["time"] or "", "mat_hang": "DO_0,05S", "gia": g, "chenh_lech": c})
    daily = build_daily(pd.DataFrame(rows), out_csv="/tmp/_t.csv")
    v = daily.loc[daily["ngay"] == "2026-03-11", "gia"].iloc[0]
    print("Test2 (gia ngay 11/03 chon duoc):", v)
    ok &= (v == 26470)        # phai lay 22:00 (moi nhat), KHONG phai 00:00=30.710

    # Test 3: to_int
    ok &= to_int("26.970") == 26970 and to_int("- 3.880") == -3880 and to_int("0") == 0

    print("\n=> SELFTEST:", "PASS \u2705" if ok else "FAIL \u274c")
    return ok


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        df = crawl(start="2018-08-22")          # chay that (can internet)
        build_daily(df)
