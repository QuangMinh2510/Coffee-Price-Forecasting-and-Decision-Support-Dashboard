# -*- coding: utf-8 -*-
"""
Processing 2 file ngoai sinh -> bang feature theo ngay, SAN SANG merge vao gia noi dia.
Chong look-ahead: moi feature deu lay do tre 1 ngay (gia tri ngay t-1) vi:
  - London dong cua sau gio VN  -> London(t) chi anh huong VN(t+1)
  - tuong quan return manh nhat o lag 1 (~0.57)

Viec lam:
  1) Doc + parse 2 file (London ISO; USD/VND dinh dang Investing co dau phay).
  2) Dua ve lich ngay lam viec (business day), ffill khe ngan <=3 ngay.
  3) Tao feature: gia, log-return, va ban *_lag1 (chong look-ahead).
  4) Xuat external_features.csv
Chay: python3 23_process_external.py --london london_robusta_price.csv --fx usd_vnd.csv \
        --start 2020-01-01 --end 2026-06-03 --out external_features.csv
"""
import argparse
import numpy as np, pandas as pd


def load_london(p):
    d = pd.read_csv(p)
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values('Date').rename(columns={'Price': 'london_usd'})
    return d[['Date', 'london_usd']]


def load_fx(p):
    d = pd.read_csv(p)
    d['Date'] = pd.to_datetime(d['Date'], format='%m/%d/%Y')
    d['Price'] = pd.to_numeric(d['Price'].astype(str).str.replace(',', '', regex=False), errors='coerce')
    d = d.sort_values('Date').rename(columns={'Price': 'usdvnd'})
    return d[['Date', 'usdvnd']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--london', default=r'data/Processing/coffe/london_robusta_price.csv')
    ap.add_argument('--fx', default=r'data/Raw/business/USD_VND Historical Data.csv')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--end', default='2026-06-01')
    ap.add_argument('--out', default=r'data/Processing/coffe/london_features.csv')
    ap.add_argument('--ffill_limit', type=int, default=3)
    args = ap.parse_args()

    L = load_london(args.london)
    F = load_fx(args.fx)

    # lich ngay lam viec: mo rong them 5 ngay ve truoc de shift(1) co du lookback
    start_dt = pd.Timestamp(args.start)
    lookback_start = start_dt - pd.offsets.BDay(5)
    idx = pd.bdate_range(lookback_start, args.end)
    df = pd.DataFrame({'Ngay': idx})
    df = df.merge(L.rename(columns={'Date': 'Ngay'}), on='Ngay', how='left')
    df = df.merge(F.rename(columns={'Date': 'Ngay'}), on='Ngay', how='left')

    # ffill khe ngan (chi dung qua khu)
    df['london_usd'] = df['london_usd'].ffill(limit=args.ffill_limit)
    df['usdvnd'] = df['usdvnd'].ffill(limit=args.ffill_limit)

    # quy doi + return (tinh tren chuoi cung ngay)
    df['london_vnd_kg'] = df['london_usd'] * df['usdvnd'] / 1000.0
    df['london_ret'] = df['london_usd'].pct_change()
    df['usdvnd_ret'] = df['usdvnd'].pct_change()

    # ban *_lag1 chong look-ahead: dung de ghep vao target ngay t
    for c in ['london_usd', 'usdvnd', 'london_vnd_kg', 'london_ret', 'usdvnd_ret']:
        df[c + '_lag1'] = df[c].shift(1)

    # cat output ve tu start (bo cac dong lookback dung de tinh lag)
    df = df[df['Ngay'] >= start_dt].reset_index(drop=True)

    cov_l = df['london_usd'].notna().mean() * 100
    cov_f = df['usdvnd'].notna().mean() * 100
    print('Business days:', len(df), '| London phu %.1f%% | USDVND phu %.1f%%' % (cov_l, cov_f))
    df.to_csv(args.out, index=False)
    print('Cols:', list(df.columns))
    print('Saved:', args.out)
    print(df.tail(4).to_string(index=False))


if __name__ == '__main__':
    main()
