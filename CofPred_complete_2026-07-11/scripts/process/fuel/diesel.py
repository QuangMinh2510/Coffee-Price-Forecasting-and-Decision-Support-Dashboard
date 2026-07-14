# -*- coding: utf-8 -*-
"""
Processing GIA DAU DIESEL.
  - Input: diesel_merged_long.csv (cot: ngay_ky, gia [VND/lit], nguon)
Viec lam (chi LAM SACH, KHONG ve bieu do):
  1) Parse ngay_ky, sap xep tang dan, bo trung ngay (giu ban ghi dau).
  2) Kiem tra & hop nhat 2 nguon (webgia/luatvietnam) - thuc te KHONG trung ngay.
  3) Danh dau OUTLIER (khong xoa) bang Hampel + nguong bien dong %:
       - outlier_flag = 1 neu |pct_change| giua 2 ky > pct_thr (mac dinh 12%)
         HOAC lech qua k*MAD trong cua so truot (Hampel).
     LUU Y: bien dong >30% nam 2026 la THAT (dia chinh tri) -> chi danh dau
     tham khao, KHONG sua/xoa. Train cung khong dung 2026.
  4) Tao 'available_from' = ngay_ky + pub_buffer_days.
     Gia xang dau cong bo & hieu luc NGAY trong ngay dieu chinh (cong khai) ->
     KHONG can lag thang; buffer mac dinh 0.
  5) FEATURE CHO ABLATION (tinh theo NGAY THUC, chong look-ahead):
       - diesel_current : gia ky hien hanh (= gia)
       - diesel_lag1m/2m/3m : gia diesel cach day 1/2/3 THANG
            (lay gia ky gan nhat <= moc thoi gian do, qua merge_asof backward)
       - diesel_chg_1m  : % thay doi current so voi 1 thang truoc (kenh nhanh)
       - diesel_chg_3m  : % thay doi current so voi 3 thang truoc
            (KENH TRUYEN DAN CHI PHI CHAM: dau -> cuoc van tai -> gia, tre vai thang)
     => De WALK-FORWARD ABLATION tu chon do tre nao dong gop, thay vi ap dat.
     LUU Y: diesel rat muot -> cac lag tuong quan cao (da cong tuyen); dung
     ablation giu lai 1-2 dang tot nhat (thuong diesel_chg_3m + diesel_current),
     KHONG nhoi tat ca vao model cuoi.
  6) LOC output ve [start_year, end_year] (mac dinh 2020-2025). Tinh feature tren
     TOAN BO chuoi TRUOC roi moi loc -> dau 2020 van co lookback lag 1-3 thang (ve 2019).
  7) Xuat diesel_clean.csv (chuoi KHONG DEU + available_from + feature ablation).

CACH GHEP SANG GIA NGAY (chong look-ahead) - forward-fill bang merge_asof backward:
    df_daily = pd.merge_asof(
        df_daily.sort_values('Ngay'),
        diesel[['available_from','diesel_current','diesel_chg_3m']]
            .sort_values('available_from'),
        left_on='Ngay', right_on='available_from', direction='backward')
    # -> moi ngay lay gia diesel cua ky gan nhat (dang bac thang), khong noi suy.
Chay: python3 31_process_diesel.py --input diesel_merged_long.csv --out diesel_clean.csv
"""
import argparse
import numpy as np, pandas as pd


def load(path):
    d = pd.read_csv(path)
    d['ngay_ky'] = pd.to_datetime(d['ngay_ky'])
    d = d.sort_values('ngay_ky').drop_duplicates('ngay_ky').reset_index(drop=True)
    return d


def hampel_mask(x, window=7, k=5.0):
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    thr = k * 1.4826 * mad
    return (x - med).abs() > thr.replace(0, np.nan)


def month_lag_value(out, months):
    """Gia diesel tai moc (ngay_ky - 'months' thang), lay ky gan nhat <= moc do."""
    base = out[['ngay_ky', 'gia']].sort_values('ngay_ky').rename(
        columns={'ngay_ky': 'src_ngay', 'gia': 'val'})
    tmp = out[['ngay_ky']].copy()
    tmp['target'] = tmp['ngay_ky'] - pd.DateOffset(months=months)
    tmp = tmp.sort_values('target')
    merged = pd.merge_asof(tmp, base, left_on='target', right_on='src_ngay',
                           direction='backward')
    return out[['ngay_ky']].merge(
        merged[['ngay_ky', 'val']], on='ngay_ky', how='left')['val']


def process(d, pct_thr=12.0, pub_buffer_days=0, hampel_window=7, hampel_k=5.0):
    out = d.copy()
    out['gia_ret'] = out['gia'].pct_change()
    out['gia_lag1'] = out['gia'].shift(1)
    # 3) danh dau outlier (khong xoa)
    by_pct = out['gia_ret'].abs() * 100 > pct_thr
    by_hampel = hampel_mask(out['gia'], window=hampel_window, k=hampel_k).fillna(False)
    out['outlier_flag'] = (by_pct | by_hampel).astype(int)
    # 4) ngay kha dung
    out['available_from'] = out['ngay_ky'] + pd.Timedelta(days=pub_buffer_days)
    # 5) feature cho ablation (tinh theo ngay thuc)
    out['diesel_current'] = out['gia']
    out['diesel_lag1m'] = month_lag_value(out, 1).values
    out['diesel_lag2m'] = month_lag_value(out, 2).values
    out['diesel_lag3m'] = month_lag_value(out, 3).values
    out['diesel_chg_1m'] = out['diesel_current'] / out['diesel_lag1m'] - 1
    out['diesel_chg_3m'] = out['diesel_current'] / out['diesel_lag3m'] - 1
    out = out[['ngay_ky', 'available_from', 'gia', 'nguon',
               'gia_lag1', 'gia_ret', 'outlier_flag',
               'diesel_current', 'diesel_lag1m', 'diesel_lag2m', 'diesel_lag3m',
               'diesel_chg_1m', 'diesel_chg_3m']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/Processing/Fuel/diesel_merged_long.csv')
    ap.add_argument('--out', default='data/Processing/Fuel/diesel_feature.csv')
    ap.add_argument('--pct_thr', type=float, default=12.0)
    ap.add_argument('--pub_buffer_days', type=int, default=0)
    ap.add_argument('--hampel_window', type=int, default=7)
    ap.add_argument('--hampel_k', type=float, default=5.0)
    ap.add_argument('--start_year', type=int, default=2020)
    ap.add_argument('--end_year', type=int, default=2025)
    ap.add_argument('--daily_out', default='data/Processing/Fuel/diesel_daily.csv',
                    help='File gia theo NGAY (forward-fill bac thang); de trong de bo qua')
    ap.add_argument('--daily_freq', default='B',
                    help="Lich ngay: 'B'=ngay lam viec, 'D'=moi ngay lich")
    args = ap.parse_args()

    d = load(args.input)
    # Tinh feature tren TOAN BO chuoi truoc (de lag/chg 1-3 thang dau 2020 van co
    # du lookback ve 2019), SAU DO moi loc output ve [start_year, end_year].
    out = process(d, pct_thr=args.pct_thr, pub_buffer_days=args.pub_buffer_days,
                  hampel_window=args.hampel_window, hampel_k=args.hampel_k)

    # === (TUY CHON) Fill ra GIA HANG NGAY bang forward-fill (bac thang) ===
    # merge_asof backward theo 'available_from' tren TOAN BO chuoi (truoc khi loc nam)
    # -> moi ngay (ke ca dau 2020) lay gia ky gan nhat DA cong bo. KHONG noi suy:
    # trong 1 ky dieu chinh gia di ngang. Sau do moi loc ve [start_year, end_year].
    if args.daily_out:
        bdays = pd.date_range(f'{args.start_year}-01-01', f'{args.end_year}-12-31',
                              freq=args.daily_freq)
        feats = (out[['available_from', 'diesel_current', 'diesel_chg_1m', 'diesel_chg_3m']]
                 .sort_values('available_from'))
        daily = pd.merge_asof(pd.DataFrame({'Ngay': bdays}).sort_values('Ngay'),
                              feats, left_on='Ngay', right_on='available_from',
                              direction='backward')
        daily = daily.rename(columns={'diesel_current': 'diesel'})[
            ['Ngay', 'diesel', 'diesel_chg_1m', 'diesel_chg_3m']]
        daily.to_csv(args.daily_out, index=False)
        print('Daily (%s): %s -> %s | rows %d -> %s'
              % (args.daily_freq, daily.Ngay.min().date(), daily.Ngay.max().date(),
                 len(daily), args.daily_out))

    yr = out['ngay_ky'].dt.year
    out = out[(yr >= args.start_year) & (yr <= args.end_year)].reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print('Loc:', args.start_year, '-', args.end_year)
    print('Range:', out.ngay_ky.min().date(), '->', out.ngay_ky.max().date(), '| rows', len(out))
    print('So outlier danh dau:', int(out['outlier_flag'].sum()))
    print('\nVi du feature ablation (6 dong cuoi):')
    print(out[['ngay_ky', 'diesel_current', 'diesel_lag1m', 'diesel_lag3m',
               'diesel_chg_1m', 'diesel_chg_3m']].tail(6).to_string(index=False))
    print('\nSaved:', args.out)


if __name__ == '__main__':
    main()
