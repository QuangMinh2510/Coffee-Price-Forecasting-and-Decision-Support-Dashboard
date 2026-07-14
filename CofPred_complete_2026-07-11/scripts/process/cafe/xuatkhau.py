# -*- coding: utf-8 -*-
"""
Processing XUAT KHAU ca phe Viet Nam theo THANG.
  - Input: xuatkhau_cafe.xlsx (sheet 'data')
Viec lam (chi LAM SACH, KHONG ve bieu do):
  1) Doi ten cot ve dang khong dau, gon.
  2) Tao cot Ngay (dau thang) tu (Nam, Thang) de join theo thoi gian.
  3) Tinh Don gia XK (USD/tan) = Kim Ngach*1e6 / (Luong*1000).
  4) Sap xep tang dan theo thoi gian.
  5) Tao feature dang LAG-1-THANG (de tranh look-ahead khi join voi gia ngay):
       - Luong_lag1m, DonGia_lag1m      (gia tri cua THANG TRUOC)
       - DonGia_ret_lag1m               (% thay doi don gia thang truoc so voi thang truoc nua)
  6) Tao cot 'available_from' = NGAY DU LIEU THUC SU KHA DUNG
       = dau thang KE TIEP + pub_buffer_days (do tre cong bo).
       So XK thang t chi cong bo dau thang t+1 -> moi dung tu luc do tro di.
  7) Xuat xuatkhau_clean.csv

CACH GHEP SANG GIA NGAY (chong look-ahead) - dung merge_asof backward:
    df_daily = pd.merge_asof(
        df_daily.sort_values('Ngay'),
        xk[['available_from','Luong_nghin_tan','DonGia_USD_tan']]
            .rename(columns={'Luong_nghin_tan':'xk_luong','DonGia_USD_tan':'xk_dongia'})
            .sort_values('available_from'),
        left_on='Ngay', right_on='available_from', direction='backward')
    # -> moi ngay lay so thang gan nhat DA cong bo (dang bac thang), khong noi suy.
Chay: python3 29_process_xuatkhau.py --input xuatkhau_cafe.xlsx --out xuatkhau_clean.csv
"""
import argparse
import numpy as np, pandas as pd


def load(path):
    d = pd.read_excel(path, sheet_name='data')
    d.columns = ['Nam', 'Thang', 'Luong_nghin_tan', 'KimNgach_trieu_USD']
    return d


def process(d, pub_buffer_days=5):
    out = d.sort_values(['Nam', 'Thang']).reset_index(drop=True)
    # 2) cot ngay dau thang
    out['Ngay'] = pd.to_datetime(dict(year=out.Nam, month=out.Thang, day=1))
    # 3) don gia XK USD/tan
    out['DonGia_USD_tan'] = out['KimNgach_trieu_USD'] * 1e6 / (out['Luong_nghin_tan'] * 1000)
    # 5) feature lag 1 thang (chong look-ahead o cap thang)
    out['Luong_lag1m'] = out['Luong_nghin_tan'].shift(1)
    out['DonGia_lag1m'] = out['DonGia_USD_tan'].shift(1)
    out['DonGia_ret_lag1m'] = out['DonGia_USD_tan'].pct_change().shift(1)
    # 6) ngay du lieu thuc su kha dung = dau thang ke tiep + buffer cong bo
    out['available_from'] = (out['Ngay'] + pd.offsets.MonthBegin(1)
                             + pd.Timedelta(days=pub_buffer_days))
    # sap xep cot
    out = out[['Ngay', 'Nam', 'Thang', 'available_from',
               'Luong_nghin_tan', 'KimNgach_trieu_USD', 'DonGia_USD_tan',
               'Luong_lag1m', 'DonGia_lag1m', 'DonGia_ret_lag1m']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=r"data/Raw/coffe/xuatkhua_cafe.xlsx")
    ap.add_argument('--out', default=r"data/Processing/coffe/xuatkhau_clean.csv")
    ap.add_argument('--pub_buffer_days', type=int, default=5,
                    help='So ngay tre cong bo sau dau thang ke tiep (mac dinh 5)')
    args = ap.parse_args()

    d = load(args.input)
    out = process(d, pub_buffer_days=args.pub_buffer_days)
    out.to_csv(args.out, index=False)
    print('Range:', out.Ngay.min().date(), '->', out.Ngay.max().date(), '| rows', len(out))
    print('pub_buffer_days =', args.pub_buffer_days)
    print(out[['Ngay', 'available_from', 'Luong_nghin_tan', 'DonGia_USD_tan',
               'Luong_lag1m', 'DonGia_lag1m']].tail(6).to_string(index=False))
    print('\nSaved:', args.out)


if __name__ == '__main__':
    main()
