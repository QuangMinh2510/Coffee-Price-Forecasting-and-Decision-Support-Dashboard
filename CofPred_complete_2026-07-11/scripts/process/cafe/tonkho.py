# -*- coding: utf-8 -*-
"""
Processing TON KHO (ending stocks) ca phe Viet Nam.
  - Input: tonkho_vietnam.xlsx (sheet 'data'), don vi: nghin bao 60kg, theo nien vu.
Viec lam (chi LAM SACH, KHONG ve bieu do):
  1) Doi ten cot ve dang khong dau, gon.
  2) Tach nien vu '2023/2024' -> Nam_ket_thuc = 2024 (so nguyen, de sap xep/join).
  3) Doi don vi: nghin bao 60kg -> tan  (x60).
  4) Sap xep tang dan theo nam.
  5) Xuat tonkho_clean.csv (Nien_vu, Nam_ket_thuc, TonKho_nghin_bao, TonKho_tan).
Chay: python3 27_process_tonkho.py --input tonkho_vietnam.xlsx --out tonkho_clean.csv
"""
import argparse
import pandas as pd


def load(path):
    d = pd.read_excel(path, sheet_name='data')
    # 1) doi ten cot
    d.columns = ['Nien_vu', 'Quoc_gia', 'TonKho_nghin_bao', 'DVT']
    return d


def process(d):
    out = d.copy()
    # 2) tach nam ket thuc nien vu: '2023/2024' -> 2024
    out['Nam_ket_thuc'] = out['Nien_vu'].str.split('/').str[1].astype(int)
    # 3) doi don vi: nghin bao * 60 = tan
    #    (1 nghin bao = 1000 bao * 60kg = 60000 kg = 60 tan)
    out['TonKho_tan'] = out['TonKho_nghin_bao'] * 60.0
    # 4) sap xep tang dan theo nam
    out = out.sort_values('Nam_ket_thuc').reset_index(drop=True)
    # 5) chon & sap xep cot
    out = out[['Nien_vu', 'Nam_ket_thuc', 'TonKho_nghin_bao', 'TonKho_tan']]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=r"data/Raw/coffe/tonkho_vietnam.xlsx")
    ap.add_argument('--out', default=r"data/Processing/coffe/tonkho_clean.csv")
    args = ap.parse_args()

    d = load(args.input)
    out = process(d)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))
    print('\nSaved:', args.out)


if __name__ == '__main__':
    main()
