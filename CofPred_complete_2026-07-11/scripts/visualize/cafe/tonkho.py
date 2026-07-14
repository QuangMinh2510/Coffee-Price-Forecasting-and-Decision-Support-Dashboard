# -*- coding: utf-8 -*-
"""
EDA TON KHO (ending stocks) ca phe Viet Nam.
  - Input: tonkho_vietnam.xlsx (sheet 'data'), don vi: nghin bao 60kg, theo nien vu.
  - Chi VE BIEU DO, khong sua/xuat du lieu (phan do o 27_process_tonkho.py).
Chay: python3 26_eda_tonkho.py --input tonkho_vietnam.xlsx --outdir tonkho_out
"""
import argparse, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    d = pd.read_excel(path, sheet_name='data')
    d.columns = ['Nien_vu', 'Quoc_gia', 'TonKho_nghin_bao', 'DVT']
    d['Nam_ket_thuc'] = d['Nien_vu'].str.split('/').str[1].astype(int)
    d['TonKho_tan'] = d['TonKho_nghin_bao'] * 60.0
    return d.sort_values('Nam_ket_thuc').reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=r"data/Raw/coffe/tonkho_vietnam.xlsx")
    ap.add_argument('--outdir', default=r"reports/figures/cafe/tonkho")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = load(args.input)
    print('Nien vu:', d['Nien_vu'].iloc[0], '->', d['Nien_vu'].iloc[-1], '| n', len(d))
    print('Ton kho (nghin bao): min %d (%s) | max %d (%s)' % (
        d['TonKho_nghin_bao'].min(), d.loc[d['TonKho_nghin_bao'].idxmin(), 'Nien_vu'],
        d['TonKho_nghin_bao'].max(), d.loc[d['TonKho_nghin_bao'].idxmax(), 'Nien_vu']))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    a = ax[0]
    a.bar(d['Nam_ket_thuc'], d['TonKho_nghin_bao'], color='#d35400', alpha=0.85)
    a.plot(d['Nam_ket_thuc'], d['TonKho_nghin_bao'], color='#2c3e50', lw=1.5, marker='o', ms=4)
    a.set_title('Ton kho cuoi ky ca phe Viet Nam (nghin bao 60kg)', fontweight='bold')
    a.set_xlabel('Nam ket thuc nien vu'); a.set_ylabel('nghin bao'); a.grid(alpha=0.3, axis='y')
    mn = d.loc[d['TonKho_nghin_bao'].idxmin()]
    a.annotate('Thap ky luc %d/%d' % (mn['Nam_ket_thuc']-1, mn['Nam_ket_thuc']),
               xy=(mn['Nam_ket_thuc'], mn['TonKho_nghin_bao']),
               xytext=(mn['Nam_ket_thuc']-3, 2500), fontsize=8,
               arrowprops=dict(arrowstyle='->', color='#c0392b'))
    a2 = ax[1]
    a2.bar(d['Nam_ket_thuc'], d['TonKho_tan']/1000, color='#16a085', alpha=0.85)
    a2.set_title('Ton kho quy ra nghin tan', fontweight='bold')
    a2.set_xlabel('Nam ket thuc nien vu'); a2.set_ylabel('nghin tan'); a2.grid(alpha=0.3, axis='y')
    fig.suptitle('EDA TON KHO CA PHE VIET NAM (2011-2024)', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pth = os.path.join(args.outdir, 'tonkho_eda.png')
    fig.savefig(pth, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('Saved:', pth)


if __name__ == '__main__':
    main()
