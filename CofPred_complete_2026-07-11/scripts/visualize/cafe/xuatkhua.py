# -*- coding: utf-8 -*-
"""
EDA XUAT KHAU ca phe Viet Nam theo THANG (2009-2025).
  - Input: xuatkhau_cafe.xlsx (sheet 'data')
  - Cot: Nam, Thang, Luong (nghin tan), Kim Ngach (trieu USD)
  - Don gia XK (USD/tan) = Kim Ngach*1e6 / (Luong*1000)
  - Chi VE BIEU DO, khong xuat du lieu (phan do o 29_process_xuatkhau.py).
Chay: python3 28_eda_xuatkhau.py --input xuatkhau_cafe.xlsx --outdir xuatkhau_out
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(path):
    d = pd.read_excel(path, sheet_name='data')
    d.columns = ['Nam', 'Thang', 'Luong_nghin_tan', 'KimNgach_trieu_USD']
    d = d.sort_values(['Nam', 'Thang']).reset_index(drop=True)
    d['Ngay'] = pd.to_datetime(dict(year=d.Nam, month=d.Thang, day=1))
    d['DonGia_USD_tan'] = d['KimNgach_trieu_USD'] * 1e6 / (d['Luong_nghin_tan'] * 1000)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=r"data/Raw/coffe/xuatkhua_cafe.xlsx")
    ap.add_argument('--outdir', default=r"reports/figures/cafe/xuatkhua")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = load(args.input)
    print('Range:', d.Ngay.min().date(), '->', d.Ngay.max().date(), '| rows', len(d))
    print('Don gia XK USD/tan: min %.0f max %.0f' % (d.DonGia_USD_tan.min(), d.DonGia_USD_tan.max()))

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Luong XK theo thang
    a = ax[0, 0]
    a.plot(d['Ngay'], d['Luong_nghin_tan'], color='#27ae60', lw=1.3)
    a.set_title('Luong xuat khau theo thang (nghin tan)', fontweight='bold')
    a.set_xlabel('Thoi gian'); a.set_ylabel('nghin tan'); a.grid(alpha=0.3)

    # (2) Kim ngach theo thang
    a = ax[0, 1]
    a.plot(d['Ngay'], d['KimNgach_trieu_USD'], color='#2980b9', lw=1.3)
    a.set_title('Kim ngach xuat khau theo thang (trieu USD)', fontweight='bold')
    a.set_xlabel('Thoi gian'); a.set_ylabel('trieu USD'); a.grid(alpha=0.3)

    # (3) Don gia XK USD/tan
    a = ax[1, 0]
    a.plot(d['Ngay'], d['DonGia_USD_tan'], color='#c0392b', lw=1.5)
    a.set_title('Don gia xuat khau (USD/tan) - bam sat gia the gioi', fontweight='bold')
    a.set_xlabel('Thoi gian'); a.set_ylabel('USD/tan'); a.grid(alpha=0.3)

    # (4) Mua vu: luong XK trung binh theo thang 1-12
    a = ax[1, 1]
    m = d.groupby('Thang')['Luong_nghin_tan'].mean()
    a.bar(m.index, m.values, color='#e67e22', alpha=0.85)
    a.set_xticks(range(1, 13))
    a.set_title('Mua vu: luong XK trung binh theo thang (2009-2025)', fontweight='bold')
    a.set_xlabel('Thang'); a.set_ylabel('nghin tan (TB)'); a.grid(alpha=0.3, axis='y')
    a.annotate('Cao diem sau thu hoach\n(T1-T4)', xy=(3, m.loc[3]), xytext=(6, m.max()*0.95),
               fontsize=8, color='#d35400',
               arrowprops=dict(arrowstyle='->', color='#d35400'))

    fig.suptitle('EDA XUAT KHAU CA PHE VIET NAM THEO THANG (2009-2025)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pth = os.path.join(args.outdir, 'xuatkhau_eda.png')
    fig.savefig(pth, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('Saved:', pth)


if __name__ == '__main__':
    main()
