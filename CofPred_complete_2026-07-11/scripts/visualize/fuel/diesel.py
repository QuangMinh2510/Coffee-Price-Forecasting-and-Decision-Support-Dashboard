# -*- coding: utf-8 -*-
"""
EDA GIA DAU DIESEL (chi phi dau vao: van chuyen, say, may moc).
  - Input: diesel_merged_long.csv  (cot: ngay_ky, gia [VND/lit], nguon)
  - Lich KHONG DEU: theo ky dieu chinh gia xang dau (~7 hoac 15 ngay/lan).
  - 2 nguon noi tiep nhau (webgia 2017-2018 -> luatvietnam 2018-2026), KHONG trung ngay.
  - Chi VE BIEU DO, khong xuat du lieu (phan do o 31_process_diesel.py).
Chay: python3 30_eda_diesel.py --input diesel_merged_long.csv --outdir diesel_out
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PCT_THR = 12.0  # nguong canh bao bien dong bat thuong giua 2 ky lien tiep (%)


def load(path):
    d = pd.read_csv(path)
    d['ngay_ky'] = pd.to_datetime(d['ngay_ky'])
    d = d.sort_values('ngay_ky').drop_duplicates('ngay_ky').reset_index(drop=True)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='data/Processing/Fuel/diesel_merged_long.csv')
    ap.add_argument('--outdir', default='reports/figures/Fuel/')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = load(args.input)
    d['pct'] = d['gia'].pct_change() * 100
    d['gap'] = d['ngay_ky'].diff().dt.days
    print('Range:', d.ngay_ky.min().date(), '->', d.ngay_ky.max().date(), '| rows', len(d))
    print('Gia VND/lit: min', d.gia.min(), 'max', d.gia.max())

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    # (1) Gia theo thoi gian, to mau theo nguon
    a = ax[0, 0]
    for ng, c in [('webgia', '#7f8c8d'), ('luatvietnam', '#2980b9')]:
        s = d[d['nguon'] == ng]
        a.plot(s['ngay_ky'], s['gia'], '.-', ms=3, lw=0.9, color=c, label=ng)
    a.set_title('Gia diesel theo ky dieu chinh (VND/lit)', fontweight='bold')
    a.set_xlabel('Thoi gian'); a.set_ylabel('VND/lit'); a.grid(alpha=0.3); a.legend()

    # (2) Bien dong % giua cac ky + nguong canh bao
    a = ax[0, 1]
    a.bar(d['ngay_ky'], d['pct'], width=8, color='#16a085')
    a.axhline(PCT_THR, ls='--', color='red', lw=0.8)
    a.axhline(-PCT_THR, ls='--', color='red', lw=0.8)
    a.set_title('Bien dong %% giua 2 ky (nguong +-%.0f%%)' % PCT_THR, fontweight='bold')
    a.set_xlabel('Thoi gian'); a.set_ylabel('%'); a.grid(alpha=0.3)

    # (3) Phan bo khoang cach ngay giua cac ky
    a = ax[1, 0]
    a.hist(d['gap'].dropna(), bins=range(0, 24), color='#e67e22', alpha=0.85)
    a.set_title('Phan bo khoang cach ngay giua cac ky (chu ky ~7/15 ngay)', fontweight='bold')
    a.set_xlabel('So ngay'); a.set_ylabel('So lan'); a.grid(alpha=0.3, axis='y')

    # (4) Zoom bat thuong 2026 (bien dong +-20-31%)
    a = ax[1, 1]
    z = d[(d['ngay_ky'] >= '2025-10-01')]
    a.plot(z['ngay_ky'], z['gia'], 'o-', color='#c0392b', ms=4)
    flag = z[z['pct'].abs() > PCT_THR]
    a.scatter(flag['ngay_ky'], flag['gia'], color='black', zorder=5, s=40, label='|%|>nguong')
    a.set_title('Zoom bat thuong cuoi chuoi (2025-10 -> 2026-05)', fontweight='bold')
    a.set_xlabel('Thoi gian'); a.set_ylabel('VND/lit'); a.grid(alpha=0.3); a.legend()
    # Rotate nhan truc X cho tung axis co ngay thang
    for ax_date in [ax[0,0], ax[0,1], ax[1,1]]:
        ax_date.tick_params(axis='x', labelrotation=30)
        for lbl in ax_date.get_xticklabels():
            lbl.set_ha('right')

    fig.suptitle('EDA GIA DAU DIESEL VIET NAM (2017-2026)', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    pth = os.path.join(args.outdir, 'diesel_eda.png')
    fig.savefig(pth, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('So ky bien dong |%%|>%.0f%%:' % PCT_THR, int((d['pct'].abs() > PCT_THR).sum()))
    print('Saved:', pth)


if __name__ == '__main__':
    main()
