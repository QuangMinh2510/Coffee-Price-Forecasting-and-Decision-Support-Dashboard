# -*- coding: utf-8 -*-
"""
EDA cho 2 file ngoai sinh tu Investing:
  - london_robusta_price.csv  (USD/tonne)
  - usd_vnd.csv               (ty gia USD/VND, dinh dang Investing)
Tao 1 hinh tong quan 2x2 + in thong ke + tuong quan co do tre voi gia noi dia.
Chay: python3 22_eda_external.py --london london_robusta_price.csv --fx usd_vnd.csv \
        --domestic gia_cafe_processed.csv --outdir ext_out
"""
import argparse, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_london(p):
    d = pd.read_csv(p)
    d['Date'] = pd.to_datetime(d['Date'])
    d = d.sort_values('Date').reset_index(drop=True)
    d = d.rename(columns={'Price': 'london_usd'})
    return d[['Date', 'london_usd']]


def load_fx(p):
    d = pd.read_csv(p)
    d['Date'] = pd.to_datetime(d['Date'], format='%m/%d/%Y')
    for c in ['Price', 'Open', 'High', 'Low']:
        if c in d.columns:
            d[c] = d[c].astype(str).str.replace(',', '', regex=False)
            d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.sort_values('Date').reset_index(drop=True)
    d = d.rename(columns={'Price': 'usdvnd'})
    return d[['Date', 'usdvnd']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--london', default=r'data/Processing/coffe/london_robusta_price.csv')
    ap.add_argument('--fx', default=r'data/Raw/business/USD_VND Historical Data.csv')
    ap.add_argument('--domestic', default=r'data/Processing/coffe/gia_taynguyen_proxy_daily_filled.csv')
    ap.add_argument('--outdir', default=r'reports/figures/cafe/london')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    L = load_london(args.london)
    F = load_fx(args.fx)
    m = pd.merge(L, F, on='Date', how='inner')
    m['london_vnd_kg'] = m['london_usd'] * m['usdvnd'] / 1000.0

    print('LONDON :', L.Date.min().date(), '->', L.Date.max().date(), '| rows', len(L),
          '| USD/tonne', L.london_usd.min(), '-', L.london_usd.max())
    print('USD/VND:', F.Date.min().date(), '->', F.Date.max().date(), '| rows', len(F),
          '| rate', F.usdvnd.min(), '-', F.usdvnd.max())

    # tuong quan co do tre voi gia noi dia
    lag_txt = ''
    dom = None
    if os.path.exists(args.domestic):
        P = pd.read_csv(args.domestic); P['Ngay'] = pd.to_datetime(P['Ngay'])
        # Tim cot gia noi dia: uu tien 'target', sau do 'proxy', roi 'gia_taynguyen'
        tcol = [c for c in P.columns if 'target' in c.lower()]
        if not tcol:
            tcol = [c for c in P.columns if 'proxy' in c.lower() or 'taynguyen' in c.lower()]
        if not tcol:
            # Fallback: lay cot so cuoi cung (khong phai 'Ngay')
            num_cols = P.select_dtypes(include='number').columns.tolist()
            if num_cols:
                tcol = [num_cols[-1]]
        if tcol:
            t = tcol[0]
            dom = P[['Ngay', t]].merge(m, left_on='Ngay', right_on='Date', how='left').sort_values('Ngay')
            rd = dom[t].pct_change()
            lags = {k: rd.corr(dom['london_usd'].pct_change().shift(k)) for k in range(4)}
            lag_txt = ' | '.join([f'lag{k}={v:.2f}' for k, v in lags.items()])
            print('Corr return noi dia vs London return:', lag_txt)

    fig, ax = plt.subplots(2, 2, figsize=(16, 10))
    # (1) London USD/tonne
    a = ax[0, 0]
    a.plot(L.Date, L.london_usd, color='#8e44ad', lw=1.2)
    a.set_title('London Robusta (USD/tonne)', fontweight='bold'); a.grid(alpha=0.3)
    # (2) USD/VND
    a = ax[0, 1]
    a.plot(F.Date, F.usdvnd, color='#16a085', lw=1.2)
    a.set_title('Ty gia USD/VND', fontweight='bold'); a.grid(alpha=0.3)
    # (3) London quy VND/kg vs noi dia
    a = ax[1, 0]
    a.plot(m.Date, m.london_vnd_kg, color='#8e44ad', lw=1.0, label='London quy VND/kg')
    if dom is not None:
        a.plot(dom.Ngay, dom[tcol[0]], color='#c0392b', lw=1.2, label='Gia noi dia (TB2)')
    a.set_title('London quy VND/kg vs gia noi dia', fontweight='bold')
    a.legend(fontsize=8); a.grid(alpha=0.3)
    # (4) Lag correlation
    a = ax[1, 1]
    if dom is not None:
        ks = list(range(4))
        vals = [rd.corr(dom['london_usd'].pct_change().shift(k)) for k in ks]
        bars = a.bar([str(k) for k in ks], vals, color=['#bdc3c7', '#27ae60', '#bdc3c7', '#bdc3c7'])
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width()/2, v + 0.01, f'{v:.2f}', ha='center', fontsize=9)
        a.set_title('Corr: return noi dia(t) vs London return(t-k)', fontweight='bold')
        a.set_xlabel('do tre k (ngay)'); a.set_ylabel('correlation'); a.grid(alpha=0.3, axis='y')
        a.axhline(0, color='k', lw=0.8)
    fig.suptitle('EDA DU LIEU NGOAI SINH: London Robusta + USD/VND', fontsize=15, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path = os.path.join(args.outdir, 'external_eda.png')
    fig.savefig(path, dpi=120, bbox_inches='tight'); plt.close(fig)
    print('Saved:', path)


if __name__ == '__main__':
    main()
