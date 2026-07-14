import numpy as np
import pandas as pd

df = pd.read_csv('/data/gia_cafe_master_full.csv')
y = df['Gia_target'].dropna().values.astype(float)
n = len(y)
print(f"Series: Gia_target  n={n}  min={y.min():.0f}  max={y.max():.0f}  mean={y.mean():.0f}")

# ── ADF (hand-coded, AIC lag selection) ──────────────────────────────────────
def adf_test(series, max_lags=15, regression='c'):
    y = series
    n = len(y)
    dy = np.diff(y)
    best_aic, best_p = np.inf, 0
    for p in range(0, min(max_lags+1, len(dy)-10)):
        T = len(dy) - p
        y_lag = y[p:n-1]
        cols = [y_lag]
        for i in range(1, p+1):
            cols.append(dy[p-i:len(dy)-i])
        cols.append(np.ones(T))
        if regression == 'ct':
            cols.append(np.arange(p+1, p+T+1, dtype=float))
        X = np.column_stack(cols)
        yy = dy[p:]
        beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ beta
        aic = T * np.log(np.sum(resid**2)/T) + 2*X.shape[1]
        if aic < best_aic:
            best_aic, best_p = aic, p
    p = best_p
    T = len(dy) - p
    y_lag = y[p:n-1]
    cols = [y_lag]
    for i in range(1, p+1):
        cols.append(dy[p-i:len(dy)-i])
    cols.append(np.ones(T))
    if regression == 'ct':
        cols.append(np.arange(p+1, p+T+1, dtype=float))
    X = np.column_stack(cols)
    yy = dy[p:]
    beta, _, _, _ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    sigma2 = np.sum(resid**2) / (T - X.shape[1])
    XtX_inv = np.linalg.pinv(X.T @ X)
    se = np.sqrt(sigma2 * np.diag(XtX_inv))
    stat = beta[0] / se[0]
    if regression == 'c':
        cv = {'1%': -3.430, '5%': -2.862, '10%': -2.567}
    else:
        cv = {'1%': -3.960, '5%': -3.410, '10%': -3.127}
    if stat < cv['1%']:   p_str = '<0.01'
    elif stat < cv['5%']: p_str = '<0.05'
    elif stat < cv['10%']:p_str = '<0.10'
    else:                 p_str = '>0.10'
    return stat, p_str, p, cv

# ── KPSS (Kwiatkowski et al. 1992) ───────────────────────────────────────────
def kpss_test(series, regression='c'):
    y = series
    n = len(y)
    lags = int(np.floor(4*(n/100)**0.25))
    if regression == 'c':
        resid = y - np.mean(y)
        cv = {'10%':0.347,'5%':0.463,'2.5%':0.574,'1%':0.739}
    else:
        t = np.arange(n, dtype=float)
        X = np.column_stack([np.ones(n), t])
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        cv = {'10%':0.119,'5%':0.146,'2.5%':0.176,'1%':0.216}
    S = np.cumsum(resid)
    s2 = np.sum(resid**2)/n
    for j in range(1, lags+1):
        w = 1 - j/(lags+1)
        s2 += 2*w*np.sum(resid[j:]*resid[:-j])/n
    stat = np.sum(S**2)/(n**2*s2)
    if stat > cv['1%']:   conc = 'KHÔNG dừng (bác bỏ H0 ở 1%)'
    elif stat > cv['5%']: conc = 'KHÔNG dừng (bác bỏ H0 ở 5%)'
    elif stat > cv['10%']:conc = 'KHÔNG dừng (bác bỏ H0 ở 10%)'
    else:                 conc = 'Dừng (không bác bỏ H0)'
    return stat, conc, lags, cv

# ═══════════════════════════════════════════════════════
print('\n' + '='*55)
print('CHUỖI GỐC (mức giá — LEVEL)')
print('='*55)

s, p, lag, cv = adf_test(y, regression='c')
print(f'\nADF (hằng số):          stat={s:8.4f}  p≈{p}  lags={lag}')
print(f'  CV  1%={cv["1%"]}  5%={cv["5%"]}  10%={cv["10%"]}')
print(f'  → {"KHÔNG dừng" if s > cv["5%"] else "Dừng"} theo ADF')

s2, p2, lag2, cv2 = adf_test(y, regression='ct')
print(f'\nADF (hằng số + xu thế): stat={s2:8.4f}  p≈{p2}  lags={lag2}')
print(f'  CV  1%={cv2["1%"]}  5%={cv2["5%"]}  10%={cv2["10%"]}')
print(f'  → {"KHÔNG dừng" if s2 > cv2["5%"] else "Dừng quanh xu thế"} theo ADF')

ks, kconc, klags, kcv = kpss_test(y, regression='c')
print(f'\nKPSS (level):           stat={ks:8.4f}  lags={klags}')
print(f'  CV  5%={kcv["5%"]}  1%={kcv["1%"]}')
print(f'  → {kconc}')

kts, ktconc, ktlags, ktcv = kpss_test(y, regression='ct')
print(f'\nKPSS (trend):           stat={kts:8.4f}  lags={ktlags}')
print(f'  CV  5%={ktcv["5%"]}  1%={ktcv["1%"]}')
print(f'  → {ktconc}')

# ═══════════════════════════════════════════════════════
dy = np.diff(y)
print('\n' + '='*55)
print('SAI PHÂN BẬC 1 (delta = giá_t - giá_{t-1})')
print('='*55)

ds, dp, dlag, dcv = adf_test(dy, regression='c')
print(f'\nADF (hằng số):          stat={ds:8.4f}  p≈{dp}  lags={dlag}')
print(f'  CV  1%={dcv["1%"]}  5%={dcv["5%"]}  10%={dcv["10%"]}')
print(f'  → {"Dừng" if ds < dcv["5%"] else "KHÔNG dừng"} theo ADF')

dks, dkconc, dklags, dkcv = kpss_test(dy, regression='c')
print(f'\nKPSS (level):           stat={dks:8.4f}  lags={dklags}')
print(f'  CV  5%={dkcv["5%"]}  1%={dkcv["1%"]}')
print(f'  → {dkconc}')

# ═══════════════════════════════════════════════════════
print('\n' + '='*55)
print('KẾT LUẬN')
print('='*55)
level_nd_adf  = s  > cv['5%']
level_nd_kpss = ks > kcv['5%']
diff_d_adf    = ds < dcv['5%']
diff_d_kpss   = dks < dkcv['5%']
print(f'Gốc KHÔNG dừng  — ADF : {"✓" if level_nd_adf  else "✗"}')
print(f'Gốc KHÔNG dừng  — KPSS: {"✓" if level_nd_kpss else "✗"}')
print(f'Delta DỪng      — ADF : {"✓" if diff_d_adf    else "✗"}')
print(f'Delta DỪng      — KPSS: {"✓" if diff_d_kpss   else "✗"}')
if level_nd_adf and level_nd_kpss and diff_d_adf and diff_d_kpss:
    print('\n✅ XÁC NHẬN I(1): chuỗi gốc không dừng, sai phân bậc 1 dừng (ADF + KPSS đồng thuận)')
elif level_nd_adf and diff_d_adf:
    print('\n✅ LIKELY I(1): ADF xác nhận cả hai chiều')
elif not level_nd_adf:
    print('\n⚠️  ADF bác bỏ nghiệm đơn vị ở chuỗi gốc — cần xem xét lại')
else:
    print('\n⚠️  Kết quả không đồng nhất — cần phân tích thêm')
