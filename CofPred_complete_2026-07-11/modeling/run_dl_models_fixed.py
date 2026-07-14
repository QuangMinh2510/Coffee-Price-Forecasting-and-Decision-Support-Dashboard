#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark NHITS, NBEATSx va Chronos; xuat prediction va checkpoint NeuralForecast."""
from __future__ import annotations

import argparse
import os
import time
import warnings
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from export_preds import save_preds
except ImportError:
    def save_preds(model_name, horizon, dates, y_true, y_pred, anchor=None, outdir="results/preds"):
        outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(model_name))
        frame = pd.DataFrame({"date": pd.to_datetime(list(dates)), "y_true": y_true, "y_pred": y_pred})
        if anchor is not None: frame["anchor"] = anchor
        path = outdir / f"preds_{safe}_h{int(horizon)}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        return path

TARGET = "Gia_target"
HORIZONS = [1, 5, 21, 63]
INITIAL_FRAC = 0.6
STEP = 5
ALIGN_FEATURES = [
    "target_lag1", "target_lag2", "target_lag3", "target_ret_lag1",
    "MA5", "MA10", "std5", "dayofweek", "month", "london_vnd_kg_lag1",
    "usdvnd_lag1", "diesel", "diesel_chg_1m", "diesel_chg_3m",
    "Luong_lag1m", "rain_90d", "oni", "waterbal_90d",
]


def now(): return time.strftime("%H:%M:%S")


def discover_project_root(explicit: str | None = None) -> Path:
    candidates = []
    if explicit: candidates.append(Path(explicit).expanduser())
    if os.getenv("COFPRED_ROOT"): candidates.append(Path(os.environ["COFPRED_ROOT"]).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parent])
    if os.name == "nt": candidates.append(Path(r"E:\FPT\AI\SEM8_AI\DAP391m\project\CofPred"))
    for base in candidates:
        for candidate in [base, *base.parents]:
            if (candidate / "data" / "processed" / "gia_cafe_master_full.csv").exists():
                return candidate.resolve()
    return Path(explicit).expanduser().resolve() if explicit else Path.cwd().resolve()


def resolve_data_path(root: Path, explicit: str | None) -> Path:
    path = Path(explicit).expanduser() if explicit else root / "data" / "processed" / "gia_cafe_master_full.csv"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay CSV: {path}. Dung --data <file.csv> hoac COFPRED_ROOT.")
    return path


def detect_date_column(columns) -> str:
    for col in ("date", "Date", "Ngay", "ngay", "DATE"):
        if col in columns: return col
    raise KeyError("Khong tim thay cot ngay: date/Date/Ngay/ngay/DATE")


def load_data(path: Path):
    df = pd.read_csv(path)
    date_col = detect_date_column(df.columns)
    if TARGET not in df.columns: raise KeyError(f"Khong co cot {TARGET}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last").reset_index(drop=True)
    df["prev_price"] = df[TARGET].shift(1)
    have = [c for c in ALIGN_FEATURES if c in df.columns]
    if have: df[have] = df[have].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna(subset=[TARGET, "prev_price"] + have).reset_index(drop=True)
    if len(df) < 100: raise ValueError(f"Chi con {len(df)} dong hop le; du lieu qua it.")
    print(f"[load] {path}")
    print(f"[load] giu {len(df)}/{before} dong; date_col={date_col}; align_features={len(have)}")
    return df, date_col


def dm_test(error_model, error_naive, horizon=1):
    e1, e2 = np.asarray(error_model, float), np.asarray(error_naive, float)
    valid = np.isfinite(e1) & np.isfinite(e2)
    d = np.abs(e1[valid]) - np.abs(e2[valid])
    n = len(d)
    if n < 2: return float("nan"), float("nan")
    var = float(np.var(d, ddof=0))
    for k in range(1, min(horizon, n)):
        if n - k > 1:
            var += 2 * (1 - k / horizon) * float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
    if not np.isfinite(var) or var <= 0: return float("nan"), float("nan")
    dm = float(d.mean() / sqrt(var / n))
    p = float(2 * (1 - 0.5 * (1 + erf(abs(dm) / sqrt(2)))))
    return dm, p


def make_nf_models(horizon, max_steps, accelerator, devices):
    from neuralforecast.models import NBEATSx, NHITS
    common = dict(
        h=horizon, input_size=max(2 * horizon, 30), max_steps=max_steps,
        scaler_type="robust", accelerator=accelerator, devices=devices,
        enable_progress_bar=False, enable_model_summary=False, random_seed=42,
    )
    return [
        NHITS(**common, n_blocks=[1, 1, 1], alias="NHITS"),
        NBEATSx(**common, alias="NBEATSx"),
    ]


def save_final_nf_models(long_df, horizon, max_steps, accelerator, devices, out_dir: Path):
    from neuralforecast import NeuralForecast
    models = make_nf_models(horizon, max_steps, accelerator, devices)
    nf = NeuralForecast(models=models, freq=1)
    val_size = min(max(horizon, 1), max(1, len(long_df) // 10))
    nf.fit(df=long_df, val_size=val_size)
    path = out_dir / f"nhits_nbeatsx_h{horizon}"
    path.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(path), model_index=None, overwrite=True, save_dataset=True)
    return path


def append_metrics(rows, tag, horizon, y_true, y_pred, y_naive):
    model_mae = float(np.mean(np.abs(y_true - y_pred)))
    naive_mae = float(np.mean(np.abs(y_true - y_naive)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    da = float(np.mean(np.sign(y_true - y_naive) == np.sign(y_pred - y_naive)))
    dm, p = dm_test(y_true - y_pred, y_true - y_naive, horizon)
    row = {
        "model": tag, "h": horizon, "n": int(len(y_true)),
        "MAE": round(model_mae, 1), "MAE_naive": round(naive_mae, 1),
        "dMAE_pct": round(100 * (model_mae / naive_mae - 1), 2) if naive_mae > 0 else np.nan,
        "RMSE": round(rmse, 1), "DA": round(da, 3),
        "DM": round(dm, 3) if np.isfinite(dm) else np.nan,
        "DM_p": round(p, 4) if np.isfinite(p) else np.nan,
    }
    rows.append(row)
    return row


def run_nhits_nbeatsx(df, date_col, horizons, rows, *, initial_frac, step, refit_every, max_steps, save_models, results_root):
    try:
        import torch
        from neuralforecast import NeuralForecast
    except ImportError as exc:
        raise RuntimeError("Thieu neuralforecast. Cai: pip install neuralforecast torch") from exc

    n = len(df)
    start = int(n * initial_frac)
    series = df[TARGET].astype(float).to_numpy()
    long_df = pd.DataFrame({"unique_id": "cafe", "ds": np.arange(n, dtype=int), "y": series})
    dates = pd.to_datetime(df[date_col]).to_numpy()
    price_at = {int(i): float(series[i]) for i in range(n)}

    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1
    if use_gpu: torch.set_float32_matmul_precision("high")
    print(f"[device] CUDA={use_gpu} -> accelerator={accelerator}, devices={devices}")

    preds_dir = results_root / "preds" / "dl"
    checkpoint_dir = results_root / "saved_models" / "neuralforecast"
    preds_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for horizon in horizons:
        if horizon <= 0 or horizon >= n:
            print(f"[skip] horizon khong hop le: {horizon}")
            continue
        n_windows = len(range(start, n - horizon, step))
        if n_windows <= 0:
            print(f"[skip] h={horizon}: khong co window")
            continue
        print(f"\n[{now()}] NHITS/NBEATSx h={horizon}, n_windows={n_windows}")
        try:
            nf = NeuralForecast(models=make_nf_models(horizon, max_steps, accelerator, devices), freq=1)
            kwargs = dict(df=long_df, n_windows=n_windows, step_size=step, verbose=False)
            if refit_every > 0: kwargs["refit"] = refit_every
            try:
                cv = nf.cross_validation(**kwargs)
            except TypeError:
                kwargs.pop("refit", None)
                print("[warn] Phien ban NeuralForecast khong nhan refit; dung mac dinh.")
                cv = nf.cross_validation(**kwargs)

            cv_h = cv.sort_values(["cutoff", "ds"]).groupby("cutoff", sort=False).tail(1).reset_index(drop=True)
            for tag in ("NHITS", "NBEATSx"):
                if tag not in cv_h.columns:
                    print(f"[warn] Khong co cot {tag}; cot hien co: {list(cv_h.columns)}")
                    continue
                subset = cv_h.loc[cv_h[tag].notna() & cv_h["y"].notna() & cv_h["cutoff"].notna()].copy()
                cutoffs = pd.to_numeric(subset["cutoff"], errors="coerce").to_numpy()
                ok = np.isfinite(cutoffs)
                subset = subset.iloc[np.flatnonzero(ok)].copy(); cutoffs = cutoffs[ok].astype(int)
                in_bounds = (cutoffs >= 0) & (cutoffs + horizon < n)
                subset = subset.iloc[np.flatnonzero(in_bounds)].copy(); cutoffs = cutoffs[in_bounds]
                y_true = subset["y"].to_numpy(float)
                y_pred = subset[tag].to_numpy(float)
                y_naive = np.asarray([price_at[c] for c in cutoffs], float)
                save_preds(tag, horizon, dates[cutoffs + horizon], y_true, y_pred, anchor=y_naive, outdir=preds_dir)
                row = append_metrics(rows, tag, horizon, y_true, y_pred, y_naive)
                print(f"  {tag:8s}: MAE={row['MAE']}, Naive={row['MAE_naive']}, DA={row['DA']}, n={row['n']}")

            if save_models:
                path = save_final_nf_models(long_df, horizon, max_steps, accelerator, devices, checkpoint_dir)
                print(f"  [saved model] {path}")
        except Exception as exc:
            print(f"  [LOI] NHITS/NBEATSx h={horizon}: {type(exc).__name__}: {exc}")


def load_chronos_pipeline(model_id, device):
    import torch
    from chronos import ChronosPipeline
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    errors = []
    for kwargs in ({"torch_dtype": dtype}, {"dtype": dtype}, {}):
        try:
            return ChronosPipeline.from_pretrained(model_id, device_map=device, **kwargs)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("Khong load duoc Chronos: " + " | ".join(errors))


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def run_chronos(df, date_col, horizons, rows, *, initial_frac, step, context, model_id, results_root):
    try:
        import torch
        import chronos  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Thieu chronos-forecasting. Cai: pip install chronos-forecasting torch") from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[{now()}] Load Chronos '{model_id}' tren {device.upper()}")
    pipe = load_chronos_pipeline(model_id, device)

    series = df[TARGET].astype(float).to_numpy()
    dates = pd.to_datetime(df[date_col]).to_numpy()
    n = len(series)
    start = int(n * initial_frac)
    preds_dir = results_root / "preds" / "dl"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for horizon in horizons:
        if horizon <= 0 or horizon >= n:
            print(f"[skip] horizon khong hop le: {horizon}")
            continue
        y_true, y_pred, y_naive, origins = [], [], [], []
        t0 = time.time()
        try:
            for d in sorted(range(n - 1 - horizon, start - 1, -step)):
                ctx = torch.tensor(series[max(0, d - context + 1): d + 1], dtype=torch.float32)
                forecast = pipe.predict(context=ctx, prediction_length=horizon)
                first = forecast[0]
                arr = tensor_to_numpy(first)
                # Shape thuong la [num_samples, prediction_length].
                median_path = np.quantile(arr, 0.5, axis=0) if arr.ndim >= 2 else arr
                pred = float(np.asarray(median_path).reshape(-1)[-1])
                y_pred.append(pred)
                y_true.append(float(series[d + horizon]))
                y_naive.append(float(series[d]))
                origins.append(d)

            yt = np.asarray(y_true, float); yp = np.asarray(y_pred, float); yn = np.asarray(y_naive, float)
            origins_arr = np.asarray(origins, int)
            save_preds("Chronos", horizon, dates[origins_arr + horizon], yt, yp, anchor=yn, outdir=preds_dir)
            row = append_metrics(rows, "Chronos", horizon, yt, yp, yn)
            print(f"  Chronos h={horizon}: MAE={row['MAE']}, Naive={row['MAE_naive']}, DA={row['DA']}, n={row['n']}, time={time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"  [LOI] Chronos h={horizon}: {type(exc).__name__}: {exc}")


def save_comparisons(rows, results_root: Path):
    out_dir = results_root / "DL_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    dl_path = out_dir / "dl_comparison.csv"
    out.to_csv(dl_path, index=False, encoding="utf-8-sig")

    ts_path = results_root / "TS_model" / "ts_model_comparison.csv"
    all_path = out_dir / "all_comparison.csv"
    if ts_path.exists():
        stat = pd.read_csv(ts_path)
        combined = pd.concat([stat, out], ignore_index=True, sort=False)
        sort_cols = [c for c in ("h", "MAE") if c in combined.columns]
        if sort_cols: combined = combined.sort_values(sort_cols)
        combined.to_csv(all_path, index=False, encoding="utf-8-sig")
    else:
        out.to_csv(all_path, index=False, encoding="utf-8-sig")
    return out, dl_path, all_path


def parse_args():
    parser = argparse.ArgumentParser(description="Chay NHITS, NBEATSx, Chronos va xuat model")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--initial", type=float, default=INITIAL_FRAC)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--refit-every", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--chronos-model", default="amazon/chronos-bolt-base")
    parser.add_argument("--skip-neuralforecast", action="store_true")
    parser.add_argument("--skip-chronos", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    args = parser.parse_args()
    if not 0.1 <= args.initial < 1: parser.error("--initial phai trong [0.1,1)")
    if args.step < 1: parser.error("--step phai >=1")
    if args.max_steps < 1: parser.error("--max-steps phai >=1")
    if args.context < 2: parser.error("--context phai >=2")
    return args


def main():
    args = parse_args()
    root = discover_project_root(args.project_root)
    df, date_col = load_data(resolve_data_path(root, args.data))
    rows = []
    results_root = root / "results"

    if not args.skip_neuralforecast:
        try:
            run_nhits_nbeatsx(
                df, date_col, args.horizons, rows, initial_frac=args.initial,
                step=args.step, refit_every=args.refit_every, max_steps=args.max_steps,
                save_models=not args.no_save_models, results_root=results_root,
            )
        except Exception as exc:
            print(f"[LOI NeuralForecast] {type(exc).__name__}: {exc}")

    if not args.skip_chronos:
        try:
            run_chronos(
                df, date_col, args.horizons, rows, initial_frac=args.initial,
                step=args.step, context=args.context, model_id=args.chronos_model,
                results_root=results_root,
            )
        except Exception as exc:
            print(f"[LOI Chronos] {type(exc).__name__}: {exc}")

    out, dl_path, all_path = save_comparisons(rows, results_root)
    print(f"\n[done] {dl_path}")
    print(f"[done] {all_path}")
    print(out.to_string(index=False) if not out.empty else "Khong co ket qua.")


if __name__ == "__main__":
    main()
