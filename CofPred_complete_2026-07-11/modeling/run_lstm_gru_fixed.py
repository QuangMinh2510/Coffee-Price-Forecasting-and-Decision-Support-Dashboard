#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark LSTM/GRU bang NeuralForecast va luu checkpoint da train."""
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


def now() -> str:
    return time.strftime("%H:%M:%S")


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
    if TARGET not in df.columns:
        raise KeyError(f"Khong co cot {TARGET}")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last").reset_index(drop=True)
    df["prev_price"] = df[TARGET].shift(1)
    have = [c for c in ALIGN_FEATURES if c in df.columns]
    if have:
        df[have] = df[have].apply(pd.to_numeric, errors="coerce")
    before = len(df)
    df = df.dropna(subset=[TARGET, "prev_price"] + have).reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"Chi con {len(df)} dong hop le; du lieu qua it.")
    print(f"[load] {path}")
    print(f"[load] giu {len(df)}/{before} dong; date_col={date_col}; align_features={len(have)}")
    return df, date_col


def dm_test(error_model, error_naive, horizon=1):
    e1 = np.asarray(error_model, float)
    e2 = np.asarray(error_naive, float)
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


def make_model(model_class, tag, horizon, max_steps, accelerator, devices):
    return model_class(
        h=horizon,
        input_size=max(2 * horizon, 30),
        max_steps=max_steps,
        encoder_hidden_size=128,
        scaler_type="robust",
        alias=tag,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=False,
        enable_model_summary=False,
        random_seed=42,
    )


def save_final_neural_model(model_class, tag, horizon, long_df, *, max_steps, accelerator, devices, out_dir: Path):
    from neuralforecast import NeuralForecast
    model = make_model(model_class, tag, horizon, max_steps, accelerator, devices)
    nf = NeuralForecast(models=[model], freq=1)
    val_size = min(max(horizon, 1), max(1, len(long_df) // 10))
    nf.fit(df=long_df, val_size=val_size)
    path = out_dir / f"{tag.lower()}_h{horizon}"
    path.mkdir(parents=True, exist_ok=True)
    nf.save(path=str(path), model_index=None, overwrite=True, save_dataset=True)
    return path


def run_lstm_gru(df, date_col, horizons, *, initial_frac, step, refit_every, max_steps, save_models, results_root):
    try:
        import torch
        from neuralforecast import NeuralForecast
        from neuralforecast.models import GRU, LSTM
    except ImportError as exc:
        raise RuntimeError("Thieu neuralforecast. Cai: pip install neuralforecast torch") from exc

    n = len(df)
    start = int(n * initial_frac)
    series = df[TARGET].astype(float).to_numpy()
    long_df = pd.DataFrame({"unique_id": "cafe", "ds": np.arange(n, dtype=int), "y": series})

    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1
    if use_gpu:
        torch.set_float32_matmul_precision("high")
    print(f"[device] CUDA={use_gpu} -> accelerator={accelerator}, devices={devices}")

    preds_dir = results_root / "preds" / "dl"
    compare_dir = results_root / "DL_model"
    checkpoints_dir = results_root / "saved_models" / "neuralforecast"
    for p in (preds_dir, compare_dir, checkpoints_dir): p.mkdir(parents=True, exist_ok=True)

    rows = []
    model_specs = [("LSTM", LSTM), ("GRU", GRU)]
    date_values = pd.to_datetime(df[date_col]).to_numpy()
    price_at = {int(i): float(series[i]) for i in range(n)}

    for horizon in horizons:
        if horizon <= 0 or horizon >= n:
            print(f"[skip] horizon khong hop le: {horizon}")
            continue
        n_windows = len(range(start, n - horizon, step))
        if n_windows <= 0:
            print(f"[skip] h={horizon}: khong co window")
            continue
        print(f"\n[{now()}] h={horizon}, n_windows={n_windows}, refit={refit_every}")

        for tag, model_class in model_specs:
            t0 = time.time()
            try:
                model = make_model(model_class, tag, horizon, max_steps, accelerator, devices)
                nf = NeuralForecast(models=[model], freq=1)
                cv_kwargs = dict(df=long_df, n_windows=n_windows, step_size=step, verbose=False)
                if refit_every > 0:
                    cv_kwargs["refit"] = refit_every
                try:
                    cv = nf.cross_validation(**cv_kwargs)
                except TypeError:
                    cv_kwargs.pop("refit", None)
                    print(f"[warn] NeuralForecast hien tai khong nhan refit; chay cross_validation mac dinh cho {tag}.")
                    cv = nf.cross_validation(**cv_kwargs)

                if tag not in cv.columns:
                    raise KeyError(f"Khong tim thay cot output '{tag}'. Cot co san: {list(cv.columns)}")
                cv_h = cv.sort_values(["cutoff", "ds"]).groupby("cutoff", sort=False).tail(1).reset_index(drop=True)
                valid = cv_h[tag].notna() & cv_h["y"].notna() & cv_h["cutoff"].notna()
                cv_h = cv_h.loc[valid].copy()
                cutoffs = pd.to_numeric(cv_h["cutoff"], errors="coerce").to_numpy()
                ok = np.isfinite(cutoffs)
                cv_h = cv_h.iloc[np.flatnonzero(ok)].copy()
                cutoffs = cutoffs[ok].astype(int)
                in_bounds = (cutoffs >= 0) & (cutoffs + horizon < n)
                cv_h = cv_h.iloc[np.flatnonzero(in_bounds)].copy()
                cutoffs = cutoffs[in_bounds]

                y_true = cv_h["y"].to_numpy(float)
                y_pred = cv_h[tag].to_numpy(float)
                y_naive = np.asarray([price_at[c] for c in cutoffs], float)
                target_dates = date_values[cutoffs + horizon]
                save_preds(tag, horizon, target_dates, y_true, y_pred, anchor=y_naive, outdir=preds_dir)

                model_mae = float(np.mean(np.abs(y_true - y_pred)))
                naive_mae = float(np.mean(np.abs(y_true - y_naive)))
                model_rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
                da = float(np.mean(np.sign(y_true - y_naive) == np.sign(y_pred - y_naive)))
                dm, p_value = dm_test(y_true - y_pred, y_true - y_naive, horizon=horizon)
                rows.append({
                    "model": tag, "h": horizon, "n": len(y_true),
                    "MAE": round(model_mae, 1), "MAE_naive": round(naive_mae, 1),
                    "dMAE_pct": round(100 * (model_mae / naive_mae - 1), 2) if naive_mae > 0 else np.nan,
                    "RMSE": round(model_rmse, 1), "DA": round(da, 3),
                    "DM": round(dm, 3) if np.isfinite(dm) else np.nan,
                    "DM_p": round(p_value, 4) if np.isfinite(p_value) else np.nan,
                })
                pd.DataFrame(rows).to_csv(compare_dir / "lstm_gru_comparison.csv", index=False, encoding="utf-8-sig")
                print(f"  {tag:5s}: MAE={model_mae:.1f}, Naive={naive_mae:.1f}, DA={da:.3f}, n={len(y_true)}, time={time.time()-t0:.1f}s")

                if save_models:
                    saved_path = save_final_neural_model(
                        model_class, tag, horizon, long_df, max_steps=max_steps,
                        accelerator=accelerator, devices=devices, out_dir=checkpoints_dir,
                    )
                    print(f"  [saved model] {saved_path}")
            except Exception as exc:
                print(f"  [LOI] {tag} h={horizon}: {type(exc).__name__}: {exc}")

    out = pd.DataFrame(rows)
    out.to_csv(compare_dir / "lstm_gru_comparison.csv", index=False, encoding="utf-8-sig")
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Chay LSTM/GRU va xuat checkpoint")
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--horizons", type=int, nargs="+", default=HORIZONS)
    parser.add_argument("--initial", type=float, default=INITIAL_FRAC)
    parser.add_argument("--step", type=int, default=STEP)
    parser.add_argument("--refit-every", type=int, default=1, help="0=khong refit; 1=refit moi window")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--no-save-models", action="store_true", help="Khong fit lai full data de luu checkpoint")
    args = parser.parse_args()
    if not 0.1 <= args.initial < 1: parser.error("--initial phai trong [0.1,1)")
    if args.step < 1: parser.error("--step phai >=1")
    if args.max_steps < 1: parser.error("--max-steps phai >=1")
    return args


def main():
    args = parse_args()
    root = discover_project_root(args.project_root)
    data_path = resolve_data_path(root, args.data)
    df, date_col = load_data(data_path)
    try:
        out = run_lstm_gru(
            df, date_col, args.horizons, initial_frac=args.initial, step=args.step,
            refit_every=args.refit_every, max_steps=args.max_steps,
            save_models=not args.no_save_models, results_root=root / "results",
        )
    except Exception as exc:
        print(f"[LOI] {type(exc).__name__}: {exc}")
        raise SystemExit(1) from None
    print("\n[done]")
    print(out.to_string(index=False) if not out.empty else "Khong co ket qua.")


if __name__ == "__main__":
    main()
