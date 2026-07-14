#!/usr/bin/env python3
"""Train and export CofPred models for the Streamlit dashboard.

The script uses a chronological holdout set, reports out-of-sample metrics,
then refits every requested model on all usable rows and saves a joblib bundle.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
TARGET = "Gia_target"
DATE_CANDIDATES = ("Ngay", "date", "Date", "ngay", "DATE")
FEATURES = [
    "target_lag1", "target_lag2", "target_lag3", "target_ret_lag1",
    "MA5", "MA10", "std5", "dayofweek", "month",
    "london_vnd_kg_lag1", "usdvnd_lag1", "diesel",
    "diesel_chg_1m", "diesel_chg_3m", "Luong_lag1m", "rain_90d",
    "oni", "waterbal_90d", "area_tn", "prod_tn", "yield_tn",
    "tonkho_tan", "dongia_lag1m", "dongia_ret_lag1m",
]
DEFAULT_MODELS = ["LinearRegression", "RandomForest", "XGBoost", "LightGBM"]


def detect_date_column(columns) -> str:
    for candidate in DATE_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(f"Khong tim thay cot ngay. Can mot trong: {DATE_CANDIDATES}")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def load_data(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay du lieu: {path}")

    frame = pd.read_csv(path)
    date_col = detect_date_column(frame.columns)
    if TARGET not in frame.columns:
        raise KeyError(f"Du lieu khong co cot target '{TARGET}'")

    available = [feature for feature in FEATURES if feature in frame.columns]
    missing = [feature for feature in FEATURES if feature not in frame.columns]
    if not available:
        raise ValueError("Khong tim thay feature nao de train model.")
    if missing:
        print(f"[canh bao] Thieu {len(missing)} feature: {missing}")

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    numeric = [TARGET, *available]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = (
        frame.dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(date_col, keep="last")
        .reset_index(drop=True)
    )
    return frame, date_col, available


def make_model(name: str, rf_estimators: int):
    if name == "LinearRegression":
        return LinearRegression()
    if name == "RandomForest":
        return RandomForestRegressor(
            n_estimators=rf_estimators,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
    if name == "XGBoost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Thieu xgboost. Chay: pip install xgboost") from exc
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
        )
    if name == "LightGBM":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("Thieu lightgbm. Chay: pip install lightgbm") from exc
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"Model khong ho tro: {name}")


def metrics(actual: np.ndarray, predicted: np.ndarray, anchor: np.ndarray) -> dict[str, float]:
    error = actual - predicted
    denom = np.where(np.abs(actual) > 1e-9, np.abs(actual), np.nan)
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_pct": float(np.nanmean(np.abs(error) / denom) * 100),
        "DA_pct": float(np.mean(np.sign(actual - anchor) == np.sign(predicted - anchor)) * 100),
    }


def train(args: argparse.Namespace) -> pd.DataFrame:
    data_path = Path(args.data).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    prediction_dir = ROOT / "results" / "preds" / "web"
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    frame, date_col, features = load_data(data_path)
    rows: list[dict] = []

    for horizon in args.horizons:
        if horizon < 1 or horizon >= len(frame):
            print(f"[bo qua] horizon khong hop le: {horizon}")
            continue

        work = frame[[date_col, TARGET, *features]].copy()
        work["future_price"] = work[TARGET].shift(-horizon)
        work["future_date"] = work[date_col].shift(-horizon)
        work = work.dropna(subset=[TARGET, "future_price", *features]).reset_index(drop=True)
        if len(work) < 100:
            print(f"[bo qua] h={horizon}: chi co {len(work)} dong hop le")
            continue

        X = work[features]
        anchor = work[TARGET].to_numpy(float)
        actual = work["future_price"].to_numpy(float)
        y = actual - anchor if args.mode == "delta" else actual
        split = int(len(work) * (1.0 - args.test_size))
        split = min(max(split, 50), len(work) - 20)

        for model_name in args.models:
            try:
                model = make_model(model_name, args.rf_estimators)
            except RuntimeError as exc:
                print(f"[bo qua] {model_name}: {exc}")
                continue

            model.fit(X.iloc[:split], y[:split])
            raw_test = np.asarray(model.predict(X.iloc[split:]), dtype=float)
            predicted = anchor[split:] + raw_test if args.mode == "delta" else raw_test
            score = metrics(actual[split:], predicted, anchor[split:])

            pred_file = prediction_dir / f"preds_{safe_name(model_name)}_h{horizon}_{args.mode}.csv"
            pd.DataFrame({
                "date": work["future_date"].iloc[split:].dt.strftime("%Y-%m-%d"),
                "y_true": actual[split:],
                "y_pred": predicted,
                "anchor": anchor[split:],
            }).to_csv(pred_file, index=False, encoding="utf-8-sig")

            model.fit(X, y)
            latest = frame.dropna(subset=[TARGET, *features]).iloc[-1]
            latest_x = latest[features].astype(float).to_frame().T
            raw_latest = float(model.predict(latest_x)[0])
            latest_anchor = float(latest[TARGET])
            latest_prediction = latest_anchor + raw_latest if args.mode == "delta" else raw_latest

            model_file = output_dir / f"model_{safe_name(model_name)}_h{horizon}_{args.mode}.joblib"
            bundle = {
                "model": model,
                "model_name": model_name,
                "horizon": horizon,
                "target_mode": args.mode,
                "features": features,
                "target": TARGET,
                "date_column": date_col,
                "trained_until": pd.Timestamp(latest[date_col]).strftime("%Y-%m-%d"),
                "data_path": str(data_path.relative_to(ROOT)) if data_path.is_relative_to(ROOT) else str(data_path),
                "metrics": score,
            }
            joblib.dump(bundle, model_file, compress=3)

            row = {
                "model": model_name,
                "horizon": horizon,
                "mode": args.mode,
                **score,
                "test_rows": len(work) - split,
                "trained_until": bundle["trained_until"],
                "latest_prediction": latest_prediction,
                "model_path": model_file.relative_to(ROOT).as_posix(),
                "prediction_path": pred_file.relative_to(ROOT).as_posix(),
            }
            rows.append(row)
            print(
                f"[da luu] {model_name:16s} h={horizon:<2d} "
                f"MAE={score['MAE']:.1f} RMSE={score['RMSE']:.1f} -> {model_file.name}"
            )

    if not rows:
        raise RuntimeError("Khong model nao duoc train. Kiem tra thu vien va tham so.")

    registry = pd.DataFrame(rows).sort_values(["horizon", "RMSE", "MAE"]).reset_index(drop=True)
    registry.to_csv(output_dir / "model_registry.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "data": str(data_path),
        "rows": int(len(frame)),
        "date_min": str(frame[date_col].min().date()),
        "date_max": str(frame[date_col].max().date()),
        "features": features,
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[hoan tat] Registry: {output_dir / 'model_registry.csv'}")
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train va xuat model cho CofPred Streamlit")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--output-dir", default=str(ROOT / "models"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 21, 63])
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS)
    parser.add_argument("--mode", choices=("delta", "level"), default="delta")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--rf-estimators", type=int, default=300)
    args = parser.parse_args()
    if not 0.05 <= args.test_size <= 0.4:
        parser.error("--test-size phai trong [0.05, 0.4]")
    if args.rf_estimators < 10:
        parser.error("--rf-estimators phai >= 10")
    return args


if __name__ == "__main__":
    train(parse_args())
