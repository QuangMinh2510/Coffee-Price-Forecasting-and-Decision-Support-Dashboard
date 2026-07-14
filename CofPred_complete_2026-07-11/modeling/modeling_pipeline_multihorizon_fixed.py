#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark ML da chan troi cho du bao gia ca phe.

Sua chinh:
- Khong con loi escape duong dan Windows.
- Tu dong tim project root, file CSV va cot ngay.
- Khong bat buoc phai co export_preds.py o mot duong dan hard-code.
- Xu ly an toan ARIMAX khi khong co exogenous feature.
- Bao ve metric MASE/DM khi mau so bang 0.
- Luu prediction, bang ket qua, bieu do va model ML tot nhat bang joblib.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    XGBRegressor = None
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGB = True
except Exception:
    LGBMRegressor = None
    HAS_LGB = False

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    HAS_SM = True
except Exception:
    SARIMAX = None
    HAS_SM = False

try:
    from export_preds import save_preds
except ImportError:
    # Cho phep chay khi chi copy mot file vao project.
    def save_preds(model_name, horizon, dates, y_true, y_pred, anchor=None, outdir="results/preds"):
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in str(model_name))
        df = pd.DataFrame({
            "date": pd.to_datetime(list(dates)),
            "y_true": np.asarray(list(y_true), float),
            "y_pred": np.asarray(list(y_pred), float),
        })
        if anchor is not None:
            df["anchor"] = np.asarray(list(anchor), float)
        path = outdir / f"preds_{safe}_h{int(horizon)}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

TARGET = "Gia_target"
HORIZON_NAMES = {1: "1 ngay", 5: "1 tuan", 10: "2 tuan", 21: "1 thang", 63: "1 quy"}
AR = ["target_lag1", "target_lag2", "target_lag3", "target_ret_lag1", "MA5", "MA10", "std5", "dayofweek", "month"]
EXO = ["london_vnd_kg_lag1", "usdvnd_lag1", "diesel", "diesel_chg_1m", "diesel_chg_3m", "Luong_lag1m", "rain_90d", "oni", "waterbal_90d"]
SUPPLY = ["area_tn", "prod_tn", "yield_tn", "tonkho_tan", "dongia_lag1m", "dongia_ret_lag1m"]
WANTED_FEATURES = AR + EXO + SUPPLY


def discover_project_root(explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_root = os.getenv("COFPRED_ROOT")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend([Path.cwd(), Path(__file__).resolve().parent])
    if os.name == "nt":
        candidates.append(Path(r"E:\FPT\AI\SEM8_AI\DAP391m\project\CofPred"))

    checked: set[Path] = set()
    for base in candidates:
        for candidate in [base, *base.parents]:
            candidate = candidate.resolve()
            if candidate in checked:
                continue
            checked.add(candidate)
            if (candidate / "data" / "processed" / "gia_cafe_master_full.csv").exists():
                return candidate
    return Path(explicit).expanduser().resolve() if explicit else Path.cwd().resolve()


def resolve_data_path(project_root: Path, explicit: str | None = None) -> Path:
    path = Path(explicit).expanduser() if explicit else project_root / "data" / "processed" / "gia_cafe_master_full.csv"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay du lieu: {path}\n"
            "Hay dung --data <duong_dan_csv> hoac dat bien moi truong COFPRED_ROOT."
        )
    return path


def detect_date_column(columns) -> str:
    for name in ("date", "Date", "Ngay", "ngay", "DATE"):
        if name in columns:
            return name
    raise KeyError("Khong tim thay cot ngay. Can mot trong: date, Date, Ngay, ngay, DATE")


def load_data(csv_path: Path):
    raw = pd.read_csv(csv_path)
    date_col = detect_date_column(raw.columns)
    if TARGET not in raw.columns:
        raise KeyError(f"Khong tim thay cot target '{TARGET}' trong file CSV")

    raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
    raw = raw.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last")
    raw = raw.set_index(date_col)

    features = [c for c in WANTED_FEATURES if c in raw.columns]
    missing = [c for c in WANTED_FEATURES if c not in raw.columns]
    if not features:
        raise ValueError("Khong co feature nao trong danh sach 24 feature du kien.")
    if missing:
        print(f"[CANH BAO] Thieu {len(missing)} feature: {missing}")

    numeric_cols = [TARGET] + features
    raw[numeric_cols] = raw[numeric_cols].apply(pd.to_numeric, errors="coerce")
    data = raw[numeric_cols].copy()
    data["prev_price"] = data[TARGET].shift(1)
    before = len(data)
    data = data.dropna(subset=numeric_cols + ["prev_price"])
    if len(data) < 100:
        raise ValueError(f"Chi con {len(data)} dong hop le sau dropna; qua it de benchmark.")

    weekends = int((data.index.dayofweek >= 5).sum())
    print(f"[load] {csv_path}")
    print(f"[load] giu {len(data)}/{before} dong, {len(features)} feature, weekend={weekends}")
    return data, features


def evaluation_origins(n: int, horizon: int, initial: float, step: int) -> list[int]:
    start = int(n * initial)
    return sorted(range(n - 1 - horizon, start - 1, -step))


def mae(actual, pred) -> float:
    return float(np.mean(np.abs(np.asarray(actual) - np.asarray(pred))))


def rmse(actual, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred)) ** 2)))


def smape(actual, pred) -> float:
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    denom = (np.abs(actual) + np.abs(pred)) / 2.0
    valid = denom > 0
    return float(np.mean(np.abs(actual[valid] - pred[valid]) / denom[valid]) * 100) if valid.any() else float("nan")


def mase(actual, pred, y_train) -> float:
    scale = float(np.mean(np.abs(np.diff(np.asarray(y_train, float)))))
    return float(mae(actual, pred) / scale) if np.isfinite(scale) and scale > 0 else float("nan")


def directional_accuracy(actual, pred, anchor) -> float:
    return float(np.mean(np.sign(np.asarray(actual) - np.asarray(anchor)) == np.sign(np.asarray(pred) - np.asarray(anchor))) * 100)


def diebold_mariano(actual, pred_model, pred_naive, horizon=1) -> float:
    actual = np.asarray(actual, float)
    p1 = np.asarray(pred_model, float)
    p2 = np.asarray(pred_naive, float)
    d = (actual - p1) ** 2 - (actual - p2) ** 2
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return float("nan")
    var = float(np.var(d, ddof=0))
    for k in range(1, min(horizon, n)):
        if n - k > 1:
            cov = float(np.cov(d[k:], d[:-k], ddof=0)[0, 1])
            var += 2.0 * (1.0 - k / horizon) * cov
    if not np.isfinite(var) or var <= 0:
        return float("nan")
    return float(d.mean() / np.sqrt(var / n))


def model_factories(horizon: int) -> Dict[str, Callable[[], object]]:
    cfg = {
        1: dict(ridge=0.5, lasso=0.0005, enet=0.0005, l1=0.3, rf_est=400, rf_leaf=2, rf_depth=None, xgb_est=600, xgb_lr=0.02, xgb_depth=5, lgb_est=700, lgb_lr=0.02, lgb_leaves=40, svr_c=20, svr_eps=0.05, knn=5),
        5: dict(ridge=1.0, lasso=0.001, enet=0.001, l1=0.5, rf_est=400, rf_leaf=3, rf_depth=None, xgb_est=500, xgb_lr=0.03, xgb_depth=4, lgb_est=600, lgb_lr=0.03, lgb_leaves=31, svr_c=10, svr_eps=0.1, knn=10),
        21: dict(ridge=5.0, lasso=0.01, enet=0.01, l1=0.5, rf_est=500, rf_leaf=5, rf_depth=10, xgb_est=400, xgb_lr=0.05, xgb_depth=3, lgb_est=500, lgb_lr=0.05, lgb_leaves=20, svr_c=5, svr_eps=0.5, knn=15),
        63: dict(ridge=10.0, lasso=0.1, enet=0.1, l1=0.7, rf_est=500, rf_leaf=10, rf_depth=8, xgb_est=300, xgb_lr=0.05, xgb_depth=3, lgb_est=400, lgb_lr=0.05, lgb_leaves=15, svr_c=1, svr_eps=1.0, knn=20),
    }.get(horizon)
    if cfg is None:
        cfg = model_factories(5)  # pragma: no cover
        return cfg

    factories: Dict[str, Callable[[], object]] = {
        "Ridge": lambda: Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=cfg["ridge"]))]),
        "Lasso": lambda: Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=cfg["lasso"], max_iter=50000))]),
        "ElasticNet": lambda: Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(alpha=cfg["enet"], l1_ratio=cfg["l1"], max_iter=50000))]),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=cfg["rf_est"], min_samples_leaf=cfg["rf_leaf"], max_depth=cfg["rf_depth"], n_jobs=-1, random_state=42),
        "SVR": lambda: TransformedTargetRegressor(
            regressor=Pipeline([("scaler", StandardScaler()), ("model", SVR(C=cfg["svr_c"], epsilon=cfg["svr_eps"], kernel="rbf"))]),
            transformer=StandardScaler(),
        ),
        "kNN": lambda: Pipeline([("scaler", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=cfg["knn"]))]),
    }
    if HAS_XGB:
        factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=cfg["xgb_est"], learning_rate=cfg["xgb_lr"], max_depth=cfg["xgb_depth"],
            subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1,
        )
    if HAS_LGB:
        factories["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=cfg["lgb_est"], learning_rate=cfg["lgb_lr"], num_leaves=cfg["lgb_leaves"],
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, n_jobs=-1, verbose=-1,
        )
    return factories


def walk_forward(factory, X, future, anchor, *, initial, step, horizon, mode):
    records = []
    for d in evaluation_origins(len(X), horizon, initial, step):
        train_end = d - horizon + 1
        if train_end <= 20:
            continue
        x_train = X.iloc[:train_end]
        y_train = (future - anchor).iloc[:train_end] if mode == "delta" else future.iloc[:train_end]
        valid = y_train.notna() & np.isfinite(y_train.to_numpy(float))
        if valid.sum() <= 20:
            continue
        model = factory()
        model.fit(x_train.loc[valid], y_train.loc[valid])
        raw = float(model.predict(X.iloc[[d]])[0])
        pred = float(anchor.iloc[d]) + raw if mode == "delta" else raw
        records.append((X.index[d], float(future.iloc[d]), float(anchor.iloc[d]), pred))
    return pd.DataFrame(records, columns=["origin_date", "y_true", "anchor", "pred"]).set_index("origin_date")


def run_arimax(price, X, future, anchor, *, horizon, initial, step, exog_cols):
    records = []
    for d in evaluation_origins(len(X), horizon, initial, step):
        try:
            train_y = price.iloc[: d + 1]
            if exog_cols:
                train_x = X[exog_cols].iloc[: d + 1]
                result = SARIMAX(train_y, exog=train_x, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                future_x = np.tile(X[exog_cols].iloc[d].to_numpy(float), (horizon, 1))
                pred = float(np.asarray(result.forecast(steps=horizon, exog=future_x))[-1])
            else:
                result = SARIMAX(train_y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                pred = float(np.asarray(result.forecast(steps=horizon))[-1])
        except Exception:
            pred = float(anchor.iloc[d])
        records.append((X.index[d], float(future.iloc[d]), float(anchor.iloc[d]), pred))
    return pd.DataFrame(records, columns=["origin_date", "y_true", "anchor", "pred"]).set_index("origin_date")


def get_feature_importance(model, features):
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]
    values = getattr(estimator, "feature_importances_", None)
    if values is None:
        values = getattr(estimator, "coef_", None)
    if values is None:
        return None
    values = np.asarray(values).reshape(-1)
    return pd.Series(values, index=features) if len(values) == len(features) else None


def fit_and_save_best_trainable(res, factories, X, future, anchor, mode, horizon, features, model_dir):
    candidates = [name for name in res["Model"] if name in factories]
    if not candidates:
        print("[CANH BAO] Khong co model ML nao de luu.")
        return None, None
    best_name = candidates[0]
    target = (future - anchor) if mode == "delta" else future
    valid = target.notna() & np.isfinite(target.to_numpy(float))
    model = factories[best_name]()
    model.fit(X.loc[valid], target.loc[valid])

    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"best_{best_name}_h{horizon}_{mode}.joblib"
    joblib.dump({
        "model": model,
        "model_name": best_name,
        "horizon": horizon,
        "target_mode": mode,
        "features": list(features),
        "target": TARGET,
        "inference_note": "Neu target_mode='delta', cong output model voi gia hien tai (anchor).",
    }, path)
    return best_name, path


def run(args) -> None:
    project_root = discover_project_root(args.project_root)
    csv_path = resolve_data_path(project_root, args.data)
    data, features = load_data(csv_path)

    X = data[features].copy()
    price = data[TARGET].copy()
    results_root = project_root / "results"
    ml_dir = results_root / "ML_model"
    preds_dir = results_root / "preds"
    saved_dir = results_root / "saved_models"
    ml_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    registry = []
    print(f"[deps] xgboost={HAS_XGB} lightgbm={HAS_LGB} statsmodels={HAS_SM}")

    for horizon in args.horizons:
        if horizon <= 0 or horizon >= len(X):
            print(f"[skip] horizon khong hop le: {horizon}")
            continue
        for mode in args.modes:
            print(f"\n=== h={horizon} ({HORIZON_NAMES.get(horizon, str(horizon))}) | mode={mode} ===")
            future = price.shift(-horizon)
            anchor = price.copy()
            origins = evaluation_origins(len(X), horizon, args.initial, args.step)
            if not origins:
                print("[skip] Khong co diem danh gia.")
                continue

            naive = pd.DataFrame(
                [(X.index[d], float(future.iloc[d]), float(anchor.iloc[d]), float(anchor.iloc[d])) for d in origins],
                columns=["origin_date", "y_true", "anchor", "pred"],
            ).set_index("origin_date")
            blocks = {"Naive": naive}

            if HAS_SM and not args.skip_arimax:
                blocks["ARIMAX(1,1,1)"] = run_arimax(
                    price, X, future, anchor, horizon=horizon, initial=args.initial, step=args.step,
                    exog_cols=[c for c in EXO if c in features],
                )

            factories = model_factories(horizon)
            if args.models:
                requested = set(args.models)
                unknown = sorted(requested - set(factories))
                if unknown:
                    print(f"[CANH BAO] Model khong ton tai/khong cai thu vien: {unknown}")
                factories = {name: factory for name, factory in factories.items() if name in requested}
                if not factories:
                    print("[skip] Khong con model ML nao sau khi loc --models.")
            for name, factory in factories.items():
                try:
                    blocks[name] = walk_forward(factory, X, future, anchor, initial=args.initial, step=args.step, horizon=horizon, mode=mode)
                    print(f"[ok] {name}: {len(blocks[name])} diem")
                except Exception as exc:
                    print(f"[loi] {name}: {type(exc).__name__}: {exc}")

            eval_index = naive.index
            train_scale = price.iloc[: int(len(price) * args.initial)].to_numpy(float)
            rows = []
            for name, block in blocks.items():
                aligned = block.reindex(eval_index)
                actual = aligned["y_true"].to_numpy(float)
                pred = aligned["pred"].to_numpy(float)
                anc = aligned["anchor"].to_numpy(float)
                valid = np.isfinite(actual) & np.isfinite(pred) & np.isfinite(anc)
                if not valid.any():
                    continue
                rows.append({
                    "Model": name,
                    "MAE": mae(actual[valid], pred[valid]),
                    "RMSE": rmse(actual[valid], pred[valid]),
                    "sMAPE(%)": smape(actual[valid], pred[valid]),
                    "MASE": mase(actual[valid], pred[valid], train_scale),
                    "DA(%)": directional_accuracy(actual[valid], pred[valid], anc[valid]),
                    "n": int(valid.sum()),
                })
                target_dates = [price.index[price.index.get_loc(origin) + horizon] for origin in aligned.index[valid]]
                save_preds(name, horizon, target_dates, actual[valid], pred[valid], anchor=anc[valid], outdir=preds_dir / mode)

            if not rows:
                print("[skip] Khong model nao co ket qua hop le.")
                continue
            res = pd.DataFrame(rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)
            res_path = ml_dir / f"model_comparison_h{horizon}_{mode}.csv"
            res.to_csv(res_path, index=False, encoding="utf-8-sig")
            print(res.to_string(index=False))

            best_overall = str(res.iloc[0]["Model"])
            best_block = blocks[best_overall].reindex(eval_index)
            plt.figure(figsize=(13, 5))
            plt.plot(eval_index, naive["y_true"], label="Thuc te gia(t+h)", linewidth=1.2)
            plt.plot(eval_index, best_block["pred"], label=f"Du bao ({best_overall})", linewidth=1.0)
            plt.plot(eval_index, naive["pred"], label="Naive", linewidth=0.8, alpha=0.65)
            plt.title(f"Forecast vs actual - {best_overall} - h={horizon} - {mode}")
            plt.ylabel("Gia (VND/kg)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(ml_dir / f"forecast_best_h{horizon}_{mode}.png", dpi=140)
            plt.close()

            best_trainable, model_path = fit_and_save_best_trainable(
                res, factories, X, future, anchor, mode, horizon, features, saved_dir
            )
            if model_path:
                print(f"[saved model] {model_path}")
                registry.append({"horizon": horizon, "mode": mode, "best_overall": best_overall, "saved_model": best_trainable, "path": str(model_path)})

                bundle = joblib.load(model_path)
                fi = get_feature_importance(bundle["model"], features)
                if fi is not None:
                    fi.sort_values().plot(kind="barh", figsize=(8, 7), title=f"Feature importance - {best_trainable} - h={horizon}")
                    plt.tight_layout()
                    plt.savefig(ml_dir / f"feature_importance_h{horizon}_{mode}.png", dpi=140)
                    plt.close()
                    fi.sort_values(ascending=False).to_csv(ml_dir / f"feature_importance_h{horizon}_{mode}.csv", header=["importance"], encoding="utf-8-sig")

            naive_pred = naive["pred"].to_numpy(float)
            best_pred = best_block["pred"].to_numpy(float)
            actual = naive["y_true"].to_numpy(float)
            valid = np.isfinite(actual) & np.isfinite(best_pred) & np.isfinite(naive_pred)
            dm = diebold_mariano(actual[valid], best_pred[valid], naive_pred[valid], horizon=horizon)
            print(f"DM ({best_overall} vs Naive) = {dm:.3f}" if np.isfinite(dm) else "DM khong tinh duoc")

    if registry:
        pd.DataFrame(registry).to_csv(saved_dir / "model_registry.csv", index=False, encoding="utf-8-sig")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark va xuat model ML multi-horizon")
    parser.add_argument("--project-root", default=None, help="Thu muc goc CofPred")
    parser.add_argument("--data", default=None, help="Duong dan gia_cafe_master_full.csv")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 21, 63])
    parser.add_argument("--modes", nargs="+", choices=["delta", "level"], default=["delta", "level"])
    parser.add_argument("--initial", type=float, default=0.6)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--skip-arimax", action="store_true", help="Bo qua ARIMAX de chay nhanh")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Chi chay model duoc chon, vd: --models Ridge RandomForest XGBoost")
    args = parser.parse_args()
    if not 0.1 <= args.initial < 1.0:
        parser.error("--initial phai nam trong [0.1, 1.0)")
    if args.step < 1:
        parser.error("--step phai >= 1")
    return args


if __name__ == "__main__":
    run(parse_args())
