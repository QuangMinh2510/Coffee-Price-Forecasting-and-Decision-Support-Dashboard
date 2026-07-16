from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "results"
PREDICTION_DIR = RESULT_DIR / "predictions"
IMPORTANCE_DIR = RESULT_DIR / "feature_importance"

DATE_CANDIDATES = ("Ngay", "date", "Date", "ngay")
TARGET_CANDIDATES = ("Gia_target", "Gia", "price", "Price")

# Đúng 10 model trong các file model_comparison.
PROJECT_MODELS = (
    "Naive",
    "SVR",
    "ElasticNet",
    "Ridge",
    "Lasso",
    "ARIMAX(1,1,1)",
    "RandomForest",
    "XGBoost",
    "LightGBM",
    "kNN",
)

# Đúng nhóm feature thể hiện trong các biểu đồ LightGBM.
FEATURE_ORDER = (
    "target_ret_lag1",
    "std5",
    "target_lag1",
    "usdvnd_lag1",
    "target_lag3",
    "london_vnd_kg_lag1",
    "target_lag2",
    "MA10",
    "MA5",
    "diesel_chg_1m",
    "waterbal_90d",
    "dayofweek",
    "rain_90d",
    "diesel_chg_3m",
    "dongia_lag1m",
    "diesel",
    "dongia_ret_lag1m",
    "Luong_lag1m",
    "oni",
    "month",
    "area_tn",
    "tonkho_tan",
    "prod_tn",
    "yield_tn",
)


def detect_column(columns, candidates):
    """Tìm tên cột ngày hoặc cột giá."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Không tìm thấy cột trong {candidates}")


def find_column(frame: pd.DataFrame, wanted: str) -> str | None:
    """Tìm cột không phân biệt chữ hoa/thường."""
    lookup = {str(column).lower(): str(column) for column in frame.columns}
    return lookup.get(wanted.lower())


def smape(actual, predicted) -> float:
    """Tính Symmetric MAPE."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual) + np.abs(predicted)
    mask = denominator > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(200.0 * np.abs(actual[mask] - predicted[mask]) / denominator[mask]))


def mase(actual, predicted, train_actual) -> float:
    """Tính Mean Absolute Scaled Error."""
    train_actual = np.asarray(train_actual, dtype=float)
    scale = np.mean(np.abs(np.diff(train_actual)))
    if not np.isfinite(scale) or scale <= 1e-9:
        return np.nan
    return float(mean_absolute_error(actual, predicted) / scale)


def directional_accuracy(actual, predicted, anchor) -> float:
    """Đoán đúng hướng tăng/giảm."""
    actual_direction = np.sign(np.asarray(actual) - np.asarray(anchor))
    predicted_direction = np.sign(np.asarray(predicted) - np.asarray(anchor))
    return float(np.mean(actual_direction == predicted_direction) * 100)


def scaled_model(model):
    """Pipeline cho model cần chuẩn hóa."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def tree_model(model):
    """Pipeline cho model cây."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def model_factories():
    """Tạo 8 model học máy; Naive và ARIMAX xử lý riêng."""
    return {
        "SVR": lambda: scaled_model(SVR(C=30.0, epsilon=0.1, gamma="scale")),
        "ElasticNet": lambda: scaled_model(
            ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=20_000)
        ),
        "Ridge": lambda: scaled_model(Ridge(alpha=10.0)),
        "Lasso": lambda: scaled_model(Lasso(alpha=0.001, max_iter=20_000)),
        "RandomForest": lambda: tree_model(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "XGBoost": lambda: tree_model(
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.025,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
        ),
        "LightGBM": lambda: tree_model(
            LGBMRegressor(
                n_estimators=500,
                learning_rate=0.025,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )
        ),
        "kNN": lambda: scaled_model(
            KNeighborsRegressor(n_neighbors=10, weights="distance")
        ),
    }


def prepare_features(frame: pd.DataFrame, date_col: str, target_col: str):
    """Tạo đúng nhóm feature trong ảnh."""
    work = frame.copy()

    # Feature giá quá khứ, không dùng giá tương lai.
    work["target_lag1"] = work[target_col].shift(1)
    work["target_lag2"] = work[target_col].shift(2)
    work["target_lag3"] = work[target_col].shift(3)
    work["target_ret_lag1"] = work[target_col].pct_change().shift(1)
    work["MA5"] = work[target_col].shift(1).rolling(5).mean()
    work["MA10"] = work[target_col].shift(1).rolling(10).mean()
    work["std5"] = work[target_col].shift(1).rolling(5).std()
    work["dayofweek"] = work[date_col].dt.dayofweek
    work["month"] = work[date_col].dt.month

    # Chuẩn hóa tên các feature đã có trong master.
    generated = {
        "target_lag1", "target_lag2", "target_lag3",
        "target_ret_lag1", "MA5", "MA10", "std5",
        "dayofweek", "month",
    }
    for feature in FEATURE_ORDER:
        if feature in generated:
            continue
        source = find_column(work, feature)
        if source is not None and source != feature:
            work[feature] = work[source]

    features = []
    for feature in FEATURE_ORDER:
        if feature not in work.columns:
            continue
        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        if work[feature].notna().any():
            features.append(feature)

    missing = [feature for feature in FEATURE_ORDER if feature not in features]
    if missing:
        print("[WARN] Feature không có dữ liệu và sẽ bỏ qua:", ", ".join(missing))

    if not features:
        raise ValueError("Không tìm thấy feature hợp lệ trong dữ liệu master.")

    return work, features


def build_dataset(path: Path, horizon: int):
    """Đọc master và tạo target cho h=1,5,21,63."""
    frame = pd.read_csv(path)
    date_col = detect_column(frame.columns, DATE_CANDIDATES)
    target_col = detect_column(frame.columns, TARGET_CANDIDATES)

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = (
        frame.dropna(subset=[date_col, target_col])
        .sort_values(date_col)
        .drop_duplicates(date_col, keep="last")
        .reset_index(drop=True)
    )

    frame, features = prepare_features(frame, date_col, target_col)
    frame["target_future"] = frame[target_col].shift(-horizon)
    frame["anchor_price"] = frame[target_col]
    frame["target_level"] = frame["target_future"]
    frame["target_delta"] = frame["target_future"] - frame["anchor_price"]

    dataset = frame.dropna(
        subset=["target_future", "anchor_price"]
    ).reset_index(drop=True)

    return dataset, date_col, target_col, features


def get_feature_importance(model, features):
    """Lấy feature importance để web tự vẽ."""
    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return None

    if len(values) != len(features):
        return None

    return (
        pd.DataFrame({"Feature": features, "Importance": values})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def evaluate(actual, predicted, anchor, train_actual):
    """Tính đúng các chỉ số trong file comparison."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "sMAPE(%)": smape(actual, predicted),
        "MASE": mase(actual, predicted, train_actual),
        "DA(%)": directional_accuracy(actual, predicted, anchor),
    }


def train_naive(test, train_actual):
    """Naive: giá tương lai bằng giá hiện tại."""
    predicted = test["anchor_price"].to_numpy(float)
    actual = test["target_future"].to_numpy(float)
    anchor = test["anchor_price"].to_numpy(float)
    return predicted, evaluate(actual, predicted, anchor, train_actual), {
        "model_type": "Naive"
    }, None


def train_arimax(train, test, features, mode, train_actual):
    """Train ARIMAX(1,1,1) với feature ngoại sinh."""
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # ARIMAX chỉ dùng nhóm feature chính để train nhanh và ổn định hơn.
    arimax_features = [
        feature for feature in (
            "target_lag1", "target_lag2", "target_lag3",
            "MA5", "MA10", "std5",
            "london_vnd_kg_lag1", "usdvnd_lag1",
        )
        if feature in features
    ]
    if not arimax_features:
        arimax_features = features[:8]

    x_train = imputer.fit_transform(train[arimax_features])
    x_test = imputer.transform(test[arimax_features])
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = train[f"target_{mode}"].to_numpy(float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = SARIMAX(
            y_train,
            exog=x_train,
            order=(1, 1, 1),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=80)

    raw_prediction = np.asarray(
        result.get_forecast(steps=len(test), exog=x_test).predicted_mean,
        dtype=float,
    )
    anchor = test["anchor_price"].to_numpy(float)
    predicted = anchor + raw_prediction if mode == "delta" else raw_prediction
    actual = test["target_future"].to_numpy(float)

    bundle = {
        "model_type": "ARIMAX(1,1,1)",
        "model": result,
        "imputer": imputer,
        "scaler": scaler,
        "features": arimax_features,
        "mode": mode,
    }
    return predicted, evaluate(actual, predicted, anchor, train_actual), bundle, None


def train_ml_model(train, test, features, mode, model_name, factory, train_actual):
    """Train một model học máy."""
    model = factory()
    y_train = train[f"target_{mode}"].to_numpy(float)

    model.fit(train[features], y_train)
    raw_prediction = np.asarray(model.predict(test[features]), dtype=float)

    anchor = test["anchor_price"].to_numpy(float)
    predicted = anchor + raw_prediction if mode == "delta" else raw_prediction
    actual = test["target_future"].to_numpy(float)

    bundle = {
        "model_type": model_name,
        "model": model,
        "features": features,
        "mode": mode,
    }
    importance = get_feature_importance(model, features)
    return predicted, evaluate(actual, predicted, anchor, train_actual), bundle, importance


def clean_old_outputs():
    """Xóa kết quả cũ để web không đọc nhầm model."""
    patterns = [
        (MODEL_DIR, "model_*.joblib"),
        (RESULT_DIR, "model_comparison_*.csv"),
        (PREDICTION_DIR, "prediction_*.csv"),
        (IMPORTANCE_DIR, "*.csv"),
    ]
    for folder, pattern in patterns:
        if folder.exists():
            for path in folder.glob(pattern):
                path.unlink()


def safe_model_name(name: str) -> str:
    return (
        name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(" ", "_")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 21, 63])
    parser.add_argument("--modes", nargs="+", choices=["delta", "level"], default=["delta", "level"])
    parser.add_argument("--models", nargs="*", choices=PROJECT_MODELS, default=[])
    parser.add_argument("--keep-old", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu: {args.data}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.keep_old:
        clean_old_outputs()

    selected_models = args.models or list(PROJECT_MODELS)
    factories = model_factories()
    registry_rows = []

    for horizon in args.horizons:
        dataset, date_col, target_col, features = build_dataset(args.data, horizon)
        split = max(int(len(dataset) * 0.8), 1)
        train = dataset.iloc[:split].copy()
        test = dataset.iloc[split:].copy()

        if len(test) < 10:
            raise ValueError(f"h={horizon}: tập test quá nhỏ.")

        train_actual = train["target_future"].to_numpy(float)

        for mode in args.modes:
            comparison_rows = []
            lightgbm_importance = None

            for model_name in selected_models:
                print(f"Training {model_name} | h={horizon} | mode={mode}...")

                try:
                    if model_name == "Naive":
                        prediction, metrics, bundle, importance = train_naive(
                            test, train_actual
                        )
                    elif model_name == "ARIMAX(1,1,1)":
                        prediction, metrics, bundle, importance = train_arimax(
                            train, test, features, mode, train_actual
                        )
                    else:
                        prediction, metrics, bundle, importance = train_ml_model(
                            train,
                            test,
                            features,
                            mode,
                            model_name,
                            factories[model_name],
                            train_actual,
                        )
                except Exception as exc:
                    print(f"[ERROR] {model_name} thất bại: {type(exc).__name__}: {exc}")
                    continue

                row = {"Model": model_name, **metrics}
                comparison_rows.append(row)
                registry_rows.append({
                    "model": model_name,
                    "horizon": horizon,
                    "mode": mode,
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "MAPE_pct": metrics["sMAPE(%)"],
                    "DA_pct": metrics["DA(%)"],
                    "sMAPE(%)": metrics["sMAPE(%)"],
                    "MASE": metrics["MASE"],
                    "DA(%)": metrics["DA(%)"],
                    "train_rows": len(train),
                    "test_rows": len(test),
                })

                safe_name = safe_model_name(model_name)
                bundle.update({
                    "model_name": model_name,
                    "horizon": horizon,
                    "mode": mode,
                    "date_column": date_col,
                    "target_column": target_col,
                    "metrics": metrics,
                })
                joblib.dump(
                    bundle,
                    MODEL_DIR / f"model_{safe_name}_h{horizon}_{mode}.joblib",
                )

                pd.DataFrame({
                    "date": test[date_col].values,
                    "actual": test["target_future"].to_numpy(float),
                    "prediction": prediction,
                    "anchor": test["anchor_price"].to_numpy(float),
                    "error": test["target_future"].to_numpy(float) - prediction,
                    "model": model_name,
                    "horizon": horizon,
                    "mode": mode,
                }).to_csv(
                    PREDICTION_DIR / f"prediction_{safe_name}_h{horizon}_{mode}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

                if importance is not None:
                    importance.to_csv(
                        IMPORTANCE_DIR / f"importance_{safe_name}_h{horizon}_{mode}.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
                    if model_name == "LightGBM":
                        lightgbm_importance = importance

            comparison = (
                pd.DataFrame(comparison_rows)
                .sort_values("RMSE")
                .reset_index(drop=True)
            )
            comparison.to_csv(
                RESULT_DIR / f"model_comparison_h{horizon}_{mode}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # File này thay cho ảnh feature_importance_h...png.
            if lightgbm_importance is not None:
                lightgbm_importance.to_csv(
                    IMPORTANCE_DIR / f"feature_importance_h{horizon}_{mode}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

            print(f"\n=== h={horizon}, mode={mode} ===")
            print(comparison.to_string(index=False))

    registry = (
        pd.DataFrame(registry_rows)
        .sort_values(["horizon", "mode", "RMSE"])
        .reset_index(drop=True)
    )
    registry.to_csv(
        MODEL_DIR / "model_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\nĐã tạo model cho h=1,5,21,63 và mode delta/level.")
    print("Dashboard vẽ feature importance từ CSV, không dùng ảnh PNG.")


if __name__ == "__main__":
    main()
