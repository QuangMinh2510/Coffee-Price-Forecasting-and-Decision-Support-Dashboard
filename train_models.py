from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
MODEL_DIR = ROOT / "models"
PREDICTION_DIR = ROOT / "results" / "predictions"
IMPORTANCE_DIR = ROOT / "results" / "feature_importance"
DATE_CANDIDATES = ("Ngay", "date", "Date", "ngay")
TARGET_CANDIDATES = ("Gia_target", "Gia", "price", "Price")
PROJECT_MODELS = ("LinearRegression", "RandomForest", "XGBoost", "LightGBM")


def detect(columns, candidates):
    """Tìm cột phù hợp."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Không tìm thấy cột trong {candidates}")


def directional_accuracy(actual, predicted, anchor):
    """Tính độ chính xác xu hướng."""
    actual_direction = np.sign(np.asarray(actual) - np.asarray(anchor))
    predicted_direction = np.sign(np.asarray(predicted) - np.asarray(anchor))
    return float((actual_direction == predicted_direction).mean() * 100)


def mape(actual, predicted):
    """Tính sai số phần trăm."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def model_factories():
    """Tạo đúng 4 model của project."""
    scaled_linear = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    return {
        "LinearRegression": lambda: scaled_linear,
        "RandomForest": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=350,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=500,
                learning_rate=0.025,
                max_depth=5,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )),
        ]),
        "LightGBM": lambda: Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMRegressor(
                n_estimators=500,
                learning_rate=0.025,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1,
                verbosity=-1,
            )),
        ]),
    }


def feature_importance(model, feature_names):
    """Lấy độ quan trọng của đặc trưng."""
    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return None

    if len(values) != len(feature_names):
        return None

    return pd.DataFrame({
        "feature": feature_names,
        "importance": values,
    }).sort_values("importance", ascending=False)


def build_dataset(path: Path, horizon: int):
    """Đọc dữ liệu và tạo feature."""
    frame = pd.read_csv(path)
    date_col = detect(frame.columns, DATE_CANDIDATES)
    target_col = detect(frame.columns, TARGET_CANDIDATES)

    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")
    frame = (
        frame.dropna(subset=[date_col, target_col])
        .sort_values(date_col)
        .drop_duplicates(date_col, keep="last")
    )

    numeric = frame.select_dtypes(include="number").columns.tolist()
    forbidden = {target_col}
    forbidden.update(
        column for column in numeric
        if column.lower().startswith(("target", "future", "lead"))
    )
    features = [column for column in numeric if column not in forbidden]

    # Feature giá quá khứ.
    for lag in (1, 2, 5, 10, 21, 63):
        name = f"price_lag_{lag}"
        frame[name] = frame[target_col].shift(lag)
        features.append(name)

    # Feature trung bình và độ lệch chuẩn.
    for window in (5, 10, 21, 63):
        mean_name = f"price_roll_mean_{window}"
        std_name = f"price_roll_std_{window}"
        frame[mean_name] = frame[target_col].shift(1).rolling(window).mean()
        frame[std_name] = frame[target_col].shift(1).rolling(window).std()
        features.extend([mean_name, std_name])

    frame["target_future"] = frame[target_col].shift(-horizon)
    frame["anchor_price"] = frame[target_col]
    dataset = frame.dropna(subset=["target_future"]).copy()

    return dataset, date_col, target_col, features


def train_one(dataset, date_col, target_col, features, horizon, model_name, factory):
    """Train và đánh giá một model."""
    split = max(int(len(dataset) * 0.8), 1)
    train = dataset.iloc[:split]
    test = dataset.iloc[split:]

    if len(test) < 10:
        raise ValueError("Tập test quá nhỏ; cần thêm dữ liệu")

    model = factory()
    model.fit(train[features], train["target_future"])
    prediction = model.predict(test[features])

    actual = test["target_future"].to_numpy(float)
    anchor = test["anchor_price"].to_numpy(float)

    metrics = {
        "model": model_name,
        "horizon": horizon,
        "MAE": mean_absolute_error(actual, prediction),
        "RMSE": mean_squared_error(actual, prediction) ** 0.5,
        "MAPE_pct": mape(actual, prediction),
        "DA_pct": directional_accuracy(actual, prediction, anchor),
        "train_rows": len(train),
        "test_rows": len(test),
    }

    predictions = pd.DataFrame({
        "date": test[date_col].values,
        "actual": actual,
        "prediction": prediction,
        "anchor": anchor,
        "error": actual - prediction,
    })

    bundle = {
        "model": model,
        "model_name": model_name,
        "horizon": horizon,
        "features": features,
        "date_column": date_col,
        "target_column": target_col,
        "metrics": metrics,
    }

    return metrics, predictions, bundle, feature_importance(model, features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 21, 63])
    parser.add_argument("--models", nargs="*", choices=PROJECT_MODELS, default=[])
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    factories = model_factories()
    selected = args.models or list(PROJECT_MODELS)
    registry_rows = []

    for horizon in args.horizons:
        dataset, date_col, target_col, features = build_dataset(args.data, horizon)

        for model_name in selected:
            print(f"Training {model_name}, horizon={horizon}...")
            metrics, predictions, bundle, importance = train_one(
                dataset,
                date_col,
                target_col,
                features,
                horizon,
                model_name,
                factories[model_name],
            )

            registry_rows.append(metrics)
            safe_name = model_name.lower()

            joblib.dump(
                bundle,
                MODEL_DIR / f"model_{safe_name}_h{horizon}.joblib",
            )
            predictions.to_csv(
                PREDICTION_DIR / f"prediction_{safe_name}_h{horizon}.csv",
                index=False,
                encoding="utf-8-sig",
            )

            if importance is not None:
                importance.to_csv(
                    IMPORTANCE_DIR / f"importance_{safe_name}_h{horizon}.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

    registry = pd.DataFrame(registry_rows).sort_values(["horizon", "RMSE"])
    registry.to_csv(
        MODEL_DIR / "model_registry.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(registry.to_string(index=False))
    print("\nDashboard đọc kết quả từ CSV, không dùng ảnh model.")


if __name__ == "__main__":
    main()
