from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

from live_data import fetch_live_coffee, fetch_live_fuel, fetch_live_weather


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
REGISTRY_PATH = ROOT / "models" / "model_registry.csv"
FEATURE_IMPORTANCE_DIR = ROOT / "results" / "feature_importance"
PREDICTION_DIR = ROOT / "results" / "predictions"

FUEL_PATHS = (
    ROOT / "data" / "Processing" / "Fuel" / "diesel_2017_now_daily.csv",
    ROOT / "data" / "processed" / "diesel_2017_now_daily.csv",
)
WEATHER_PATHS = (
    ROOT / "data" / "Raw" / "weather" / "weather_all_regions.csv",
    ROOT / "data" / "processed" / "weather_all_regions.csv",
)

DATE_CANDIDATES = ("Ngay", "date", "Date", "ngay", "datetime", "time")
TARGET_CANDIDATES = ("Gia_target", "Gia", "price", "Price")
FUEL_VALUE_CANDIDATES = ("gia", "diesel", "fuel_price", "diesel_price", "Price", "Gia")
WEATHER_VALUE_CANDIDATES = (
    "temperature_2m_mean",
    "temperature_2m",
    "temperature",
    "temp_mean",
    "temperature_c",
)
PERIOD_OFFSETS = {
    "1 tháng": pd.DateOffset(months=1),
    "3 tháng": pd.DateOffset(months=3),
    "6 tháng": pd.DateOffset(months=6),
    "1 năm": pd.DateOffset(years=1),
}


st.set_page_config(page_title="CofPred", page_icon="☕", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#fffaf3 0%,#f5ecdf 100%)}
    .block-container {padding-top: 2rem}
    [data-testid='stMetric'] {
        background:white;border:1px solid #ead8c2;padding:16px;
        border-radius:14px;box-shadow:0 5px 18px rgba(75,45,25,.05)
    }
    [data-testid='stVegaLiteChart'] {
        background:white;border:1px solid #ead8c2;padding:10px;border-radius:14px
    }
    h1,h2,h3 {color:#4a2f20}
    </style>
    """,
    unsafe_allow_html=True,
)


def detect_column(columns, candidates):
    """Tìm cột phù hợp."""
    for name in candidates:
        if name in columns:
            return name
    return None


def first_existing(paths):
    """Lấy file đầu tiên tồn tại."""
    return next((path for path in paths if path.exists()), None)


@st.cache_data
def load_master(path: Path):
    """Đọc dữ liệu cà phê lịch sử."""
    if not path.exists():
        return pd.DataFrame(), None, None

    frame = pd.read_csv(path)
    date_col = detect_column(frame.columns, DATE_CANDIDATES)
    target_col = detect_column(frame.columns, TARGET_CANDIDATES)

    if date_col:
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    if target_col:
        frame[target_col] = pd.to_numeric(frame[target_col], errors="coerce")

    return frame, date_col, target_col


@st.cache_data
def load_history_series(paths, value_candidates):
    """Đọc chuỗi dữ liệu lịch sử."""
    path = first_existing(paths)
    if path is None:
        return pd.DataFrame(columns=["date", "value"]), None

    frame = pd.read_csv(path)
    date_col = detect_column(frame.columns, DATE_CANDIDATES)
    value_col = detect_column(frame.columns, value_candidates)
    if not date_col or not value_col:
        return pd.DataFrame(columns=["date", "value"]), path

    frame["date"] = pd.to_datetime(frame[date_col], errors="coerce")
    frame["value"] = pd.to_numeric(frame[value_col], errors="coerce")
    series = (
        frame.dropna(subset=["date", "value"])
        .groupby("date", as_index=False)["value"]
        .mean()
        .sort_values("date")
    )
    return series, path


@st.cache_data(ttl=1800, show_spinner=False)
def load_live_snapshot():
    """Lấy ba nguồn live cùng lúc."""
    with ThreadPoolExecutor(max_workers=3) as pool:
        coffee = pool.submit(fetch_live_coffee)
        fuel = pool.submit(fetch_live_fuel)
        weather = pool.submit(fetch_live_weather)
        return {
            "coffee": coffee.result(),
            "fuel": fuel.result(),
            "weather": weather.result(),
        }


@st.cache_data
def load_registry(path: Path):
    """Đọc bảng đánh giá model."""
    if not path.exists():
        return pd.DataFrame()

    frame = pd.read_csv(path)
    for column in ("MAE", "RMSE", "MAPE_pct", "DA_pct", "horizon"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def coffee_history(master, date_col, target_col):
    """Đổi master thành chuỗi giá."""
    if master.empty or not date_col or not target_col:
        return pd.DataFrame(columns=["date", "value"])

    return (
        master[[date_col, target_col]]
        .rename(columns={date_col: "date", target_col: "value"})
        .dropna()
        .groupby("date", as_index=False)["value"]
        .mean()
        .sort_values("date")
    )


def append_live_value(frame, value, observed_at=None):
    """Nối điểm live vào dữ liệu cũ."""
    if value is None:
        return frame.copy()

    timestamp = pd.to_datetime(observed_at, errors="coerce")
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now()
    if getattr(timestamp, "tzinfo", None) is not None:
        timestamp = timestamp.tz_localize(None)

    live_row = pd.DataFrame({"date": [timestamp], "value": [float(value)]})
    return (
        pd.concat([frame, live_row], ignore_index=True)
        .dropna(subset=["date", "value"])
        .sort_values("date")
        .drop_duplicates("date", keep="last")
    )


def filter_period(frame, label):
    """Lọc theo khoảng thời gian."""
    if frame.empty or label == "Tất cả":
        return frame.copy()
    end_date = frame["date"].max()
    return frame[frame["date"] >= end_date - PERIOD_OFFSETS[label]].copy()


def format_vnd(value, unit="kg"):
    return f"{float(value):,.0f} VND/{unit}".replace(",", ".")


def format_temperature(value):
    return f"{float(value):.1f} °C"


def render_period_dashboard(frame, key, metric_word, formatter):
    """Hiển thị 4 thẻ và biểu đồ."""
    if frame.empty:
        st.warning("Chưa có dữ liệu để vẽ biểu đồ.")
        return

    period = st.selectbox(
        "Khoảng thời gian",
        ["1 tháng", "3 tháng", "6 tháng", "1 năm", "Tất cả"],
        index=3,
        key=f"period_{key}",
    )
    shown = filter_period(frame, period).sort_values("date")
    if shown.empty:
        st.warning("Không có dữ liệu trong khoảng đã chọn.")
        return

    first = shown.iloc[0]
    last = shown.iloc[-1]
    change = float(last["value"] - first["value"])
    percent = change / abs(float(first["value"])) * 100 if first["value"] else 0.0
    trend = "Tăng" if change > 0 else "Giảm" if change < 0 else "Ổn định"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        f"{metric_word} cuối kỳ",
        formatter(last["value"]),
        f"{last['date']:%d/%m/%Y}",
        delta_color="off",
    )
    c2.metric(
        f"{metric_word} đầu kỳ",
        formatter(first["value"]),
        f"{first['date']:%d/%m/%Y}",
        delta_color="off",
    )
    c3.metric("Thay đổi trong kỳ", formatter(change), f"{percent:+.2f}%")
    c4.metric(
        "Xu hướng giai đoạn",
        trend,
        f"{len(shown):,} phiên".replace(",", "."),
        delta_color="off",
    )

    chart_data = shown.set_index("date").rename(columns={"value": metric_word})
    st.line_chart(chart_data, height=430)


st.title("☕ CofPred — Coffee Price Forecasting & Decision Support")
st.caption(
    "Dữ liệu lịch sử + dữ liệu live cà phê · dầu diesel · thời tiết "
    "+ so sánh nhiều mô hình dự báo"
)

if st.button("🔄 Làm mới dữ liệu live"):
    st.cache_data.clear()
    st.rerun()

# Nạp dữ liệu dùng chung.
master, date_col, target_col = load_master(DATA_PATH)
fuel_history, fuel_path = load_history_series(FUEL_PATHS, FUEL_VALUE_CANDIDATES)
weather_history, weather_path = load_history_series(
    WEATHER_PATHS, WEATHER_VALUE_CANDIDATES
)

live_tab, overview_tab, model_tab, data_tab = st.tabs(
    ["🔴 Dữ liệu live", "📈 Tổng quan", "🤖 Mô hình dự đoán", "🗂 Dữ liệu"]
)

with live_tab:
    with st.spinner("Đang lấy dữ liệu live..."):
        live = load_live_snapshot()

    coffee = live["coffee"]
    fuel = live["fuel"]
    weather = live["weather"]
    coffee_tab, fuel_tab, weather_tab = st.tabs(
        ["☕ Cà phê", "⛽ Dầu diesel", "🌦 Thời tiết"]
    )

    with coffee_tab:
        st.subheader("Giá cà phê Việt Nam")
        series = coffee_history(master, date_col, target_col)

        if coffee.get("ok"):
            series = append_live_value(
                series,
                coffee.get("average_vnd_kg"),
                coffee.get("fetched_at"),
            )
            st.caption(
                f"Nguồn live: {coffee.get('source_name', 'Nguồn cà phê')} · "
                f"cập nhật: {coffee.get('fetched_at', '')}"
            )
        else:
            st.warning("Không lấy được giá live; biểu đồ dùng dữ liệu lịch sử.")
            with st.expander("Chi tiết lỗi"):
                st.code(coffee.get("error", "Không rõ lỗi"))

        render_period_dashboard(
            series,
            key="coffee",
            metric_word="Giá",
            formatter=lambda value: format_vnd(value, "kg"),
        )

        if coffee.get("ok") and coffee.get("rows"):
            province_frame = pd.DataFrame(coffee["rows"]).rename(
                columns={
                    "province": "Tỉnh/thị trường",
                    "price_vnd_kg": "Giá (VND/kg)",
                }
            )
            st.subheader("Giá live theo tỉnh/thị trường")
            st.dataframe(province_frame, use_container_width=True, hide_index=True)
            st.bar_chart(
                province_frame.set_index("Tỉnh/thị trường")["Giá (VND/kg)"]
            )

    with fuel_tab:
        st.subheader("Giá dầu diesel DO 0,05S")
        series = fuel_history.copy()

        if fuel.get("ok"):
            series = append_live_value(
                series,
                fuel.get("price_vnd_litre"),
                fuel.get("fetched_at"),
            )
            st.caption(
                f"Nguồn live: {fuel.get('source_name', 'Nguồn dầu diesel')} · "
                f"cập nhật: {fuel.get('fetched_at', '')}"
            )
        else:
            st.warning("Không lấy được giá dầu diesel live.")
            with st.expander("Chi tiết lỗi"):
                st.code(fuel.get("error", "Không rõ lỗi"))

        if fuel_path:
            st.caption(f"Dữ liệu lịch sử: `{fuel_path.relative_to(ROOT)}`")

        render_period_dashboard(
            series,
            key="fuel",
            metric_word="Giá",
            formatter=lambda value: format_vnd(value, "lít"),
        )

    with weather_tab:
        st.subheader("Thời tiết vùng cà phê")
        series = weather_history.copy()

        if weather.get("ok"):
            series = append_live_value(
                series,
                weather.get("temperature_c"),
                weather.get("observed_at") or weather.get("fetched_at"),
            )
            st.caption(
                f"Nguồn live: {weather.get('source_name', 'Open-Meteo')} · "
                f"khu vực: {weather.get('location', 'Buôn Ma Thuột')} · "
                f"cập nhật: {weather.get('observed_at', '')}"
            )
        else:
            st.warning("Không lấy được thời tiết live.")
            with st.expander("Chi tiết lỗi"):
                st.code(weather.get("error", "Không rõ lỗi"))

        if weather_path:
            st.caption(f"Dữ liệu lịch sử: `{weather_path.relative_to(ROOT)}`")

        render_period_dashboard(
            series,
            key="weather",
            metric_word="Nhiệt độ",
            formatter=format_temperature,
        )

        if weather.get("ok"):
            w1, w2, w3 = st.columns(3)
            w1.metric("Độ ẩm hiện tại", f"{weather.get('humidity_pct', 0):.0f}%")
            w2.metric("Mưa hiện tại", f"{weather.get('rain_mm', 0):.1f} mm")
            w3.metric("Tốc độ gió", f"{weather.get('wind_kmh', 0):.1f} km/h")

            forecast = pd.DataFrame(weather.get("forecast", []))
            if not forecast.empty:
                forecast["date"] = pd.to_datetime(forecast["date"], errors="coerce")
                st.subheader("Dự báo thời tiết 7 ngày")
                st.dataframe(forecast, use_container_width=True, hide_index=True)
                st.line_chart(
                    forecast.set_index("date")[["temp_max_c", "temp_min_c"]]
                )
                st.bar_chart(forecast.set_index("date")[["rain_mm"]])

    st.info(
        "Điểm live được nối vào dữ liệu lịch sử để hiển thị. "
        "Muốn retrain model cần lưu và chuẩn hóa dữ liệu trước."
    )

with overview_tab:
    if master.empty or not date_col or not target_col:
        st.warning(f"Chưa tìm thấy dữ liệu master tại `{DATA_PATH}`.")
    else:
        clean = master.dropna(subset=[target_col]).copy()
        latest = clean.iloc[-1]
        previous = clean.iloc[-2] if len(clean) > 1 else latest
        delta = latest[target_col] - previous[target_col]

        a, b, c, d = st.columns(4)
        a.metric("Giá mới nhất trong master", format_vnd(latest[target_col]), format_vnd(delta))
        b.metric("Số dòng", f"{len(clean):,}".replace(",", "."))
        c.metric("Bắt đầu", f"{clean[date_col].min():%d/%m/%Y}")
        d.metric("Kết thúc", f"{clean[date_col].max():%d/%m/%Y}")

        period = st.selectbox(
            "Khoảng hiển thị",
            ["3 tháng", "6 tháng", "1 năm", "Tất cả"],
            key="overview_period",
        )
        filtered = clean
        if period != "Tất cả":
            months = {"3 tháng": 3, "6 tháng": 6, "1 năm": 12}[period]
            filtered = clean[
                clean[date_col] >= clean[date_col].max() - pd.DateOffset(months=months)
            ]

        st.line_chart(filtered.set_index(date_col)[target_col])
        yearly = (
            clean.assign(Năm=clean[date_col].dt.year)
            .groupby("Năm")[target_col]
            .agg(["mean", "min", "max", "count"])
        )
        st.subheader("Thống kê giá theo năm")
        st.dataframe(
            yearly.rename(
                columns={
                    "mean": "Trung bình",
                    "min": "Thấp nhất",
                    "max": "Cao nhất",
                    "count": "Số phiên",
                }
            ),
            use_container_width=True,
        )

with model_tab:
    registry = load_registry(REGISTRY_PATH)
    if registry.empty:
        st.warning("Chưa có `models/model_registry.csv`. Chạy file train model trước.")
        st.code("python train_models.py --horizons 1 5 21 63", language="bash")
    else:
        horizons = (
            sorted(registry["horizon"].dropna().astype(int).unique())
            if "horizon" in registry
            else []
        )
        selected_horizon = st.selectbox("Chân trời dự báo", horizons) if horizons else None
        shown = (
            registry[registry["horizon"].astype(int) == selected_horizon].copy()
            if selected_horizon
            else registry.copy()
        )
        if "RMSE" in shown:
            shown = shown.sort_values("RMSE")

        st.subheader("Bảng xếp hạng mô hình")
        st.dataframe(shown, use_container_width=True, hide_index=True)

        if not shown.empty and "model" in shown and "RMSE" in shown:
            best = shown.iloc[0]
            x1, x2, x3, x4 = st.columns(4)
            x1.metric("Model tốt nhất", str(best["model"]))
            x2.metric("RMSE", f"{best.get('RMSE', 0):,.0f}")
            x3.metric("MAE", f"{best.get('MAE', 0):,.0f}")
            x4.metric("Directional Accuracy", f"{best.get('DA_pct', 0):.1f}%")
            columns = [column for column in ("MAE", "RMSE") if column in shown]
            st.bar_chart(shown.set_index("model")[columns])

        st.subheader("Biểu đồ dự báo từ CSV")
        prediction_files = sorted(PREDICTION_DIR.glob("*.csv")) if PREDICTION_DIR.exists() else []
        if prediction_files:
            selected_file = st.selectbox(
                "File kết quả", prediction_files, format_func=lambda path: path.name
            )
            predictions = pd.read_csv(selected_file)
            prediction_date = detect_column(predictions.columns, DATE_CANDIDATES)
            if prediction_date:
                predictions[prediction_date] = pd.to_datetime(
                    predictions[prediction_date], errors="coerce"
                )
                predictions = predictions.set_index(prediction_date)
            st.line_chart(predictions.select_dtypes("number"))
            st.dataframe(predictions.tail(50), use_container_width=True)
        else:
            st.caption("Sau khi train, dự báo nằm trong `results/predictions/*.csv`.")

        st.subheader("Feature importance")
        importance_files = (
            sorted(FEATURE_IMPORTANCE_DIR.glob("*.csv"))
            if FEATURE_IMPORTANCE_DIR.exists()
            else []
        )
        if importance_files:
            selected_importance = st.selectbox(
                "File feature importance",
                importance_files,
                format_func=lambda path: path.name,
            )
            importance = pd.read_csv(selected_importance)
            feature_col = importance.columns[0]
            value_col = importance.columns[-1]
            importance[value_col] = pd.to_numeric(importance[value_col], errors="coerce")
            importance = (
                importance.dropna(subset=[value_col])
                .sort_values(value_col, ascending=False)
                .head(20)
            )
            st.bar_chart(importance.set_index(feature_col)[value_col])
            st.dataframe(importance, use_container_width=True, hide_index=True)
        else:
            st.caption("Feature importance được web vẽ từ CSV.")

with data_tab:
    if master.empty:
        st.warning("Chưa có dữ liệu master để hiển thị.")
    else:
        st.dataframe(
            master.tail(100).sort_values(date_col, ascending=False),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Tải dữ liệu master",
            data=master.to_csv(index=False).encode("utf-8-sig"),
            file_name="gia_cafe_master_full.csv",
            mime="text/csv",
        )

st.caption(
    "CofPred phục vụ học tập và hỗ trợ quyết định, không phải khuyến nghị tài chính."
)
