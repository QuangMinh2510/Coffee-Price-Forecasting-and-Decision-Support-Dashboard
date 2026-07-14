from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from live_data import fetch_live_coffee, fetch_live_fuel, fetch_live_weather

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
REGISTRY_PATH = ROOT / "models" / "model_registry.csv"
FEATURE_IMPORTANCE_DIR = ROOT / "results" / "feature_importance"
PREDICTION_DIR = ROOT / "results" / "predictions"
DATE_CANDIDATES = ("Ngay", "date", "Date", "ngay")
TARGET_CANDIDATES = ("Gia_target", "Gia", "price", "Price")

st.set_page_config(page_title="CofPred", page_icon="☕", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#fffaf3 0%,#f5ecdf 100%)}
    [data-testid='stMetric'] {background:white;border:1px solid #ead8c2;padding:14px;border-radius:14px}
    h1,h2,h3 {color:#4a2f20}
    </style>
    """,
    unsafe_allow_html=True,
)


def detect_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


@st.cache_data
def load_master(path: Path):
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


@st.cache_data(ttl=1800, show_spinner=False)
def load_live_snapshot():
    with ThreadPoolExecutor(max_workers=3) as pool:
        coffee = pool.submit(fetch_live_coffee)
        fuel = pool.submit(fetch_live_fuel)
        weather = pool.submit(fetch_live_weather)
        return {"coffee": coffee.result(), "fuel": fuel.result(), "weather": weather.result()}


@st.cache_data
def load_registry(path: Path):
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    for column in ("MAE", "RMSE", "MAPE_pct", "DA_pct", "horizon"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def format_vnd(value, unit="kg"):
    return f"{float(value):,.0f} VND/{unit}".replace(",", ".")


def render_live_card(title, payload, metric_name, metric_value, source_label):
    st.subheader(title)
    if not payload.get("ok"):
        st.warning("Chưa lấy được dữ liệu live. Web không dùng dữ liệu cũ để giả làm dữ liệu hiện tại.")
        with st.expander("Chi tiết lỗi"):
            st.code(payload.get("error", "Không rõ lỗi"))
        return
    st.metric(metric_name, metric_value)
    st.caption(f"Nguồn: {payload.get('source_name', source_label)} · cập nhật: {payload.get('fetched_at', '')}")


st.title("☕ CofPred — Coffee Price Forecasting & Decision Support")
st.caption("Dữ liệu lịch sử + dữ liệu live cà phê · dầu diesel · thời tiết + so sánh nhiều mô hình dự báo")

if st.button("🔄 Làm mới dữ liệu live"):
    st.cache_data.clear()
    st.rerun()

live_tab, overview_tab, model_tab, data_tab = st.tabs([
    "🔴 Dữ liệu live", "📈 Tổng quan", "🤖 Mô hình dự đoán", "🗂 Dữ liệu",
])

with live_tab:
    with st.spinner("Đang lấy dữ liệu live..."):
        live = load_live_snapshot()

    coffee, fuel, weather = live["coffee"], live["fuel"], live["weather"]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_live_card(
            "☕ Cà phê Việt Nam",
            coffee,
            "Giá trung bình",
            format_vnd(coffee.get("average_vnd_kg", 0)),
            "Nguồn giá cà phê",
        )
    with c2:
        render_live_card(
            "⛽ Dầu diesel",
            fuel,
            "DO 0,05S",
            format_vnd(fuel.get("price_vnd_litre", 0), "lít"),
            "Nguồn giá nhiên liệu",
        )
    with c3:
        render_live_card(
            "🌦 Thời tiết vùng cà phê",
            weather,
            "Nhiệt độ hiện tại",
            f"{weather.get('temperature_c', 0):.1f} °C" if weather.get("temperature_c") is not None else "N/A",
            "Open-Meteo",
        )

    if coffee.get("ok") and coffee.get("rows"):
        st.subheader("Giá cà phê theo tỉnh/thị trường")
        coffee_frame = pd.DataFrame(coffee["rows"]).rename(columns={
            "province": "Tỉnh/thị trường", "price_vnd_kg": "Giá (VND/kg)",
        })
        st.dataframe(coffee_frame, use_container_width=True, hide_index=True)
        st.bar_chart(coffee_frame.set_index("Tỉnh/thị trường")["Giá (VND/kg)"])

    if weather.get("ok"):
        w1, w2, w3 = st.columns(3)
        w1.metric("Độ ẩm", f"{weather.get('humidity_pct', 0):.0f}%")
        w2.metric("Mưa hiện tại", f"{weather.get('rain_mm', 0):.1f} mm")
        w3.metric("Gió", f"{weather.get('wind_kmh', 0):.1f} km/h")
        forecast = pd.DataFrame(weather.get("forecast", []))
        if not forecast.empty:
            forecast["date"] = pd.to_datetime(forecast["date"])
            st.subheader("Dự báo thời tiết 7 ngày")
            st.dataframe(forecast, use_container_width=True, hide_index=True)
            st.line_chart(forecast.set_index("date")[["temp_max_c", "temp_min_c"]])
            st.bar_chart(forecast.set_index("date")[["rain_mm"]])

    st.info("Dữ liệu live dùng để theo dõi. Muốn dùng để retrain model, cần lưu snapshot, chuẩn hóa feature và kiểm tra chất lượng trước khi ghép vào master.")

master, date_col, target_col = load_master(DATA_PATH)

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
        period = st.selectbox("Khoảng hiển thị", ["3 tháng", "6 tháng", "1 năm", "Tất cả"])
        filtered = clean
        if period != "Tất cả":
            months = {"3 tháng": 3, "6 tháng": 6, "1 năm": 12}[period]
            filtered = clean[clean[date_col] >= clean[date_col].max() - pd.DateOffset(months=months)]
        st.line_chart(filtered.set_index(date_col)[target_col])
        yearly = clean.assign(Năm=clean[date_col].dt.year).groupby("Năm")[target_col].agg(["mean", "min", "max", "count"])
        st.subheader("Thống kê giá theo năm")
        st.dataframe(yearly.rename(columns={"mean": "Trung bình", "min": "Thấp nhất", "max": "Cao nhất", "count": "Số phiên"}), use_container_width=True)

with model_tab:
    registry = load_registry(REGISTRY_PATH)
    if registry.empty:
        st.warning("Chưa có `models/model_registry.csv`. Chạy `python train_models.py` để train nhiều model.")
        st.code("python train_models.py --horizons 1 5 21 63", language="bash")
    else:
        horizons = sorted(registry["horizon"].dropna().astype(int).unique()) if "horizon" in registry else []
        selected_horizon = st.selectbox("Chân trời dự báo", horizons) if horizons else None
        shown = registry[registry["horizon"].astype(int) == selected_horizon].copy() if selected_horizon else registry.copy()
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
            chart_data = shown.set_index("model")[[column for column in ["MAE", "RMSE"] if column in shown]]
            st.bar_chart(chart_data)

        st.subheader("Biểu đồ dự báo được tạo trực tiếp từ CSV")
        prediction_files = sorted(PREDICTION_DIR.glob("*.csv")) if PREDICTION_DIR.exists() else []
        if prediction_files:
            selected_file = st.selectbox("File kết quả", prediction_files, format_func=lambda p: p.name)
            predictions = pd.read_csv(selected_file)
            pred_date = detect_column(predictions.columns, DATE_CANDIDATES)
            if pred_date:
                predictions[pred_date] = pd.to_datetime(predictions[pred_date], errors="coerce")
                predictions = predictions.set_index(pred_date)
            numeric = predictions.select_dtypes("number")
            st.line_chart(numeric)
            st.dataframe(predictions.tail(50), use_container_width=True)
        else:
            st.caption("Sau khi train, các dự báo sẽ nằm trong `results/predictions/*.csv`; web không cần ảnh PNG model.")

        st.subheader("Feature importance")
        importance_files = sorted(FEATURE_IMPORTANCE_DIR.glob("*.csv")) if FEATURE_IMPORTANCE_DIR.exists() else []
        if importance_files:
            selected_importance = st.selectbox("File feature importance", importance_files, format_func=lambda p: p.name)
            importance = pd.read_csv(selected_importance)
            feature_col = importance.columns[0]
            value_col = importance.columns[-1]
            importance[value_col] = pd.to_numeric(importance[value_col], errors="coerce")
            importance = importance.dropna(subset=[value_col]).sort_values(value_col, ascending=False).head(20)
            st.bar_chart(importance.set_index(feature_col)[value_col])
            st.dataframe(importance, use_container_width=True, hide_index=True)
        else:
            st.caption("Feature importance sẽ được web vẽ từ CSV thay vì lưu ảnh tĩnh.")

with data_tab:
    if master.empty:
        st.warning("Chưa có dữ liệu master để hiển thị.")
    else:
        st.dataframe(master.tail(100).sort_values(date_col, ascending=False), use_container_width=True, hide_index=True)
        st.download_button(
            "Tải dữ liệu master",
            data=master.to_csv(index=False).encode("utf-8-sig"),
            file_name="gia_cafe_master_full.csv",
            mime="text/csv",
        )

st.caption("CofPred phục vụ học tập và hỗ trợ quyết định, không phải khuyến nghị tài chính.")
