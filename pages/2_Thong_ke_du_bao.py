from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "models" / "model_registry.csv"
PREDICTION_DIR = ROOT / "results" / "predictions"
HORIZON_LABELS = {
    1: "1 ngày — ngày mai",
    5: "1 tuần",
    21: "1 tháng",
    63: "1 quý",
}

st.set_page_config(page_title="Thống kê dự báo", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg,#fffaf3 0%,#f5ecdf 100%)}
    [data-testid='stMetric'] {
        background:white;border:1px solid #ead8c2;padding:14px;
        border-radius:14px;box-shadow:0 5px 18px rgba(75,45,25,.05)
    }
    h1,h2,h3 {color:#4a2f20}
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_model_name(name: str) -> str:
    """Đổi tên model thành tên file."""
    return (
        name.lower()
        .replace("(", "")
        .replace(")", "")
        .replace(",", "_")
        .replace(" ", "_")
    )


@st.cache_data
def load_registry() -> pd.DataFrame:
    """Đọc bảng đánh giá model."""
    if not REGISTRY_PATH.exists():
        return pd.DataFrame()

    frame = pd.read_csv(REGISTRY_PATH)
    numeric_columns = [
        "horizon", "MAE", "RMSE", "MAPE_pct", "DA_pct",
        "sMAPE(%)", "MASE", "DA(%)", "train_rows", "test_rows",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


@st.cache_data
def load_prediction(path: Path) -> pd.DataFrame:
    """Đọc kết quả dự báo."""
    frame = pd.read_csv(path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ("actual", "prediction", "anchor", "error"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "actual", "prediction"]).sort_values("date")


def prediction_path(model: str, horizon: int, mode: str) -> Path:
    """Tạo đường dẫn file dự báo."""
    return PREDICTION_DIR / f"prediction_{safe_model_name(model)}_h{horizon}_{mode}.csv"


def smape(actual: pd.Series, predicted: pd.Series) -> float:
    """Tính sMAPE từ dữ liệu hiển thị."""
    denominator = actual.abs() + predicted.abs()
    valid = denominator > 1e-9
    if not valid.any():
        return float("nan")
    return float((200 * (actual[valid] - predicted[valid]).abs() / denominator[valid]).mean())


def directional_accuracy(frame: pd.DataFrame) -> float:
    """Tính tỷ lệ đoán đúng hướng tăng giảm."""
    if "anchor" not in frame.columns:
        return float("nan")
    actual_direction = np.sign(frame["actual"] - frame["anchor"])
    predicted_direction = np.sign(frame["prediction"] - frame["anchor"])
    return float((actual_direction == predicted_direction).mean() * 100)


def metric_value(row: pd.Series, *names: str) -> float:
    """Lấy chỉ số theo tên cột có sẵn."""
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return float("nan")


st.title("📊 Thống kê dự báo giá cà phê")
st.caption(
    "So sánh giá thực tế, giá dự báo và Naive theo từng model, "
    "chân trời và chế độ dự báo."
)

registry = load_registry()
if registry.empty:
    st.warning("Chưa có kết quả model. Hãy chạy train trước.")
    st.code(
        "python train_models.py --horizons 1 5 21 63 --modes delta level",
        language="bash",
    )
    st.stop()

horizons = sorted(registry["horizon"].dropna().astype(int).unique().tolist())
modes = [mode for mode in ("delta", "level") if mode in registry["mode"].astype(str).unique()]

control_1, control_2, control_3 = st.columns(3)
selected_horizon = control_1.selectbox(
    "Chân trời dự báo",
    horizons,
    format_func=lambda value: HORIZON_LABELS.get(value, f"{value} ngày"),
)
selected_mode = control_2.selectbox(
    "Chế độ dự báo",
    modes,
    format_func=lambda value: "Dự báo mức thay đổi (delta)" if value == "delta" else "Dự báo trực tiếp mức giá (level)",
)

available = registry[
    (registry["horizon"].astype(int) == selected_horizon)
    & (registry["mode"].astype(str) == selected_mode)
].copy()
available = available.sort_values("RMSE")
models = available["model"].astype(str).tolist()
selected_model = control_3.selectbox("Model", models)

selected_row = available[available["model"].astype(str) == selected_model].iloc[0]
selected_path = prediction_path(selected_model, selected_horizon, selected_mode)

if not selected_path.exists():
    st.error(f"Không tìm thấy file dự báo: `{selected_path.relative_to(ROOT)}`")
    st.stop()

prediction = load_prediction(selected_path)
if prediction.empty:
    st.warning("File dự báo không có dữ liệu hợp lệ.")
    st.stop()

# Thêm đường Naive giống các biểu đồ gốc.
naive_path = prediction_path("Naive", selected_horizon, selected_mode)
chart = prediction[["date", "actual", "prediction"]].copy()
chart = chart.rename(
    columns={
        "actual": "Thực tế giá(t+h)",
        "prediction": f"Dự báo ({selected_model})",
    }
)
if selected_model != "Naive" and naive_path.exists():
    naive = load_prediction(naive_path)[["date", "prediction"]].rename(
        columns={"prediction": "Naive"}
    )
    chart = chart.merge(naive, on="date", how="left")

mae = metric_value(selected_row, "MAE")
rmse = metric_value(selected_row, "RMSE")
smape_value = metric_value(selected_row, "sMAPE(%)", "MAPE_pct")
mase_value = metric_value(selected_row, "MASE")
da_value = metric_value(selected_row, "DA(%)", "DA_pct")

st.subheader(
    f"{selected_model} · {HORIZON_LABELS.get(selected_horizon, str(selected_horizon))} "
    f"· mode={selected_mode}"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("MAE", f"{mae:,.0f} VND/kg")
m2.metric("RMSE", f"{rmse:,.0f} VND/kg")
m3.metric("sMAPE", f"{smape_value:.2f}%")
m4.metric("MASE", f"{mase_value:.2f}")
m5.metric("Đúng xu hướng", f"{da_value:.2f}%")

st.line_chart(chart.set_index("date"), height=480)

# Thống kê chi tiết từ từng điểm dự báo.
prediction["absolute_error"] = (prediction["actual"] - prediction["prediction"]).abs()
prediction["percentage_error"] = np.where(
    prediction["actual"].abs() > 1e-9,
    prediction["absolute_error"] / prediction["actual"].abs() * 100,
    np.nan,
)
prediction["bias"] = prediction["prediction"] - prediction["actual"]

last = prediction.iloc[-1]
mean_actual = prediction["actual"].mean()
mean_prediction = prediction["prediction"].mean()
mean_bias = prediction["bias"].mean()
median_error = prediction["absolute_error"].median()
max_error = prediction["absolute_error"].max()
last_difference = last["prediction"] - last["actual"]

st.subheader("Thống kê chi tiết")
s1, s2, s3, s4 = st.columns(4)
s1.metric("Giá thực tế trung bình", f"{mean_actual:,.0f} VND/kg")
s2.metric("Giá dự báo trung bình", f"{mean_prediction:,.0f} VND/kg")
s3.metric("Độ lệch trung bình", f"{mean_bias:+,.0f} VND/kg")
s4.metric("Số phiên kiểm thử", f"{len(prediction):,}".replace(",", "."))

s5, s6, s7, s8 = st.columns(4)
s5.metric("Sai số trung vị", f"{median_error:,.0f} VND/kg")
s6.metric("Sai số lớn nhất", f"{max_error:,.0f} VND/kg")
s7.metric("Dự báo cuối kỳ", f"{last['prediction']:,.0f} VND/kg")
s8.metric("Chênh lệch cuối kỳ", f"{last_difference:+,.0f} VND/kg")

st.caption(
    f"Khoảng kiểm thử: {prediction['date'].min():%d/%m/%Y} – "
    f"{prediction['date'].max():%d/%m/%Y}. "
    f"Directional Accuracy tính lại từ file: {directional_accuracy(prediction):.2f}%. "
    f"sMAPE tính lại: {smape(prediction['actual'], prediction['prediction']):.2f}%."
)

error_chart = prediction[["date", "bias"]].rename(columns={"bias": "Sai số dự báo"})
st.subheader("Sai số theo thời gian")
st.line_chart(error_chart.set_index("date"), height=260)

st.subheader("10 phiên dự báo lệch nhiều nhất")
worst = (
    prediction.nlargest(10, "absolute_error")
    [["date", "actual", "prediction", "bias", "absolute_error", "percentage_error"]]
    .rename(
        columns={
            "date": "Ngày",
            "actual": "Thực tế",
            "prediction": "Dự báo",
            "bias": "Dự báo - thực tế",
            "absolute_error": "Sai số tuyệt đối",
            "percentage_error": "Sai số (%)",
        }
    )
)
st.dataframe(worst, use_container_width=True, hide_index=True)

st.subheader("So sánh model cùng chân trời")
display_columns = [
    column for column in
    ["model", "MAE", "RMSE", "sMAPE(%)", "MASE", "DA(%)", "test_rows"]
    if column in available.columns
]
st.dataframe(
    available[display_columns].rename(
        columns={
            "model": "Model",
            "sMAPE(%)": "sMAPE (%)",
            "DA(%)": "Directional Accuracy (%)",
            "test_rows": "Số dòng test",
        }
    ),
    use_container_width=True,
    hide_index=True,
)

st.caption("Kết quả phục vụ học tập và hỗ trợ quyết định, không phải khuyến nghị tài chính.")
