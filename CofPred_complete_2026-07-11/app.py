from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import BDay

from live_data import fetch_live_coffee, fetch_live_fuel, fetch_live_weather


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "gia_cafe_master_full.csv"
PROVINCE_COFFEE_PATH = ROOT / "data" / "Processing" / "coffe" / "gia_cafe.csv"
INTERNATIONAL_COFFEE_PATH = ROOT / "data" / "Processing" / "coffe" / "london_robusta_price.csv"
FUEL_PATH = ROOT / "data" / "Processing" / "Fuel" / "diesel_2017_now_daily.csv"
WEATHER_PATH = ROOT / "data" / "Raw" / "weather" / "weather_all_regions.csv"
REGISTRY_PATH = ROOT / "models" / "model_registry.csv"
TARGET = "Gia_target"
DATE_CANDIDATES = ("Ngay", "date", "Date", "ngay", "DATE")
HORIZON_LABELS = {1: "Ngày làm việc tiếp theo", 5: "1 tuần", 21: "1 tháng", 63: "1 quý"}
DEFAULT_WEATHER_LOCATION = {
    "name": "Đắk Lắk — Buôn Ma Thuột",
    "latitude": 12.688928,
    "longitude": 108.016312,
}
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
    .stApp {background: linear-gradient(180deg, #fffaf3 0%, #f7efe4 100%);}
    [data-testid="stMetric"] {background: #ffffff; border: 1px solid #ead8c2;
        padding: 14px; border-radius: 14px; box-shadow: 0 5px 18px rgba(75,45,25,.06);}
    h1, h2, h3 {color: #4a2f20;}
    </style>
    """,
    unsafe_allow_html=True,
)


def detect_date_column(columns) -> str:
    for candidate in DATE_CANDIDATES:
        if candidate in columns:
            return candidate
    raise KeyError(f"Không tìm thấy cột ngày: {DATE_CANDIDATES}")


@st.cache_data
def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    date_col = detect_date_column(frame.columns)
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last")
    return frame, date_col


@st.cache_data
def load_province_coffee_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Ngay"] = pd.to_datetime(frame["Ngay"], errors="coerce")
    frame["Gia"] = pd.to_numeric(frame["Gia"], errors="coerce")
    frame["Thi_truong"] = frame["Thi_truong"].astype("string").str.strip()
    frame = frame.dropna(subset=["Ngay", "Thi_truong", "Gia"])
    # Một tỉnh có thể có nhiều loại đơn vị thu mua trong cùng ngày. Dùng trung
    # bình theo tỉnh/ngày để biểu đồ và thống kê không bị đếm trùng.
    return (
        frame.groupby(["Ngay", "Thi_truong"], as_index=False)["Gia"]
        .mean()
        .sort_values(["Ngay", "Thi_truong"])
        .reset_index(drop=True)
    )


@st.cache_data
def load_international_coffee_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame["Price"] = pd.to_numeric(frame["Price"], errors="coerce")
    return (
        frame.dropna(subset=["Date", "Price"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )


@st.cache_data
def load_fuel_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["ngay"] = pd.to_datetime(frame["ngay"], errors="coerce")
    frame["gia"] = pd.to_numeric(frame["gia"], errors="coerce")
    return (
        frame.dropna(subset=["ngay", "gia"])
        .sort_values("ngay")
        .drop_duplicates("ngay", keep="last")
        .reset_index(drop=True)
    )


@st.cache_data
def load_weather_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    parsed = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["date"] = parsed.dt.tz_convert("Asia/Bangkok").dt.tz_localize(None)
    numeric_columns = [
        "temperature_2m_mean", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "rain_sum", "relative_humidity_2m_mean",
        "wind_speed_10m_mean", "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
    ]
    existing = [column for column in numeric_columns if column in frame.columns]
    frame[existing] = frame[existing].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["date", "region", "province"]).sort_values(["region", "date"])
    frame["location"] = frame["province"].astype(str) + " — " + frame["region"].str.replace("_", " ")
    return frame.reset_index(drop=True)


@st.cache_data
def load_registry(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_bundle(path: Path) -> dict:
    return joblib.load(path)


@st.cache_data(ttl=1800, show_spinner=False)
def load_live_snapshot() -> dict:
    with ThreadPoolExecutor(max_workers=3) as executor:
        coffee_future = executor.submit(fetch_live_coffee)
        fuel_future = executor.submit(fetch_live_fuel)
        weather_future = executor.submit(
            fetch_live_weather,
            DEFAULT_WEATHER_LOCATION["latitude"],
            DEFAULT_WEATHER_LOCATION["longitude"],
            DEFAULT_WEATHER_LOCATION["name"],
        )
        return {
            "coffee": coffee_future.result(),
            "fuel": fuel_future.result(),
            "weather": weather_future.result(),
        }


@st.cache_data(ttl=900, show_spinner=False)
def load_location_live_weather(latitude: float, longitude: float, location: str) -> dict:
    return fetch_live_weather(latitude, longitude, location)


def format_vnd(value: float) -> str:
    return f"{value:,.0f} VND/kg".replace(",", ".")


def format_fuel(value: float) -> str:
    return f"{value:,.0f} VND/lít".replace(",", ".")


def format_usd_tonne(value: float) -> str:
    return f"{value:,.0f} USD/tấn".replace(",", ".")


def filter_period(frame: pd.DataFrame, date_column: str, range_label: str) -> pd.DataFrame:
    if frame.empty or range_label == "Tất cả":
        return frame.copy()
    end_date = pd.Timestamp(frame[date_column].max())
    start_date = end_date - PERIOD_OFFSETS[range_label]
    return frame[frame[date_column] >= start_date].copy()


def province_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values(["Thi_truong", "Ngay"]).copy()
    result = work.groupby("Thi_truong").agg(
        **{
            "Số phiên": ("Gia", "count"),
            "Giá đầu kỳ": ("Gia", "first"),
            "Giá cuối kỳ": ("Gia", "last"),
            "Trung bình": ("Gia", "mean"),
            "Thấp nhất": ("Gia", "min"),
            "Cao nhất": ("Gia", "max"),
        }
    ).reset_index().rename(columns={"Thi_truong": "Tỉnh / thị trường"})
    result["Thay đổi (VND/kg)"] = result["Giá cuối kỳ"] - result["Giá đầu kỳ"]
    result["Thay đổi (%)"] = np.where(
        result["Giá đầu kỳ"].abs() > 1e-9,
        result["Thay đổi (VND/kg)"] / result["Giá đầu kỳ"] * 100,
        np.nan,
    )
    numeric = result.columns.difference(["Tỉnh / thị trường", "Số phiên"])
    result[numeric] = result[numeric].round(2)
    return result.sort_values("Giá cuối kỳ", ascending=False).reset_index(drop=True)


def yearly_statistics(frame: pd.DataFrame, date_column: str, value_column: str) -> pd.DataFrame:
    work = frame.dropna(subset=[date_column, value_column]).sort_values(date_column).copy()
    work["Năm"] = work[date_column].dt.year
    grouped = work.groupby("Năm")[value_column]
    result = grouped.agg(
        **{
            "Số quan sát": "count",
            "Trung bình": "mean",
            "Trung vị": "median",
            "Thấp nhất": "min",
            "Cao nhất": "max",
            "Đầu năm": "first",
            "Cuối năm": "last",
        }
    ).reset_index()
    result["Thay đổi (%)"] = np.where(
        result["Đầu năm"].abs() > 1e-9,
        (result["Cuối năm"] / result["Đầu năm"] - 1.0) * 100,
        np.nan,
    )
    numeric = result.columns.difference(["Năm", "Số quan sát"])
    result[numeric] = result[numeric].round(2)
    return result


def get_importance(bundle: dict) -> pd.DataFrame | None:
    model = bundle["model"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        values = getattr(model, "coef_", None)
    if values is None:
        return None
    values = np.asarray(values, dtype=float).reshape(-1)
    features = bundle["features"]
    if len(values) != len(features):
        return None
    result = pd.DataFrame({"Feature": features, "Mức ảnh hưởng": np.abs(values)})
    return result.sort_values("Mức ảnh hưởng", ascending=False).head(10)


def predict_latest(frame: pd.DataFrame, bundle: dict) -> tuple[float, pd.Series]:
    features = bundle["features"]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        raise KeyError(f"Dữ liệu thiếu feature: {missing}")
    usable = frame.dropna(subset=[TARGET, *features])
    if usable.empty:
        raise ValueError("Không còn dòng dữ liệu hợp lệ để dự báo.")
    latest = usable.iloc[-1]
    X_latest = latest[features].astype(float).to_frame().T
    raw = float(bundle["model"].predict(X_latest)[0])
    prediction = float(latest[TARGET]) + raw if bundle["target_mode"] == "delta" else raw
    return prediction, latest


if not DATA_PATH.exists():
    st.error(f"Không tìm thấy dữ liệu: {DATA_PATH}")
    st.stop()

data, date_col = load_data(DATA_PATH)
price_data = data.dropna(subset=[TARGET]).copy()
latest = price_data.iloc[-1]
latest_price = float(latest[TARGET])

st.title("☕ CofPred")
st.caption(
    "Coffee Price Forecasting and Decision Support Dashboard · "
    f"Dữ liệu mới nhất: {latest[date_col]:%d/%m/%Y}"
)

refresh_col, status_col = st.columns([1, 4])
if refresh_col.button("↻ Làm mới live", use_container_width=True):
    load_live_snapshot.clear()
    load_location_live_weather.clear()
    st.rerun()
status_col.caption(
    "Live được lưu cache để hạn chế gọi nguồn quá nhiều. Nút làm mới sẽ yêu cầu lại cả ba nguồn."
)

live_snapshot = load_live_snapshot()
live_errors = []
with st.expander("Dữ liệu live: cà phê · dầu diesel · thời tiết", expanded=True):
    live_1, live_2, live_3 = st.columns(3)
    live_coffee = live_snapshot["coffee"]
    if live_coffee.get("ok"):
        coffee_delta = live_coffee.get("delta")
        live_1.metric(
            "Cà phê Tây Nguyên live",
            format_vnd(float(live_coffee["value"])),
            f"{coffee_delta:+,.0f} VND/kg".replace(",", ".") if coffee_delta is not None else None,
        )
        live_1.caption(f"Ngày nguồn: {pd.to_datetime(live_coffee['observed_at']):%d/%m/%Y}")
    else:
        live_1.metric("Cà phê Tây Nguyên live", "Không lấy được")
        live_errors.append(f"Cà phê: {live_coffee.get('error', 'không rõ lỗi')}")

    live_fuel = live_snapshot["fuel"]
    if live_fuel.get("ok"):
        fuel_delta = live_fuel.get("delta")
        live_2.metric(
            "Dầu DO 0,05S live",
            format_fuel(float(live_fuel["value"])),
            f"{fuel_delta:+,.0f} VND/lít".replace(",", ".") if fuel_delta is not None else None,
        )
        live_2.caption(f"Ngày nguồn: {pd.to_datetime(live_fuel['observed_at']):%d/%m/%Y}")
    else:
        live_2.metric("Dầu DO 0,05S live", "Không lấy được")
        live_errors.append(f"Dầu: {live_fuel.get('error', 'không rõ lỗi')}")

    live_weather = live_snapshot["weather"]
    if live_weather.get("ok"):
        live_3.metric(
            "Thời tiết Buôn Ma Thuột live",
            f"{float(live_weather['temperature']):.1f} °C",
            f"Độ ẩm {float(live_weather['humidity']):.0f}%",
            delta_color="off",
        )
        live_3.caption(
            f"Gió {float(live_weather['wind_speed']):.1f} km/h · "
            f"Mưa {float(live_weather.get('rain') or 0):.1f} mm · "
            f"{pd.to_datetime(live_weather['observed_at']):%d/%m/%Y %H:%M}"
        )
    else:
        live_3.metric("Thời tiết Buôn Ma Thuột live", "Không lấy được")
        live_errors.append(f"Thời tiết: {live_weather.get('error', 'không rõ lỗi')}")

    coffee_source_label = live_coffee.get("source_name", "Nguồn cà phê dự phòng")
    coffee_source_url = live_coffee.get("source", "https://nhabeagri.com/gia-nong-san/gia-ca-phe/")
    st.caption(
        f"Nguồn live: [{coffee_source_label}]({coffee_source_url}) · "
        "[LuatVietnam](https://luatvietnam.vn/bang-gia-xang-dau-hom-nay.html) · "
        "[Open-Meteo](https://open-meteo.com/en/docs) · tự làm mới sau tối đa 30 phút."
    )
    if live_errors:
        st.warning(" | ".join(live_errors))

(
    overview_tab,
    yearly_tab,
    fuel_tab,
    weather_tab,
    prediction_tab,
    data_tab,
) = st.tabs([
    "Tổng quan giá cà phê",
    "Thống kê theo năm",
    "Dầu diesel",
    "Thời tiết",
    "Dự báo & mô hình",
    "Dữ liệu",
])

with overview_tab:
    range_label = st.selectbox(
        "Khoảng thời gian",
        ["1 tháng", "3 tháng", "6 tháng", "1 năm", "Tất cả"],
        index=3,
    )
    history = filter_period(price_data, date_col, range_label)

    period_first = history.iloc[0]
    period_last = history.iloc[-1]
    period_first_price = float(period_first[TARGET])
    period_last_price = float(period_last[TARGET])
    period_change = period_last_price - period_first_price
    period_change_pct = period_change / period_first_price * 100 if period_first_price else 0.0
    trend = "Tăng" if period_change_pct > 0.25 else "Giảm" if period_change_pct < -0.25 else "Đi ngang"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá cuối kỳ", format_vnd(period_last_price), f"{period_last[date_col]:%d/%m/%Y}", delta_color="off")
    col2.metric("Giá đầu kỳ", format_vnd(period_first_price), f"{period_first[date_col]:%d/%m/%Y}", delta_color="off")
    col3.metric("Thay đổi trong kỳ", format_vnd(period_change), f"{period_change_pct:+.2f}%")
    col4.metric("Xu hướng giai đoạn", trend, f"{len(history):,} phiên".replace(",", "."), delta_color="off")

    chart = history.set_index(date_col)[[TARGET]].rename(columns={TARGET: "Giá cà phê (VND/kg)"})
    st.line_chart(chart, height=420)

    st.info(
        f"Các thẻ và biểu đồ đang được tính lại theo đúng khoảng {range_label}: "
        f"{period_first[date_col]:%d/%m/%Y} – {period_last[date_col]:%d/%m/%Y}. "
        "Dữ liệu live được hiển thị riêng và chưa đưa vào model cho đến khi chạy lại feature engineering."
    )

    st.divider()
    st.subheader("Giá cà phê theo tỉnh tại Việt Nam")
    st.caption(
        "Thống kê từ các tỉnh/thị trường có trong file dự án. "
        "Mỗi tỉnh trong một ngày được lấy trung bình nếu có nhiều bản ghi thu mua."
    )
    if not PROVINCE_COFFEE_PATH.exists():
        st.error(f"Không tìm thấy dữ liệu giá theo tỉnh: {PROVINCE_COFFEE_PATH}")
    else:
        province_data = load_province_coffee_data(PROVINCE_COFFEE_PATH)
        province_names = sorted(province_data["Thi_truong"].unique().tolist())
        preferred_provinces = [
            province for province in ["Đắk Lắk", "Lâm Đồng", "Gia Lai", "Đắk Nông"]
            if province in province_names
        ]
        selected_provinces = st.multiselect(
            "Chọn tỉnh / thị trường",
            province_names,
            default=preferred_provinces or province_names,
            key="overview_provinces",
        )

        if not selected_provinces:
            st.warning("Hãy chọn ít nhất một tỉnh để xem thống kê.")
        else:
            selected_province_data = province_data[
                province_data["Thi_truong"].isin(selected_provinces)
            ].copy()
            province_period = filter_period(selected_province_data, "Ngay", range_label)

            if province_period.empty:
                st.warning(f"Không có dữ liệu tỉnh trong khoảng {range_label} đã chọn.")
            else:
                province_stats = province_statistics(province_period)
                highest_province_row = province_period.loc[province_period["Gia"].idxmax()]
                lowest_province_row = province_period.loc[province_period["Gia"].idxmin()]
                province_daily_average = province_period.groupby("Ngay")["Gia"].mean().sort_index()
                province_first_average = float(province_daily_average.iloc[0])
                province_last_average = float(province_daily_average.iloc[-1])
                province_average_change = province_last_average - province_first_average
                province_average_pct = (
                    province_average_change / province_first_average * 100
                    if province_first_average else 0.0
                )

                pv1, pv2, pv3, pv4 = st.columns(4)
                pv1.metric(
                    "Giá tỉnh TB cuối kỳ",
                    format_vnd(province_last_average),
                    f"{province_average_pct:+.2f}%",
                )
                pv2.metric(
                    "Cao nhất trong kỳ",
                    format_vnd(float(highest_province_row["Gia"])),
                    str(highest_province_row["Thi_truong"]),
                    delta_color="off",
                )
                pv3.metric(
                    "Thấp nhất trong kỳ",
                    format_vnd(float(lowest_province_row["Gia"])),
                    str(lowest_province_row["Thi_truong"]),
                    delta_color="off",
                )
                pv4.metric(
                    "Tỉnh có dữ liệu",
                    int(province_period["Thi_truong"].nunique()),
                    f"{province_period['Ngay'].min():%d/%m/%Y} – {province_period['Ngay'].max():%d/%m/%Y}",
                    delta_color="off",
                )

                province_chart = province_period.pivot_table(
                    index="Ngay", columns="Thi_truong", values="Gia", aggfunc="mean"
                ).sort_index()
                st.line_chart(province_chart, height=390)
                st.markdown("##### Bảng thống kê giá theo tỉnh trong khoảng đã chọn")
                st.dataframe(province_stats, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Giá cà phê quốc tế — Robusta London")
    st.caption(
        "Giá hợp đồng Robusta London theo dữ liệu có sẵn trong dự án, đơn vị USD/tấn. "
        "Phần này dùng cùng lựa chọn khoảng thời gian ở phía trên."
    )
    if not INTERNATIONAL_COFFEE_PATH.exists():
        st.error(f"Không tìm thấy dữ liệu giá quốc tế: {INTERNATIONAL_COFFEE_PATH}")
    else:
        international_data = load_international_coffee_data(INTERNATIONAL_COFFEE_PATH)
        international_period = filter_period(international_data, "Date", range_label)
        international_first = international_period.iloc[0]
        international_last = international_period.iloc[-1]
        international_change = float(international_last["Price"] - international_first["Price"])
        international_pct = (
            international_change / float(international_first["Price"]) * 100
            if float(international_first["Price"]) else 0.0
        )

        int1, int2, int3, int4 = st.columns(4)
        int1.metric(
            "Robusta London cuối kỳ",
            format_usd_tonne(float(international_last["Price"])),
            f"{international_pct:+.2f}%",
        )
        int2.metric(
            "Robusta London đầu kỳ",
            format_usd_tonne(float(international_first["Price"])),
            f"{international_first['Date']:%d/%m/%Y}",
            delta_color="off",
        )
        int3.metric(
            "Cao nhất trong kỳ",
            format_usd_tonne(float(international_period["Price"].max())),
        )
        int4.metric(
            "Thấp nhất trong kỳ",
            format_usd_tonne(float(international_period["Price"].min())),
        )

        international_chart = international_period.set_index("Date")[["Price"]].rename(
            columns={"Price": "Robusta London (USD/tấn)"}
        )
        st.line_chart(international_chart, height=390)
        international_summary = pd.DataFrame([
            {
                "Thị trường": "Robusta London",
                "Số phiên": len(international_period),
                "Đầu kỳ": float(international_first["Price"]),
                "Cuối kỳ": float(international_last["Price"]),
                "Trung bình": float(international_period["Price"].mean()),
                "Trung vị": float(international_period["Price"].median()),
                "Thấp nhất": float(international_period["Price"].min()),
                "Cao nhất": float(international_period["Price"].max()),
                "Thay đổi (USD/tấn)": international_change,
                "Thay đổi (%)": international_pct,
            }
        ]).round(2)
        st.markdown("##### Bảng thống kê thị trường quốc tế trong khoảng đã chọn")
        st.dataframe(international_summary, use_container_width=True, hide_index=True)

with yearly_tab:
    st.subheader("Thống kê giá cà phê qua các năm")
    coffee_years = sorted(price_data[date_col].dt.year.unique().astype(int).tolist())
    selected_coffee_years = st.multiselect(
        "Chọn năm để so sánh",
        coffee_years,
        default=coffee_years,
        key="coffee_years",
    )

    if not selected_coffee_years:
        st.warning("Hãy chọn ít nhất một năm.")
    else:
        coffee_filtered = price_data[price_data[date_col].dt.year.isin(selected_coffee_years)].copy()
        coffee_stats = yearly_statistics(coffee_filtered, date_col, TARGET)
        best_average = coffee_stats.loc[coffee_stats["Trung bình"].idxmax()]
        largest_gain = coffee_stats.loc[coffee_stats["Thay đổi (%)"].idxmax()]
        highest_row = coffee_filtered.loc[coffee_filtered[TARGET].idxmax()]

        y1, y2, y3 = st.columns(3)
        y1.metric("Năm có giá trung bình cao nhất", int(best_average["Năm"]), format_vnd(best_average["Trung bình"]))
        y2.metric("Năm tăng mạnh nhất", int(largest_gain["Năm"]), f"{largest_gain['Thay đổi (%)']:+.2f}%")
        y3.metric("Mức giá cao nhất", format_vnd(float(highest_row[TARGET])), f"{highest_row[date_col]:%d/%m/%Y}")

        annual_average = coffee_stats.set_index("Năm")[["Trung bình"]].rename(
            columns={"Trung bình": "Giá trung bình (VND/kg)"}
        )
        st.markdown("##### Giá trung bình từng năm")
        st.bar_chart(annual_average, height=350)

        coffee_filtered["Tháng"] = coffee_filtered[date_col].dt.month
        coffee_filtered["Năm"] = coffee_filtered[date_col].dt.year
        monthly_compare = coffee_filtered.pivot_table(
            index="Tháng", columns="Năm", values=TARGET, aggfunc="mean"
        ).sort_index()
        st.markdown("##### So sánh giá trung bình theo tháng giữa các năm")
        st.line_chart(monthly_compare, height=390)

        st.markdown("##### Bảng thống kê chi tiết")
        st.dataframe(coffee_stats, use_container_width=True, hide_index=True)

with fuel_tab:
    st.subheader("Lịch sử giá dầu diesel")
    if not FUEL_PATH.exists():
        st.error(f"Không tìm thấy dữ liệu dầu: {FUEL_PATH}")
    else:
        fuel_data = load_fuel_data(FUEL_PATH)
        fuel_years = sorted(fuel_data["ngay"].dt.year.unique().astype(int).tolist())
        default_fuel_years = fuel_years[-5:]
        selected_fuel_years = st.multiselect(
            "Chọn năm dầu diesel",
            fuel_years,
            default=default_fuel_years,
            key="fuel_years",
        )

        if not selected_fuel_years:
            st.warning("Hãy chọn ít nhất một năm để xem dữ liệu dầu.")
        else:
            fuel_filtered = fuel_data[fuel_data["ngay"].dt.year.isin(selected_fuel_years)].copy()
            fuel_latest = fuel_filtered.iloc[-1]
            fuel_previous = fuel_filtered.iloc[-2] if len(fuel_filtered) > 1 else fuel_latest
            fuel_delta = float(fuel_latest["gia"] - fuel_previous["gia"])
            fuel_stats = yearly_statistics(fuel_filtered, "ngay", "gia")

            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Giá dầu mới nhất", format_fuel(float(fuel_latest["gia"])), f"{fuel_latest['ngay']:%d/%m/%Y}")
            f2.metric("Thay đổi phiên gần nhất", format_fuel(fuel_delta))
            f3.metric("Cao nhất giai đoạn", format_fuel(float(fuel_filtered["gia"].max())))
            f4.metric("Thấp nhất giai đoạn", format_fuel(float(fuel_filtered["gia"].min())))

            fuel_chart = fuel_filtered.set_index("ngay")[["gia"]].rename(
                columns={"gia": "Giá diesel (VND/lít)"}
            )
            st.line_chart(fuel_chart, height=400)

            st.markdown("##### Giá dầu trung bình từng năm")
            st.bar_chart(
                fuel_stats.set_index("Năm")[["Trung bình"]].rename(
                    columns={"Trung bình": "Giá trung bình (VND/lít)"}
                ),
                height=330,
            )
            st.dataframe(fuel_stats, use_container_width=True, hide_index=True)
            st.caption(
                f"Nguồn trong file: {', '.join(sorted(fuel_filtered['nguon'].dropna().astype(str).unique()))}. "
                f"Dữ liệu hiện có đến {fuel_data['ngay'].max():%d/%m/%Y}."
            )

with weather_tab:
    st.subheader("Thời tiết các tỉnh và vùng trồng cà phê")
    if not WEATHER_PATH.exists():
        st.error(f"Không tìm thấy dữ liệu thời tiết: {WEATHER_PATH}")
    else:
        weather_data = load_weather_data(WEATHER_PATH)
        locations = sorted(weather_data["location"].unique().tolist())
        selected_location = st.selectbox("Tỉnh / khu vực", locations, key="weather_location")
        location_data = weather_data[weather_data["location"] == selected_location].copy()
        coordinate = location_data.iloc[-1]
        current_location_weather = load_location_live_weather(
            float(coordinate["latitude"]),
            float(coordinate["longitude"]),
            selected_location,
        )
        if current_location_weather.get("ok"):
            current_1, current_2, current_3, current_4 = st.columns(4)
            current_1.metric("Nhiệt độ live", f"{float(current_location_weather['temperature']):.1f} °C")
            current_2.metric("Độ ẩm live", f"{float(current_location_weather['humidity']):.0f}%")
            current_3.metric("Gió live", f"{float(current_location_weather['wind_speed']):.1f} km/h")
            current_4.metric("Mưa live", f"{float(current_location_weather.get('rain') or 0):.1f} mm")
            st.caption(f"Thời điểm Open-Meteo: {pd.to_datetime(current_location_weather['observed_at']):%d/%m/%Y %H:%M}")
        else:
            st.warning(f"Không lấy được thời tiết live cho {selected_location}: {current_location_weather.get('error')}")
        weather_years = sorted(location_data["date"].dt.year.unique().astype(int).tolist())
        selected_weather_years = st.multiselect(
            "Chọn năm thời tiết",
            weather_years,
            default=weather_years[-5:],
            key="weather_years",
        )

        weather_metrics = {
            "Nhiệt độ trung bình (°C)": ("temperature_2m_mean", "mean"),
            "Lượng mưa (mm)": ("rain_sum", "sum"),
            "Độ ẩm trung bình (%)": ("relative_humidity_2m_mean", "mean"),
            "Tốc độ gió trung bình": ("wind_speed_10m_mean", "mean"),
            "Bức xạ mặt trời": ("shortwave_radiation_sum", "mean"),
            "Bốc thoát hơi nước ET0": ("et0_fao_evapotranspiration", "sum"),
        }
        selected_metric_label = st.selectbox("Chỉ số biểu đồ", list(weather_metrics), key="weather_metric")

        if not selected_weather_years:
            st.warning("Hãy chọn ít nhất một năm thời tiết.")
        else:
            weather_filtered = location_data[
                location_data["date"].dt.year.isin(selected_weather_years)
            ].copy()
            w1, w2, w3, w4 = st.columns(4)
            w1.metric("Nhiệt độ trung bình", f"{weather_filtered['temperature_2m_mean'].mean():.1f} °C")
            w2.metric("Tổng lượng mưa", f"{weather_filtered['rain_sum'].sum():,.0f} mm".replace(",", "."))
            w3.metric("Độ ẩm trung bình", f"{weather_filtered['relative_humidity_2m_mean'].mean():.1f}%")
            w4.metric("Gió trung bình", f"{weather_filtered['wind_speed_10m_mean'].mean():.1f}")

            metric_column, aggregation = weather_metrics[selected_metric_label]
            weather_filtered["Tháng"] = weather_filtered["date"].dt.to_period("M").dt.to_timestamp()
            monthly_weather = weather_filtered.groupby("Tháng")[metric_column].agg(aggregation).to_frame(
                selected_metric_label
            )
            st.line_chart(monthly_weather, height=400)

            annual_weather = weather_filtered.assign(Năm=weather_filtered["date"].dt.year).groupby("Năm").agg(
                **{
                    "Số ngày": ("date", "count"),
                    "Nhiệt độ TB (°C)": ("temperature_2m_mean", "mean"),
                    "Nhiệt độ cao nhất (°C)": ("temperature_2m_max", "max"),
                    "Nhiệt độ thấp nhất (°C)": ("temperature_2m_min", "min"),
                    "Tổng mưa (mm)": ("rain_sum", "sum"),
                    "Độ ẩm TB (%)": ("relative_humidity_2m_mean", "mean"),
                    "Gió TB": ("wind_speed_10m_mean", "mean"),
                    "Tổng ET0": ("et0_fao_evapotranspiration", "sum"),
                }
            ).reset_index()
            numeric_weather = annual_weather.columns.difference(["Năm", "Số ngày"])
            annual_weather[numeric_weather] = annual_weather[numeric_weather].round(2)
            st.markdown("##### Thống kê thời tiết từng năm")
            st.dataframe(annual_weather, use_container_width=True, hide_index=True)
            st.caption(
                f"Dữ liệu {selected_location} từ {location_data['date'].min():%d/%m/%Y} "
                f"đến {location_data['date'].max():%d/%m/%Y}."
            )

with prediction_tab:
    if not REGISTRY_PATH.exists():
        st.warning("Chưa có model đã xuất. Hãy chạy lệnh train trước khi mở chức năng dự báo.")
        st.code(
            "python train_models.py --horizons 1 5 21 63 "
            "--models LinearRegression RandomForest XGBoost LightGBM",
            language="bash",
        )
    else:
        registry = load_registry(REGISTRY_PATH)
        horizons = sorted(registry["horizon"].astype(int).unique().tolist())
        selected_horizon = st.selectbox(
            "Chân trời dự báo",
            horizons,
            format_func=lambda value: HORIZON_LABELS.get(value, f"{value} phiên"),
        )
        available = registry[registry["horizon"].astype(int) == selected_horizon].copy()
        available = available.sort_values(["RMSE", "MAE"])
        selected_model = st.selectbox("Mô hình", available["model"].tolist())
        selected = available[available["model"] == selected_model].iloc[0]

        try:
            bundle = load_bundle(ROOT / selected["model_path"])
            forecast, forecast_anchor = predict_latest(data, bundle)
            anchor_price = float(forecast_anchor[TARGET])
            forecast_change = forecast - anchor_price
            forecast_pct = forecast_change / anchor_price * 100 if anchor_price else 0.0
            forecast_date = pd.Timestamp(forecast_anchor[date_col]) + BDay(int(selected_horizon))

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Giá dự báo", format_vnd(forecast))
            p2.metric("Mốc dự báo", f"{forecast_date:%d/%m/%Y}")
            p3.metric("So với giá cuối", format_vnd(forecast_change), f"{forecast_pct:+.2f}%")
            p4.metric("RMSE kiểm thử", format_vnd(float(selected["RMSE"])))

            if forecast_pct > 1.0:
                recommendation = "Mô hình nghiêng về xu hướng tăng. Nên theo dõi cơ hội bán và kiểm tra thêm dữ liệu thị trường mới."
            elif forecast_pct < -1.0:
                recommendation = "Mô hình nghiêng về xu hướng giảm. Nên thận trọng với quyết định mua và theo dõi biến động đầu vào."
            else:
                recommendation = "Mô hình cho thấy giá có khả năng đi ngang. Chưa có tín hiệu đủ mạnh để kết luận tăng hoặc giảm."
            st.success(recommendation)

            importance = get_importance(bundle)
            if importance is not None:
                st.subheader("Feature ảnh hưởng nhiều nhất")
                st.bar_chart(importance.set_index("Feature"), height=330)

            st.subheader("So sánh model cùng chân trời")
            display_columns = ["model", "MAE", "RMSE", "MAPE_pct", "DA_pct", "test_rows"]
            st.dataframe(
                available[display_columns].rename(columns={
                    "model": "Model", "MAPE_pct": "MAPE (%)",
                    "DA_pct": "Directional Accuracy (%)", "test_rows": "Số dòng test",
                }),
                use_container_width=True,
                hide_index=True,
            )
        except Exception as exc:
            st.error(f"Không thể nạp hoặc dự báo bằng model: {type(exc).__name__}: {exc}")

    st.caption("Kết quả chỉ hỗ trợ học tập và ra quyết định; không phải khuyến nghị tài chính.")

with data_tab:
    d1, d2, d3 = st.columns(3)
    d1.metric("Số dòng", f"{len(data):,}".replace(",", "."))
    d2.metric("Số cột", len(data.columns))
    d3.metric("Khoảng dữ liệu", f"{data[date_col].min():%d/%m/%Y} – {data[date_col].max():%d/%m/%Y}")

    st.subheader("30 dòng gần nhất")
    st.dataframe(data.tail(30).sort_values(date_col, ascending=False), use_container_width=True, hide_index=True)
    st.subheader("Tải các nguồn dữ liệu")
    download_1, download_2, download_3 = st.columns(3)
    download_1.download_button(
        "Tải dữ liệu master",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="gia_cafe_master_full.csv",
        mime="text/csv",
    )
    if FUEL_PATH.exists():
        download_2.download_button(
            "Tải dữ liệu dầu",
            data=FUEL_PATH.read_bytes(),
            file_name=FUEL_PATH.name,
            mime="text/csv",
        )
    if WEATHER_PATH.exists():
        download_3.download_button(
            "Tải dữ liệu thời tiết",
            data=WEATHER_PATH.read_bytes(),
            file_name=WEATHER_PATH.name,
            mime="text/csv",
        )
    download_4, download_5 = st.columns(2)
    if PROVINCE_COFFEE_PATH.exists():
        download_4.download_button(
            "Tải giá cà phê theo tỉnh",
            data=PROVINCE_COFFEE_PATH.read_bytes(),
            file_name=PROVINCE_COFFEE_PATH.name,
            mime="text/csv",
        )
    if INTERNATIONAL_COFFEE_PATH.exists():
        download_5.download_button(
            "Tải giá Robusta London",
            data=INTERNATIONAL_COFFEE_PATH.read_bytes(),
            file_name=INTERNATIONAL_COFFEE_PATH.name,
            mime="text/csv",
        )
