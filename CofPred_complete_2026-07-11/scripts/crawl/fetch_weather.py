"""
Script lấy dữ liệu thời tiết & lượng mưa lịch sử từ Open-Meteo
cho các vùng trồng cà phê chính ở Việt Nam.

Tọa độ được tra tự động qua Open-Meteo Geocoding API.

Cài đặt trước khi chạy:
    pip install openmeteo-requests requests-cache retry-requests numpy pandas
"""

import os
import time
import requests
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# ============================================================
# CẤU HÌNH
# ============================================================

# Danh sách vùng trồng cà phê — chỉ cần tên thành phố & tỉnh
# Tọa độ sẽ được tra tự động qua Geocoding API
COFFEE_REGIONS = [
    {"search": "Buon Ma Thuot",  "region": "Buon_Ma_Thuot", "province": "Đắk Lắk"},
    {"search": "Da Lat",         "region": "Da_Lat",         "province": "Lâm Đồng"},
    {"search": "Pleiku",         "region": "Pleiku",         "province": "Gia Lai"},
    {"search": "Kon Tum",        "region": "Kon_Tum",        "province": "Kon Tum"},
    {"search": "Gia Nghia",      "region": "Dak_Nong",       "province": "Đắk Nông"},
    {"search": "Son La",         "region": "Son_La",         "province": "Sơn La"},
]

START_DATE = "2008-01-01"
END_DATE   = "2026-06-01"
TIMEZONE   = "Asia/Bangkok"   # UTC+7

# Thư mục lưu kết quả (cùng thư mục với script này)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# KHỞI TẠO CLIENT (có cache + tự retry khi lỗi mạng)
# ============================================================
cache_session = requests_cache.CachedSession(
    os.path.join(OUTPUT_DIR, ".cache_openmeteo"),
    expire_after=-1   # cache vĩnh viễn (dữ liệu lịch sử không thay đổi)
)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# ============================================================
# CÁC BIẾN THỜI TIẾT CẦN LẤY
# ============================================================
DAILY_VARIABLES = [
    "temperature_2m_mean",           # Nhiệt độ TB ngày (°C)
    "temperature_2m_max",            # Nhiệt độ cao nhất (°C)
    "temperature_2m_min",            # Nhiệt độ thấp nhất (°C)
    "precipitation_sum",             # Tổng lượng mưa ngày (mm)
    "precipitation_hours",           # Số giờ có mưa trong ngày
    "rain_sum",                      # Lượng mưa thực (không tính tuyết) (mm)
    "relative_humidity_2m_mean",     # Độ ẩm TB (%)
    "relative_humidity_2m_max",      # Độ ẩm cao nhất (%)
    "relative_humidity_2m_min",      # Độ ẩm thấp nhất (%)
    "wind_speed_10m_mean",           # Tốc độ gió TB (km/h)
    "shortwave_radiation_sum",       # Bức xạ mặt trời (MJ/m²)
    "et0_fao_evapotranspiration",    # Bốc thoát hơi nước (mm) - quan trọng cho cà phê
]


# ============================================================
# HÀM TRA TỌA ĐỘ QUA GEOCODING API
# ============================================================
def geocode(city_name: str, country_code: str = "VN") -> dict:
    """
    Tra tọa độ (lat, lon, elevation) của một địa điểm
    qua Open-Meteo Geocoding API (miễn phí, không cần key).

    Args:
        city_name   : Tên thành phố/địa điểm (tiếng Anh không dấu)
        country_code: Mã quốc gia ISO 3166 (mặc định VN = Việt Nam)

    Returns:
        dict với các key: name, lat, lon, elevation, admin1, country
    """
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city_name,
        "count": 5,           # lấy 5 kết quả để lọc theo country
        "language": "en",
        "format": "json",
    }

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "results" not in data or len(data["results"]) == 0:
        raise ValueError(f"Không tìm thấy địa điểm: '{city_name}'")

    # Ưu tiên kết quả trong Việt Nam
    results = data["results"]
    vn_results = [r for r in results if r.get("country_code") == country_code]
    best = vn_results[0] if vn_results else results[0]

    return {
        "name":      best.get("name", city_name),
        "lat":       best["latitude"],
        "lon":       best["longitude"],
        "elevation": best.get("elevation", 0),
        "admin1":    best.get("admin1", ""),   # tỉnh/bang
        "country":   best.get("country", ""),
    }


def resolve_all_coordinates(regions: list) -> list:
    """
    Tra tọa độ cho tất cả vùng, in bảng kết quả để kiểm tra.
    Thêm lat/lon/elevation vào mỗi dict trong danh sách.
    """
    print("\n🔍 Tra tọa độ qua Open-Meteo Geocoding API...")
    print(f"{'Vùng':<20} {'Tìm kiếm':<18} {'Tên thực':<20} {'Lat':>8} {'Lon':>9} {'Elev(m)':>8} {'Tỉnh API'}")
    print("-" * 100)

    resolved = []
    for item in regions:
        try:
            geo = geocode(item["search"])
            row = {**item, "lat": geo["lat"], "lon": geo["lon"], "elevation": geo["elevation"]}
            resolved.append(row)
            print(
                f"{item['region']:<20} {item['search']:<18} {geo['name']:<20}"
                f" {geo['lat']:>8.4f} {geo['lon']:>9.4f} {geo['elevation']:>8.1f}"
                f"  {geo['admin1']}"
            )
            time.sleep(1.0)   # tránh gọi quá nhanh
        except Exception as e:
            print(f"{'❌ ' + item['region']:<20} Lỗi: {e}")

    print()
    return resolved


# ============================================================
# HÀM XỬ LÝ DỮ LIỆU NGÀY (DAILY)
# ============================================================
def process_daily_response(response, region_name, province) -> pd.DataFrame:
    """Chuyển đổi response daily thành DataFrame."""
    daily = response.Daily()

    date_range = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    ).tz_convert(response.Timezone().decode())

    daily_data = {"date": date_range}

    # Gán từng biến theo đúng thứ tự trong DAILY_VARIABLES
    for i, var_name in enumerate(DAILY_VARIABLES):
        daily_data[var_name] = daily.Variables(i).ValuesAsNumpy()

    df = pd.DataFrame(data=daily_data)
    df.insert(1, "region", region_name)
    df.insert(2, "province", province)
    df.insert(3, "latitude", response.Latitude())
    df.insert(4, "longitude", response.Longitude())
    df.insert(5, "elevation_m", response.Elevation())

    return df


# ============================================================
# HÀM LẤY DỮ LIỆU 1 VÙNG
# ============================================================
def fetch_region(region_name, lat, lon, province, max_retries: int = 3) -> pd.DataFrame:
    """
    Gọi Archive API và trả về DataFrame daily cho một vùng.
    Tự động chờ và thử lại nếu bị rate limit.
    """
    print(f"📡 {region_name} ({province}) — {lat:.4f}°N, {lon:.4f}°E")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": DAILY_VARIABLES,
        "timezone": TIMEZONE,
    }

    for attempt in range(1, max_retries + 1):
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]

            print(
                f"   → Grid point: {response.Latitude():.4f}°N {response.Longitude():.4f}°E"
                f" | Cao: {response.Elevation():.0f}m"
            )

            df = process_daily_response(response, region_name, province)
            print(f"   → {len(df):,} ngày dữ liệu\n")
            return df

        except Exception as e:
            err_msg = str(e)
            if "Minutely API request limit" in err_msg or "rate limit" in err_msg.lower():
                wait_sec = 65  # chờ hơn 1 phút để reset limit
                print(f"   ⏳ Rate limit! Chờ {wait_sec}s rồi thử lại (lần {attempt}/{max_retries})...")
                time.sleep(wait_sec)
            else:
                raise   # lỗi khác thì throw luôn

    raise RuntimeError(f"Vẫn bị rate limit sau {max_retries} lần thử: {region_name}")


# ============================================================
# CHẠY CHÍNH
# ============================================================
def main():
    print("=" * 60)
    print("🌧️  Thu thập dữ liệu thời tiết - CofPred Project")
    print("=" * 60)
    print(f"Giai đoạn : {START_DATE} → {END_DATE}")
    print(f"Số vùng   : {len(COFFEE_REGIONS)}")
    print(f"Lưu vào   : {OUTPUT_DIR}")

    # --- Bước 1: Tra tọa độ tự động ---
    regions_with_coords = resolve_all_coordinates(COFFEE_REGIONS)

    if not regions_with_coords:
        print("❌ Không tra được tọa độ nào!")
        return

    # --- Bước 2: Lấy dữ liệu thời tiết ---
    print("=" * 60)
    print("🌡️  Lấy dữ liệu thời tiết lịch sử...")
    print("=" * 60)

    all_daily = []
    for item in regions_with_coords:
        try:
            df = fetch_region(
                region_name=item["region"],
                lat=item["lat"],
                lon=item["lon"],
                province=item["province"],
            )
            all_daily.append(df)
        except Exception as e:
            print(f"❌ Lỗi khi lấy {item['region']}: {e}\n")

    # --- Bước 3: Lưu file ---
    print("=" * 60)
    print("💾 Lưu dữ liệu...")

    if all_daily:
        df_all = pd.concat(all_daily, ignore_index=True)

        # File tổng hợp
        out_all = os.path.join(OUTPUT_DIR, "weather_all_regions.csv")
        df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
        print(f"✅ Tổng hợp  : {out_all}")
        print(f"   → {len(df_all):,} dòng | {df_all['region'].nunique()} vùng")

        # File riêng từng vùng
        for region_name, group in df_all.groupby("region"):
            out_region = os.path.join(OUTPUT_DIR, f"weather_{region_name}.csv")
            group.to_csv(out_region, index=False, encoding="utf-8-sig")
            print(f"✅ {region_name:<20}: {len(group):,} dòng")

        # Thống kê nhanh
        print("\n📈 Thống kê tổng quan:")
        summary = df_all.groupby("region").agg(
            province=("province", "first"),
            n_days=("date", "count"),
            precip_mean_mm=("precipitation_sum", "mean"),
            precip_max_mm=("precipitation_sum", "max"),
            temp_mean_c=("temperature_2m_mean", "mean"),
            humidity_mean=("relative_humidity_2m_mean", "mean"),
        ).round(2)
        print(summary.to_string())
    else:
        print("❌ Không có dữ liệu nào được tải về!")

    print("\n✅ Hoàn thành!")


if __name__ == "__main__":
    main()
