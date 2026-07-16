from __future__ import annotations

import re
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup

REQUEST_TIMEOUT = 15
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
}

COFFEE_SOURCES = [
    ("Nhà Bè Agri", "https://nhabeagri.com/gia-nong-san/gia-ca-phe/"),
    ("Chợ Giá", "https://chogia.vn/gia-ca-phe/"),
    ("Giacaphe.com", "https://giacaphe.com/gia-ca-phe-noi-dia/"),
]

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _get_text(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)


def _number(text: str) -> float:
    cleaned = re.sub(r"[^0-9,.-]", "", text).replace(".", "").replace(",", ".")
    return float(cleaned)


def _coffee_values(text: str) -> list[dict[str, Any]]:
    provinces = ["Đắk Lắk", "Lâm Đồng", "Gia Lai", "Đắk Nông"]
    rows: list[dict[str, Any]] = []
    for province in provinces:
        match = re.search(rf"{province}.{{0,100}}?([0-9]{{2,3}}[.,][0-9]{{3}})", text, re.I)
        if match:
            value = _number(match.group(1))
            if 20_000 <= value <= 300_000:
                rows.append({"province": province, "price_vnd_kg": value})
    if not rows:
        candidates = re.findall(r"\b([0-9]{2,3}[.,][0-9]{3})\b", text)
        values = [_number(value) for value in candidates]
        values = [value for value in values if 20_000 <= value <= 300_000]
        if values:
            rows.append({"province": "Việt Nam", "price_vnd_kg": values[0]})
    return rows


def fetch_live_coffee() -> dict[str, Any]:
    errors: list[str] = []
    for source_name, source_url in COFFEE_SOURCES:
        try:
            rows = _coffee_values(_get_text(source_url))
            if not rows:
                raise ValueError("Không tìm thấy giá phù hợp")
            return {
                "ok": True,
                "source_name": source_name,
                "source": source_url,
                "rows": rows,
                "average_vnd_kg": sum(row["price_vnd_kg"] for row in rows) / len(rows),
                "fetched_at": _now(),
            }
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
    return {"ok": False, "error": " | ".join(errors), "fetched_at": _now()}


def fetch_live_fuel() -> dict[str, Any]:
    # Giá dầu trong nước không có API công khai ổn định. Collector đọc nguồn HTML
    # và thất bại an toàn thay vì hiển thị một con số cũ như dữ liệu live.
    sources = [
        ("LuatVietnam", "https://luatvietnam.vn/bang-gia-xang-dau-hom-nay.html"),
        ("Petrolimex", "https://www.petrolimex.com.vn/nd/thong-cao-bao-chi.html"),
    ]
    errors: list[str] = []
    for source_name, source_url in sources:
        try:
            text = _get_text(source_url)
            patterns = [
                r"DO\s*0[,.]05S[^0-9]{0,80}([0-9]{1,2}[.,][0-9]{3})",
                r"dầu diesel[^0-9]{0,80}([0-9]{1,2}[.,][0-9]{3})",
            ]
            for pattern in patterns:
                match = re.search(pattern, text, re.I)
                if match:
                    value = _number(match.group(1))
                    if 10_000 <= value <= 40_000:
                        return {
                            "ok": True,
                            "source_name": source_name,
                            "source": source_url,
                            "product": "Dầu diesel DO 0,05S",
                            "price_vnd_litre": value,
                            "fetched_at": _now(),
                        }
            raise ValueError("Không tìm thấy giá dầu diesel")
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__}: {exc}")
    return {"ok": False, "error": " | ".join(errors), "fetched_at": _now()}


def fetch_live_weather(latitude: float = 12.6889, longitude: float = 108.0163, location: str = "Buôn Ma Thuột") -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum",
        "timezone": "Asia/Bangkok",
        "forecast_days": 7,
    }
    try:
        response = requests.get(WEATHER_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current", {})
        daily = payload.get("daily", {})
        rows = []
        dates = daily.get("time", [])
        for index, date in enumerate(dates):
            rows.append({
                "date": date,
                "temp_max_c": daily.get("temperature_2m_max", [None] * len(dates))[index],
                "temp_min_c": daily.get("temperature_2m_min", [None] * len(dates))[index],
                "rain_mm": daily.get("rain_sum", [None] * len(dates))[index],
                "precipitation_mm": daily.get("precipitation_sum", [None] * len(dates))[index],
            })
        return {
            "ok": True,
            "source_name": "Open-Meteo",
            "source": WEATHER_URL,
            "location": location,
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "rain_mm": current.get("rain"),
            "wind_kmh": current.get("wind_speed_10m"),
            "forecast": rows,
            "observed_at": current.get("time"),
            "fetched_at": _now(),
        }
    except Exception as exc:
        return {"ok": False, "source": WEATHER_URL, "error": f"{type(exc).__name__}: {exc}", "fetched_at": _now()}
