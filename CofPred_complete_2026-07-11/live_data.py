"""Fetch small, cache-friendly live snapshots for the CofPred dashboard.

These functions do not rewrite the modeling master dataset. Live observations
are displayed separately because the master file needs feature engineering
before it is safe to use for model inference or retraining.
"""
from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup


COFFEE_SOURCES = [
    ("Nhà Bè Agri", "https://nhabeagri.com/gia-nong-san/gia-ca-phe/"),
    ("Chợ Giá", "https://chogia.vn/gia-ca-phe/"),
    ("HCT", "https://hct.vn/kien-thuc-dau-tu/gia-ca-phe-truc-tuyen-san-london-new-york-cap-nhat-realtime-1240"),
    ("Giacaphe.com", "https://giacaphe.com/gia-ca-phe-noi-dia/"),
]
FUEL_URL = "https://luatvietnam.vn/bang-gia-xang-dau-hom-nay.html"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi,en;q=0.9",
}


def _get_text(url: str, timeout: int = 20) -> str:
    response = requests.get(url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml").get_text(" ", strip=True)


def _fold(value: str) -> str:
    value = value.replace("đ", "d").replace("Đ", "D")
    value = unicodedata.normalize("NFD", value)
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    digits = re.sub(r"[^0-9]", "", value)
    return int(digits) if digits else None


def _result_error(source: str, exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "source": source,
        "error": f"{type(exc).__name__}: {exc}",
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_coffee_text(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", text)
    date_match = re.search(r"(?:ngày|ngay)\s*(\d{1,2}/\d{1,2}/\d{4})", normalized, re.I)
    patterns = [
        r"trung\s*bình.{0,35}?(\d{2,3}[.,]\d{3})\s*(?:đ|vnđ|đồng)\s*/?\s*kg",
        r"trung\s*bình.{0,20}?(\d{4,6})\s*(?:đ|vnđ|đồng)\s*/?\s*kg",
    ]
    price_match = next((match for pattern in patterns if (match := re.search(pattern, normalized, re.I))), None)
    if price_match is None:
        raise ValueError("Không tìm thấy giá cà phê trung bình trong trang nguồn")

    price = _integer(price_match.group(1))
    if price is None or not 20_000 <= price <= 300_000:
        raise ValueError(f"Giá cà phê ngoài phạm vi hợp lệ: {price}")

    delta = None
    direction = re.search(r"\b(tăng|giảm)(?:\s+mạnh|\s+nhẹ)?\s*([\d.,]+)", normalized, re.I)
    if direction:
        magnitude = _integer(direction.group(2))
        if magnitude is not None:
            delta = magnitude if direction.group(1).lower() == "tăng" else -magnitude

    observed_at = (
        datetime.strptime(date_match.group(1), "%d/%m/%Y")
        if date_match
        else datetime.now()
    )
    return {"value": price, "delta": delta, "observed_at": observed_at.isoformat(timespec="seconds")}


def parse_coffee_province_table(text: str) -> dict[str, Any]:
    """Parse province prices and calculate an equal-weight Tây Nguyên average."""
    normalized = re.sub(r"\s+", " ", _fold(text))
    locations = ["Dak Lak", "Lam Dong", "Gia Lai", "Dak Nong"]
    values: list[int] = []
    deltas: list[int] = []
    number_pattern = r"[+-]?\s*(?:\d{1,3}(?:[.,]\d{3})+|\d{1,6})"

    for location in locations:
        price_matches = list(re.finditer(
            rf"{re.escape(location)}[^0-9]{{0,30}}({number_pattern})", normalized, re.I
        ))
        selected = None
        for price_match in price_matches:
            price = _integer(price_match.group(1))
            if price is not None and 20_000 <= price <= 300_000:
                selected = (price_match, price)
                break
        if selected is None:
            continue

        price_match, price = selected
        values.append(price)
        tail = normalized[price_match.end(): price_match.end() + 35]
        delta_match = re.search(number_pattern, tail)
        if delta_match:
            raw_delta = delta_match.group(0).strip()
            delta = _integer(raw_delta)
            if delta is not None and delta <= 20_000:
                deltas.append(-delta if raw_delta.startswith("-") else delta)

    if len(values) < 2:
        raise ValueError("Không tìm đủ giá của các tỉnh Tây Nguyên")

    date_match = re.search(r"(\d{1,2}/\d{1,2}/\d{4})", normalized)
    observed_at = datetime.strptime(date_match.group(1), "%d/%m/%Y") if date_match else datetime.now()
    return {
        "value": round(sum(values) / len(values)),
        "delta": round(sum(deltas) / len(deltas)) if deltas else None,
        "observed_at": observed_at.isoformat(timespec="seconds"),
        "province_count": len(values),
    }


def fetch_live_coffee() -> dict[str, Any]:
    errors = []
    for source_name, source_url in COFFEE_SOURCES:
        try:
            text = _get_text(source_url)
            try:
                parsed = parse_coffee_text(text)
            except ValueError:
                parsed = parse_coffee_province_table(text)
            return {
                "ok": True,
                "name": "Giá cà phê Tây Nguyên trung bình",
                "unit": "VND/kg",
                "source_name": source_name,
                "source": source_url,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
                **parsed,
            }
        except Exception as exc:
            errors.append(f"{source_name}: {type(exc).__name__} {exc}")

    return {
        "ok": False,
        "source": COFFEE_SOURCES[0][1],
        "error": " | ".join(errors),
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def parse_fuel_text(text: str) -> dict[str, Any]:
    normalized = re.sub(r"\s+", " ", _fold(text))
    header_pattern = re.compile(
        r"Gia\s*dieu\s*chinh\s*(?:(\d{1,2}:\d{2})\s*)?ngay\s*(\d{2}/\d{2}/\d{4})",
        re.I,
    )
    headers = list(header_pattern.finditer(normalized))
    candidates = []
    for index, header in enumerate(headers):
        block = normalized[
            header.end(): headers[index + 1].start() if index + 1 < len(headers) else len(normalized)
        ]
        product = re.search(r"(?:Dau\s*)?(?:DO|Diezen)\s*0[.,]0?5S(?:-II)?", block, re.I)
        if not product:
            continue
        numbers = re.findall(r"[-–]?\s*\d[\d.]*", block[product.end():])
        price = _integer(numbers[0]) if numbers else None
        if price is None or not 5_000 <= price <= 100_000:
            continue
        raw_delta = numbers[1] if len(numbers) > 1 else None
        delta = _integer(raw_delta)
        if raw_delta and raw_delta.strip().startswith(("-", "–")) and delta is not None:
            delta = -delta
        observed_at = datetime.strptime(header.group(2), "%d/%m/%Y")
        candidates.append((observed_at, price, delta))

    if not candidates:
        raise ValueError("Không tìm thấy giá dầu DO 0,05S trong trang nguồn")
    observed_at, price, delta = max(candidates, key=lambda item: item[0])
    return {"value": price, "delta": delta, "observed_at": observed_at.isoformat(timespec="seconds")}


def fetch_live_fuel() -> dict[str, Any]:
    try:
        parsed = parse_fuel_text(_get_text(FUEL_URL))
        return {
            "ok": True,
            "name": "Dầu DO 0,05S-II",
            "unit": "VND/lít",
            "source": FUEL_URL,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
            **parsed,
        }
    except Exception as exc:
        return _result_error(FUEL_URL, exc)


def fetch_live_weather(latitude: float, longitude: float, location: str) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,rain,"
            "wind_speed_10m"
        ),
        "timezone": "Asia/Bangkok",
        "forecast_days": 1,
    }
    try:
        response = requests.get(WEATHER_URL, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
        payload = response.json()
        current = payload.get("current") or {}
        if "temperature_2m" not in current:
            raise ValueError("Open-Meteo không trả về dữ liệu current")
        return {
            "ok": True,
            "name": location,
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "rain": current.get("rain"),
            "wind_speed": current.get("wind_speed_10m"),
            "observed_at": current.get("time", datetime.now().isoformat(timespec="minutes")),
            "source": WEATHER_URL,
            "fetched_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:
        return _result_error(WEATHER_URL, exc)
