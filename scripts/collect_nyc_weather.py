#!/usr/bin/env python3
"""Download and prepare historical NYC borough-point weather, 2014-2025.

Source: Open-Meteo historical archive, ERA5 fixed reanalysis (not station measurements).
The coordinates are a single representative sample point per borough, NOT an area average.

Place at scripts/collect_nyc_weather.py in the Air-Pollution repo and run:
    python scripts/collect_nyc_weather.py

Writes under datasets/raw/weather/:
  - nyc_weather_daily_2014_2025.csv
  - nyc_weather_borough_year_2014_2025.csv
  - weather_source_metadata.json
  - _api_cache/weather_<fips>_era5_2014_2025.json

Does not alter existing datasets/processed files. Run prepare_model_data.py next.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_URL = "https://archive-api.open-meteo.com/v1/archive"
START, END = "2014-01-01", "2025-12-31"
TIMEZONE = "America/New_York"
MODEL = "era5"
# Central/representative coordinates for each county; not an areal average.
LOCATIONS = [
    ("36005", "Bronx", 40.8448, -73.8648),
    ("36047", "Brooklyn", 40.6782, -73.9442),
    ("36061", "Manhattan", 40.7831, -73.9712),
    ("36081", "Queens", 40.7282, -73.7949),
    ("36085", "Staten Island", 40.5795, -74.1502),
]
VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "precipitation_sum",
    "wind_speed_10m_mean",
    "relative_humidity_2m_mean",
]
DAILY_NAMES = {
    "temperature_2m_mean": "temperature_mean_c",
    "temperature_2m_max": "temperature_max_c",
    "precipitation_sum": "precipitation_mm",
    "wind_speed_10m_mean": "wind_speed_mean_kmh",
    "relative_humidity_2m_mean": "relative_humidity_mean_pct",
}
OUTPUT_FEATURES = [
    "weather_days_complete",
    "weather_temp_mean_c",
    "weather_mean_daily_max_temp_c",
    "weather_precip_total_mm",
    "weather_wind_mean_kmh",
    "weather_humidity_mean_pct",
]


def session_with_retries() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5, connect=5, read=4,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        respect_retry_after_header=True,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "UTD-NYC-Air-Pollution-Weather-Research/1.0"})
    return s


def expected_dates() -> list[str]:
    d = date.fromisoformat(START)
    end = date.fromisoformat(END)
    values = []
    while d <= end:
        values.append(d.isoformat())
        d += timedelta(days=1)
    return values


EXPECTED_DATES = expected_dates()


def validate_response(doc: dict, fips: str) -> None:
    if not isinstance(doc, dict) or doc.get("error"):
        raise ValueError(f"{fips}: API returned error: {str(doc)[:150]}")
    if doc.get("timezone") != TIMEZONE:
        raise ValueError(f"{fips}: unexpected timezone {doc.get('timezone')!r}")
    daily = doc.get("daily")
    if not isinstance(daily, dict):
        raise ValueError(f"{fips}: API response missing daily object")
    if daily.get("time") != EXPECTED_DATES:
        raise ValueError(f"{fips}: daily date list incomplete, duplicate, or not ordered")
    for field in VARIABLES:
        if not isinstance(daily.get(field), list) or len(daily[field]) != len(EXPECTED_DATES):
            raise ValueError(f"{fips}: missing/short variable {field}")
    units = doc.get("daily_units") or {}
    required_units = {"temperature_2m_mean": "°C", "temperature_2m_max": "°C",
                      "precipitation_sum": "mm", "wind_speed_10m_mean": "km/h",
                      "relative_humidity_2m_mean": "%"}
    for field, expected in required_units.items():
        if units.get(field) != expected:
            raise ValueError(f"{fips}: {field} has unit {units.get(field)!r}, expected {expected!r}")


def get_data(session: requests.Session, cache_path: Path, fips: str, lat: float, lon: float) -> dict:
    if cache_path.is_file():
        try:
            doc = json.loads(cache_path.read_text(encoding="utf-8"))
            validate_response(doc, fips)
            print("  Using verified cached API response.", flush=True)
            return doc
        except (ValueError, OSError, json.JSONDecodeError) as e:
            print(f"  Cache invalid ({e}); fetching again.", flush=True)

    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START, "end_date": END,
        "daily": ",".join(VARIABLES),
        "models": MODEL, "timezone": TIMEZONE,
        "temperature_unit": "celsius", "precipitation_unit": "mm", "wind_speed_unit": "kmh",
    }
    try:
        r = session.get(API_URL, params=params, timeout=(20, 150))
        r.raise_for_status()
        doc = r.json()
    except (requests.RequestException, ValueError) as e:
        raise RuntimeError(f"{fips}: weather download failed: {type(e).__name__}: {str(e)[:180]}") from e
    validate_response(doc, fips)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    tmp.replace(cache_path)
    return doc


def make_daily(doc: dict, fips: str, borough: str, lat: float, lon: float) -> pd.DataFrame:
    daily = doc["daily"]
    frame = pd.DataFrame({"county_fips": fips, "borough": borough, "date": daily["time"]})
    for raw, col in DAILY_NAMES.items():
        frame[col] = pd.to_numeric(pd.Series(daily[raw]), errors="coerce")
    frame["sample_latitude"] = lat
    frame["sample_longitude"] = lon
    frame["year"] = pd.to_datetime(frame["date"], errors="raise").dt.year
    frame["county_fips"] = frame["county_fips"].astype("string")
    return frame


def annual_aggregate(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    inputs = list(DAILY_NAMES.values())
    for (fips, borough, year), group in daily.groupby(["county_fips", "borough", "year"], sort=True):
        total_days = 366 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 365
        if len(group) != total_days or group["date"].nunique() != total_days:
            raise ValueError(f"{fips}/{year}: expected {total_days} unique days, got {len(group)}")
        complete = int(group[inputs].notna().all(axis=1).sum())
        # If any model field has a missing day, deliberately suppress the
        # associated annual statistic rather than silently undercount rainfall.
        def statistic(col: str, kind: str) -> float | None:
            s = group[col]
            if s.isna().any():
                return None
            val = s.mean() if kind == "mean" else s.sum()
            return round(float(val), 4)
        rows.append({
            "county_fips": str(fips), "borough": borough, "year": int(year),
            "weather_days_complete": complete,
            "weather_temp_mean_c": statistic("temperature_mean_c", "mean"),
            "weather_mean_daily_max_temp_c": statistic("temperature_max_c", "mean"),
            "weather_precip_total_mm": statistic("precipitation_mm", "sum"),
            "weather_wind_mean_kmh": statistic("wind_speed_mean_kmh", "mean"),
            "weather_humidity_mean_pct": statistic("relative_humidity_mean_pct", "mean"),
        })
    result = pd.DataFrame(rows)
    if len(result) != 60 or result.duplicated(["county_fips", "year"]).any():
        raise ValueError("Expected precisely 35 unique NYC borough-year weather rows")
    return result.sort_values(["county_fips", "year"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect NYC historical reanalysis weather 2014-2025")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    dest = root / "datasets" / "raw" / "weather"
    cache = dest / "_api_cache"
    cache.mkdir(parents=True, exist_ok=True)
    all_days = []
    session = session_with_retries()
    for fips, borough, lat, lon in LOCATIONS:
        print(f"Collecting {borough} ({fips}), {START} through {END}...", flush=True)
        doc = get_data(session, cache / f"weather_{fips}_{MODEL}_2014_2025.json", fips, lat, lon)
        all_days.append(make_daily(doc, fips, borough, lat, lon))
        print(f"  Verified {len(all_days[-1])} daily dates.", flush=True)
        time.sleep(.5)

    daily = pd.concat(all_days, ignore_index=True).sort_values(["county_fips", "date"])
    if daily.duplicated(["county_fips", "date"]).any():
        raise ValueError("Duplicate county/date in combined daily weather")
    annual = annual_aggregate(daily)
    metadata = {
        "source": API_URL,
        "documentation": "https://open-meteo.com/en/docs/historical-weather-api",
        "source_type": "ERA5 weather reanalysis / grid-cell estimates, NOT measured at 5 stations",
        "spatial_method": "one representative location (sample point) per borough, NOT borough-area mean",
        "model": MODEL,
        "start_date": START, "end_date": END, "timezone": TIMEZONE,
        "units": {"temp": "deg_C", "precipitation": "millimeters", "wind": "km/h", "relative_humidity": "percent"},
        "locations_requested": [{"county_fips": fips, "borough": b, "latitude": lat, "longitude": lon}
                               for fips, b, lat, lon in LOCATIONS],
        "api_grid_coordinates_may_not_match_requested_coordinates": True,
        "output_daily_rows": len(daily), "output_annual_rows": len(annual),
        "missing_daily_values_suppress_affected_yearly_aggregate": True,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "notice": "A fixed ERA5 grid may represent more than one nearby borough; weather is a regional proxy."
    }
    dest.mkdir(parents=True, exist_ok=True)
    daily_path = dest / "nyc_weather_daily_2014_2025.csv"
    annual_path = dest / "nyc_weather_borough_year_2014_2025.csv"
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    annual.to_csv(annual_path, index=False, encoding="utf-8-sig")
    (dest / "weather_source_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nSUCCESS: downloaded 5 borough-location samples for 2014-2025.")
    print(f"Daily rows: {len(daily):,}; annual borough-year rows: {len(annual)}")
    print(f"Annual output: {annual_path}")
    print("NEXT: python scripts/prepare_model_data.py")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
