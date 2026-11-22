"""Read-only API for the AQI records committed with this project."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "processed" / "master_borough_year_2014_2025.csv"
FORECAST_SOURCE = ROOT / "datasets" / "processed" / "model_results" / "annual_aqi_forecast_2026.csv"
ANNUAL_SCENARIO_SOURCE = ROOT / "datasets" / "processed" / "model_results" / "annual_aqi_forecast_2026_2030.csv"
MONTHLY_SOURCE = ROOT / "datasets" / "processed" / "monthly" / "monthly_aqi_targets.csv"
MONTHLY_FORECAST_SOURCE = ROOT / "datasets" / "processed" / "monthly" / "model_results" / "monthly_aqi_forecast_2026_2030.csv"
FIPS = {"36005", "36047", "36061", "36081", "36085"}

app = FastAPI(title="Airshed NYC AQI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


def observed_aqi_rows(path: Path = SOURCE) -> list[dict[str, Any]]:
    """Load the committed annual median AQI observations without fabricating forecasts."""
    frame = pd.read_csv(path, dtype={"county_fips": "string"})
    required = {"county_fips", "borough", "year", "aqi_median"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"AQI source is missing columns: {', '.join(sorted(missing))}")
    rows = []
    for row in frame.loc[:, ["county_fips", "borough", "year", "aqi_median"]].itertuples(index=False):
        fips = str(row.county_fips).zfill(5)
        if fips in FIPS and not pd.isna(row.aqi_median):
            rows.append({"fips": fips, "borough": str(row.borough), "year": int(row.year),
                         "aqi": float(row.aqi_median), "forecast": False,
                         "metric": "annual_median_daily_aqi"})
    if not rows:
        raise ValueError("AQI source contains no usable NYC borough observations.")
    return sorted(rows, key=lambda item: (item["year"], item["fips"]))


def forecast_aqi_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Load a complete, explicitly marked forecast artifact."""
    path = path or FORECAST_SOURCE
    if not path.exists():
        return []
    frame = pd.read_csv(path, dtype={"fips": "string"})
    required = {"fips", "borough", "year", "aqi", "forecast"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Forecast source is missing columns: {', '.join(sorted(missing))}")
    frame = frame.loc[:, ["fips", "borough", "year", "aqi", "forecast"]].copy()
    frame["fips"] = frame["fips"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["aqi"] = pd.to_numeric(frame["aqi"], errors="coerce")
    if (frame["forecast"] != True).any() or frame[["fips", "borough", "year", "aqi"]].isna().any().any():
        raise ValueError("Forecast source contains invalid required values.")
    expected_years = set(range(2026, 2031)) if path == ANNUAL_SCENARIO_SOURCE else {2026}
    if not frame["fips"].isin(FIPS).all() or not set(frame["year"]).issubset(expected_years):
        raise ValueError("Forecast source contains unsupported annual borough/year rows.")
    if frame.duplicated(["fips", "year"]).any() or set(frame["fips"]) != FIPS or set(frame["year"]) != expected_years or len(frame) != len(FIPS)*len(expected_years):
        raise ValueError("Forecast source does not have complete annual borough/year coverage.")
    if not pd.api.types.is_numeric_dtype(frame["aqi"]) or not np.isfinite(frame["aqi"]).all() or (frame["aqi"] < 0).any():
        raise ValueError("Forecast source contains invalid AQI values.")
    return sorted(({
        "fips": str(row.fips), "borough": str(row.borough), "year": int(row.year),
        "aqi": float(row.aqi), "forecast": True, "metric": "annual_median_daily_aqi",
        "forecast_method": "observed_2025_inputs" if int(row.year) == 2026 else "recursive_fixed_2025_external_features_scenario",
    } for row in frame.itertuples(index=False)), key=lambda item: item["fips"])


def available_aqi_rows() -> tuple[list[dict[str, Any]], str]:
    """Keep observations available if an optional forecast artifact is absent or invalid."""
    observed = observed_aqi_rows()
    try:
        # Retain the original one-year artifact as the compatibility fallback.  Tests
        # and deployments can still override FORECAST_SOURCE to disable forecasts.
        source = ANNUAL_SCENARIO_SOURCE if FORECAST_SOURCE == ROOT / "datasets" / "processed" / "model_results" / "annual_aqi_forecast_2026.csv" and ANNUAL_SCENARIO_SOURCE.exists() else FORECAST_SOURCE
        forecasts = forecast_aqi_rows(source)
    except ValueError as error:
        return observed, f"unavailable: {error}"
    return observed + forecasts, "available" if forecasts else "not generated"


def monthly_aqi_rows() -> tuple[list[dict[str, Any]], str]:
    """Load observed monthly AQI and an optional complete recursive artifact."""
    observed = pd.read_csv(MONTHLY_SOURCE, dtype={"county_fips": "string"})
    observed["month"] = pd.to_datetime(observed["target_month"]).dt.strftime("%Y-%m")
    historical = [{"fips": str(r.county_fips).zfill(5), "borough": str(r.borough), "month": r.month,
                   "aqi": float(r.aqi_median_daily), "forecast": False, "metric": "monthly_median_daily_aqi"}
                  for r in observed.itertuples() if str(r.county_fips).zfill(5) in FIPS]
    if not MONTHLY_FORECAST_SOURCE.exists(): return historical, "not generated"
    try:
        f = pd.read_csv(MONTHLY_FORECAST_SOURCE, dtype={"fips": "string"})
        required={"fips","borough","month","aqi","forecast","metric"}
        if required-set(f.columns): raise ValueError("missing required columns")
        f["fips"]=f.fips.astype("string").str.zfill(5); f["aqi"]=pd.to_numeric(f.aqi,errors="coerce")
        expected={f"{year}-{month:02d}" for year in range(2026,2031) for month in range(1,13)}
        if set(f.fips)!=FIPS or set(f.month)!=expected or len(f)!=300 or f.duplicated(["fips","month"]).any() or f.aqi.isna().any() or not np.isfinite(f.aqi).all() or (f.aqi<0).any() or not f.forecast.eq(True).all(): raise ValueError("incomplete or invalid monthly forecast coverage")
        return historical + f.to_dict("records"), "available"
    except ValueError as error: return historical, f"unavailable: {error}"


@app.get("/health")
def health() -> dict[str, Any]:
    data, forecast_status = available_aqi_rows()
    monthly, monthly_status = monthly_aqi_rows()
    return {"status": "ok", "records": len(data), "forecast_records": sum(row["forecast"] for row in data),
            "forecast_status": forecast_status, "monthly_records":len(monthly), "monthly_forecast_status":monthly_status}


@app.get("/api/aqi")
def get_aqi(start_year: int | None = Query(None, ge=1900, le=3000),
            end_year: int | None = Query(None, ge=1900, le=3000)) -> dict[str, Any]:
    if start_year is not None and end_year is not None and start_year > end_year:
        raise HTTPException(status_code=422, detail="start_year must not exceed end_year")
    all_data, forecast_status = available_aqi_rows()
    data = [row for row in all_data
            if (start_year is None or row["year"] >= start_year)
            and (end_year is None or row["year"] <= end_year)]
    return {"data": data, "metric": "annual_median_daily_aqi",
            "forecast_status": forecast_status}


@app.get("/api/aqi/monthly")
def get_monthly_aqi(start_month: str | None = None, end_month: str | None = None) -> dict[str, Any]:
    def valid(value: str | None) -> str | None:
        if value is None: return None
        try: return pd.Period(value, freq="M").strftime("%Y-%m")
        except ValueError: raise HTTPException(status_code=422, detail="month filters must be YYYY-MM")
    start, end = valid(start_month), valid(end_month)
    if start and end and start > end: raise HTTPException(status_code=422, detail="start_month must not exceed end_month")
    rows, status = monthly_aqi_rows()
    return {"data":[r for r in rows if (start is None or r["month"]>=start) and (end is None or r["month"]<=end)],"metric":"monthly_median_daily_aqi","forecast_status":status}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
