"""Read-only API for the AQI records committed with this project."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "processed" / "master_borough_year_2014_2025.csv"
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


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "records": len(observed_aqi_rows()), "forecast_records": 0}


@app.get("/api/aqi")
def get_aqi(start_year: int | None = Query(None, ge=1900, le=3000),
            end_year: int | None = Query(None, ge=1900, le=3000)) -> dict[str, Any]:
    if start_year is not None and end_year is not None and start_year > end_year:
        raise HTTPException(status_code=422, detail="start_year must not exceed end_year")
    data = [row for row in observed_aqi_rows()
            if (start_year is None or row["year"] >= start_year)
            and (end_year is None or row["year"] <= end_year)]
    return {"data": data, "metric": "annual_median_daily_aqi",
            "forecast_status": "No future forecast artifact is committed in this release."}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
