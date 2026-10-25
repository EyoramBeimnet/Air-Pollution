#!/usr/bin/env python3
"""Collect and audit NYC county-year EV registration counts (Zenodo record 12773413).

This is a SOURCE COVERAGE check, not a model-training merge. The original
EV_data.csv has columns: unnamed row index, county (5-digit FIPS), n, year.
Count represents the EV registrations reported by the study's source compilation;
this file alone cannot establish the fraction of vehicles that are EVs.

Run from project root: python scripts/collect_nyc_ev.py
Requires: python -m pip install requests
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "datasets" / "raw" / "ev"
SOURCE_URL = "https://zenodo.org/records/12773413/files/EV_data.csv?download=1"
RECORD_URL = "https://zenodo.org/records/12773413"
RAW_NAME = "EV_data_zenodo_12773413_original.csv"
NYC = {"36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan",
       "36081": "Queens", "36085": "Staten Island"}
YEARS = range(2014, 2026)


def save_csv(path: Path, records: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def read_source(path: Path) -> tuple[list[dict], dict]:
    """Strictly validate only the NYC rows; leave other counties unmodified."""
    text = path.read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not {"county", "n", "year"}.issubset(reader.fieldnames):
        raise ValueError(f"Unexpected EV_data.csv headers: {reader.fieldnames}")
    observations: dict[tuple[str, int], int] = {}
    all_nyc_years = set()
    excluded_years = Counter()
    other_county_rows = 0
    for row_number, item in enumerate(reader, start=2):
        code = str(item.get("county") or "").strip().zfill(5)
        if code not in NYC:
            other_county_rows += 1
            continue
        try:
            year = int(str(item["year"]).strip())
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid NYC year on source row {row_number}: {item}") from exc
        all_nyc_years.add(year)
        if year not in YEARS:
            excluded_years[year] += 1
            continue
        raw_n = str(item.get("n") or "").strip()
        try:
            count = int(raw_n)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid NYC registration count on source row {row_number}: {item}") from exc
        if count < 0:
            raise ValueError(f"Negative EV count on source row {row_number}")
        key = (code, year)
        if key in observations:
            raise ValueError(f"Duplicate NYC county-year {key}; do not sum without checking source")
        observations[key] = count
    rows = [{"county_fips": code, "borough": NYC[code], "year": year,
             "ev_reported_registrations_count": count}
            for (code, year), count in sorted(observations.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
    details = {"all_nyc_years_in_original": sorted(all_nyc_years),
               "excluded_nyc_source_rows_by_year": dict(sorted(excluded_years.items())),
               "non_nyc_source_rows": other_county_rows,
               "usable_nyc_county_year_rows": len(rows)}
    return rows, details


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit published county-year EV count coverage for NYC")
    parser.add_argument("--refresh", action="store_true", help="Redownload original Zenodo CSV")
    opts = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = OUTPUT / RAW_NAME
    if opts.refresh or not raw.exists():
        print("Downloading the original EV_data.csv from Zenodo (about 184 KB)...", flush=True)
        sess = requests.Session()
        retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=frozenset(["GET"]))
        sess.mount("https://", HTTPAdapter(max_retries=retry))
        sess.headers["User-Agent"] = "Student-Research-NYC-Air-Pollution/1.0"
        response = sess.get(SOURCE_URL, timeout=(20, 90))
        response.raise_for_status()
        data = response.content
        if len(data) < 100 or len(data) > 2_000_000:
            raise ValueError("Unexpected EV source file size; did not overwrite the cached original")
        # Validate before replacing any existing raw file.
        content = data.decode("utf-8-sig")
        headers = csv.DictReader(io.StringIO(content)).fieldnames or []
        if not {"county", "n", "year"}.issubset(headers):
            raise ValueError(f"Unexpected downloaded headers: {headers}")
        temp = raw.with_suffix(".csv.tmp")
        temp.write_bytes(data)
        temp.replace(raw)
    else:
        print(f"Using previously downloaded source: {raw}", flush=True)
    observations, detail = read_source(raw)
    coverage = []
    observed = {(r["county_fips"], r["year"]): r for r in observations}
    for year in YEARS:
        for fips, borough in NYC.items():
            matching = observed.get((fips, year))
            coverage.append({"county_fips": fips, "borough": borough, "year": year,
                             "has_ev_source_record": "Yes" if matching else "No",
                             "ev_reported_registrations_count": matching["ev_reported_registrations_count"] if matching else ""})
    save_csv(OUTPUT / "nyc_ev_registration_CANDIDATE_borough_year_2019_2025.csv", observations,
             ["county_fips", "borough", "year", "ev_reported_registrations_count"])
    save_csv(OUTPUT / "nyc_ev_source_coverage_report.csv", coverage,
             ["county_fips", "borough", "year", "has_ev_source_record", "ev_reported_registrations_count"])
    notes = {
        "source_url": RECORD_URL, "download_url": SOURCE_URL,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CANDIDATE_ONLY_NOT_YET_MODEL_READY", **detail,
        "limitations": [
            "Source compiled from state EV registration records; confirm meaning and consistency of counts in study documentation.",
            "n is a reported count, not the share of all registered vehicles. No denominator is supplied.",
            "No missing county-year record should be interpreted as zero registrations.",
            "Do not forward-fill unavailable years or extrapolate from county data to other boroughs.",
            "Do not automatically merge this candidate into the master until source year and definition are audited.",
        ],
    }
    (OUTPUT / "nyc_ev_source_report.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    print(f"\nSUCCESS: inspected {len(observations)} observed NYC county-year records out of 35 possible.")
    print("Coverage by year:")
    for year in YEARS:
        n = sum(r["year"] == year for r in observations)
        print(f"  {year}: {n}/5 boroughs recorded")
    print("Original NYC source years:", detail["all_nyc_years_in_original"])
    print("Coverage report:", OUTPUT / "nyc_ev_source_coverage_report.csv")
    print("Important: do NOT merge yet; missing years are missing, not zero.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, requests.RequestException) as exc:
        print(f"EV COLLECTION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
