#!/usr/bin/env python3
"""Collect NYC DOT sampled traffic counts, 2019-2025, for COVERAGE REVIEW.

Source: NYC Open Data Automated Traffic Volume Counts (7ym2-wayt).

Run from your Air-Pollution repository root:
    python -m pip install requests
    python scripts/collect_nyc_traffic.py

IMPORTANT: This dataset contains sampled 15-minute intervals at selected roads,
not all vehicles or all roads. It is NOT an annual traffic-volume census.
The output is intentionally NOT merged into model data until coverage and
sampling comparability have been reviewed.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASET_URL = "https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt"
API_URL = "https://data.cityofnewyork.us/resource/7ym2-wayt.json"
COUNTIES = {
    "bronx": ("36005", "Bronx"),
    "brooklyn": ("36047", "Brooklyn"),
    "manhattan": ("36061", "Manhattan"),
    "queens": ("36081", "Queens"),
    "staten island": ("36085", "Staten Island"),
}
YEAR_FIRST, YEAR_LAST = 2019, 2025
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "raw" / "traffic"


def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4, connect=4, read=3, status=4, backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "UTD-Air-Pollution-Student-Research/1.0"
    token = os.getenv("SOCRATA_APP_TOKEN", "").strip()
    if token:
        session.headers["X-App-Token"] = token
    return session


def fetch_grouped_counts(session: requests.Session) -> list[dict]:
    """Request grouped counts on server; avoids downloading ~1.8 million rows."""
    base = {
        "$where": "vol >= 0 AND yr IS NOT NULL AND boro IS NOT NULL",
        "$group": "yr,boro",
        "$order": "yr,boro",
        "$limit": 1000,
    }
    # Count of distinct locations/requests is useful for detecting sampling bias.
    # A few Socrata API deployments cannot process count(distinct ...), so
    # fall back to a simpler aggregate without inventing those counts.
    selects = [
        ("yr,boro,count(*) as observed_15min_intervals,"
         "avg(vol) as mean_15min_vehicle_count,"
         "count(distinct segmentid) as observed_segments,"
         "count(distinct requestid) as sampling_requests"),
        ("yr,boro,count(*) as observed_15min_intervals,"
         "avg(vol) as mean_15min_vehicle_count"),
    ]
    failures = []
    for attempt, select in enumerate(selects, start=1):
        params = {**base, "$select": select}
        try:
            print(f"Querying NYC DOT sample-count API (aggregation {attempt}/2) ...", flush=True)
            response = session.get(API_URL, params=params, timeout=(20, 240))
            response.raise_for_status()
            records = response.json()
            if not isinstance(records, list):
                raise ValueError(f"Expected JSON array; got {str(records)[:250]}")
            if len(records) == 1000:
                raise ValueError("Reached 1,000-group safety limit. Refine query before trusting coverage.")
            return records
        except (requests.RequestException, ValueError) as exc:
            failures.append(f"query {attempt}: {type(exc).__name__} ({str(exc)[:250]})")
            # Only downgrade SQL features for a 400 query error. A network problem
            # should be retried by HTTP adapter, not hidden by a simpler query.
            if isinstance(exc, requests.HTTPError) and exc.response is not None and exc.response.status_code == 400 and attempt == 1:
                print("  Advanced aggregation rejected; retrying basic aggregation.", flush=True)
                continue
            raise RuntimeError("Unable to fetch grouped traffic data: " + "; ".join(failures)) from exc
    raise RuntimeError("Unable to fetch grouped traffic data: " + "; ".join(failures))


def normalize_year(value: object) -> int | None:
    try:
        yr = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if 0 <= yr <= 99:
        yr += 2000
    return yr if 2000 <= yr <= 2100 else None


def safe_nonnegative_float(value: object) -> float | None:
    try:
        number = float(value)
    except (ValueError, TypeError):
        return None
    if number < 0 or not (number < float("inf")):
        return None
    return number


def compile_rows(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    seen = set()
    collected = []
    other_years = set()
    unknown_boroughs = set()
    for record in records:
        year = normalize_year(record.get("yr"))
        if year is None:
            continue
        if not YEAR_FIRST <= year <= YEAR_LAST:
            other_years.add(year)
            continue
        name = str(record.get("boro", "")).strip().lower()
        if name not in COUNTIES:
            unknown_boroughs.add(name)
            continue
        fips, borough = COUNTIES[name]
        key = (fips, year)
        if key in seen:
            raise ValueError(f"More than one grouped record for {borough} {year}. Check raw year encoding.")
        seen.add(key)
        n = safe_nonnegative_float(record.get("observed_15min_intervals"))
        mean = safe_nonnegative_float(record.get("mean_15min_vehicle_count"))
        if n is None or n < 1 or int(n) != n or mean is None:
            raise ValueError(f"Bad volume or count in {borough} {year}: {record}")
        locations = safe_nonnegative_float(record.get("observed_segments"))
        requests_n = safe_nonnegative_float(record.get("sampling_requests"))
        collected.append({
            "county_fips": fips,
            "borough": borough,
            "year": year,
            "traffic_sample_mean_15min_vehicle_count": round(mean, 4),
            "traffic_sample_observed_15min_intervals": int(n),
            "traffic_sample_distinct_segments": int(locations) if locations is not None else "",
            "traffic_sample_distinct_requests": int(requests_n) if requests_n is not None else "",
        })
    collected.sort(key=lambda r: (r["year"], r["county_fips"]))
    cover = []
    by_key = {(r["county_fips"], r["year"]): r for r in collected}
    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        for _, (fips, borough) in COUNTIES.items():
            row = by_key.get((fips, year))
            cover.append({
                "county_fips": fips, "borough": borough, "year": year,
                "has_traffic_sample": "Yes" if row else "No",
                "observed_15min_intervals": row["traffic_sample_observed_15min_intervals"] if row else "",
                "distinct_segments": row["traffic_sample_distinct_segments"] if row else "",
                "distinct_requests": row["traffic_sample_distinct_requests"] if row else "",
            })
    diagnostics = {"other_source_years": sorted(other_years),
                   "unrecognized_boroughs": sorted(unknown_boroughs),
                   "available_borough_years": len(collected)}
    return collected, cover, diagnostics


def write_csv(path: Path, rows: list[dict], names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def main() -> int:
    session = http_session()
    records = fetch_grouped_counts(session)
    rows, coverage, diagnostics = compile_rows(records)
    OUT.mkdir(parents=True, exist_ok=True)
    feature_file = OUT / "nyc_dot_traffic_SAMPLE_borough_year_2019_2025.csv"
    coverage_file = OUT / "nyc_dot_traffic_sample_coverage_report.csv"
    write_csv(feature_file, rows, ["county_fips", "borough", "year",
        "traffic_sample_mean_15min_vehicle_count", "traffic_sample_observed_15min_intervals",
        "traffic_sample_distinct_segments", "traffic_sample_distinct_requests"])
    write_csv(coverage_file, coverage, ["county_fips", "borough", "year", "has_traffic_sample",
                                               "observed_15min_intervals", "distinct_segments", "distinct_requests"])
    info = {
        "source_url": DATASET_URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "method": "NYC DOT ATR sampled traffic volumes; server-side aggregate by borough and sampling year",
        "year_range_requested": [YEAR_FIRST, YEAR_LAST],
        "data_use_status": "CANDIDATE_ONLY_NOT_YET_MODEL_READY",
        "limitations": [
            "Roadway samples are not collected daily or uniformly across locations/years.",
            "Mean is per available 15-minute sample record, not annual average daily traffic.",
            "Sample sizes and locations may vary by borough-year and induce misleading trends.",
            "Do not use summed counts as total NYC vehicles or extrapolate missing years.",
            "Do not merge until coverage and location comparability are inspected.",
            "NULL or missing source years are NOT zeros.",
        ],
        **diagnostics,
    }
    (OUT / "nyc_dot_traffic_source_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nTRAFFIC SOURCE CHECK COMPLETE")
    print(f"NYC DOT grouped source observations: {len(records)}")
    print(f"NYC borough-year observations for 2019–2025: {len(rows)} / 35")
    print(f"Feature candidate CSV: {feature_file}")
    print(f"Coverage CSV: {coverage_file}")
    print("By year (boroughs sampled / 5):")
    for year in range(YEAR_FIRST, YEAR_LAST + 1):
        entries = [r for r in rows if r["year"] == year]
        print(f"  {year}: {len(entries)}/5 boroughs sampled")
    if diagnostics["unrecognized_boroughs"]:
        print("Unknown borough codes: " + ", ".join(diagnostics["unrecognized_boroughs"]))
    print("NOTE: Source contains intermittent road sampling, not comparable complete annual totals.")
    print("IMPORTANT: Do not merge into master CSV until coverage is reviewed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"TRAFFIC COLLECTION ERROR: {error}", file=sys.stderr)
        sys.exit(1)
