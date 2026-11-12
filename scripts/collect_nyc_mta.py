#!/usr/bin/env python3

"""Collect annual NYC subway station entries by borough, 2017-2025.
 
Source: MTA Subway Hourly Ridership on data.ny.gov. Three datasets, same measure
(estimated entries by station complex), so the series is continuous:
    2017-2019  t69i-h2me
    2020-2024  wujg-7c2s
    2025+      5wq4-mkjj
 
Run from the repository root:
    python -m pip install requests
    python scripts/collect_nyc_mta.py
 
Entries are assigned to the STATION's borough (not the rider's home borough) and
include every mode in these datasets (subway, Staten Island Railway, tram), which
is why Staten Island is not zero. A borough-year the source lacks is left out,
never filled with zero. Years before 2017 have no hourly dataset, so they stay blank.
"""

from __future__ import annotations

import csv
import sys 
from collections import defaultdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BOROUGHS = {
    "bronx": ("36005", "Bronx"),
    "brooklyn": ("36047", "Brooklyn"),
    "manhattan": ("36061", "Manhattan"),
    "queens": ("36081", "Queens"),
    "staten island": ("36085", "Staten Island"),
}
# mta datasets go as far back as 2017
FIRST_YEAR, LAST_YEAR = 2017, 2025
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "raw" / "mta"
OLD_SOLVER_FILE = ROOT / "datasets" / "solvers" / "nyc_mta_station_entries_by_borough_year.csv"
COLUMNS = ["county_fips", "borough", "year", "mta_annual_station_entries_est", "mta_months_observed"]


def dataset_for(year: int) -> str:
    if year <= 2019:
        return "t69i-h2me"
    return "wujg-7c2s" if year <= 2024 else "5wq4-mkjj"


def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=2, 
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "UTD-Air-Pollution-Student-Research/1.0"
    return session

def fetch_year(session: requests.Session, year: int) -> list[dict]:
    """One server-side aggregate: total entries per month and borough for one year."""
    params = {
        "$select": "date_extract_m(transit_timestamp) as mth, borough, sum(ridership) as entries",
        "$where": f"transit_timestamp >= '{year}-01-01T00:00:00' AND transit_timestamp < '{year + 1}-01-01T00:00:00'",
        "$group": "mth,borough",
        "$limit": 1000,
    }
    response = session.get(f"https://data.ny.gov/resource/{dataset_for(year)}.json",
                           params=params, timeout=(20, 600))
    if response.status_code >= 400:
        raise RuntimeError(f"{year}: HTTP {response.status_code} from {dataset_for(year)}: {response.text[:300]}")
    records = response.json()
    if not isinstance(records, list) or len(records) >= 1000:
        raise ValueError(f"{year}: unexpected response (not a list, or hit the 1,000-group limit)")
    return records

def annual_rows(year: int, records: list[dict]) -> tuple[list[dict], dict]:
    """Sum monthly records to borough-year rows. Returns (rows, unrecognized borough totals)."""
    entries, months, unknown = defaultdict(float), defaultdict(set), defaultdict(float)
    for record in records:
        name = str(record.get("borough", "")).strip().lower()
        value = float(record["entries"])
        if value < 0:
            raise ValueError(f"{year} {name}: negative entries {value}")
        if name not in BOROUGHS:
            unknown[name] += value
            continue
        entries[name] += value
        months[name].add(int(record["mth"]))
    rows = [{"county_fips": fips, "borough": borough, "year": year,
             "mta_annual_station_entries_est": round(entries[name]),
             "mta_months_observed": len(months[name])}
            for name, (fips, borough) in BOROUGHS.items() if name in entries]
    return rows, dict(unknown)

def annual_rows(year: int, records: list[dict]) -> tuple[list[dict], dict]:
    """Sum monthly records to borough-year rows. Returns (rows, unrecognized borough totals)."""
    entries, months, unknown = defaultdict(float), defaultdict(set), defaultdict(float)
    for record in records:
        name = str(record.get("borough", "")).strip().lower()
        value = float(record["entries"])
        if value < 0:
            raise ValueError(f"{year} {name}: negative entries {value}")
        if name not in BOROUGHS:
            unknown[name] += value
            continue
        entries[name] += value
        months[name].add(int(record["mth"]))
    rows = [{"county_fips": fips, "borough": borough, "year": year,
             "mta_annual_station_entries_est": round(entries[name]),
             "mta_months_observed": len(months[name])}
            for name, (fips, borough) in BOROUGHS.items() if name in entries]
    return rows, dict(unknown)


def compare_with_existing(rows: list[dict]) -> None:
    """Sanity check: 2020-2025 totals should match the values already in the repo."""
    if not OLD_SOLVER_FILE.exists():
        return
    with OLD_SOLVER_FILE.open(encoding="utf-8-sig", newline="") as file:
        old = {(r["county_fips"], int(r["year"])): float(r["mta_annual_station_entries_est"])
               for r in csv.DictReader(file) if r["mta_annual_station_entries_est"]}
    diffs = [abs(r["mta_annual_station_entries_est"] - old[k]) / old[k]
             for r in rows if (k := (r["county_fips"], r["year"])) in old and old[k]]
    if not diffs:
        return
    print(f"Overlap check vs existing solver file: {len(diffs)} borough-years, "
          f"largest difference {max(diffs):.2%}")
    if max(diffs) > 0.01:
        print("  WARNING: differs by more than 1%. The measure may not match what was used before "
              "(e.g. different modes or ridership column). Investigate before merging.")



def main() -> int:
    session = http_session()
    all_rows, all_unknown = [], defaultdict(float)
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        print(f"Fetching {year} from {dataset_for(year)} ...", flush=True)
        rows, unknown = annual_rows(year, fetch_year(session, year))
        all_rows += rows
        for name, total in unknown.items():
            all_unknown[name] += total
        print(f"  {len(rows)}/5 boroughs, months observed: "
              f"{sorted({r['mta_months_observed'] for r in rows})}")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"nyc_mta_station_entries_borough_year_{FIRST_YEAR}_{LAST_YEAR}.csv"
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    tmp.replace(path)
    print(f"\nWrote {len(all_rows)} borough-year rows (of {5 * (LAST_YEAR - FIRST_YEAR + 1)}) to {path}")
    partial = sorted({r["year"] for r in all_rows if r["mta_months_observed"] < 12})
    if partial:
        print(f"Years with fewer than 12 months of data (totals are understated): {partial}")
    if all_unknown:
        print("Unrecognized borough values (NOT included): "
              + ", ".join(f"{n!r}={t:,.0f}" for n, t in all_unknown.items()))
    compare_with_existing(all_rows)
    return 0
 
 
if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"MTA COLLECTION ERROR: {error}", file=sys.stderr)
        sys.exit(1)