#!/usr/bin/env python3
"""Collect recorded NYC tree plantings by borough and year, 2014-2025.

Source: NYC Parks "Forestry Tree Points" (hn5i-inap). The dataset has no borough column and
stores the planted date as text, so this script works around both:
  * Borough: Socrata adds built-in "computed region" columns (zip code, council district, ...).
    The one with exactly five values is the borough. Each value is matched to a borough by
    counting trees within 2 km of that borough's center.
  * Year: read from the start of the planted-date text (2019-05-14... or 05/14/2019 ...).
The script prints what it found at each step so you can check it.

    python -m pip install requests
    python scripts/collect_nyc_trees.py

CAUTION: this is the CURRENT inventory (active and retired trees) of NYC Parks' ForMS system.
Trees that were removed before their records reached the system may be missing, so older years
are probably undercounted and a rise over time can reflect record-keeping, not only planting.
A borough-year with no records is left blank, never zero. The coverage CSV shows every year.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://data.cityofnewyork.us"
DATASET = "hn5i-inap"
FIRST_YEAR, LAST_YEAR = 2014, 2025
COVERAGE_FROM = 2000  # also count earlier years, only to show how far back records go
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / "raw" / "trees"
OLD_SOLVER_FILE = ROOT / "datasets" / "solvers" / "nyc_recorded_tree_plantings_by_borough_year.csv"
CENTERS = {  # borough: (county FIPS, latitude, longitude); a 2 km circle around each stays inside it
    "Bronx": ("36005", 40.8448, -73.8648),
    "Brooklyn": ("36047", 40.6782, -73.9442),
    "Manhattan": ("36061", 40.7831, -73.9712),
    "Queens": ("36081", 40.7282, -73.7949),
    "Staten Island": ("36085", 40.5795, -74.1502),
}
FIPS_TO_BOROUGH = {fips: borough for borough, (fips, _, _) in CENTERS.items()}


def http_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "UTD-Air-Pollution-Student-Research/1.0"
    return session


def get_json(session: requests.Session, url: str, params: dict | None = None) -> object:
    response = session.get(url, params=params, timeout=(20, 300))
    if response.status_code >= 400:
        raise RuntimeError(f"HTTP {response.status_code} from {url}: {response.text[:300]}")
    return response.json()


def query(session: requests.Session, params: dict) -> list[dict]:
    rows = get_json(session, f"{BASE}/resource/{DATASET}.json", params)
    if not isinstance(rows, list):
        raise ValueError(f"Unexpected response: {str(rows)[:200]}")
    return rows


def pick_field(columns: list[dict], keyword: str) -> str:
    """First column whose API field name contains keyword."""
    for column in columns:
        if keyword in column["fieldName"].lower():
            return column["fieldName"]
    available = ", ".join(f"{c['fieldName']} ({c.get('dataTypeName')})" for c in columns)
    raise ValueError(f"No column containing {keyword!r}. Columns are: {available}")


def find_borough_regions(session: requests.Session, columns: list[dict]) -> tuple[str, dict]:
    """Find the computed-region column that splits trees into the five boroughs.
    Returns (field name, {region value: (county FIPS, borough)})."""
    for column in columns:
        field = column["fieldName"]
        if not field.startswith(":@computed_region"):
            continue
        values = query(session, {"$select": f"{field} as v, count(*) as n", "$group": "v", "$limit": 1000})
        if len({row["v"] for row in values if "v" in row}) != 5:
            continue  # zip codes, council districts, etc. have many more values
        mapping = {}
        for borough, (fips, lat, lon) in CENTERS.items():
            near = query(session, {"$select": f"{field} as v, count(*) as n",
                                   "$where": f"within_circle(location, {lat}, {lon}, 2000)",
                                   "$group": "v", "$order": "n DESC", "$limit": 5})
            if near and "v" in near[0]:
                mapping[str(near[0]["v"])] = (fips, borough)
        if len(mapping) == 5:  # five boroughs matched to five different values
            return field, mapping
    raise ValueError("Could not find a computed-region column that splits trees into the five boroughs. "
                     "Columns: " + ", ".join(c["fieldName"] for c in columns))


def count_year(session: requests.Session, date_field: str, region_field: str, year: int) -> list[dict]:
    """Tree counts per region value for trees whose planted-date text mentions this year."""
    where = (f"({date_field} like '{year}%' OR {date_field} like '%/{year}%') "
             f"AND {region_field} IS NOT NULL")
    return query(session, {"$select": f"{region_field} as v, count(*) as n", "$where": where,
                           "$group": "v", "$limit": 100})


def write_csv(path: Path, rows: list[dict], names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def compare_with_existing(counts: dict) -> None:
    """Sanity check: 2019-2025 counts should be close to the values already in the repo."""
    if not OLD_SOLVER_FILE.exists():
        return
    with OLD_SOLVER_FILE.open(encoding="utf-8-sig", newline="") as file:
        old = {(r["county_fips"], int(r["year"])): float(r["recorded_tree_plantings"])
               for r in csv.DictReader(file) if r["recorded_tree_plantings"]}
    diffs = [abs(counts[k] - v) / v for k, v in old.items() if k in counts and v]
    if diffs:
        print(f"Overlap check vs existing solver file: {len(diffs)} borough-years, "
              f"largest difference {max(diffs):.2%}")
        if max(diffs) > 0.05:
            print("  WARNING: more than 5% different. The source may have been edited since the old pull, "
                  "or the borough matching above is wrong. Check the mapping printed earlier.")


def main() -> int:
    session = http_session()
    columns = get_json(session, f"{BASE}/api/views/{DATASET}/columns.json")
    date_field = pick_field(columns, "planted")
    samples = [row.get(date_field) for row in query(session, {
        "$select": date_field, "$where": f"{date_field} IS NOT NULL", "$limit": 3})]
    print(f"Planted-date column {date_field!r}; sample values: {samples}", flush=True)
    region_field, mapping = find_borough_regions(session, columns)
    print(f"Borough column {region_field!r}; values -> boroughs: "
          + ", ".join(f"{v}={b}" for v, (_, b) in sorted(mapping.items())), flush=True)

    counts, unknown = defaultdict(int), defaultdict(int)
    for year in range(COVERAGE_FROM, LAST_YEAR + 1):
        for row in count_year(session, date_field, region_field, year):
            value, n = str(row["v"]), int(row["n"])
            if value in mapping:
                counts[(mapping[value][0], year)] += n
            else:
                unknown[value] += n
    if not counts:
        raise ValueError(f"No planted dates matched any year. Sample values were {samples}; the queries "
                         "expect text that starts with YYYY or contains /YYYY.")

    rows = [{"county_fips": fips, "borough": FIPS_TO_BOROUGH[fips], "year": year,
             "recorded_tree_plantings": counts[(fips, year)]}
            for year in range(FIRST_YEAR, LAST_YEAR + 1) for fips in FIPS_TO_BOROUGH
            if (fips, year) in counts]
    coverage = [{"year": year, "total_recorded_plantings": sum(n for (_, y), n in counts.items() if y == year)}
                for year in sorted({y for _, y in counts})]
    path = OUT / f"nyc_tree_plantings_borough_year_{FIRST_YEAR}_{LAST_YEAR}.csv"
    write_csv(path, rows, ["county_fips", "borough", "year", "recorded_tree_plantings"])
    write_csv(OUT / "nyc_tree_plantings_year_coverage.csv", coverage, ["year", "total_recorded_plantings"])

    print(f"\nWrote {len(rows)} borough-year rows (of {5 * (LAST_YEAR - FIRST_YEAR + 1)}) to {path}")
    print("Recorded plantings per year (citywide):")
    for row in coverage:
        if row["year"] >= FIRST_YEAR - 4:
            print(f"  {row['year']}: {row['total_recorded_plantings']:,}")
    print("A steep rise in early years can mean missing older records, not fewer plantings.")
    if unknown:
        print("Region values not matched to a borough (NOT included): "
              + ", ".join(f"{k!r}={v:,}" for k, v in unknown.items()))
    compare_with_existing(counts)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"TREE COLLECTION ERROR: {error}", file=sys.stderr)
        sys.exit(1)