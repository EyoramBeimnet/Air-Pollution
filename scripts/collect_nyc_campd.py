#!/usr/bin/env python3
"""Collect NYC power-sector emissions by county and year (2014-2025) from EPA CAMPD.

Needs a free API key (EPA "Clean Air Markets API Portal", https://www.epa.gov/node/277325).
    export CAMPD_API_KEY=your_key          (PowerShell: $env:CAMPD_API_KEY="your_key")
    python -m pip install requests
    python scripts/collect_nyc_campd.py

Method: two unit-level annual downloads for New York State, joined on facility + unit:
  1. apportioned emissions (mass, gross load, heat input), which has no county
  2. facility attributes, which carry each unit's county
then summed by NYC county and year. Missing values stay blank, never zero.
The output goes in datasets/ under the filename prepare_model_data.py looks for.
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import date
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.epa.gov/easey/streaming-services"
COUNTIES = {"005": "Bronx", "047": "Kings", "061": "New York", "081": "Queens", "085": "Richmond"}
FIRST_YEAR, LAST_YEAR = 2014, 2025
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "datasets" / f"campd_industrial_emissions_county_year_{FIRST_YEAR}_{LAST_YEAR}.csv"
OLD_FILE = ROOT / "datasets" / "campd_industrial_emissions_county_year_2020_2026.csv"
# API field -> output column (masses are short tons)
SUMS = {"co2Mass": "total_co2_mass_tons", "so2Mass": "total_so2_mass_tons", "noxMass": "total_nox_mass_tons",
        "grossLoad": "total_gross_load_mwh", "heatInput": "total_heat_input_mmbtu"}
UNIT_FIELDS = [*SUMS, "sumOpTime"]  # summed per unit so "active" can be defined below
COLUMNS = ["state_abbr", "county", "year", "is_partial_year", "num_active_units", *SUMS.values(),
           "co2_rate_tons_per_mwh", "so2_rate_tons_per_mwh", "nox_rate_tons_per_mwh"]


def number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def http_session(api_key: str) -> requests.Session:
    session = requests.Session()
    retry = Retry(total=4, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["x-api-key"] = api_key
    session.headers["User-Agent"] = "UTD-Air-Pollution-Student-Research/1.0"
    return session


def fetch(session: requests.Session, path: str, year: int) -> list[dict]:
    """All New York State unit rows for one year from one streaming endpoint."""
    response = session.get(f"{API}/{path}", params={"year": year, "stateCode": "NY"}, timeout=(20, 300))
    if response.status_code >= 400:
        raise RuntimeError(f"{path} {year}: HTTP {response.status_code}: {response.text[:300]}")
    rows = response.json()
    if not isinstance(rows, list):
        raise ValueError(f"{path} {year}: expected a JSON list, got {str(rows)[:200]}")
    return rows


def unit_key(row: dict) -> tuple | None:
    try:
        return int(float(row["facilityId"])), str(row["unitId"]).strip()
    except (KeyError, TypeError, ValueError):
        return None


def county_code(row: dict) -> str | None:
    """3-digit county FIPS from a facility-attributes row (countyCode looks like 'NY005')."""
    code = str(row.get("countyCode") or "")
    if len(code) == 5 and code[:2] in ("NY", "36"):
        return code[2:]
    fips = str(row.get("fipsCode") or "")
    return fips.zfill(3) if fips.isdigit() else None


def aggregate(emissions_by_year: dict, county_of: dict) -> tuple[dict, dict, int]:
    """Sum unit rows to (county, year). Returns (totals, per-unit sums, rows with unknown county)."""
    totals, units, unknown = {}, {}, 0
    for year, rows in emissions_by_year.items():
        for row in rows:
            key = unit_key(row)
            county = county_of.get(key)
            if county is None:
                unknown += 1
                continue
            if county not in COUNTIES:
                continue
            sums = totals.setdefault((county, year), {field: None for field in SUMS})
            for field in SUMS:
                value = number(row.get(field))
                if value is not None:
                    sums[field] = (sums[field] or 0.0) + value
            unit = units.setdefault((county, year), {}).setdefault(key, {field: 0.0 for field in UNIT_FIELDS})
            for field in UNIT_FIELDS:
                value = number(row.get(field))
                if value is not None:
                    unit[field] += value
    return totals, units, unknown


def load_existing() -> dict:
    """NYC complete-year rows of the existing CAMPD file, keyed by (county name, year)."""
    if not OLD_FILE.exists():
        return {}
    with OLD_FILE.open(encoding="utf-8-sig", newline="") as file:
        return {(r["county"], int(r["year"])): r for r in csv.DictReader(file)
                if r["state_abbr"] == "NY" and r["county"] in COUNTIES.values()
                and r["is_partial_year"].upper() != "TRUE"}


# The API does not define an "active" unit, so these candidates are tested against the existing file.
DEFINITIONS = {
    "all units reported": lambda u: True,
    "heat input > 0": lambda u: u["heatInput"] > 0,
    "gross load > 0": lambda u: u["grossLoad"] > 0,
    "operating time > 0": lambda u: u["sumOpTime"] > 0,
    "CO2 mass > 0": lambda u: u["co2Mass"] > 0,
}


def unit_counts(units: dict, definition: str) -> dict:
    return {key: sum(1 for u in per_unit.values() if DEFINITIONS[definition](u))
            for key, per_unit in units.items()}


def choose_definition(units: dict) -> str:
    """Pick the unit-count definition that best reproduces num_active_units in the existing file."""
    old = load_existing()
    if not old:
        print("No existing file to calibrate unit counts against; counting every unit that reported.")
        return "all units reported"
    diffs = {}
    for name in DEFINITIONS:
        pairs = [(n, number(old[(COUNTIES[c], y)]["num_active_units"]))
                 for (c, y), n in unit_counts(units, name).items() if (COUNTIES[c], y) in old]
        diffs[name] = max((abs(n - o) / o for n, o in pairs if o), default=1.0)
    best = min(diffs, key=diffs.get)  # ties go to the first (simplest) definition
    print("Unit-count definition vs existing file (largest difference):")
    for name, diff in diffs.items():
        print(f"  {name}: {diff:.2%}" + ("   <-- used" if name == best else ""))
    if diffs[best] > 0.01:
        print("  WARNING: no definition reproduces the old counts; num_active_units is not comparable across files.")
    return best


def rate(mass: float | None, load: float | None) -> float | str:
    return round(mass / load, 9) if mass is not None and load else ""


def build_rows(totals: dict, units: dict, definition: str) -> list[dict]:
    counts = unit_counts(units, definition)
    rows = []
    for (county, year), sums in sorted(totals.items(), key=lambda item: (item[0][1], item[0][0])):
        load = sums["grossLoad"]
        row = {"state_abbr": "NY", "county": COUNTIES[county], "year": year,
               "is_partial_year": str(year >= date.today().year).upper(),
               "num_active_units": counts[(county, year)]}
        for field, column in SUMS.items():
            row[column] = "" if sums[field] is None else round(sums[field], 3)
        row["co2_rate_tons_per_mwh"] = rate(sums["co2Mass"], load)
        row["so2_rate_tons_per_mwh"] = rate(sums["so2Mass"], load)
        row["nox_rate_tons_per_mwh"] = rate(sums["noxMass"], load)
        rows.append(row)
    return rows


def compare_with_existing(rows: list[dict]) -> None:
    """Sanity check: 2020-2025 sums should match the file already in the repo."""
    old = load_existing()
    worst, matched = {}, 0
    for row in rows:
        ref = old.get((row["county"], row["year"]))
        if not ref:
            continue
        matched += 1
        for column in ["num_active_units", *SUMS.values()]:
            new_value, old_value = number(row[column]), number(ref[column])
            if new_value is not None and old_value:
                worst[column] = max(worst.get(column, 0.0), abs(new_value - old_value) / old_value)
    if not matched:
        return
    print(f"Overlap check vs existing file ({matched} county-years); largest difference per column:")
    for column, diff in worst.items():
        print(f"  {column}: {diff:.2%}" + ("   <-- investigate" if diff > 0.01 else ""))


def main() -> int:
    api_key = os.getenv("CAMPD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set CAMPD_API_KEY first (free key: https://www.epa.gov/node/277325).")
    session = http_session(api_key)
    years = range(FIRST_YEAR, LAST_YEAR + 1)

    county_of = {}
    for year in years:
        print(f"Facility attributes {year} ...", flush=True)
        for row in fetch(session, "facilities/attributes", year):
            key, code = unit_key(row), county_code(row)
            if key and code:
                county_of[key] = code
    emissions = {}
    for year in years:
        print(f"Annual emissions {year} ...", flush=True)
        emissions[year] = fetch(session, "emissions/apportioned/annual", year)

    totals, units, unknown = aggregate(emissions, county_of)
    definition = choose_definition(units)
    rows = build_rows(totals, units, definition)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(OUT)

    expected = {(c, y) for c in COUNTIES for y in years}
    missing = sorted((COUNTIES[c], y) for c, y in expected - set(totals))
    print(f"\nWrote {len(rows)} county-year rows (of {len(expected)}) to {OUT}")
    if missing:
        print(f"County-years with no unit data (left blank, NOT zero): {missing}")
    if unknown:
        print(f"{unknown} New York unit-year rows had no county in the facility attributes and were skipped.")
    compare_with_existing(rows)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(f"CAMPD COLLECTION ERROR: {error}", file=sys.stderr)
        sys.exit(1)