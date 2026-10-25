#!/usr/bin/env python3
"""Download NYC ACS B08301 2019–2024 five-year estimates with safe retries.

- Bounded retries for temporary Census API connection failures / 429 / 5xx.
- Saves each fully validated year's observations to a local cache for resuming.
- Publishes the complete CSV only when all 30 borough-year records are verified.
- Never prints or writes the API key.
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

FIPS = {
    "36005": "Bronx", "36047": "Brooklyn", "36061": "Manhattan",
    "36081": "Queens", "36085": "Staten Island",
}
FIELDS = {
    "B08301_003E": "drive_alone_pct",
    "B08301_004E": "carpool_pct",
    "B08301_010E": "public_transit_pct",
    "B08301_018E": "bicycle_commute_pct",
    "B08301_019E": "walk_commute_pct",
    "B08301_021E": "work_from_home_pct",
}
COLUMNS = ["county_fips", "borough", "year", "acs_commuters_est", *FIELDS.values()]
YEARS = range(2014, 2025)
ATTEMPTS = 5
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def validate_year(year: int, rows: list[dict]) -> list[dict]:
    """Fail instead of creating a corrupted / incomplete historical year."""
    if len(rows) != 5 or {str(row.get("county_fips")) for row in rows} != set(FIPS):
        raise ValueError(f"ACS {year}: expected exactly five unique NYC counties")
    clean = []
    for row in rows:
        fips = str(row["county_fips"])
        if row.get("borough") != FIPS[fips] or int(row["year"]) != year:
            raise ValueError(f"ACS {year}: borough or year does not match county FIPS")
        total = int(row["acs_commuters_est"])
        if total <= 0:
            raise ValueError(f"ACS {year}/{fips}: nonpositive commuter total")
        normalized = {"county_fips": fips, "borough": FIPS[fips], "year": year,
                      "acs_commuters_est": total}
        for feature in FIELDS.values():
            pct = float(row[feature])
            if not (0 <= pct <= 100):
                raise ValueError(f"ACS {year}/{fips}: invalid {feature}={pct}")
            normalized[feature] = pct
        clean.append(normalized)
    return sorted(clean, key=lambda r: r["county_fips"])


def request_json(session: requests.Session, year: int, api_key: str):
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {"get": ",".join(["NAME", "B08301_001E", *FIELDS]),
              "for": "county:*", "in": "state:36", "key": api_key}
    for attempt in range(1, ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=(12, 40))
        except (requests.ConnectionError, requests.Timeout) as exc:
            reason = type(exc).__name__  # Do not print exception: may contain authenticated URL.
        except requests.RequestException as exc:
            raise RuntimeError(f"Census {year}: request failed ({type(exc).__name__})") from None
        else:
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise RuntimeError(f"Census {year}: response is not JSON") from None
            if response.status_code not in RETRYABLE_HTTP:
                raise RuntimeError(
                    f"Census {year}: HTTP {response.status_code}; verify API key/access."
                )
            reason = f"HTTP {response.status_code}"
        if attempt == ATTEMPTS:
            raise RuntimeError(
                f"Census {year}: {reason} after {ATTEMPTS} attempts. "
                "Year caches were kept; retry later."
            )
        delay = 2 ** attempt
        print(f"  {year}: {reason}; retrying in {delay}s "
              f"(attempt {attempt + 1}/{ATTEMPTS})...", flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def parse_response(year: int, payload) -> list[dict]:
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], list):
        raise ValueError(f"Census {year}: unexpected JSON response")
    headers = payload[0]
    required = {"state", "county", "B08301_001E", *FIELDS}
    if not required.issubset(headers):
        raise ValueError(f"Census {year}: missing required ACS columns")
    year_rows = []
    for values in payload[1:]:
        row = dict(zip(headers, values))
        fips = str(row.get("state", "")) + str(row.get("county", ""))
        if fips not in FIPS:
            continue
        try:
            total = int(row["B08301_001E"])
            if total <= 0:
                raise ValueError("nonpositive total")
            out = {"county_fips": fips, "borough": FIPS[fips],
                   "year": year, "acs_commuters_est": total}
            for variable, feature in FIELDS.items():
                count = int(row[variable])
                if not (0 <= count <= total):
                    raise ValueError(f"invalid count for {variable}")
                out[feature] = round(count / total * 100, 4)
        except (ValueError, TypeError, KeyError):
            raise ValueError(f"Census {year}/{fips}: invalid ACS worker estimate") from None
        year_rows.append(out)
    return validate_year(year, year_rows)


def atomic_json_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp.replace(path)


def collect(api_key: str, root: Path, session: requests.Session | None = None) -> list[dict]:
    cache_dir = root / "datasets" / "raw" / "acs" / "_year_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = requests.Session()
    session.headers.update({"User-Agent": "UTD-Air-Pollution-ACS-Collection/1.1"})
    all_rows = []
    for year in YEARS:
        cache = cache_dir / f"acs_{year}.json"
        if cache.exists():
            try:
                stored = json.loads(cache.read_text(encoding="utf-8"))
                if stored.get("year") != year or stored.get("source") != "US_CENSUS_ACS5_B08301":
                    raise ValueError("Unexpected cached year/source")
                year_rows = validate_year(year, stored["rows"])
                print(f"ACS {year}: using verified local cache (5 NYC counties).", flush=True)
                all_rows.extend(year_rows)
                continue
            except (OSError, ValueError, TypeError, KeyError) as exc:
                # Invalid caches must not be trusted. New API fetch can recover.
                print(f"ACS {year}: invalid cache ({type(exc).__name__}); downloading again.", flush=True)
        print(f"Collecting ACS {year} 5-year vintage...", flush=True)
        payload = request_json(session, year, api_key)
        year_rows = parse_response(year, payload)
        atomic_json_save(cache, {
            "source": "US_CENSUS_ACS5_B08301", "year": year,
            "url": f"https://api.census.gov/data/{year}/acs/acs5",
            "collected_utc": datetime.now(timezone.utc).isoformat(), "rows": year_rows,
        })
        print(f"  Saved ACS {year}: 5 verified NYC counties.", flush=True)
        all_rows.extend(year_rows)
    if len(all_rows) != 55 or len({(x["county_fips"], x["year"]) for x in all_rows}) != 30:
        raise ValueError("Historical ACS output failed 30-record uniqueness check")
    return sorted(all_rows, key=lambda x: (x["county_fips"], x["year"]))


def main() -> None:
    api_key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("CENSUS_API_KEY is missing from this terminal session. Set it locally and retry.")
    root = Path(__file__).resolve().parents[1]
    destination = root / "datasets" / "raw" / "acs" / "nyc_acs_commuting_2014_2024.csv"
    try:
        rows = collect(api_key, root)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"NOT COMPLETE: {exc}\nExisting validated year caches remain saved. Rerun when connected.") from None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(destination)
    print(f"SUCCESS: saved 30 verified NYC county-year ACS estimates to {destination}")
    print("Now run: python scripts/prepare_model_data.py")


if __name__ == "__main__":
    main()
