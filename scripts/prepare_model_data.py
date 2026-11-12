#!/usr/bin/env python3
"""Prepare the Air-Pollution team's existing NYC datasets for borough-year modeling.

Run from the GitHub repository root (place this file at scripts/prepare_model_data.py):
    pip install pandas requests
    python scripts/prepare_model_data.py

Inputs remain unchanged. Writes datasets/processed/ only. Downloads official EPA
annual AQI ZIP files (FIRST_YEAR-LAST_YEAR) to datasets/raw/epa_aqi/ for reproducibility.
No missing values are invented or set to zero, and no static snapshot is
backfilled across years. When available, a verified ACS CSV from
collect_historical_acs.py replaces the original sparse 2024-only solver ACS values.
Years a source does not cover (e.g. solver features before 2019) stay blank.
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

COUNTIES = {
    "Bronx": ("36005", "Bronx"),
    "Kings": ("36047", "Brooklyn"),
    "New York": ("36061", "Manhattan"),
    "Queens": ("36081", "Queens"),
    "Richmond": ("36085", "Staten Island"),
}
FIPS_TO_BOROUGH = {fips: borough for fips, borough in COUNTIES.values()}
FIRST_YEAR, LAST_YEAR = 2014, 2025
ACS_LAST_YEAR = 2024  # newest ACS 5-year vintage available
TREE_FIRST_YEAR = 2016  # the tree inventory has almost no planting records before 2015 (2015 looks partial)
SPAN = f"{FIRST_YEAR}_{LAST_YEAR}"  # used in output filenames


def earliest_file(folder: Path, pattern: str) -> Path | None:
    """Of the files matching pattern, pick the one whose name starts at the earliest year."""
    return min(folder.glob(pattern), default=None) if folder.exists() else None


def fips_str(col: pd.Series) -> pd.Series:
    """Normalize 36005, '36005', and 36005.0 without creating fake codes."""
    return col.astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(5)


def clean_numeric(frame: pd.DataFrame, fields: list[str], issues: list[str], source: str) -> None:
    for field in fields:
        original = frame[field]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = int(original.notna().sum() - converted.notna().sum())
        if invalid:
            issues.append(f"{source}: {invalid} nonnumeric values in {field} -> missing")
        frame[field] = converted


def validate_key(df: pd.DataFrame, source: str) -> None:
    if df.empty:
        raise ValueError(f"{source}: no usable rows")
    if df[["county_fips", "year"]].isna().any().any():
        raise ValueError(f"{source}: missing county/year join key")
    duplicates = df.duplicated(["county_fips", "year"], keep=False)
    if duplicates.any():
        sample = df.loc[duplicates, ["county_fips", "year"]].head(6).to_dict("records")
        raise ValueError(f"{source}: duplicate borough/year rows: {sample}")
    if not df["county_fips"].isin(FIPS_TO_BOROUGH).all():
        raise ValueError(f"{source}: unknown NYC county FIPS")


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_solvers(path: Path, issues: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing solver combined CSV: {path}")
    df = pd.read_csv(path, dtype={"county_fips": "string"}, low_memory=False)
    for key in ("county_fips", "year"):
        if key not in df:
            raise ValueError(f"solvers: missing required column {key}")
    df["county_fips"] = fips_str(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df = df[df["county_fips"].isin(FIPS_TO_BOROUGH) & df["year"].between(FIRST_YEAR, LAST_YEAR)].copy()
    df = df.drop(columns=["borough"], errors="ignore")
    features = [c for c in df if c not in ("county_fips", "year")]
    clean_numeric(df, features, issues, "solvers")
    validate_key(df, "solvers")
    return df



ACS_COLUMNS = ["acs_commuters_est", "drive_alone_pct", "carpool_pct",
               "public_transit_pct", "bicycle_commute_pct", "walk_commute_pct",
               "work_from_home_pct"]


def load_historical_acs(path: Path, issues: list[str]) -> pd.DataFrame:
    """Read separately collected verified ACS 5-year vintage features."""
    df = pd.read_csv(path, dtype={"county_fips": "string"}, low_memory=False)
    required = {"county_fips", "year", *ACS_COLUMNS}
    if not required.issubset(df.columns):
        raise ValueError(f"ACS file missing columns {sorted(required - set(df.columns))}")
    df["county_fips"] = fips_str(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    if not df["year"].between(FIRST_YEAR, ACS_LAST_YEAR).all():
        raise ValueError(f"ACS should contain {FIRST_YEAR}-{ACS_LAST_YEAR} vintages only")
    data = df[["county_fips", "year", *ACS_COLUMNS]].copy()
    clean_numeric(data, ACS_COLUMNS, issues, "acs_transport")
    validate_key(data, "acs_transport")
    # The file must be a complete grid from its own first year through ACS_LAST_YEAR.
    first = int(data["year"].min())
    keys = {(fips, year) for fips in FIPS_TO_BOROUGH for year in range(first, ACS_LAST_YEAR + 1)}
    actual = set(zip(data["county_fips"], data["year"]))
    if actual != keys:
        raise ValueError(f"ACS file is incomplete: expected {len(keys)} borough-years, got {len(actual & keys)}")
    if data[ACS_COLUMNS].isna().any().any():
        raise ValueError("ACS source has missing feature values; investigate rather than silently impute")
    pct_columns = [x for x in ACS_COLUMNS if x.endswith("_pct")]
    if (data[pct_columns].lt(0) | data[pct_columns].gt(100)).any().any():
        raise ValueError("ACS contains an out-of-range percentage")
    return data

WEATHER_FEATURES = ["weather_days_complete", "weather_temp_mean_c",
                    "weather_mean_daily_max_temp_c", "weather_precip_total_mm",
                    "weather_wind_mean_kmh", "weather_humidity_mean_pct"]


def load_weather(path: Path, issues: list[str]) -> pd.DataFrame:
    """Read validated ERA5 point-sampled weather (not borough-area means)."""
    data = pd.read_csv(path, dtype={"county_fips": "string"}, low_memory=False)
    required = {"county_fips", "year", *WEATHER_FEATURES}
    if not required.issubset(data.columns):
        raise ValueError(f"Weather CSV missing columns: {sorted(required - set(data.columns))}")
    data["county_fips"] = fips_str(data["county_fips"])
    data["year"] = pd.to_numeric(data["year"], errors="raise").astype(int)
    data = data[["county_fips", "year", *WEATHER_FEATURES]].copy()
    clean_numeric(data, WEATHER_FEATURES, issues, "weather")
    validate_key(data, "weather")
    # Must be a complete grid from the file's own first year through LAST_YEAR.
    first = int(data["year"].min())
    expected = {(f, y) for f in FIPS_TO_BOROUGH for y in range(first, LAST_YEAR + 1)}
    if set(zip(data["county_fips"], data["year"])) != expected:
        raise ValueError(f"Weather CSV does not cover all {len(expected)} expected NYC borough-year keys")
    if data["weather_days_complete"].isna().any() or not data["weather_days_complete"].between(0, 366).all():
        raise ValueError("Weather days_complete column contains invalid counts")
    if data["weather_humidity_mean_pct"].dropna().lt(0).any() or data["weather_humidity_mean_pct"].dropna().gt(100).any():
        raise ValueError("Weather mean humidity must be between 0 and 100 percent")
    if data["weather_wind_mean_kmh"].dropna().lt(0).any() or data["weather_precip_total_mm"].dropna().lt(0).any():
        raise ValueError("Weather contains negative wind speed or precipitation")
    issues.append("Weather ERA5 historical reanalysis sampled at a representative point for each borough; NOT borough-average measured weather")
    return data


MTA_COLUMNS = ["mta_annual_station_entries_est", "mta_months_observed"]


def load_mta(path: Path, issues: list[str]) -> pd.DataFrame:
    """Read annual subway station entries by borough from collect_nyc_mta.py."""
    df = pd.read_csv(path, dtype={"county_fips": "string"}, low_memory=False)
    required = {"county_fips", "year", *MTA_COLUMNS}
    if not required.issubset(df.columns):
        raise ValueError(f"MTA file missing columns {sorted(required - set(df.columns))}")
    df["county_fips"] = fips_str(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    data = df.loc[df["year"].between(FIRST_YEAR, LAST_YEAR), ["county_fips", "year", *MTA_COLUMNS]].copy()
    clean_numeric(data, MTA_COLUMNS, issues, "mta")
    validate_key(data, "mta")
    if data["mta_annual_station_entries_est"].lt(0).any():
        raise ValueError("MTA entries must not be negative")
    partial = sorted(int(y) for y in data.loc[data["mta_months_observed"] < 12, "year"].unique())
    if partial:
        issues.append(f"MTA: fewer than 12 months of data in {partial}; those totals are understated")
    issues.append("MTA: hourly-ridership datasets only start in 2017; earlier years stay blank")
    return data


TRAFFIC_COLUMNS = ["traffic_sample_mean_15min_vehicle_count", "traffic_sample_observed_15min_intervals",
                   "traffic_sample_distinct_segments", "traffic_sample_distinct_requests"]


def load_traffic(path: Path, issues: list[str]) -> pd.DataFrame:
    """Read NYC DOT sampled traffic counts from collect_nyc_traffic.py (gaps allowed)."""
    df = pd.read_csv(path, dtype={"county_fips": "string"}, low_memory=False)
    required = {"county_fips", "year", *TRAFFIC_COLUMNS}
    if not required.issubset(df.columns):
        raise ValueError(f"Traffic file missing columns {sorted(required - set(df.columns))}")
    df["county_fips"] = fips_str(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    data = df.loc[df["year"].between(FIRST_YEAR, LAST_YEAR), ["county_fips", "year", *TRAFFIC_COLUMNS]].copy()
    clean_numeric(data, TRAFFIC_COLUMNS, issues, "traffic")
    validate_key(data, "traffic")
    issues.append("Traffic: sampled 15-minute counts at a changing set of road segments, NOT annual traffic "
                  "volume; check the sample-size columns before trusting year-to-year trends")
    return data


def load_trees(path: Path, issues: list[str]) -> pd.DataFrame:
    """Read recorded tree plantings by borough-year from collect_nyc_trees.py (gaps allowed)."""
    df = pd.read_csv(path, dtype={"county_fips": "string"}, low_memory=False)
    required = {"county_fips", "year", "recorded_tree_plantings"}
    if not required.issubset(df.columns):
        raise ValueError(f"Tree file missing columns {sorted(required - set(df.columns))}")
    df["county_fips"] = fips_str(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    data = df.loc[df["year"].between(FIRST_YEAR, LAST_YEAR), ["county_fips", "year", "recorded_tree_plantings"]].copy()
    clean_numeric(data, ["recorded_tree_plantings"], issues, "trees")
    early = data["year"] < TREE_FIRST_YEAR
    if early.any():
        issues.append(f"Tree plantings: {int(early.sum())} borough-year values before {TREE_FIRST_YEAR} left blank; "
                      "the inventory has almost no planting records before 2015, and 2015 looks partial")
        data = data[~early].copy()
    validate_key(data, "trees")
    if data["recorded_tree_plantings"].lt(0).any():
        raise ValueError("Tree plantings must not be negative")
    issues.append("Tree plantings: trees in the current Forestry inventory with a planted date; trees removed "
                  "before records were migrated may be missing, so older years are probably undercounted")
    return data


def load_lucid(folder: Path, issues: list[str]) -> pd.DataFrame:
    files = sorted(folder.glob("*-19-22.csv")) if folder.exists() else []
    if not files:
        raise FileNotFoundError(f"No LUCID borough CSVs found in {folder}")
    frames = []
    for p in files:
        df = pd.read_csv(p)
        required = {"State", "County", "Year", "Forsestland", "Urban Builtup", "Pastureland"}
        if not required.issubset(df.columns):
            raise ValueError(f"LUCID {p.name}: missing columns: {required - set(df.columns)}")
        df = df[df["State"].astype(str).str.strip().eq("New York")].copy()
        df["county_fips"] = df["County"].map({k: v[0] for k, v in COUNTIES.items()})
        df = df[df["county_fips"].notna()].copy()
        df["year"] = pd.to_numeric(df["Year"], errors="raise").astype(int)
        # The supplied CSV literally spells Forestland as 'Forsestland'.
        rename = {"Forsestland": "landuse_forestland_pct",
                  "Urban Builtup": "landuse_urban_builtup_pct",
                  "Pastureland": "landuse_pastureland_pct"}
        data = df[["county_fips", "year", *rename]].rename(columns=rename)
        clean_numeric(data, list(rename.values()), issues, p.name)
        if ((data[list(rename.values())] < 0) | (data[list(rename.values())] > 100)).any().any():
            issues.append(f"LUCID {p.name}: some land-use percentages are outside 0-100; inspect source")
        frames.append(data)
    out = pd.concat(frames, ignore_index=True)
    out = out[out["year"].between(FIRST_YEAR, LAST_YEAR)].copy()
    validate_key(out, "LUCID")
    seen = set(out["county_fips"])
    missing_boroughs = set(FIPS_TO_BOROUGH) - seen
    if missing_boroughs:
        issues.append(f"LUCID: missing boroughs {sorted(missing_boroughs)}")
    # LUCID observations currently stop at 2022. Do not forward-fill them.
    return out


def load_campd(path: Path, issues: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CAMPD CSV: {path}")
    df = pd.read_csv(path, low_memory=False)
    required = {"state_abbr", "county", "year", "is_partial_year"}
    if not required.issubset(df.columns):
        raise ValueError(f"CAMPD: missing {required - set(df.columns)}")
    df = df[df["state_abbr"].astype(str).str.strip().eq("NY")].copy()
    df["county_fips"] = df["county"].map({k: v[0] for k, v in COUNTIES.items()})
    df = df[df["county_fips"].notna()].copy()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    is_partial = df["is_partial_year"].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])
    if is_partial.any():
        issues.append(f"CAMPD: excluded {int(is_partial.sum())} partial-year NYC records")
    df = df[~is_partial & df["year"].between(FIRST_YEAR, LAST_YEAR)].copy()
    selected = ["num_active_units", "total_co2_mass_tons", "total_so2_mass_tons",
                "total_nox_mass_tons", "total_gross_load_mwh", "total_heat_input_mmbtu",
                "co2_rate_tons_per_mwh", "so2_rate_tons_per_mwh", "nox_rate_tons_per_mwh"]
    missing = set(selected) - set(df.columns)
    if missing:
        raise ValueError(f"CAMPD: missing feature columns: {sorted(missing)}")
    df = df[["county_fips", "year", *selected]].copy()
    clean_numeric(df, selected, issues, "CAMPD")
    df = df.rename(columns={c: "campd_" + c for c in selected})
    validate_key(df, "CAMPD")
    return df


def load_residential_snapshot(path: Path, issues: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing residential CSV: {path}")
    df = pd.read_csv(path, dtype={"fips code": "string"}, low_memory=False)
    required = {"fips code", "sector", "pollutant desc", "total emissions", "emissions uom"}
    if not required.issubset(df.columns):
        raise ValueError(f"Residential: missing {required - set(df.columns)}")
    df["county_fips"] = fips_str(df["fips code"])
    df = df[df["county_fips"].isin(FIPS_TO_BOROUGH)].copy()
    df["emissions_uom_original"] = df["emissions uom"].astype("string").str.strip().str.upper()
    df["value"] = pd.to_numeric(df["total emissions"], errors="coerce")
    valid_units = df["emissions_uom_original"].isin(["LB", "TON"])
    invalid_count = int((~valid_units | df["value"].isna()).sum())
    if invalid_count:
        issues.append(f"Residential: excluded {invalid_count} rows with invalid units/amounts")
    df = df[valid_units & df["value"].notna()].copy()
    # 'TON' and 'LB' in the source are US short tons and pounds: 1 short ton = 2000 lb.
    df["emissions_short_tons"] = df["value"] / df["emissions_uom_original"].map({"LB": 2000, "TON": 1})
    df = df.rename(columns={"pollutant desc": "pollutant", "sector": "sector"})
    out = df.groupby(["county_fips", "sector", "pollutant"], dropna=False, as_index=False)["emissions_short_tons"].sum(min_count=1)
    out.insert(1, "borough", out["county_fips"].map(FIPS_TO_BOROUGH))
    # Deliberately no year column: source has no documented year in the CSV.
    # Do not sum unrelated pollutant types into one total or add these to annual features.
    return out


def download_epa_aqi(raw_dir: Path, year: int, download: bool) -> pd.DataFrame:
    """Read cached EPA annual county AQI ZIP; download from EPA if requested/needed."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"annual_aqi_by_county_{year}.zip"
    if not dest.exists():
        if not download:
            raise FileNotFoundError(f"No cached EPA ZIP for {year} (download disabled)")
        url = f"https://aqs.epa.gov/aqsweb/airdata/annual_aqi_by_county_{year}.zip"
        response = requests.get(url, timeout=70, headers={"User-Agent": "UTD-NYC-Air-Pollution-ETL/1.0"})
        response.raise_for_status()
        try:
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                if not any(n.lower().endswith(".csv") for n in z.namelist()):
                    raise ValueError("EPA ZIP contains no CSV")
        except zipfile.BadZipFile as exc:
            raise ValueError(f"EPA download for {year} was not a ZIP") from exc
        dest.write_bytes(response.content)
    with zipfile.ZipFile(dest) as z:
        candidates = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not candidates:
            raise ValueError(f"EPA archive {dest.name} has no CSV")
        with z.open(candidates[0]) as f:
            return pd.read_csv(f, low_memory=False)


def load_epa_targets(raw_dir: Path, download: bool, issues: list[str]) -> pd.DataFrame:
    frames = []
    wanted = {k: v[0] for k, v in COUNTIES.items()}
    for year in range(FIRST_YEAR, LAST_YEAR + 1):
        try:
            df = download_epa_aqi(raw_dir, year, download)
            required = {"State", "County", "Year", "Days with AQI", "Median AQI", "Max AQI", "90th Percentile AQI"}
            if not required.issubset(df):
                raise ValueError(f"Missing EPA columns: {sorted(required - set(df))}")
            nyc = df[df["State"].astype(str).str.strip().eq("New York")].copy()
            nyc["county_fips"] = nyc["County"].map(wanted)
            nyc = nyc[nyc["county_fips"].notna()].copy()
            nyc["year"] = pd.to_numeric(nyc["Year"], errors="raise").astype(int)
            if not nyc["year"].eq(year).all():
                raise ValueError("EPA ZIP year disagrees with Year column")
            if nyc.empty:
                raise ValueError("No five-NYC-county observations found")
            rename = {"Days with AQI": "aqi_days_covered", "Median AQI": "aqi_median",
                      "Max AQI": "aqi_max", "90th Percentile AQI": "aqi_p90"}
            nyc = nyc[["county_fips", "year", *rename]].rename(columns=rename)
            clean_numeric(nyc, list(rename.values()), issues, f"EPA {year}")
            validate_key(nyc, f"EPA {year}")
            if nyc.shape[0] < 5:
                issues.append(f"EPA {year}: only {len(nyc)} of five NYC counties have an AQI observation")
            frames.append(nyc)
        except Exception as exc:
            issues.append(f"EPA {year} FAILED: {type(exc).__name__}: {exc}")
    if not frames:
        # Valid feature-only staging table is still useful; never invent labels.
        return pd.DataFrame(columns=["county_fips", "year", "aqi_days_covered", "aqi_median", "aqi_max", "aqi_p90"])
    out = pd.concat(frames, ignore_index=True)
    validate_key(out, f"EPA {FIRST_YEAR}-{LAST_YEAR}")
    return out


def run(root: Path, download: bool) -> None:
    datasets = root / "datasets"
    processed = datasets / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    issues: list[str] = []
    source_stats: list[dict] = []
    baseline = pd.MultiIndex.from_product([sorted(FIPS_TO_BOROUGH), range(FIRST_YEAR, LAST_YEAR + 1)], names=["county_fips", "year"]).to_frame(index=False)
    baseline.insert(1, "borough", baseline["county_fips"].map(FIPS_TO_BOROUGH))
    master = baseline.copy()

    # Preserve the original GitHub solver CSV. If historical ACS was collected,
    # use its dedicated source instead of the older 2024-only columns. If several
    # ACS / CAMPD files exist, the one starting in the earliest year is used.
    historical_acs_file = earliest_file(datasets / "raw" / "acs", f"nyc_acs_commuting_*_{ACS_LAST_YEAR}.csv")
    use_historical_acs = historical_acs_file is not None
    campd_file = (earliest_file(datasets, "campd_industrial_emissions_county_year_*.csv")
                  or datasets / "campd_industrial_emissions_county_year_2020_2026.csv")
    sources = [
        ("solvers", lambda: load_solvers(datasets / "solvers" / "combined_solver_features_by_borough_year.csv", issues)),
        ("lucid", lambda: load_lucid(datasets / "LUCID", issues)),
        ("campd", lambda: load_campd(campd_file, issues)),
    ]
    weather_path = earliest_file(datasets / "raw" / "weather", f"nyc_weather_borough_year_*_{LAST_YEAR}.csv")
    if weather_path is not None:
        sources.append(("weather", lambda: load_weather(weather_path, issues)))
    else:
        issues.append("Weather: no cleaned borough-year file yet; run python scripts/collect_nyc_weather.py")
    mta_path = earliest_file(datasets / "raw" / "mta", f"nyc_mta_station_entries_borough_year_*_{LAST_YEAR}.csv")
    if mta_path is not None:
        sources.append(("mta", lambda: load_mta(mta_path, issues)))
    else:
        issues.append("MTA: no collected file; using the solver 2020-2025 values. Run python scripts/collect_nyc_mta.py")
    traffic_path = earliest_file(datasets / "raw" / "traffic", f"nyc_dot_traffic_SAMPLE_borough_year_*_{LAST_YEAR}.csv")
    if traffic_path is not None:
        sources.append(("traffic", lambda: load_traffic(traffic_path, issues)))
    else:
        issues.append("Traffic: no collected file yet; run python scripts/collect_nyc_traffic.py")
    trees_path = earliest_file(datasets / "raw" / "trees", f"nyc_tree_plantings_borough_year_*_{LAST_YEAR}.csv")
    if trees_path is not None:
        sources.append(("trees", lambda: load_trees(trees_path, issues)))
    else:
        issues.append("Tree plantings: no collected file; using the solver 2019-2025 values. Run python scripts/collect_nyc_trees.py")
    if use_historical_acs:
        # Validate ACS *before* dropping the existing values; never lose data
        # because of a partial or invalid newly downloaded source.
        try:
            historical_acs = load_historical_acs(historical_acs_file, issues)
        except Exception as exc:
            raise RuntimeError(f"The historical ACS file is invalid; fix or remove it before rerunning: {exc}") from exc
        issues.append(f"ACS: historical five-year vintages loaded from {historical_acs_file.name}; {ACS_LAST_YEAR + 1} remains unavailable")
        sources.append(("acs_transport", lambda: historical_acs))
    for name, fn in sources:
        try:
            data = fn()
            if name == "solvers":
                # ACS, MTA and tree plantings have their own collectors; each should appear only once after the join.
                if use_historical_acs:
                    data = data.drop(columns=ACS_COLUMNS, errors="ignore")
                if mta_path is not None:
                    data = data.drop(columns=["mta_annual_station_entries_est"], errors="ignore")
                if trees_path is not None:
                    data = data.drop(columns=["recorded_tree_plantings"], errors="ignore")
            save_csv(data.sort_values(["county_fips", "year"]), processed / f"clean_{name}_borough_year.csv")
            overlaps = (set(master.columns) & set(data.columns)) - {"county_fips", "year"}
            if overlaps:
                raise ValueError(f"{name}: conflicting feature names {sorted(overlaps)}")
            master = master.merge(data, on=["county_fips", "year"], how="left", validate="one_to_one")
            source_stats.append({"source": name, "status": "OK", "rows": len(data), "columns": list(data.columns)})
        except Exception as exc:
            issues.append(f"{name} FAILED: {type(exc).__name__}: {exc}")
            source_stats.append({"source": name, "status": "FAILED", "error": str(exc)})
            if name == "solvers":
                raise  # The main integration input is mandatory.

    try:
        residential = load_residential_snapshot(datasets / "NYC_Residential_Combustion.csv", issues)
        save_csv(residential, processed / "residential_emissions_snapshot_NOT_ANNUAL.csv")
        source_stats.append({"source": "residential_combustion", "status": "SNAPSHOT_ONLY", "rows": len(residential)})
        issues.append("Residential snapshot has no source year: excluded from annual training merge")
    except Exception as exc:
        issues.append(f"residential FAILED: {exc}")
        source_stats.append({"source": "residential_combustion", "status": "FAILED", "error": str(exc)})

    labels = load_epa_targets(datasets / "raw" / "epa_aqi", download, issues)
    save_csv(labels.sort_values(["county_fips", "year"]), processed / f"clean_epa_annual_aqi_{SPAN}.csv")
    source_stats.append({"source": "epa_aqi", "status": "OK" if not labels.empty else "FAILED", "rows": len(labels)})
    master = master.merge(labels, on=["county_fips", "year"], how="left", validate="one_to_one")
    master = master.sort_values(["county_fips", "year"]).reset_index(drop=True)
    validate_key(master, "master")
    if len(master) != 5 * (LAST_YEAR - FIRST_YEAR + 1):
        raise ValueError(f"Unexpected merged row count; expected {len(baseline)} borough-year rows")
    master_path = processed / f"master_borough_year_{SPAN}.csv"
    save_csv(master, master_path)

    # The following represents a ONE-YEAR-AHEAD task, not same-year feature leakage.
    # Features in year t are used to predict the AQI measured in year t+1.
    label_cols = ["aqi_days_covered", "aqi_median", "aqi_max", "aqi_p90"]
    features = [c for c in master.columns if c not in {"county_fips", "borough", "year", *label_cols}]
    forecast = master[["county_fips", "borough", "year", *features]].copy()
    forecast = forecast.rename(columns={"year": "feature_year"})
    forecast["forecast_year"] = forecast["feature_year"] + 1
    forecast["last_observed_median_aqi"] = master["aqi_median"]
    forecast["target_next_year_median_aqi"] = master.groupby("county_fips")["aqi_median"].shift(-1)
    forecast = forecast[forecast["forecast_year"].le(LAST_YEAR)].copy()
    cols = ["county_fips", "borough", "feature_year", "forecast_year", "target_next_year_median_aqi", "last_observed_median_aqi", *features]
    save_csv(forecast[cols], processed / "next_year_aqi_model_STAGING.csv")

    feature_cols = [c for c in master if c not in {"county_fips", "borough", "year", *label_cols}]
    missing = pd.DataFrame([{
        "column": c, "present_rows": int(master[c].notna().sum()),
        "missing_rows": int(master[c].isna().sum()),
        "missing_percent": round(100 * master[c].isna().mean(), 2),
    } for c in [*feature_cols, *label_cols]])
    save_csv(missing, processed / "feature_missingness_report.csv")
    coverage = master[["county_fips", "borough", "year", *label_cols]].copy()
    coverage["is_aqi_label_available"] = coverage["aqi_median"].notna()
    coverage["is_aqi_coverage_ge_300_days"] = coverage["aqi_days_covered"].ge(300).fillna(False)
    save_csv(coverage, processed / "aqi_year_coverage_report.csv")

    issues += [
        f"NYC_Residential_Combustion.csv: no year in supplied CSV, so do not join to {FIRST_YEAR}-{LAST_YEAR}",
        "PowerPlantEmissions2024 by Region.xlsx: regional 2024 only; not a borough-year feature",
        "annual_aqi_by_county_2026.csv: partial-year 2026 labels; deliberately excluded",
        "LUCID: 2019-2022 only; no backfill before 2019 or forward fill after 2022",
        "CAMPD: only complete-year rows in the supplied file are used; years it lacks stay blank",
        "No complete EV adoption or airport fact table found in the current datasets folder",
        "AQI label columns are outcomes, not predictors for same-year explanatory models",
        f"Annual data (at most {len(master)} NYC borough-year rows) is too sparse for reliable daily/weekly forecasts",
    ]
    report = {"modeling_unit": "NYC county (borough) x calendar year",
              "years": [FIRST_YEAR, LAST_YEAR], "master_rows": len(master),
              "master_feature_count": len(feature_cols),
              "aqi_label_rows": int(master["aqi_median"].notna().sum()),
              "next_year_supervised_rows": int(forecast["target_next_year_median_aqi"].notna().sum()),
              "source_status": source_stats, "issues_and_limitations": list(dict.fromkeys(issues)),
              "outputs": sorted(p.name for p in processed.glob("*.csv"))}
    (processed / "processing_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nData preparation complete.")
    print(f"Master: {master_path}")
    print(f"Rows: {len(master)}; observed AQI labels: {report['aqi_label_rows']}")
    print(f"Next-year training examples with labels: {report['next_year_supervised_rows']}")
    print(f"Quality report: {processed / 'processing_report.json'}")
    if issues:
        print(f"Read issues_and_limitations: {len(issues)} notes/failures; missing features remain blank.")


def main():
    parser = argparse.ArgumentParser(description="Clean and merge NYC Air-Pollution borough-year data")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root containing datasets/ (default: parent of scripts/)")
    parser.add_argument("--skip-aqi-download", action="store_true",
                        help="Only read previously cached EPA zip files; leave missing labels blank")
    args = parser.parse_args()
    run(args.root.resolve(), download=not args.skip_aqi_download)


if __name__ == "__main__":
    main()