#!/usr/bin/env python3
"""Final structural audit and *candidate* annual model handoff for NYC AQI.

Run from the Air-Pollution-main repository:
    python scripts/finalize_air_pollution_handoff.py

Only reads existing outputs from prepare_model_data.py and
review_ev_and_stage_data.py. Writes to datasets/processed/handoff/.
No fitting, imputation, normalization, or edits of raw/master data.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

NYC = {
    "36005": "Bronx",
    "36047": "Brooklyn",
    "36061": "Manhattan",
    "36081": "Queens",
    "36085": "Staten Island",
}
YEARS = range(2020, 2025)
TARGET = "target_next_year_median_aqi"
FEATURES = [
    "last_observed_median_aqi", "recorded_tree_plantings", "drive_alone_pct",
    "campd_total_nox_mass_tons", "weather_temp_mean_c",
    "weather_precip_total_mm",
]
KEYS = ["county_fips", "borough", "feature_year", "forecast_year"]


def require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f"{name} missing columns: {', '.join(absent)}")


def normalize_fips(df: pd.DataFrame, name: str) -> pd.DataFrame:
    data = df.copy()
    data["county_fips"] = (data["county_fips"].astype("string").str.strip()
                           .str.replace(r"\.0$", "", regex=True).str.zfill(5))
    if data["county_fips"].isna().any() or not data["county_fips"].isin(NYC).all():
        raise ValueError(f"{name} has invalid NYC county FIPS codes")
    return data


def load_data(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} not found: {path}")
    return normalize_fips(pd.read_csv(path, dtype={"county_fips": "string"}), name)


def comparable(a: pd.Series, b: pd.Series) -> bool:
    x = pd.to_numeric(a, errors="raise")
    y = pd.to_numeric(b, errors="raise")
    return bool(((x - y).abs() <= 1e-8).all() and x.notna().all() and y.notna().all())


def run(root: Path) -> dict:
    data_root = root / "datasets" / "processed"
    candidate_path = data_root / "review" / "annual_next_year_baseline_COMPLETE_CASES_REVIEW_ONLY.csv"
    master_path = data_root / "master_borough_year_2019_2025.csv"
    candidate = load_data(candidate_path, "baseline candidate")
    master = load_data(master_path, "annual master")
    require_columns(candidate, KEYS + [TARGET] + FEATURES, "baseline candidate")
    require_columns(master, ["county_fips", "borough", "year", "aqi_median", "aqi_days_covered"], "annual master")

    for name, frame, year_col in [("candidate", candidate, "feature_year"), ("master", master, "year")]:
        if frame.duplicated(["county_fips", year_col]).any():
            raise ValueError(f"Duplicate borough-year keys in {name}")
        if frame["borough"].ne(frame["county_fips"].map(NYC)).any():
            raise ValueError(f"Inconsistent borough name vs county FIPS in {name}")

    require_columns(master, [c for c in FEATURES if c != "last_observed_median_aqi"], "annual master features")
    for col in ["feature_year", "forecast_year"]:
        candidate[col] = pd.to_numeric(candidate[col], errors="raise").astype(int)
    master["year"] = pd.to_numeric(master["year"], errors="raise").astype(int)

    complete_grid = {(fips, y) for fips in NYC for y in range(2019, 2026)}
    if set(zip(master["county_fips"], master["year"])) != complete_grid or len(master) != 35:
        raise ValueError("Master must have 35 borough-year rows covering all 5 NYC boroughs and 2019-2025")

    expected = {(fips, y) for fips in NYC for y in YEARS}
    actual = set(zip(candidate["county_fips"], candidate["feature_year"]))
    if expected != actual or len(candidate) != 25:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"Expected five boroughs in feature years 2020-2024 (25 rows); "
                         f"found {len(candidate)}. Missing: {missing[:5]}; extra: {extra[:5]}")
    if not candidate["forecast_year"].eq(candidate["feature_year"] + 1).all():
        raise ValueError("forecast_year must be exactly feature_year + 1")

    numeric_cols = [TARGET, *FEATURES]
    for col in numeric_cols:
        candidate[col] = pd.to_numeric(candidate[col], errors="raise")
    if candidate[numeric_cols].isna().any().any():
        raise ValueError("Baseline candidate contains missing target/features, cannot pass complete-case audit")

    if not (candidate[TARGET].between(0, 500).all()
            and candidate["last_observed_median_aqi"].between(0, 500).all()):
        raise ValueError("AQI values outside expected AQI bounds 0..500")
    if not candidate["drive_alone_pct"].between(0, 100).all():
        raise ValueError("Drive-alone percentage outside 0..100")
    for col in ["recorded_tree_plantings", "campd_total_nox_mass_tons", "weather_precip_total_mm"]:
        if not candidate[col].ge(0).all():
            raise ValueError(f"Negative values in {col}")
    if not candidate["weather_temp_mean_c"].between(-90, 65).all():
        raise ValueError("Suspicious annual temperature value outside plausible Celsius bounds")

    # Independently verify both the historic AQI input and next-year AQI target
    # using the existing annual source. This catches accidental target shifts.
    observed = master[["county_fips", "year", "aqi_median", "aqi_days_covered"]].copy()
    current = observed.rename(columns={"year": "feature_year", "aqi_median": "observed_from_master"})
    next_year = observed.rename(columns={"year": "forecast_year", "aqi_median": "target_from_master",
                                         "aqi_days_covered": "target_days_covered"})
    check = candidate.merge(current[["county_fips", "feature_year", "observed_from_master"]],
                            on=["county_fips", "feature_year"], how="left", validate="one_to_one")
    check = check.merge(next_year[["county_fips", "forecast_year", "target_from_master", "target_days_covered"]],
                        on=["county_fips", "forecast_year"], how="left", validate="one_to_one")
    if not comparable(check["last_observed_median_aqi"], check["observed_from_master"]):
        raise ValueError("Last observed AQI does NOT match the master AQI for feature year")
    if not comparable(check[TARGET], check["target_from_master"]):
        raise ValueError("Next-year AQI target does NOT match master AQI for forecast year")
    if not pd.to_numeric(check["target_days_covered"], errors="raise").ge(300).all():
        raise ValueError("Some target-year AQI outcomes have fewer than 300 monitored days")
    # Make sure all predictor values were attached to the correct borough/year.
    predictor_columns = [c for c in FEATURES if c != "last_observed_median_aqi"]
    feature_reference = master[["county_fips", "year", *predictor_columns]].rename(columns={
        "year": "feature_year", **{col: f"master_{col}" for col in predictor_columns}
    })
    aligned = candidate.merge(feature_reference, on=["county_fips", "feature_year"],
                              how="left", validate="one_to_one")
    for col in predictor_columns:
        if not comparable(aligned[col], aligned[f"master_{col}"]):
            raise ValueError(f"Feature alignment mismatch for {col}: wrong year, borough, or data")

    # Year-level holdout: all boroughs in one forecast year must stay together.
    splits = {2020: "train", 2021: "train", 2022: "train",
              2023: "validation", 2024: "test"}
    handoff = candidate[KEYS + [TARGET] + FEATURES].copy()
    handoff["split"] = handoff["feature_year"].map(splits)
    handoff = handoff.sort_values(["feature_year", "county_fips"]).reset_index(drop=True)

    # A smaller comparison candidate without the historical ACS/CAMPD variables
    # whose exact point-in-time release availability has not been audited.
    limited_features = ["last_observed_median_aqi", "recorded_tree_plantings",
                        "weather_temp_mean_c", "weather_precip_total_mm"]
    reduced = handoff[KEYS + [TARGET] + limited_features + ["split"]].copy()

    dest = data_root / "handoff"
    dest.mkdir(parents=True, exist_ok=True)
    handoff.to_csv(dest / "annual_next_year_RESEARCH_CANDIDATE.csv", index=False, encoding="utf-8-sig")
    reduced.to_csv(dest / "annual_next_year_REDUCED_CANDIDATE.csv", index=False, encoding="utf-8-sig")
    for name in ("train", "validation", "test"):
        handoff.loc[handoff["split"] == name].to_csv(dest / f"research_{name}_CANDIDATE.csv",
                                                           index=False, encoding="utf-8-sig")

    counts = {label: int(handoff["split"].eq(label).sum()) for label in ["train", "validation", "test"]}
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "audit": "PASS - keys, borough mapping, numeric bounds, missingness, year shift, and AQI source/target alignment",
        "note": "Structural QA only. Neither a deployment-ready time-series model nor proof of forecasting skill.",
        "rows": int(len(handoff)), "rows_by_split": counts,
        "feature_years": list(YEARS),
        "forecast_years": [2021, 2022, 2023, 2024, 2025],
        "split_rule": "Train feature years 2020-2022 / validation 2023 / test 2024 (entire borough-year groups)",
        "aqi_target": TARGET,
        "research_features": FEATURES,
        "reduced_features": limited_features,
        "warnings": [
            "ACS five-year estimates include their end year but may not be published until the next calendar year or later; the drive_alone_pct column must be removed or lagged for a real-time next-year forecast.",
            "CAMPD industrial annual totals, annual tree inventories, annual weather/reanalysis, and annual AQI may also have reporting/revision lags; their real-world point-in-time availability has NOT been audited.",
            "Even the reduced candidate is not certified leakage-free or deployment-ready.",
            "No scalers, imputers, feature selection, or target-based transformations have been fitted.",
            "Only 25 correlated NYC borough-year cases; 15 train, 5 validation, and 5 test. Metrics will have high uncertainty; compare against a no-change AQI baseline and avoid claims of robust forecast accuracy.",
            "Do not treat 2024-2025 missing EV registrations as zero or backfill historical data from current snapshots.",
            "Traffic observations are nonrepresentative samples with year-varying road segments, so excluded pending comparability review.",
            "Do not commit API keys, cached private credentials, or very large raw downloads to GitHub.",
        ],
        "created_files": [p.name for p in sorted(dest.glob("*CANDIDATE.csv"))],
    }
    (dest / "audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PASS: structural quality audit and master AQI/year alignment")
    print("Candidate research dataset: 25 rows (feature years 2020-2024; AQI targets 2021-2025)")
    print("Split: train=15, validation=5, test=5 (chronological by feature year)")
    print("Outputs:", dest.resolve())
    print("IMPORTANT: time-of-release data leakage has NOT been resolved; see audit_report.json")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality audit and candidate annual AQI model handoff")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root directory")
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
