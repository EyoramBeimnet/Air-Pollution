from pathlib import Path
import argparse
import pandas as pd

COUNTIES = {"36005":"Bronx","36047":"Kings","36061":"New York","36081":"Queens","36085":"Richmond"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("datasets/raw/daily_aqi/nyc_daily_aqi_2014_2025.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("datasets/processed/monthly"))
    p.add_argument("--min-coverage", type=float, default=0.75)
    a = p.parse_args()
    if not 0 < a.min_coverage <= 1: raise ValueError("--min-coverage must be in (0,1].")
    d = pd.read_csv(a.input, dtype={"county_fips":"string"})
    if {"date","county_fips","aqi"} - set(d.columns): raise ValueError("Need date, county_fips, aqi columns.")
    d["date"] = pd.to_datetime(d.date, errors="coerce")
    d["county_fips"] = d.county_fips.astype("string").str.zfill(5)
    d["aqi"] = pd.to_numeric(d.aqi, errors="coerce")
    d = d.dropna(subset=["date","county_fips","aqi"])
    d = d[d.county_fips.isin(COUNTIES)].copy()
    if d.duplicated(["county_fips","date"]).any(): raise ValueError("Duplicate county/date rows; resolve before aggregation.")
    d["borough"] = d.county_fips.map(COUNTIES)
    d["target_month"] = d.date.dt.to_period("M").dt.to_timestamp()
    m = d.groupby(["county_fips","borough","target_month"], as_index=False).agg(
        aqi_median_daily=("aqi","median"), aqi_mean_daily=("aqi","mean"), aqi_days_observed=("aqi","count"))
    m["calendar_days_in_month"] = m.target_month.dt.days_in_month
    m["aqi_coverage_pct"] = m.aqi_days_observed / m.calendar_days_in_month
    m["meets_coverage"] = m.aqi_coverage_pct >= a.min_coverage
    a.output_dir.mkdir(parents=True, exist_ok=True)
    m.to_csv(a.output_dir/"monthly_aqi_coverage_audit.csv", index=False)
    usable = m[m.meets_coverage].copy()
    usable.to_csv(a.output_dir/"monthly_aqi_targets.csv", index=False)
    print(f"Monthly rows: {len(m)}; passing coverage: {len(usable)} ({a.min_coverage:.0%} threshold)")

if __name__ == "__main__":
    main()
