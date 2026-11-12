from pathlib import Path
import argparse
import pandas as pd

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=Path("datasets/processed/monthly/monthly_aqi_targets.csv"))
    p.add_argument("--output", type=Path, default=Path("datasets/processed/monthly/monthly_aqi_model_STAGING.csv"))
    p.add_argument("--weather", type=Path, default=None, help="Optional prior-month weather CSV with county_fips, target_month, weather_* columns.")
    a = p.parse_args()
    d = pd.read_csv(a.input, dtype={"county_fips":"string"})
    required = {"county_fips","borough","target_month","aqi_median_daily"}
    if required-set(d.columns): raise ValueError(f"Missing columns: {sorted(required-set(d.columns))}")
    d["county_fips"] = d.county_fips.astype("string").str.zfill(5)
    d["target_month"] = pd.to_datetime(d.target_month).dt.to_period("M").dt.to_timestamp()
    d["aqi_median_daily"] = pd.to_numeric(d.aqi_median_daily, errors="coerce")
    d = d.dropna(subset=["target_month","aqi_median_daily"]).sort_values(["county_fips","target_month"])
    keys = ["county_fips","target_month"]
    if d.duplicated(keys).any(): raise ValueError("Duplicate county/month rows.")
    g = d.groupby("county_fips", group_keys=False)
    d["aqi_lag_1"] = g.aqi_median_daily.shift(1)
    d["aqi_lag_12"] = g.aqi_median_daily.shift(12)
    d["issue_month"] = d.target_month - pd.offsets.MonthBegin(1)
    d["target_month_number"] = d.target_month.dt.month
    d["target_year"] = d.target_month.dt.year
    if a.weather:
        w = pd.read_csv(a.weather, dtype={"county_fips":"string"})
        if {"county_fips","target_month"}-set(w.columns): raise ValueError("Weather needs county_fips and target_month.")
        cols = [c for c in w.columns if c.startswith("weather_")]
        if not cols: raise ValueError("Weather needs numeric weather_* feature columns.")
        w["county_fips"] = w.county_fips.astype("string").str.zfill(5)
        w["target_month"] = pd.to_datetime(w.target_month).dt.to_period("M").dt.to_timestamp()
        if w.duplicated(keys).any(): raise ValueError("Duplicate weather county/month rows.")
        # Weather rows describe the observed month; shift forward to forecast the following month.
        w["target_month"] = w.target_month + pd.offsets.MonthBegin(1)
        d = d.merge(w[keys+cols], on=keys, how="left", validate="one_to_one")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(a.output, index=False)
    print(f"Saved {len(d)} candidate rows to {a.output}")

if __name__ == "__main__":
    main()
