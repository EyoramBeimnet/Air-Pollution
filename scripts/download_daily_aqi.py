from pathlib import Path
import argparse, io, zipfile, requests
import pandas as pd

COUNTIES = {"36005":"Bronx","36047":"Kings","36061":"New York","36081":"Queens","36085":"Richmond"}
URL = "https://aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2025)
    p.add_argument("--output-dir", type=Path, default=Path("datasets/raw/daily_aqi"))
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in range(a.start_year, a.end_year + 1):
        response = requests.get(URL.format(year=year), timeout=120)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            with z.open(csv_name) as f:
                d = pd.read_csv(f, low_memory=False)
        required = {"Date", "State Code", "County Code", "AQI"}
        if required - set(d.columns):
            raise ValueError(f"{year} EPA file missing {sorted(required-set(d.columns))}")
        state = pd.to_numeric(d["State Code"], errors="coerce").astype("Int64").astype("string").str.zfill(2)
        county = pd.to_numeric(d["County Code"], errors="coerce").astype("Int64").astype("string").str.zfill(3)
        d["county_fips"] = state + county
        d = d[d.county_fips.isin(COUNTIES)].copy()
        d["date"] = pd.to_datetime(d["Date"], errors="coerce")
        d["aqi"] = pd.to_numeric(d["AQI"], errors="coerce")
        d["borough"] = d.county_fips.map(COUNTIES)
        frames.append(d[["date","county_fips","borough","aqi"]].dropna())
        print(f"{year}: retained {len(frames[-1]):,} daily rows")
    result = pd.concat(frames, ignore_index=True).drop_duplicates(["county_fips","date"])
    result = result.sort_values(["county_fips","date"])
    path = a.output_dir / "nyc_daily_aqi_2014_2025.csv"
    result.to_csv(path, index=False)
    print(f"Saved {len(result):,} rows: {path}")

if __name__ == "__main__":
    main()
