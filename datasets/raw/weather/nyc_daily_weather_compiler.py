import pandas as pd 

# county key (as it appears in the filename) -> (borough name, FIPS code)
COUNTIES = {
    "bronx":     ("Bronx", 36005),
    "brooklyn":  ("Brooklyn", 36047),
    "manhatten": ("Manhattan", 36061),
    "queens":    ("Queens", 36081),
    "staten":    ("Staten Island", 36085),
}


frames = []

RENAME = {
    "time": "date",
    "temperature_2m_mean": "temp_mean_c",
    "temperature_2m_max": "temp_max_c",
    "precipitation_sum": "precipitation_mm",
    "wind_speed_10m_mean": "wind_speed_mean_kmh",
    "relative_humidity_2m_mean": "relative_humidity_mean_pct",
}

for key, (borough, fips) in COUNTIES.items():
    # Hardcoded file names, any extra data must follow this format
    path = f"weather_daily_{key}_2014-2025.csv"

    # rows 1-2: metadata (header row + one data row)
    meta = pd.read_csv(path, nrows=1)

    df = pd.read_csv(path, skiprows=3)

    # drop the " (unit)" suffix from each header, then rename
    df.columns = [c.split(" (")[0].strip() for c in df.columns]
    df = df.rename(columns=RENAME)

    df["date"] = pd.to_datetime(df["date"])
    df["county_fips"] = fips
    df["borough"] = borough
    df["sample_latitude"] = meta.loc[0, "latitude"]
    df["sample_longitude"] = meta.loc[0, "longitude"]
    df["year"] = df["date"].dt.year
    frames.append(df)

out = pd.concat(frames, ignore_index=True)

out = out[[
    "county_fips", "borough", "date",
    "temp_mean_c", "temp_max_c", "precipitation_mm",
    "wind_speed_mean_kmh", "relative_humidity_mean_pct",
    "sample_latitude", "sample_longitude", "year",
]]

out.to_csv("nyc_weather_combined.csv", index=False)
print(out.head())
print(out.shape)

