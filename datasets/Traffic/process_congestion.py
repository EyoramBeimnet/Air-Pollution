import pandas as pd
import zipfile

ZIP_PATH = "/Users/musamudesir/Downloads/archive (2).zip"
CSV_PATH = "us_congestion_2016_2022_sample_2m/us_congestion_2016_2022_sample_2m.csv"
OUTPUT_PATH = "Traffic/traffic_county_year.csv"

columns = [
    "Severity",
    "StartTime",
    "DelayFromTypicalTraffic(mins)",
    "DelayFromFreeFlowSpeed(mins)",
    "Congestion_Speed",
    "County",
    "State"
]

print("Reading traffic dataset...")

with zipfile.ZipFile(ZIP_PATH) as z:
    with z.open(CSV_PATH) as f:
        df = pd.read_csv(f, usecols=columns)

print(f"Loaded {len(df):,} traffic records")

# Extract year from StartTime
df["year"] = pd.to_datetime(
    df["StartTime"],
    errors="coerce",
    utc=True
).dt.year

# Make sure numeric columns are actually numeric
numeric_columns = [
    "Severity",
    "DelayFromTypicalTraffic(mins)",
    "DelayFromFreeFlowSpeed(mins)"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove rows missing location or year
df = df.dropna(subset=["County", "State", "year"])

# Create indicators for congestion speed categories
df["slow_events"] = (df["Congestion_Speed"] == "Slow").astype(int)
df["moderate_events"] = (df["Congestion_Speed"] == "Moderate").astype(int)
df["fast_events"] = (df["Congestion_Speed"] == "Fast").astype(int)

# Aggregate traffic data by county/state/year
traffic = (
    df.groupby(["State", "County", "year"])
    .agg(
        congestion_events=("Severity", "size"),
        avg_severity=("Severity", "mean"),
        avg_delay_typical=("DelayFromTypicalTraffic(mins)", "mean"),
        avg_delay_freeflow=("DelayFromFreeFlowSpeed(mins)", "mean"),
        slow_events=("slow_events", "sum"),
        moderate_events=("moderate_events", "sum"),
        fast_events=("fast_events", "sum")
    )
    .reset_index()
)

traffic["year"] = traffic["year"].astype(int)

# Calculate percentage of events classified as slow
traffic["slow_event_percentage"] = (
    traffic["slow_events"] / traffic["congestion_events"] * 100
)

# Round averages for cleaner output
traffic["avg_severity"] = traffic["avg_severity"].round(2)
traffic["avg_delay_typical"] = traffic["avg_delay_typical"].round(2)
traffic["avg_delay_freeflow"] = traffic["avg_delay_freeflow"].round(2)
traffic["slow_event_percentage"] = traffic["slow_event_percentage"].round(2)

# Save processed dataset
traffic.to_csv(OUTPUT_PATH, index=False)

print(f"Created {OUTPUT_PATH}")
print(f"Output contains {len(traffic):,} county-year rows")
print()
print(traffic.head())
