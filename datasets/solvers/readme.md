# Air Pollution Solvers

This folder contains datasets related to factors that may help reduce air pollution across New York City's five boroughs: Manhattan, Brooklyn, Queens, Bronx, and Staten Island.

The datasets were collected from official NYC, MTA, and U.S. Census sources and processed using Python.

## Datasets

### Trees / Vegetation
- NYC Forestry Tree Points
  - Recorded tree plantings by borough and year (2019–2025).
- NYC Forestry Work Orders
  - Tree removal work orders by borough and year.
- NYC 2015 Street Tree Census
  - Historical tree counts and health information.
- NYC Greenstreets
  - Green infrastructure locations by borough.

### Low-Emission Transportation
- NYC Bicycle Routes
  - Bicycle infrastructure records by borough.
- NYC Open Streets
  - Streets designated for pedestrian and cycling activities.
- MTA Subway Ridership
  - Estimated subway station entries by borough and year (2020–2025).
- U.S. Census ACS Transportation
  - Commuting methods, including public transit, walking, biking, and carpooling.
  - Currently contains 2024 estimates.

### Environmental Regulations
- NYC Clean Heat
  - Historical building records associated with NYC's Clean Heat program.

## Data Processing

Python was used to:
- Filter records to NYC's five boroughs.
- Remove duplicate records where appropriate.
- Standardize borough names and county FIPS codes.
- Organize available historical data by borough and year.
- Combine compatible datasets into a single CSV.
- Identify missing values and incomplete records.

## File Structure

- `annual/` - Datasets organized by borough and year.
- `snapshots/` - Historical or current datasets without complete annual coverage.
- `diagnostics/` - Data coverage and validation information.
- `combined_solver_features_by_borough_year.csv` - Combined annual features.
- `source_status_report.csv` - Dataset sources and processing status.
- `RUN_REPORT.txt` - Data collection results.

## Model Integration

The combined dataset contains 35 borough-year records covering 2019–2025.

Datasets are standardized using `county_fips` and `year` to allow integration with the team's other pollution datasets.

Missing values are preserved rather than automatically replaced with zero.

Note: Some datasets have incomplete historical coverage. Snapshot datasets are kept separate from annual observations. Additional cleaning and validation are required before model training.
