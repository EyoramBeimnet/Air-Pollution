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

## File Structure

All processed datasets are stored directly in this folder.

- `combined_solver_features_by_borough_year.csv` - Combined annual features for model integration.
- `nyc_recorded_tree_plantings_by_borough_year.csv` - Annual recorded tree plantings.
- `nyc_completed_tree_removal_orders_by_borough_year.csv` - Annual tree removal work orders.
- `nyc_mta_station_entries_by_borough_year.csv` - Annual subway ridership estimates.
- `nyc_acs_commuting_by_borough_year.csv` - Census transportation estimates.
- `nyc_bike_routes_by_status_snapshot.csv` - Bicycle route records.
- `nyc_greenstreets_current_snapshot.csv` - Green infrastructure locations.
- `nyc_open_streets_current_snapshot.csv` - Open Streets locations.
- `nyc_clean_heat_historical_snapshot.csv` - Historical Clean Heat program records.
- `nyc_street_tree_census_2015_snapshot.csv` - Historical street tree census.
- `nyc_tree_removal_coverage_check.csv` - Tree removal data coverage report.
- `source_status_report.csv` - Dataset collection and processing status.

## Model Integration

The combined dataset contains 35 borough-year records covering 2019–2025.

Datasets are standardized using `county_fips` and `year` to allow integration with the team's other pollution datasets.

Missing values are preserved rather than automatically replaced with zero.

Note: Some datasets have incomplete historical coverage. Snapshot datasets are kept separate from annual observations. Additional cleaning and validation are required before model training.
