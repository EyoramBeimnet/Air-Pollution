This folder contains all the datasets that are considered by the model.


Air Pollution Databases
----------------------------------
Labels: Environmental Protection Agency (annual_aqi_by_county)

Industrial Energy Sector:
	Emissions by Power Plant and Region: https://www.eia.gov/electricity/data/emissions/  --> NOT IDEAL (SEE NOTE)
		- Targets CO2, SO2, and NOx, which are used to measure Air Quality Index (AQI)
	NOTE:
	The original EIA emissions-by-region file (State/NERC/BA level, 2024 only)
	isn't granular enough for this project: it has no county field, so it can't
	join to our county-level AQI target without broadcasting a whole state's
	average onto every county in it, and it's a single-year snapshot, so it
	can't feed a time-series predictor.
	
	Switched to EPA's CAMPD (Clean Air Markets Program Data, https://campd.epa.gov/data/custom-data-download)
	instead. Same pollutants (CO2, SO2, NOx), but reported per facility with a
	native county field (no join needed) and continuous hourly/annual data from
	1995-present. This gives us real county-year aggregates and year-over-year
	trend features instead of a one-year state-wide average.

Vehicle Exhaust:
	- Airport traffic
		- https://www.kaggle.com/datasets/nilesh2042/airport-traffic-dataset
	- Car traffic
		- US Traffic Congestions (2016-2022) https://www.kaggle.com/datasets/sobhanmoosavi/us-traffic-congestions-2016-2022
		- Dataset: https://www.kaggle.com/datasets/sobhanmoosavi/us-traffic-congestions-2016-2022
		- Coverage: U.S. congestion event records from 2016–2022.
		- Fields used for grouping and joining: StartTime, County, and State. County and state names will be mapped to five-digit county FIPS codes before joining to the EV and AQI datasets.
		- Proposed county-year features:
			- congestion_event_count: number of recorded congestion events.
			- mean_congestion_duration_min: average time between StartTime and EndTime, for records with valid timestamps.
			- mean_delay_from_typical_min: average DelayFromTypicalTraffic(mins) for records with a valid value.
		- Additional fields available for investigation: Severity, Distance(mi), DelayFromFreeFlowSpeed(mins), and 						Congestion_Speed. These will only become model features if their meanings and missing-value rates support their use.
		- Limitations: Event counts measure recorded congestion events, not the number of cars on the road. The 2-million-row 			sample can be used to develop the processing code, but nationwide event counts must be calculated from the full dataset.
	- Electric Vehicle Adoption and Charging Infrastructure
		- Primary source: Battery-electric vehicle registrations
		- Dataset: https://zenodo.org/records/12773413
		- CSV: https://zenodo.org/records/12773413/files/EV_data.csv?download=1
		- Proposed feature: bev_count.
		- Limitation: This dataset does not include the total number of registered vehicles, so it cannot be used by itself to calculate the percentage of cars that are electric.

- Agriculture Sector
	- Land Use and Cover Inventory Database (LUCID) https://www.nrisurvey.org/lucid/

- Climate patterns
	- https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/county/mapping 
		- Would be helpful to specify by country or county
		- This website contains datasets for every county going back a given time frame for the following:
			- Average Temperatures
			- Precipitation (positive factor)
			- Palmer Hydrological Drought Index
			- Palmer Drought Severity Index

- Air Pollution Solvers (Positive Factors)
	- Trees / Vegetation
		- USDA Forest Service NLCD Tree Canopy Cover
		- https://www.mrlc.gov/data
        - Features:
            - tree_canopy_pct
            - tree_canopy_change_pct

    - Low-Emission Transportation
        - U.S. Census ACS B08301
        - Features:
            - public_transit_pct
            - walk_pct
            - bike_pct
            - carpool_pct
            - drive_alone_pct

    - Walkability / Transit Accessibility
        - EPA Smart Location Database
        - EPA National Walkability Index
        - Features:
            - walkability_index
            - transit_proximity
            - intersection_density
            - land_use_mix

    - Environmental Regulations
        - EPA Green Book
        - EPA State Implementation Plans
        - DSIRE
        - Features:
            - prior_nonattainment_status
            - years_since_regulatory_action
            - renewable_policy_present
            - clean_energy_policy_count

- Hourly Data: Ozone, SO2, CO, NO2 - (1980-2026)
	https://aqs.epa.gov/aqsweb/airdata/download_files.html#Raw
