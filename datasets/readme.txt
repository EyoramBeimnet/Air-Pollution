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
		- Electric Vehicle Adoption and Charging Infrastructure
			- Primary source: Battery-electric vehicle registrations
			- Dataset: https://zenodo.org/records/12773413
			- CSV: https://zenodo.org/records/12773413/files/EV_data.csv?download=1
			- Coverage: 2018–2023 across 20 U.S. states.
			- Original fields: county, year, and n (battery-electric vehicle count).
			- Proposed feature: bev_count.

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
