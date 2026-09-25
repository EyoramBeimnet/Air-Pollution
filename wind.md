# Feature selection findings

*Wind & vehicle-exhaust datasets — air quality forecasting model*

## Objective

Ranking which candidate features from the wind dataset and the vehicle-exhaust / mobile-sources dataset are worth building into the pipeline, and flagging the ones that carry real risk.

## Recommended, in priority order

| Feature | Priority | Why |
|---|---|---|
| `wind_u` / `wind_v` | **Must include** | Fixes the 359°→1° wraparound in raw wind direction. Neither XGBoost nor an LSTM branch can learn circular data correctly from raw degrees — nearly free to compute and fixes a real representation bug. |
| Upwind congestion impact<br>`cos(wind_dir − highway_dir) × csdi_score` | **Top pick** | Best cross-dataset feature. Trees learn thresholds well but not a cosine relationship from raw angle components — handing them the pre-computed interaction gives signal they can't reconstruct alone. |
| `stagnation_hours` | Include | Sustained low wind is one of the best-documented drivers of pollution buildup. Captures an accumulation effect a single wind reading misses. |
| Stagnant heavy-traffic multiplier<br>`CSDI × ICE density ÷ (wind speed + 0.1)` | Include | Physically sensible, but lower marginal value — XGBoost can partly reconstruct multiplicative interactions from the raw inputs on its own. Watch the +0.1 epsilon: it can spike near-zero wind speed, so a larger floor or log-transform is safer. |

## Lower priority / use with caution

| Feature | Concern |
|---|---|
| `csdi_score` | Real driver of NOx / CO / PM — but if we're predicting *future* AQI, actual future congestion isn't knowable at prediction time. Needs a forecasted or typical day-of-week/hour proxy, not the live value. |
| `ice_density` | Mostly a slow-moving county baseline; largely redundant with historical mean AQI or population density if we're already using those. |
| `flight_volume_delta` | Only meaningful for counties with a major airport — narrow coverage. Deprioritize if the team is short on time. |

## Dataset Sources for Wind & Vehicle-Exhaust Features

> **Note:** Exact public download pages for the named “wind dataset” and “vehicle-exhaust / mobile-sources dataset” (with features such as `csdi_score`, `ice_density`, `flight_volume_delta`) could not be independently verified. These appear to be internally derived features. The sources below are the standard public building blocks for rebuilding them.

### Wind Dataset (for `wind_u` / `wind_v`, stagnation_hours, direction components)

| Source | Description | Link |
|--------|-------------|------|
| **ERA5 / ERA5-Land** | High-quality reanalysis with U/V wind components at multiple levels | [Copernicus CDS – ERA5 Pressure Levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels) |
| **NOAA NCEI Blended Seawinds (NBS v2)** | Gridded 10 m neutral winds with u/v components | [NOAA CoastWatch NBS v2](https://coastwatch.noaa.gov/cwn/products/noaa-ncei-blended-seawinds-nbs-v2.html) · [DOI](https://doi.org/10.25921/mxt4-b075) |
| **NCEP NARR** | North American Regional Reanalysis u/v winds | [PSL NARR](https://psl.noaa.gov/data/gridded/data.narr.html) |
| **NOAA GFS / model archives** | Forecast and analysis wind_u / wind_v fields | [dynamical.org NOAA GFS](https://dynamical.org/catalog/noaa-gfs-forecast/) or NOMADS / AWS Open Data |

### Vehicle-Exhaust / Mobile-Sources Dataset (for CSDI-style scores, ICE density, flight volume)

| Source | Description | Link |
|--------|-------------|------|
| **NPMRDS** | Probe-based travel times & speeds on the National Highway System (congestion proxy) | [FHWA NPMRDS Overview](https://ops.fhwa.dot.gov/publications/fhwahop20028/index.htm) |
| **Uber Movement** | Historical speeds / travel times (limited cities; service has evolved) | [Uber Movement](https://movement.uber.com/) |
| **EPA MSOD** | Mobile Source Observation Database (emissions test data) | [data.gov – MSOD](https://catalog.data.gov/dataset/mobile-source-observation-database-msod) |
| **State DMV / EIA** | Vehicle registration counts for ICE density | State DMV portals or [EIA / EPA mobile-source inventories](https://www.epa.gov/air-emissions-inventories) |
| **FAA / BTS TranStats** | Flight volume / air carrier traffic (T-100 data) | [BTS TranStats](https://www.transtats.bts.gov/) |
| **EPA NEI / research inventories** | Gridded on-road mobile-source emissions | [EPA National Emissions Inventory](https://www.epa.gov/air-emissions-inventories/national-emissions-inventory-nei) |

### Cross-cutting reminder
`csdi_score`, `flight_volume_delta`, and (to a lesser extent) `ice_density` are largely contemporaneous signals. Confirm they are available (or forecastable) at prediction time to avoid leakage.
