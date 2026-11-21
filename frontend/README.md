# Airshed NYC — frontend

3D web front end for the AQI service: observed annual median daily AQI per NYC
county (borough), through 2025, plus a validated 2026 model point forecast when
the API provides it.

## Run it

The API and frontend are served together:

```
python -m uvicorn api.app:app --reload
```

Then open http://localhost:8000. Three.js loads from cdnjs, so it needs an internet connection.

## What's in it

- 3D city built from real data: NYC building footprints (~1.08M buildings, real outlines and roof heights), NYC street centerlines, USGS 3DEP terrain.
- Hand-modelled landmarks (Empire State, One WTC, Chrysler, Statue of Liberty) and suspension bridges.
- Select a borough (click or keys 1–5): the camera flies in and orbits; a side panel shows AQI, observed/forecast status, trend, rank and EPA guidance.
- Seven AQI layers from the top bar (key L cycles): particles, particle grid, wave mesh, air volume, pulse rings (selected county); heat map, columns (all counties).
- Year timeline with play, demo login (no real auth), dashboard of tracked counties.

## AQI API integration

The page requests this endpoint on startup:

```
GET /api/aqi?start_year=2020&end_year=2030
```

For a different server during development, provide its encoded endpoint through
the `api` query parameter, for example:

```
http://localhost:8000/?api=http%3A%2F%2Flocalhost%3A8001%2Fapi%2Faqi%3Fstart_year%3D2020%26end_year%3D2025
```

The endpoint may return either an array or `{ "data": [...] }`. It returns one
record per county per year for all five county FIPS values. The frontend always
requires 2020–2025 observations, and adds 2026–2030 only when all five validated
forecast records are present:

```json
{ "fips": "36085", "year": 2025, "aqi": 41, "forecast": false }
```

| field | meaning |
|---|---|
| `fips` | county FIPS: 36005 Bronx, 36047 Brooklyn, 36061 Manhattan, 36081 Queens, 36085 Staten Island |
| `year` | calendar year |
| `aqi` | observed annual median daily AQI |
| `forecast` | `false` for an observation; `true` for a model-generated forecast |

The yearly view uses the same annual median daily AQI metric as the observed
history. 2026 is an XGBoost point forecast using observed 2025 inputs; 2027–2030
are recursive scenario projections with CAMPD NOx, temperature, and precipitation
held at their observed 2025 values. It has no confidence interval. If the annual
forecast is absent or invalid, the frontend keeps working with historical data.

The Monthly control requests `GET /api/aqi/monthly?start_month=2020-01&end_month=2030-12`.
It exposes monthly median daily AQI: observations through December 2025 and
recursive XGBoost forecasts for January 2026 through December 2030. If that
artifact is absent or invalid, changing to Monthly leaves the annual view in place.

## Data files (`data/`)

| file | contents |
|---|---|
| `hero.txt` | full building outlines (Manhattan, tall or large buildings), base64 of `hero.bin` |
| `lo_2–5.txt` | low-rise buildings as oriented boxes per borough (2 Bronx, 3 Brooklyn, 4 Queens, 5 Staten Island) |
| `height.txt` | 1024×1024 terrain heightmap, base64 int16 decimetres |
| `ground.webp` | baked ground texture: roads, parks, water |

Files are base64 text because the hosted preview only serves text and image types. Locally, plain binaries would also work after changing the `.txt` paths in `index.html`.

## Rebuilding the data (`scripts/`)

Raw sources are too large for the repo (git-ignored). Download them, then run the scripts:

- Buildings: https://data.cityofnewyork.us/api/v3/views/5zhs-2jue/export.csv?accessType=DOWNLOAD → `process_buildings.py`
- Streets: https://data.cityofnewyork.us/api/v3/views/inkn-q76z/export.csv?accessType=DOWNLOAD → `process_roads.py`
- Terrain: USGS 3DEP 1/3 arc-second tiles `n41w074` and `n41w075` → `process_terrain.py`

Requires `numpy`, `shapely>=2`, `rasterio`. Then `base64 -w0 X.bin > X.txt` for each output.
