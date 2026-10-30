# Airshed NYC — frontend

3D web front end for the AQI predictor: yearly average AQI per NYC county (borough), observed 2020–2025 and forecast 2026–2030.

## Run it

The page loads its data with `fetch`, so serve the folder instead of opening the file directly:

```
cd frontend
python -m http.server 8000
```

Then open http://localhost:8000. Three.js loads from cdnjs, so it needs an internet connection.

## What's in it

- 3D city built from real data: NYC building footprints (~1.08M buildings, real outlines and roof heights), NYC street centerlines, USGS 3DEP terrain.
- Hand-modelled landmarks (Empire State, One WTC, Chrysler, Statue of Liberty) and suspension bridges.
- Select a borough (click or keys 1–5): the camera flies in and orbits; a side panel shows AQI, likely range, trend, rank and EPA guidance.
- Seven AQI layers from the top bar (key L cycles): particles, particle grid, wave mesh, air volume, pulse rings (selected county); heat map, columns (all counties).
- Year timeline with play, demo login (no real auth), dashboard of tracked counties.

## Plugging in the model's predictions

The AQI values are placeholders. Replace the `AIRSHED_DATA` object near the top of the main script in `index.html` with the model's output, one row per county per year:

```json
{ "fips": "36085", "year": 2027, "aqi": 40, "lo": 36, "hi": 44 }
```

| field | meaning |
|---|---|
| `fips` | county FIPS: 36005 Bronx, 36047 Brooklyn, 36061 Manhattan, 36081 Queens, 36085 Staten Island |
| `year` | calendar year |
| `aqi` | predicted (or observed) yearly average daily AQI |
| `lo`, `hi` | bounds of the forecast's likely range (omit for observed years) |

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
