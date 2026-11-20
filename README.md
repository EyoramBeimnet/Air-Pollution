# NYC Air-Pollution AQI Model

Forecasting air quality (AQI) for New York City's five boroughs from public data on emissions, weather, and transportation, and delivering it through a web application. The repo has three parts:

- **Modeling pipelines:** an *annual* pipeline (next-year borough AQI) and a *monthly* pipeline (next-month borough median AQI, about 700 rows instead of 55).
- **API service:** a read-only FastAPI app (`api/app.py`) that serves AQI records at `/api/aqi`.
- **Front end:** a 3D web visualization ("Airshed NYC") that loads its data from that API.

**Project goal:** an application where users see AQI forecasts through the front end, not by reading CSV files. The API and front end are built and tested. **Forecasts are not connected yet:** the current release serves observed data only (see [Project status](#project-status)).

Data covers **2014 to 2025**. Every source is keyed by county FIPS code.

| Borough | County | FIPS |
|---|---|---|
| Bronx | Bronx | 36005 |
| Brooklyn | Kings | 36047 |
| Manhattan | New York | 36061 |
| Queens | Queens | 36081 |
| Staten Island | Richmond | 36085 |

## Project status

| Stage | State |
|---|---|
| Data extension to 2014 (weather, ACS, AQI) | Done, merged (PR #4) |
| Annual and monthly modeling pipelines | Done; XGBoost is the recommended model |
| API service and front-end integration | Done for **observed** data, 2020 to 2025 (PRs #5 and #6) |
| Forecast delivery (model to API to front end) | **Not built yet** (see [Forecast integration](#forecast-integration-next)) |

What the running app does today:

- The API reads `datasets/processed/master_borough_year_2014_2025.csv` and returns annual **median daily AQI** per borough, with `forecast: false` on every record.
- The front end requests `GET /api/aqi?start_year=2020&end_year=2025`, validates the response, and shows 2020 to 2025 as observed data.
- If the API response is missing or invalid, the front end reports the error. It does **not** fall back to sample AQI values.
- No future-year predictions or uncertainty intervals are served in this release.

## Current model results

Both pipelines are compared against a "no change" persistence baseline, with chronological splits (never random) and rolling-origin backtests.

**Annual** (55 supervised rows, rolling backtest over 2019 to 2024, improvement in MAE over the no-change baseline):

| Features | Random Forest | XGBoost |
|---|---|---|
| AQI + CAMPD emissions | +6.7% | +11.8% |
| AQI + CAMPD + weather | +7.9% | **+16.3%** |
| AQI + CAMPD + drive-alone commuting | +2.7% | +10.4% |

Weather is the only feature group that helped consistently. Commuting and traffic features did not.

**Monthly** (test set: 110 borough-months, 2024-03 to 2025-12):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Persistence | 5.85 | 7.71 | -0.05 |
| Random Forest | 4.27 | 5.33 | 0.50 |
| XGBoost | 4.14 | 5.09 | 0.54 |
| 1D CNN (5-seed ensemble) | 3.87 | 4.70 | 0.61 |

A follow-up overfitting review and six-fold rolling backtest found the CNN's edge over XGBoost is within noise (mean gap about zero), so XGBoost is the recommended monthly model. Details are in [`progress.md`](progress.md) (section 7) and [`notebooks/cnn_overfitting_and_backtest_analysis.md`](notebooks/cnn_overfitting_and_backtest_analysis.md).

> With this few rows, and all five boroughs moving together in a given year, differences of about 0.1 AQI points of MAE are noise. Treat these as research results, not a deployable forecast.

## Getting started

Requires Python 3.10+. Node.js is needed only to run the front-end contract tests.

```bash
git clone https://github.com/EyoramBeimnet/Air-Pollution.git
cd Air-Pollution
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All scripts are run **from the repository root**.

## Run the application

The API and front end are served together by one command:

```bash
python -m uvicorn api.app:app --reload
```

Then open http://127.0.0.1:8000. Useful endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/aqi?start_year=2020&end_year=2025
```

Three.js loads from cdnjs, so the page needs an internet connection.

### API contract

`GET /api/aqi` returns `{ "data": [...], "metric": "annual_median_daily_aqi", "forecast_status": "..." }`. The front end also accepts a bare array. One record per borough per year:

```json
{ "fips": "36085", "year": 2025, "aqi": 41, "forecast": false, "metric": "annual_median_daily_aqi" }
```

| field | meaning |
|---|---|
| `fips` | county FIPS code (table above) |
| `year` | calendar year |
| `aqi` | annual median of the daily AQI |
| `forecast` | `false` for observed data; `true` for predicted years |
| `lo`, `hi` | forecast range; optional. If present, the front end requires `lo <= aqi <= hi` |

The front end rejects a payload that is missing any borough/year it needs, has a negative AQI, or has an invalid forecast range. To point the page at a different server during development, pass its encoded endpoint in the `api` query parameter:

```text
http://localhost:8000/?api=http%3A%2F%2Flocalhost%3A8001%2Fapi%2Faqi%3Fstart_year%3D2020%26end_year%3D2025
```

The API enables CORS so a separate dev origin can call it.

### Tests

```bash
node --test frontend/tests/aqi-api.test.js     # front-end API contract tests (9)
python -m unittest tests/test_aqi_service.py   # API data loading tests (2)
```

## Forecast integration (next)

To meet the project goal, a forecast has to flow model to API to front end. Planned path:

1. **Forecast script** (`scripts/build_forecast.py`): fit the final model on all 2014 to 2024 rows, produce predictions, and write a forecast file with `fips, year, aqi, lo, hi`.
2. **API**: `api/app.py` appends those rows with `forecast: true` and keeps observed years from the master table.
3. **Front end**: extend `YEARS` (currently 2020 to 2025) and `LAST_OBS` in `frontend/index.html`, and restore the forecast UI text.
4. **Tests**: validate the real forecast output against the front-end contract.

Gaps to resolve first:

- **Horizon.** The annual model predicts one year ahead, so multi-year forecasts need either rolling the model forward (holding future weather and emissions at assumed values) or training one model per horizon.
- **Uncertainty.** The models give point predictions only. `lo`/`hi` could come from rolling-backtest residuals, which are crude with 30 backtest rows.
- **No saved model.** `train_aqi_models.py` writes metrics and predictions, but no fitted model is persisted.
- **Metric.** The front end shows the annual **median** of daily AQI, which matches the annual model's target. Keep forecasts on the same metric.

## Running the pipelines

### 1. Collect raw data

Each collector writes under `datasets/raw/`. Two need free API keys (see [API keys](#api-keys)).

```bash
python scripts/collect_nyc_weather.py          # Open-Meteo weather (or use --from-daily, see below)
python scripts/collect_nyc_campd.py            # power-plant emissions by county (EPA CAMPD)
python scripts/collect_historical_acs.py       # ACS commuting estimates
python scripts/collect_nyc_traffic.py          # NYC DOT sampled traffic counts
python scripts/collect_nyc_mta.py              # subway station entries
python scripts/collect_nyc_trees.py            # recorded tree plantings
python scripts/collect_nyc_ev.py               # EV registration coverage audit
```

Weather can also be downloaded by hand from Open-Meteo as five per-borough CSVs, combined with `datasets/raw/weather/nyc_daily_weather_compiler.py`, and turned into the borough-year table with `python scripts/collect_nyc_weather.py --from-daily`.

### API keys

Set these as environment variables and never commit them (`.env` and `*.key` are git-ignored).

| Variable | Used by | Get one |
|---|---|---|
| `CAMPD_API_KEY` | `collect_nyc_campd.py` | [EPA Clean Air Markets API portal](https://www.epa.gov/node/277325) |
| `CENSUS_API_KEY` | `collect_historical_acs.py` | [Census API key signup](https://api.census.gov/data/key_signup.html) |

### 2. Annual pipeline (next-year AQI)

Each stage reads the previous stage's output, so keep this order:

```bash
python scripts/prepare_model_data.py            # builds the staging table, downloads EPA annual AQI ZIPs
python scripts/review_ev_and_stage_data.py      # EV coverage review + candidate datasets
python scripts/finalize_air_pollution_handoff.py  # structural audit, final splits
python scripts/train_aqi_models.py              # Random Forest + XGBoost, ablations, rolling backtest
python scripts/evaluate_aqi_predictions.py      # metrics by borough and year
```

Expected sizes: 55 supervised rows (5 boroughs x feature years 2014 to 2024), split 35 train / 10 validation / 10 test, and 30 rows per rolling-backtest experiment. Use `--skip-aqi-download` on `prepare_model_data.py` if the EPA ZIPs are already in `datasets/raw/epa_aqi/`.

### 3. Monthly pipeline (next-month AQI)

```bash
python scripts/download_daily_aqi.py --start-year 2014 --end-year 2025
python scripts/prepare_monthly_aqi.py            # daily -> monthly AQI targets (75% coverage filter)
python scripts/prepare_monthly_model_data.py     # lag features, month number
python scripts/train_monthly_aqi_models.py       # Persistence, Random Forest, XGBoost
python scripts/evaluate_monthly_aqi_predictions.py
```

Optional deep-learning and robustness experiments:

```bash
python scripts/train_monthly_cnn.py              # 1D CNN (needs tensorflow-cpu)
python scripts/diagnose_cnn_overfit.py           # model size / window comparison
python scripts/rolling_origin_backtest.py        # six-year rolling-origin backtest
```

Monthly weather features are not yet wired into the monthly staging table; see [Open items](#open-items).

## Repository layout

```
.
├── api/                  read-only FastAPI service (app.py); also serves the front end
├── tests/                Python tests for the API data loader
├── scripts/              collectors, data preparation, training, evaluation
├── datasets/
│   ├── raw/              source downloads (weather, acs, traffic, mta, trees, ev, epa_aqi, daily_aqi)
│   ├── processed/        staging tables, master table, review/handoff files, model_results/, monthly/
│   ├── solvers/          positive-factor datasets (trees, transit, bike routes, Clean Heat)
│   ├── LUCID/            land-use data for the agriculture sector
│   └── readme.txt        data source notes and rationale
├── notebooks/            results notebook, CNN findings, overfitting and backtest analysis
├── frontend/             3D AQI visualization and contract tests (see frontend/README.md)
├── integration/          service integration notes (see integration/README.md);
│                         mock_api.py is a legacy simulated API (2020-2030) kept for reference
├── progress.md           running handoff notes and detailed results
└── requirements.txt
```

## Data sources

| Source | Provides | Notes |
|---|---|---|
| EPA AQI (annual and daily by county) | Prediction target | Annual ZIPs and daily AQI, 2014 to 2025 |
| EPA CAMPD | CO2, SO2, NOx from power plants, by county and year | Replaced the EIA regional file, which had no county field |
| Open-Meteo (ERA5 reanalysis) | Daily weather per borough | One sample point per borough, not an area average |
| Census ACS B08301 | Commuting mode shares | Five-year estimates |
| NYC DOT traffic counts | Sampled traffic volume | Nonrepresentative sample |
| MTA subway ridership | Station entries by borough | Data begins 2017 |
| NYC Parks Forestry | Tree plantings | Reliable only from 2016; dropped from the models |
| Zenodo EV registrations | BEV counts by county | Missing for 2024 to 2025 |

More background and the full list of candidate sources are in [`datasets/readme.txt`](datasets/readme.txt) and [`datasets/solvers/readme.md`](datasets/solvers/readme.md).

## Caveats

- **Small data.** The annual table has 55 rows; the monthly table about 700 correlated rows.
- **Observed only for now.** The running application does not serve forecasts, and the models are not deployment-ready.
- **Not deployment-ready.** The finalize stage is a structural audit only and does not certify the model as leakage-free.
- **ACS timing.** Five-year estimates may not be published in time for a real next-year forecast, so `drive_alone_pct` would need to be removed or lagged in production.
- **Missing values are preserved,** never replaced with zero or backfilled from current snapshots (this matters for EV registrations).
- **Do not commit** API keys or very large raw downloads.

## Open items

1. **Forecast integration** (see above): choose the forecast method, build `scripts/build_forecast.py`, extend the API and front end, and add tests.
2. Decide whether to keep or delete `integration/mock_api.py`.
3. Clean up leftover forecast wording in `frontend/index.html` (the particle layer description and the dashboard card) until forecasts return.
4. Add monthly weather to the monthly training table and the rolling backtest.
5. Decide whether to drop drive-alone commuting from the annual feature set.
6. Decide whether the CNN stays in the project, and run more seeds per fold if so.
7. Add neighboring counties (Hudson, Essex, Bergen, Nassau, Suffolk, Westchester) to increase sample size.
8. Lightly tune XGBoost and the CNN with the same budget before a final comparison.
9. Merge or retire the `data` branch (EV adoption and traffic congestion datasets).
10. Add a service-integration section to `progress.md`.

## Team

Eyoram Beimnet, Nirad Bojedla, Musaab Jawed Mohammed, Jorge Plascencia, Musa Mudesir, Ashwin Prasanth.