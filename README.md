# NYC Air-Pollution AQI Model

Forecasting air quality (AQI) for New York City's five boroughs from public data on emissions, weather, and transportation, and delivering it through a web application. The repo has three parts:

- **Modeling pipelines:** an *annual* pipeline (next-year borough AQI) and a *monthly* pipeline (next-month borough median AQI, about 700 rows instead of 55).
- **API service:** a read-only FastAPI app (`api/app.py`) that serves AQI records.
- **Front end:** a 3D web visualization ("Airshed NYC") that loads its data from that API.

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
| API service and front-end integration | Done for observations and forecasts |
| Forecast delivery (model to API to front end) | Done: annual and monthly forecasts through 2030 |

The API serves annual median daily AQI observations through 2025 and documented forecast/scenario rows through 2030. `/api/aqi/monthly` serves monthly observations through 2025 and recursive forecasts through 2030. Forecast records are model-generated predictions, not observations; no confidence intervals are supplied.

## Current results

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

Requires Python 3.10+.

```bash
git clone https://github.com/EyoramBeimnet/Air-Pollution.git
cd Air-Pollution
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

All scripts are run **from the repository root**.

### API keys

Two collectors need free keys. Set them as environment variables and never commit them (`.env` and `*.key` are git-ignored).

| Variable | Used by | Get one |
|---|---|---|
| `CAMPD_API_KEY` | `collect_nyc_campd.py` | [EPA Clean Air Markets API portal](https://www.epa.gov/node/277325) |
| `CENSUS_API_KEY` | `collect_historical_acs.py` | [Census API key signup](https://api.census.gov/data/key_signup.html) |

## Run the application

Serve the API and frontend together:

```bash
python3 -m uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000`. Useful endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/aqi?start_year=2020&end_year=2030
http://127.0.0.1:8000/api/aqi/monthly?start_month=2020-01&end_month=2030-12
```

`/api/aqi` returns annual median daily AQI. `/api/aqi/monthly` returns monthly
median daily AQI. Both endpoints preserve observed history when an optional
forecast artifact is missing or invalid.

## Forecast integration

Build the documented forecast artifacts:

```bash
python3 scripts/build_annual_aqi_forecast.py
python3 scripts/build_annual_aqi_scenario.py
python3 scripts/build_monthly_aqi_forecast.py
```

The scripts leave existing experiments unchanged. The annual configuration
reuses weather-enhanced XGBoost and preserves the original 2026 forecast based
on observed 2025 AQI, CAMPD NOx, temperature, and precipitation. Annual
2027–2030 values recursively feed back predicted AQI while holding those
external features at observed 2025 values. Monthly XGBoost recursively forecasts
January 2026 through December 2030.

- `datasets/processed/model_results/annual_aqi_forecast_2026.csv`
- `datasets/processed/model_results/annual_aqi_forecast_2026_metadata.json`

Annual forecasts use the frontend's annual median daily AQI metric; monthly
forecasts use monthly median daily AQI. They are point predictions only. If an
artifact is absent or invalid, the API and frontend continue to serve observed
history.

Run the service tests with:

```bash
node --test frontend/tests/aqi-api.test.js
python3 -m unittest discover -s tests -v
```

## Running the pipelines

### 1. Collect raw data

Each collector writes under `datasets/raw/`.

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
├── scripts/              collectors, data preparation, training, evaluation
├── datasets/
│   ├── raw/              source downloads (weather, acs, traffic, mta, trees, ev, epa_aqi, daily_aqi)
│   ├── processed/        staging tables, review/handoff files, model_results/, monthly/
│   ├── solvers/          positive-factor datasets (trees, transit, bike routes, Clean Heat)
│   ├── LUCID/            land-use data for the agriculture sector
│   └── readme.txt        data source notes and rationale
├── notebooks/            results notebook, CNN findings, overfitting and backtest analysis
├── frontend/             3D AQI visualization (see frontend/README.md)
├── integration/          mock AQI API for front-end testing (see integration/README.md)
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

## Front end and mock API

```bash
python integration/mock_api.py                  # simulated AQI API on :8001
cd frontend && python -m http.server 8000       # then open http://localhost:8000
node --test frontend/tests/aqi-api.test.js      # API contract tests
```

The mock API only returns simulated data and does not serve a trained model. The
production frontend does not show sample AQI values if its API response is
missing or invalid. See [`frontend/README.md`](frontend/README.md) and
[`integration/README.md`](integration/README.md).

## Forecast artifacts and service

Build the committed forecast artifacts from the existing model configurations:

```bash
python scripts/build_annual_aqi_forecast.py  # existing 2026 annual forecast
python scripts/build_annual_aqi_scenario.py  # annual 2026–2030 artifact
python scripts/build_monthly_aqi_forecast.py # monthly Jan. 2026–Dec. 2030 artifact
python -m uvicorn api.app:app --reload
```

`/api/aqi` returns annual median daily AQI. Its 2026 values preserve the existing
weather-enhanced XGBoost forecast based on observed 2025 inputs. Annual 2027–2030
are recursive scenario projections: predicted AQI is fed back while weather,
precipitation, and CAMPD NOx stay fixed at observed 2025 values. `/api/aqi/monthly`
returns monthly median daily AQI, observed through 2025 and recursive forecasts to
2030. Neither endpoint supplies confidence intervals or fabricated observations.

The generation scripts write horizon-specific backtests beside their outputs.
They compare recursive XGBoost with persistence and are not estimates of future
accuracy; especially at multi-year horizons, the fixed-feature annual values are
conditional scenarios rather than equally reliable one-step forecasts.

## Caveats

- **Small data.** The annual table has 55 rows; the monthly table about 700 correlated rows.
- **Research forecasts.** The 2026 annual result is a point forecast; annual 2027–2030 values are fixed-feature scenario projections, not operational forecasts.
- **Not deployment-ready.** The finalize stage is a structural audit only and does not certify the model as leakage-free.
- **ACS timing.** Five-year estimates may not be published in time for a real next-year forecast, so `drive_alone_pct` would need to be removed or lagged in production.
- **Missing values are preserved,** never replaced with zero or backfilled from current snapshots (this matters for EV registrations).
- **Do not commit** API keys or very large raw downloads.

## Open items

1. Reassess the forecast after completed 2026 observations become available.
2. Add monthly weather to the monthly training table and the rolling backtest.
3. Decide whether to drop drive-alone commuting from the annual feature set.
4. Decide whether the CNN stays in the project, and run more seeds per fold if so.
5. Add neighboring counties (Hudson, Essex, Bergen, Nassau, Suffolk, Westchester) to increase sample size.
6. Lightly tune XGBoost and the CNN with the same budget before a final comparison.
7. Decide whether to keep or retire `integration/mock_api.py`.
8. Merge or retire the `data` branch (EV adoption and traffic congestion datasets).
9. Add a service-integration section to `progress.md`.

## Team

Eyoram Beimnet, Nirad Bojedla, Musaab Jawed Mohammed, Jorge Plascencia, Musa Mudesir, Ashwin Prasanth.
