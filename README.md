# NYC Air-Pollution AQI Model

Forecasting air quality (AQI) for New York City's five boroughs from public data on emissions, weather, and transportation. The repo has two modeling pipelines and a demo front end:

- **Annual pipeline:** predicts a borough's *next-year* average AQI (borough-year rows).
- **Monthly pipeline:** predicts a borough's *next-month* median AQI (borough-month rows). This gives roughly 700 rows instead of 55, so it is where most current modeling work happens.
- **Front end:** a 3D web visualization ("Airshed NYC") that reads AQI forecasts from an API.

Data covers **2014 to 2025**. Every source is keyed by county FIPS code.

| Borough | County | FIPS |
|---|---|---|
| Bronx | Bronx | 36005 |
| Brooklyn | Kings | 36047 |
| Manhattan | New York | 36061 |
| Queens | Queens | 36081 |
| Staten Island | Richmond | 36085 |

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

The front end falls back to built-in sample data if the API response is missing or invalid. The mock API only returns simulated data and does not serve a trained model. See [`frontend/README.md`](frontend/README.md) and [`integration/README.md`](integration/README.md).

## Caveats

- **Small data.** The annual table has 55 rows; the monthly table about 700 correlated rows.
- **Not deployment-ready.** The finalize stage is a structural audit only and does not certify the model as leakage-free.
- **ACS timing.** Five-year estimates may not be published in time for a real next-year forecast, so `drive_alone_pct` would need to be removed or lagged in production.
- **Missing values are preserved,** never replaced with zero or backfilled from current snapshots (this matters for EV registrations).
- **Do not commit** API keys or very large raw downloads.

## Open items

1. Add monthly weather to the monthly training table and the rolling backtest.
2. Decide whether to drop drive-alone commuting from the annual feature set.
3. Decide whether the CNN stays in the project, and run more seeds per fold if so.
4. Add neighboring counties (Hudson, Essex, Bergen, Nassau, Suffolk, Westchester) to increase sample size.
5. Lightly tune XGBoost and the CNN with the same budget before a final comparison.

## Team

Eyoram Beimnet, Nirad Bojedla, Musaab Jawed Mohammed, Jorge Plascencia, Musa Mudesir, Ashwin Prasanth.