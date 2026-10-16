# NYC Air Pollution — Data Preparation Handoff

**Status:** Structural data-quality audit passed on October 16, 2026. **This is a research dataset, not a validated operational forecasting system.**

## Goal and data grain

The current prepared dataset is **annual**, with one row per **NYC borough (county FIPS) × calendar year**, for 2019–2025. The research task is to use data associated with feature year `t` to predict **median AQI in year `t+1`**. It does **not** support daily/weekly AQI forecasting without separate daily labels and appropriately aligned daily predictors.

Five NYC counties: Bronx `36005`, Brooklyn/Kings `36047`, Manhattan/New York `36061`, Queens `36081`, and Staten Island/Richmond `36085`.

## Key outputs

| Repository path | Meaning |
| --- | --- |
| `datasets/processed/master_borough_year_2019_2025.csv` | Full 35-row annual feature/label integration table; some feature columns remain missing where source observations do not exist. |
| `datasets/processed/feature_missingness_report.csv` | Feature-by-feature observed/missing counts. |
| `datasets/processed/processing_report.json` | Source inclusion/exclusion and quality notes. |
| `datasets/processed/aqi_year_coverage_report.csv` | AQI coverage by borough and year. |
| `datasets/processed/review/master_with_ev_CANDIDATE.csv` | EV-inclusive research view; not a complete-case training table. |
| `datasets/processed/review/ev_and_baseline_review_report.json` | EV data coverage and research-baseline summary. |
| `datasets/processed/handoff/audit_report.json` | Structural verification results and unresolved limitations. |
| `datasets/processed/handoff/annual_next_year_RESEARCH_CANDIDATE.csv` | 25-row six-feature annual next-year AQI research candidate. |
| `datasets/processed/handoff/annual_next_year_REDUCED_CANDIDATE.csv` | 25-row reduced four-feature comparison candidate; **not certified leakage-free**. |
| `datasets/processed/handoff/research_{train,validation,test}_CANDIDATE.csv` | Chronological 15/5/5 split files for **exploratory** modeling. |

## Coverage and inclusion decisions

- **Annual AQI labels:** 35/35 borough-year observations, each with at least 300 recorded AQI days. AQI median is the next-year target in the research candidate, not a same-year predictor.
- **Tree-planting records:** 35/35; counts reflect recorded planting points, not necessarily all actual planting activity.
- **ACS commuting:** 30/35 (2019–2024). ACS five-year vintages cover multiyear windows; the end year is not their actual publication date.
- **CAMPD industrial emissions:** 30/35 (2020–2025); excluded partial-year 2026.
- **Weather:** 35/35 (2019–2025); ERA5 modeled historical weather at representative borough locations, **not borough-wide weather station averages**.
- **LUCID land use:** 20/35 (2019–2022); later years left missing, not fabricated.
- **Tree removal work orders:** 6/35, mostly 2025; retained with missing years blank and omitted from initial baseline.
- **Research EV registrations:** 25/35 (2019–2023). This is a count, **not percentage of total registered vehicles**. 2024–2025 left missing; EV excluded from baseline.
- **NYC DOT traffic:** samples exist for all borough-years, but roadway segments and sampling intensity change by year; excluded from baseline pending comparable-site analysis.
- **Residential combustion:** original file lacks a source year; kept as a separate snapshot, not inserted into annual rows.
- **EIA regional 2024 spreadsheet:** geographic and temporal granularity incompatible with NYC borough-year training, so excluded.

## Audit results and proposed split

The `audit_report.json` recorded `PASS` for borough/FIPS keys, numeric bounds, unique rows, year alignment, selected-feature missingness, and next-year AQI target matching. It did **not** test forecasting skill, model leakage from release dates, or feature representativeness.

| Experimental subset | Feature years | AQI target years | Rows |
| --- | --- | --- | ---: |
| Train | 2020–2022 | 2021–2023 | 15 |
| Validation | 2023 | 2024 | 5 |
| Test | 2024 | 2025 | 5 |

**Research predictors:** `last_observed_median_aqi`, `recorded_tree_plantings`, `drive_alone_pct`, `campd_total_nox_mass_tons`, `weather_temp_mean_c`, `weather_precip_total_mm`.

**Reduced comparison predictors:** `last_observed_median_aqi`, `recorded_tree_plantings`, `weather_temp_mean_c`, `weather_precip_total_mm`.

The two candidate datasets contain only **25 correlated borough-year observations**; test-set performance will be highly uncertain. Compare models against a simple **no-change baseline** (next-year AQI = most recently observed AQI), and do not random-split borough rows from the same year.

## Critical issue before claiming genuine next-year forecasting

Predictors must have been **published and available by the forecast issue date**. For example, the ACS five-year estimate labeled 2024 may have been released after January 1, 2025; using it to forecast 2025 at that date would create an information-availability problem. Annual AQI, emissions, tree inventories and historical reanalysis can also have publication or revision lags. **Neither candidate has passed a point-in-time availability audit.** Define the forecast date, verify release dates, then drop or lag features accordingly.

Do not replace unknown data with zero or fit an imputer/scaler on all train+validation+test rows. Any learned preprocessing must be fit only on training data and applied unchanged to held-out years.

## Reproducing the outputs

Install Python dependencies used by the scripts (at minimum `pandas` and `requests`) and run from the repository root, after ensuring the existing `datasets/solvers` and `datasets/LUCID` source files are present:

```powershell
python -m pip install pandas requests
python scripts/collect_historical_acs.py
python scripts/collect_nyc_weather.py
python scripts/prepare_model_data.py
python scripts/collect_nyc_ev.py
python scripts/review_ev_and_stage_data.py
python scripts/finalize_air_pollution_handoff.py
```

`collect_historical_acs.py` may require a Census API key via the `CENSUS_API_KEY` environment variable. The traffic collection is optional to reproduce the research baseline (traffic is intentionally excluded). An archived EPA annual AQI source may be downloaded by the preparation script; check `processing_report.json` after any rerun.

## Security and next steps

- Commit the **scripts**, small documented **processed outputs**, QA reports, and this README. Keep source references and provenance available.
- **Do not commit** API keys, `.env` files, private credentials, API caches, or unnecessarily large downloaded archives.
- Decide whether the final project target is *annual AQI regression*, *AQI category classification*, or *daily forecasting* before treating these as final model inputs.
- Address point-in-time availability and sample-size limitations before publishing accuracy claims.
