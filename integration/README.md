# AQI service integration

Run the frontend and API together from the repository root:

```zsh
python3 -m uvicorn api.app:app --reload
```

Open http://127.0.0.1:8000. Useful endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/aqi?start_year=2020&end_year=2026
```

The service reads committed annual median daily AQI observations and, when the
validated forecast artifact exists, appends five 2026 model-generated point
predictions. Historical records have `forecast: false`; 2026 records have
`forecast: true`. No confidence intervals are generated or displayed.

## Build the forecast artifact

```zsh
python3 scripts/build_annual_aqi_forecast.py
```

The script reuses the project’s weather-enhanced annual XGBoost configuration.
It fits on labeled 2014–2024 feature years and predicts 2026 from observed 2025
AQI, CAMPD NOx, temperature, and precipitation. It writes:

- `datasets/processed/model_results/annual_aqi_forecast_2026.csv`
- `datasets/processed/model_results/annual_aqi_forecast_2026_metadata.json`

If the artifact is absent or invalid, the API continues to serve historical
data and reports the forecast status; the frontend continues with 2020–2025.

## Tests

```zsh
node --test frontend/tests/aqi-api.test.js
python3 -m unittest tests/test_aqi_service.py
```
