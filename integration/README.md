# AQI service integration

Run the frontend and read-only API together from the repository root:

```zsh
python3 -m uvicorn api.app:app --reload
```

Open http://127.0.0.1:8000. Useful endpoints:

```text
http://127.0.0.1:8000/health
http://127.0.0.1:8000/api/aqi?start_year=2020&end_year=2025
```

The service reads committed annual median daily AQI observations and does not
invent future prediction records or uncertainty intervals.

## Tests

```zsh
node --test frontend/tests/aqi-api.test.js
python3 -m unittest tests/test_aqi_service.py
```
