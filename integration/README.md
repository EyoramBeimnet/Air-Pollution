# Local AQI integration test tools

`mock_api.py` is development-only. It does not train, load, or serve a real
model; every AQI response is explicitly marked `simulated_test_data`.

## Run the mock API

From the repository root:

```zsh
python3 integration/mock_api.py
```

Useful URLs:

```text
http://127.0.0.1:8001/api/aqi
http://127.0.0.1:8001/api/aqi?format=array
http://127.0.0.1:8001/api/aqi?scenario=http-error
http://127.0.0.1:8001/api/aqi?scenario=malformed-json
http://127.0.0.1:8001/api/aqi?scenario=missing-boroughs
http://127.0.0.1:8001/api/aqi?scenario=missing-years
http://127.0.0.1:8001/api/aqi?scenario=invalid-aqi
http://127.0.0.1:8001/api/aqi?scenario=invalid-bounds
```

## Run frontend contract tests

```zsh
node --test frontend/tests/aqi-api.test.js
```

## Browser demonstration

In separate terminals:

```zsh
python3 integration/mock_api.py
cd frontend && python3 -m http.server 8000
```

Open this URL to load simulated API data:

```text
http://localhost:8000/?api=http%3A%2F%2F127.0.0.1%3A8001%2Fapi%2Faqi
```

The top-right status changes from `Sample data` to `Live API data`. Use a
scenario URL above in the `api` parameter to verify that invalid responses fall
back to sample data.
