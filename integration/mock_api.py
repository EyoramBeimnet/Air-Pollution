#!/usr/bin/env python3
"""Development-only AQI API mock. Every returned AQI is simulated test data."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import argparse
import json

FIPS = ("36005", "36047", "36061", "36081", "36085")
YEARS = tuple(range(2020, 2031))


def simulated_rows():
    rows = []
    for borough_index, fips in enumerate(FIPS):
        for year in YEARS:
            aqi = 42 + borough_index + (year - 2020)
            record = {"fips": fips, "year": year, "aqi": aqi, "forecast": year > 2025,
                      "data_source": "simulated_test_data"}
            if record["forecast"]:
                record.update({"lo": aqi - 3, "hi": aqi + 3})
            rows.append(record)
    return rows


def scenario_payload(scenario, response_format):
    rows = simulated_rows()
    if scenario == "missing-boroughs":
        rows = [row for row in rows if row["fips"] != "36085"]
    elif scenario == "missing-years":
        rows = [row for row in rows if not (row["fips"] == "36005" and row["year"] == 2027)]
    elif scenario == "invalid-aqi":
        rows[0]["aqi"] = -1
    elif scenario == "invalid-bounds":
        next(row for row in rows if row["forecast"])["lo"] = 999
    return rows if response_format == "array" else {"data": rows, "data_source": "simulated_test_data"}


class MockHandler(BaseHTTPRequestHandler):
    def send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self.send_json(200, {"status": "ok", "data_source": "simulated_test_data"})
        if parsed.path != "/api/aqi":
            return self.send_json(404, {"error": "not found", "data_source": "simulated_test_data"})

        query = parse_qs(parsed.query)
        scenario = query.get("scenario", ["valid"])[0]
        response_format = query.get("format", ["wrapped"])[0]
        if scenario == "http-error":
            return self.send_json(503, {"error": "simulated HTTP error", "data_source": "simulated_test_data"})
        if scenario == "malformed-json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            return self.wfile.write(b'{"simulated": invalid json}')
        return self.send_json(200, scenario_payload(scenario, response_format))

    def log_message(self, format, *args):
        print("mock-api:", format % args)


def main():
    parser = argparse.ArgumentParser(description="Serve simulated AQI integration-test data only.")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), MockHandler)
    print(f"Mock AQI API: http://127.0.0.1:{args.port}/api/aqi (SIMULATED TEST DATA)")
    server.serve_forever()


if __name__ == "__main__":
    main()
