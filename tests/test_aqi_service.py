import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from api.app import available_aqi_rows, forecast_aqi_rows, observed_aqi_rows


class ObservedAqiRowsTests(unittest.TestCase):
    def test_committed_source_has_complete_2020_to_2025_borough_coverage(self) -> None:
        rows = observed_aqi_rows()
        coverage = {(row["fips"], row["year"]) for row in rows if 2020 <= row["year"] <= 2025}
        self.assertEqual(len(coverage), 30)
        self.assertTrue(all(row["forecast"] is False for row in rows))

    def test_invalid_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            observed_aqi_rows(Path(__file__))

    def test_committed_forecast_has_complete_2026_borough_coverage(self) -> None:
        rows = forecast_aqi_rows()
        self.assertEqual({row["fips"] for row in rows}, {"36005", "36047", "36061", "36081", "36085"})
        self.assertTrue(all(row["year"] == 2026 and row["forecast"] is True for row in rows))
        self.assertTrue(all(row["aqi"] >= 0 for row in rows))
        self.assertTrue(all(row["metric"] == "annual_median_daily_aqi" for row in rows))
        metadata = json.loads((Path("datasets/processed/model_results/annual_aqi_forecast_2026_metadata.json")).read_text())
        self.assertEqual(metadata["metric"], "annual_median_daily_aqi")
        self.assertEqual(metadata["forecast_feature_year"], 2025)

    def test_invalid_forecast_is_rejected_without_removing_observations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "forecast.csv"
            invalid.write_text("fips,borough,year,aqi,forecast\n36005,Bronx,2026,40,True\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                forecast_aqi_rows(invalid)
        rows, status = available_aqi_rows()
        self.assertEqual(len([row for row in rows if not row["forecast"]]), 60)
        self.assertIn(status, {"available", "not generated"})

    def test_missing_forecast_keeps_all_historical_observations_available(self) -> None:
        with patch("api.app.FORECAST_SOURCE", Path("does-not-exist.csv")):
            rows, status = available_aqi_rows()
        self.assertEqual(len(rows), 60)
        self.assertEqual(status, "not generated")
