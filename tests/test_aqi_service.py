from pathlib import Path
import unittest

from api.app import observed_aqi_rows


class ObservedAqiRowsTests(unittest.TestCase):
    def test_committed_source_has_complete_2020_to_2025_borough_coverage(self) -> None:
        rows = observed_aqi_rows()
        coverage = {(row["fips"], row["year"]) for row in rows if 2020 <= row["year"] <= 2025}
        self.assertEqual(len(coverage), 30)
        self.assertTrue(all(row["forecast"] is False for row in rows))

    def test_invalid_source_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            observed_aqi_rows(Path(__file__))
