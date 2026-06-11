import unittest
from unittest.mock import patch

import numpy as np

import model


class ModelTests(unittest.TestCase):
    def test_feature_schema_is_stable_and_finite(self):
        features = model.build_features({})
        self.assertEqual(len(features), len(model.FEATURE_NAMES))
        self.assertTrue(np.isfinite(features).all())

    def test_baseball_innings_are_not_decimal_innings(self):
        self.assertAlmostEqual(model.parse_innings("10.1"), 10 + 1 / 3)
        self.assertAlmostEqual(model.parse_innings("10.2"), 10 + 2 / 3)

    def test_fip_and_xfip_are_computed_from_raw_counts(self):
        metrics = model.compute_pitching_metrics({
            "inningsPitched": "10.1",
            "earnedRuns": 4,
            "strikeOuts": 12,
            "baseOnBalls": 3,
            "hitBatsmen": 1,
            "homeRuns": 2,
            "hits": 9,
            "airOuts": 8,
        })
        self.assertNotEqual(metrics["fip"], 4.5)
        self.assertNotEqual(metrics["xfip"], 4.5)
        self.assertAlmostEqual(metrics["ip"], 10 + 1 / 3)

    def test_no_vig_probabilities_sum_to_one(self):
        side_a, side_b = model.no_vig_probabilities(-110, -110)
        self.assertAlmostEqual(side_a + side_b, 1.0)
        self.assertAlmostEqual(side_a, 0.5)

    def test_expected_value_uses_price(self):
        value = model.expected_value(0.55, 0.45, -110)
        self.assertAlmostEqual(value, 0.05, places=3)

    def test_integer_total_and_spread_pushes(self):
        self.assertEqual(model.grade_total(8, 8, "OVER"), "PUSH")
        self.assertEqual(model.grade_total(8, 8, "UNDER"), "PUSH")
        self.assertEqual(model.grade_spread(1, "AWAY", 1), "PUSH")

    def test_run_line_semantics_price_each_side_explicitly(self):
        selection = model._select_spread(
            np.asarray([2, 2, -1, 0], dtype=float),
            np.asarray([1.8, 2.1, 1.5], dtype=float),
            {
                "home_point": -1.5,
                "away_point": 1.5,
                "home_price": -105,
                "away_price": -115,
            },
        )
        self.assertEqual(selection["side"], "HOME")
        self.assertEqual(selection["point"], -1.5)

    def test_missing_model_never_flags_a_bet(self):
        odds = {
            "total_market": {
                "point": 8.5,
                "over_price": 100,
                "under_price": -110,
            },
            "spread_market": {
                "home_point": -1.5,
                "away_point": 1.5,
                "home_price": 120,
                "away_price": -130,
            },
        }
        with patch("model.load_models", return_value=None):
            prediction = model.predict_game(
                {"has_real_pitchers": True, "data_quality": 1.0},
                odds_entry=odds,
            )
        self.assertFalse(prediction["model_ready"])
        self.assertFalse(prediction["edge_flagged"])
        self.assertFalse(prediction["rl_edge_flagged"])


if __name__ == "__main__":
    unittest.main()
