from datetime import date
import unittest

from data import _build_spread_market, _build_total_market
from historical import PitcherState, TeamState
from sheets import _merge_game_snapshots


class PipelineTests(unittest.TestCase):
    def test_team_snapshot_is_as_of_time(self):
        state = TeamState()
        before = state.snapshot(date(2025, 4, 1))
        self.assertEqual(before["stats"]["games"], 0)
        state.update(
            date(2025, 4, 1),
            6,
            3,
            {
                "atBats": 32,
                "hits": 9,
                "doubles": 2,
                "homeRuns": 1,
                "baseOnBalls": 4,
            },
            {"inningsPitched": "9.0", "earnedRuns": 3},
            {"inningsPitched": "6.0", "earnedRuns": 2},
        )
        after = state.snapshot(date(2025, 4, 2))
        self.assertEqual(after["stats"]["games"], 1)
        self.assertNotEqual(before["stats"]["rpg"], after["stats"]["rpg"])

    def test_pitcher_snapshot_excludes_current_start_until_update(self):
        state = PitcherState()
        self.assertEqual(state.snapshot()["season"]["ip"], 0.0)
        state.update({
            "inningsPitched": "6.0",
            "outs": 18,
            "earnedRuns": 2,
            "strikeOuts": 8,
            "baseOnBalls": 1,
            "homeRuns": 1,
            "hits": 5,
            "airOuts": 5,
        })
        self.assertEqual(state.snapshot()["season"]["ip"], 6.0)

    def test_first_flag_is_bet_and_latest_quote_is_close(self):
        snapshots = [
            {
                "quote_timestamp": "2025-04-01T10:00:00Z",
                "edge_flagged": True,
                "total_pred": "OVER",
                "total_line": 8.0,
                "total_price": -105,
                "over_price": -105,
                "under_price": -115,
                "rl_edge_flagged": False,
            },
            {
                "quote_timestamp": "2025-04-01T18:00:00Z",
                "edge_flagged": False,
                "total_pred": "OVER",
                "total_line": 8.5,
                "total_price": -115,
                "over_price": -115,
                "under_price": -105,
                "rl_edge_flagged": False,
            },
        ]
        merged = _merge_game_snapshots(snapshots)
        self.assertTrue(merged["edge_flagged"])
        self.assertEqual(merged["total_line"], 8.0)
        self.assertEqual(merged["closing_total_line"], 8.5)
        self.assertEqual(merged["total_clv_points"], 0.5)
        self.assertEqual(merged["closing_total_price"], -115)

    def test_market_consensus_uses_modal_line_and_best_price(self):
        total = _build_total_market([
            {"point": 8.5, "over_price": -110, "under_price": -110},
            {"point": 8.5, "over_price": -105, "under_price": -115},
            {"point": 9.0, "over_price": 100, "under_price": -120},
        ])
        self.assertEqual(total["point"], 8.5)
        self.assertEqual(total["over_price"], -105)
        self.assertEqual(total["books"], 2)

        spread = _build_spread_market([
            {"point": -1.5, "home_price": 120, "away_price": -135},
            {"point": -1.5, "home_price": 125, "away_price": -140},
        ])
        self.assertEqual(spread["home_point"], -1.5)
        self.assertEqual(spread["away_point"], 1.5)
        self.assertEqual(spread["home_price"], 125)


if __name__ == "__main__":
    unittest.main()
