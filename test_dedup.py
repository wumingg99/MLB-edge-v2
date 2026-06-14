"""
Restart-safe dedup test.

Two scenarios:
1. First session: log_prediction writes the row, _newly_logged is True.
2. Simulated restart (clear _logged_keys, row still on disk): _already_logged
   reads the disk, _newly_logged is False, no duplicate row is written.
"""
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest import mock


# ── Minimal env so sheets.py loads without real credentials ───────────────
os.environ.setdefault("SHEETS_URL", "")
os.environ.setdefault("SHEETS_SECRET", "")
os.environ.setdefault("SHEET_NAME", "test")
os.environ.setdefault("TIMEZONE", "Asia/Singapore")

import sheets  # noqa: E402
import bot     # noqa: E402


GAME = {
    "game_id": 999001,
    "home_team": "Home Team",
    "away_team": "Away Team",
    "date": "2026-01-01",
}
BASE_PREDICTION = {
    "quote_timestamp": "2026-01-01T12:00:00+00:00",
    "our_total": 9.0,
    "our_home_margin": -0.5,
    "edge_flagged": True,
    "rl_edge_flagged": False,
    "model_version": "TEST",
    "model_ready": True,
    "has_real_pitchers": True,
    "total_pred": "OVER",
    "total_line": 9.0,
    "total_conf": 0.9,
    "total_votes": 5,
    "rl_pred": "NO BET",
    "rl_conf": 0.0,
    "rl_votes": 0,
    "data_quality": 1.0,
    "total_ev": 0.05,
    "total_bet_size": "HALF",
    "total_kelly_fraction": 0.05,
    "rl_bet_size": "SKIP",
    "rl_kelly_fraction": 0.0,
    "bookmaker_count": 3,
    "total_price": -110,
    "total_win_prob": 0.55,
    "total_push_prob": 0.02,
    "total_market_prob": 0.52,
    "total_probability_edge": 0.03,
    "total_agreement": 0.8,
    "total_ensemble_std": 0.4,
    "margin_ensemble_std": 0.5,
    "rl_side": None,
    "rl_point": None,
    "rl_price": None,
    "rl_win_prob": None,
    "rl_push_prob": None,
    "rl_market_prob": None,
    "rl_probability_edge": None,
    "rl_ev": None,
    "rl_agreement": None,
}
ODDS = {
    "quote_timestamp": "2026-01-01T12:00:00+00:00",
    "total_market": {"over_price": -110, "under_price": -110},
    "spread_market": {
        "home_point": 1.5, "away_point": -1.5,
        "home_price": -110, "away_price": -110,
    },
}


def _count_rows(path):
    if not path.exists():
        return 0
    count = 0
    with path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("game_id") == GAME["game_id"]:
                count += 1
    return count


def _run_logging_block(prediction, audit):
    """Simulate the logging block inside fetch_all_games for one game."""
    game_id = GAME["game_id"]
    quote_time = prediction.get("quote_timestamp") or "no-quote"
    is_new = not bot._already_logged(game_id, quote_time)
    if is_new:
        sheets.log_prediction(GAME, prediction, ODDS)
        bot._logged_keys.add(f"{game_id}:{quote_time}")
    prediction["_newly_logged"] = is_new


def test_dedup_survives_restart():
    with tempfile.TemporaryDirectory() as tmp:
        audit = Path(tmp) / "prediction_audit.jsonl"

        with mock.patch.object(sheets, "AUDIT_PATH", audit), \
             mock.patch.object(sheets, "SHEETS_URL", ""):

            # ── SESSION 1: first time this edge is seen ───────────────────
            bot._logged_keys.clear()
            prediction = copy.deepcopy(BASE_PREDICTION)
            _run_logging_block(prediction, audit)

            assert _count_rows(audit) == 1, "Expected 1 row after first log"
            assert prediction["_newly_logged"] is True, (
                "_newly_logged should be True on first log"
            )

            # ── SIMULATE RESTART: wipe in-memory cache ────────────────────
            bot._logged_keys.clear()

            # ── SESSION 2: same (game_id, quote_timestamp) arrives again ──
            prediction2 = copy.deepcopy(BASE_PREDICTION)
            _run_logging_block(prediction2, audit)

            assert _count_rows(audit) == 1, (
                "Duplicate row written after restart — disk check failed"
            )
            assert prediction2["_newly_logged"] is False, (
                "_newly_logged should be False after restart (row already on disk) "
                "— alert would have fired incorrectly"
            )

    print("PASS — restart-safe dedup and alert gating work correctly.")


if __name__ == "__main__":
    test_dedup_survives_restart()
