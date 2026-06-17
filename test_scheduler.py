"""
Tests for the closing-capture additions in scheduler.py.

Covers four invariants:
1. Coalescing  — nearby games share one cluster; distant games get separate clusters.
2. EARLY+FINAL  — same game can hold an EARLY and a FINAL row; write-once holds per stage.
3. Cluster fail-safe — _run_closing_cluster swallows exceptions, never raises.
4. Past-window skip — schedule_closing_captures silently skips already-passed targets.
"""
import asyncio
from datetime import datetime, timezone

from scheduler import _coalesce_games


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

GAME = {
    "game_id": 777001,
    "date": "2026-06-14",
    "home_team": "Kansas City Royals",
    "away_team": "Texas Rangers",
    "home_pitcher": "Cole Ragans",
    "away_pitcher": "Nathan Eovaldi",
    "status": "Preview",
}

PREDICTION = {
    "model_version":          "v3-price-aware-2026-06",
    "total_pred":             "OVER",
    "total_line":             9.5,
    "total_win_prob":         0.5501,
    "total_push_prob":        0.0102,
    "total_market_prob":      0.5245,
    "total_probability_edge": 0.0256,
    "total_ev":               0.031,
    "total_agreement":        0.72,
    "total_ensemble_std":     0.85,
    "total_bet_size":         "HALF",
    "total_kelly_fraction":   0.015,
    "edge_flagged":           True,
    "rl_side":                "HOME",
    "rl_point":               -1.5,
    "rl_price":               -130,
    "rl_win_prob":            0.4800,
    "rl_push_prob":           0.0050,
    "rl_market_prob":         0.5480,
    "rl_probability_edge":    -0.0680,
    "rl_ev":                  -0.022,
    "rl_agreement":           0.58,
    "margin_ensemble_std":    1.10,
    "rl_bet_size":            "SKIP",
    "rl_kelly_fraction":      0.0,
    "rl_edge_flagged":        False,
    "data_quality":           0.92,
    "bookmaker_count":        3,
}

ODDS = {
    "quote_timestamp": "2026-06-14T10:30:00+00:00",
    "total_market": {"point": 9.5, "over_price": -115, "under_price": -105},
    "spread_market": {
        "home_point": -1.5, "away_point": 1.5,
        "home_price": -130, "away_price": 110,
    },
}


# ---------------------------------------------------------------------------
# Test 1 — Coalescing
# ---------------------------------------------------------------------------

def test_coalesce_groups_nearby_games_into_shared_clusters():
    """
    Games within COALESCE_WINDOW_MIN of the cluster's earliest target share
    one cluster and one API refresh; games outside start a new cluster.

    Setup (all UTC, offset_min=35, window_min=15):
      Game 1 → 18:10 → target 17:35          ┐
      Game 2 → 18:20 → target 17:45  (+10m)  ┘ cluster A, fires 17:35
      Game 3 → 20:10 → target 19:35  (+120m)  ┐
      Game 4 → 20:22 → target 19:47  (+12m)   ┘ cluster B, fires 19:35
    """
    games = [
        {"game_id": 1, "game_time_utc": "2026-06-14T18:10:00Z"},
        {"game_id": 2, "game_time_utc": "2026-06-14T18:20:00Z"},
        {"game_id": 3, "game_time_utc": "2026-06-14T20:10:00Z"},
        {"game_id": 4, "game_time_utc": "2026-06-14T20:22:00Z"},
    ]
    clusters = _coalesce_games(games, close_offset_min=35, coalesce_window_min=15)

    assert len(clusters) == 2, f"expected 2 clusters, got {len(clusters)}"

    fire_a, ids_a = clusters[0]
    assert set(ids_a) == {1, 2}, f"cluster A should hold games 1+2, got {ids_a}"
    assert fire_a == datetime(2026, 6, 14, 17, 35, tzinfo=timezone.utc), (
        f"cluster A should fire at 17:35 UTC, got {fire_a}"
    )

    fire_b, ids_b = clusters[1]
    assert set(ids_b) == {3, 4}, f"cluster B should hold games 3+4, got {ids_b}"
    assert fire_b == datetime(2026, 6, 14, 19, 35, tzinfo=timezone.utc), (
        f"cluster B should fire at 19:35 UTC, got {fire_b}"
    )


def test_coalesce_skips_games_with_no_time():
    """Games missing game_time_utc are silently skipped."""
    games = [
        {"game_id": 10, "game_time_utc": None},
        {"game_id": 11, "game_time_utc": ""},
        {"game_id": 12, "game_time_utc": "2026-06-14T18:10:00Z"},
    ]
    clusters = _coalesce_games(games, close_offset_min=35, coalesce_window_min=15)
    assert len(clusters) == 1
    assert clusters[0][1] == [12]


def test_coalesce_single_game():
    """A single game forms a single cluster."""
    games = [{"game_id": 99, "game_time_utc": "2026-06-14T19:00:00Z"}]
    clusters = _coalesce_games(games, close_offset_min=35, coalesce_window_min=15)
    assert len(clusters) == 1
    fire, ids = clusters[0]
    assert ids == [99]
    assert fire == datetime(2026, 6, 14, 18, 25, tzinfo=timezone.utc)


def test_coalesce_empty_input():
    assert _coalesce_games([]) == []


# ---------------------------------------------------------------------------
# Test 2 — EARLY + FINAL coexistence
# ---------------------------------------------------------------------------

def test_early_and_final_coexist(tmp_path):
    """
    Same game can hold one EARLY row and one FINAL row.
    The UNIQUE key is (game_id, game_date, model_version, stage), so two
    different stages are two distinct rows.  Write-once still holds per stage:
    a second FINAL insert is silently ignored and returns False.
    """
    from store import init_db, insert_pregame_snapshot

    conn = init_db(tmp_path / "coexist.db")

    r_early = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    r_final = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "FINAL")
    r_final_dup = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "FINAL")

    assert r_early is True,     "first EARLY insert should succeed"
    assert r_final is True,     "first FINAL insert should succeed"
    assert r_final_dup is False, "second FINAL insert must be ignored (write-once)"

    count = conn.execute(
        "SELECT COUNT(*) FROM pregame_snapshots"
    ).fetchone()[0]
    assert count == 2, f"expected exactly 2 rows (EARLY + FINAL), got {count}"

    stages = {
        row[0]
        for row in conn.execute("SELECT stage FROM pregame_snapshots")
    }
    assert stages == {"EARLY", "FINAL"}, f"unexpected stages: {stages}"

    conn.close()


# ---------------------------------------------------------------------------
# Test 3 — Cluster job fail-safe
# ---------------------------------------------------------------------------

def test_cluster_job_failsafe():
    """
    When fetch_all_games raises inside _run_closing_cluster, the exception
    must be swallowed and nothing must propagate out of the job.
    """
    from unittest.mock import patch, AsyncMock
    from scheduler import _run_closing_cluster

    async def _run():
        with patch(
            "bot.fetch_all_games",
            new=AsyncMock(side_effect=RuntimeError("API down")),
        ):
            await _run_closing_cluster(None, "2026-06-14", [777001])

    asyncio.run(_run())  # must not raise


# ---------------------------------------------------------------------------
# Test 4 — Past-window skipping
# ---------------------------------------------------------------------------

def test_past_windows_are_not_scheduled():
    """
    schedule_closing_captures must not add any jobs when every game's capture
    target has already passed (fire_time < now - 60 s).
    """
    from unittest.mock import MagicMock, patch
    from scheduler import schedule_closing_captures

    past_games = [
        {"game_id": 1, "game_time_utc": "2020-01-01T18:10:00Z"},
        {"game_id": 2, "game_time_utc": "2020-01-01T20:10:00Z"},
    ]
    mock_scheduler = MagicMock()

    async def _run():
        with patch("data.get_todays_games", return_value=past_games), \
             patch("data.get_todays_date", return_value=("2020-01-01", False)):
            await schedule_closing_captures(mock_scheduler, None)

    asyncio.run(_run())
    mock_scheduler.add_job.assert_not_called()
