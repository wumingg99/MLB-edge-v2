"""
Invariant tests for store.py.

Proves four structural guarantees:
1. Write-once: same (game_id, game_date, model_version, stage) twice → 1 row.
2. Finished-game guard: status "Final" or "Live" → rejected before INSERT.
3. Typed date: game_date is stored as 'YYYY-MM-DD' even when a full ISO
   timestamp is passed in.
4. latest_stage returns most recently INSERTED row by autoincrement id,
   not by stage name — so an out-of-order backfill is visible.

Plus one integration test (test_real_prediction_no_silent_nulls) that loads
a real trained-model entry from prediction_audit.jsonl and asserts the three
most critical frozen-market fields are stored as non-NULL values.

Also covers ticket_id being nullable in user_bets.
"""
import pytest

from store import (
    get_snapshot,
    init_db,
    insert_movement_event,
    insert_pregame_snapshot,
    insert_user_bet,
    latest_stage,
)


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


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test.db")
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Invariant 1 — Write-once
# ---------------------------------------------------------------------------

def test_same_key_twice_leaves_one_row(conn):
    """INSERT OR IGNORE: second call with identical key is a no-op."""
    r1 = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    r2 = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")

    assert r1 is True,  "first insert should return True (new row)"
    assert r2 is False, "second insert should return False (UNIQUE ignored)"

    count = conn.execute(
        "SELECT COUNT(*) FROM pregame_snapshots"
    ).fetchone()[0]
    assert count == 1, f"expected 1 row, got {count}"


def test_different_stage_is_a_new_row(conn):
    """Same game + different stage = two separate rows."""
    r1 = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    r2 = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "PITCHERS")

    assert r1 is True
    assert r2 is True

    count = conn.execute(
        "SELECT COUNT(*) FROM pregame_snapshots"
    ).fetchone()[0]
    assert count == 2, f"expected 2 rows for two stages, got {count}"


# ---------------------------------------------------------------------------
# Invariant 2 — Finished-game guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_status", ["Final", "Live"])
def test_finished_game_is_rejected(conn, bad_status):
    """status=Final or Live must be rejected before reaching the DB."""
    finished_game = {**GAME, "status": bad_status}

    result = insert_pregame_snapshot(conn, finished_game, PREDICTION, ODDS, "EARLY")

    assert result is False, f"status={bad_status!r} should return False"

    count = conn.execute(
        "SELECT COUNT(*) FROM pregame_snapshots"
    ).fetchone()[0]
    assert count == 0, f"no rows should exist after a {bad_status!r} rejection"


def test_preview_status_is_accepted(conn):
    """Sanity check: status=Preview passes the guard."""
    result = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    assert result is True


def test_missing_status_is_accepted(conn):
    """A game dict with no status key should not be blocked."""
    game_no_status = {k: v for k, v in GAME.items() if k != "status"}
    result = insert_pregame_snapshot(conn, game_no_status, PREDICTION, ODDS, "EARLY")
    assert result is True


# ---------------------------------------------------------------------------
# Invariant 3 — Typed date (game_date stored as 'YYYY-MM-DD')
# ---------------------------------------------------------------------------

def test_game_date_stored_as_date_string(conn):
    """game_date in the DB must be exactly 10 characters: 'YYYY-MM-DD'."""
    insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")

    raw = conn.execute(
        "SELECT game_date FROM pregame_snapshots LIMIT 1"
    ).fetchone()[0]

    assert raw == "2026-06-14", f"expected '2026-06-14', got {raw!r}"
    assert len(raw) == 10,      f"game_date must be 10 chars, got {len(raw)}"


def test_iso_timestamp_in_game_date_is_truncated(conn):
    """If a caller accidentally passes a full ISO timestamp, it's truncated."""
    iso_game = {**GAME, "date": "2026-06-14T23:59:00+08:00"}
    insert_pregame_snapshot(conn, iso_game, PREDICTION, ODDS, "EARLY")

    raw = conn.execute(
        "SELECT game_date FROM pregame_snapshots LIMIT 1"
    ).fetchone()[0]

    assert raw == "2026-06-14", f"expected '2026-06-14', got {raw!r}"


# ---------------------------------------------------------------------------
# Invariant 4 — latest_stage by insertion order
# ---------------------------------------------------------------------------

def test_latest_stage_returns_most_recently_inserted(conn):
    """
    latest_stage is ordered by autoincrement id, not stage name.
    Inserting FINAL then EARLY means EARLY is 'latest' even though
    FINAL is logically the furthest stage.
    """
    insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "FINAL")
    insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")

    row = latest_stage(conn, GAME["game_id"], GAME["date"])

    assert row is not None
    assert row["stage"] == "EARLY", (
        f"expected EARLY (last inserted), got {row['stage']!r} — "
        "latest_stage must reflect insertion order, not stage progression"
    )


def test_latest_stage_returns_none_for_unknown_game(conn):
    row = latest_stage(conn, 999999, "2026-06-14")
    assert row is None


def test_get_snapshot_returns_correct_row(conn):
    insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    row = get_snapshot(conn, GAME["game_id"], GAME["date"], "EARLY")

    assert row is not None
    assert row["game_id"] == GAME["game_id"]
    assert row["stage"] == "EARLY"
    assert row["total_line"] == 9.5
    assert row["edge_flagged"] == 1        # True stored as int
    assert row["rl_edge_flagged"] == 0     # False stored as int


# ---------------------------------------------------------------------------
# user_bets — ticket_id nullable
# ---------------------------------------------------------------------------

def test_user_bet_ticket_id_nullable(conn):
    """ticket_id can be omitted (NULL) or supplied."""
    insert_user_bet(
        conn, game_id=777001, game_date="2026-06-14",
        side="OVER", price=-115, stake=1.0,
    )
    row_no_ticket = conn.execute(
        "SELECT ticket_id FROM user_bets WHERE ticket_id IS NULL"
    ).fetchone()
    assert row_no_ticket is not None, "row without ticket_id should exist"

    insert_user_bet(
        conn, game_id=777001, game_date="2026-06-14",
        side="OVER", price=-115, stake=1.0,
        ticket_id="DK-ABC-123",
    )
    row_with_ticket = conn.execute(
        "SELECT ticket_id FROM user_bets WHERE ticket_id IS NOT NULL"
    ).fetchone()
    assert row_with_ticket is not None
    assert row_with_ticket[0] == "DK-ABC-123"


def test_user_bet_game_date_truncated(conn):
    """game_date is also truncated in user_bets."""
    insert_user_bet(
        conn, game_id=777001,
        game_date="2026-06-14T15:00:00+00:00",
        side="UNDER", price=-105, stake=0.5,
    )
    raw = conn.execute(
        "SELECT game_date FROM user_bets LIMIT 1"
    ).fetchone()[0]
    assert raw == "2026-06-14"


# ---------------------------------------------------------------------------
# movement_events — basic smoke test
# ---------------------------------------------------------------------------

def test_movement_event_inserted(conn):
    insert_movement_event(
        conn,
        game_id=777001,
        game_date="2026-06-14",
        model_version="v3-price-aware-2026-06",
        event_type="LINE_MOVE",
        magnitude=0.5,
        detail={"from": 9.0, "to": 9.5},
    )
    row = conn.execute("SELECT * FROM movement_events").fetchone()
    assert row is not None
    assert row["event_type"] == "LINE_MOVE"
    assert row["magnitude"] == 0.5

    import json
    detail = json.loads(row["detail"])
    assert detail == {"from": 9.0, "to": 9.5}
    assert row["game_date"] == "2026-06-14"


# ---------------------------------------------------------------------------
# Integration test — real prediction from prediction_audit.jsonl
# ---------------------------------------------------------------------------

def test_real_prediction_no_silent_nulls(tmp_path):
    """
    Load a real trained-model entry from the live audit log and round-trip it
    through insert_pregame_snapshot. Asserts that total_line, rl_point, and
    rl_price — the three frozen-market fields most critical to get right —
    are stored as non-NULL values matching the source prediction dict.

    This test catches key-name mismatches that hand-built fixture dicts cannot
    catch, because those dicts carry the same assumed names as the function.

    Skipped if prediction_audit.jsonl does not exist or has no trained-model
    entries (e.g. fresh dev environment before any bot run).
    """
    import json
    from pathlib import Path

    audit_path = Path(__file__).with_name("prediction_audit.jsonl")
    if not audit_path.exists():
        pytest.skip("prediction_audit.jsonl not present")

    # Find first entry where the model was actually trained (rl_point not None).
    real_entry = None
    with audit_path.open() as fh:
        for line in fh:
            try:
                row = json.loads(line)
                pred = row.get("prediction", {})
                if pred.get("model_ready") and pred.get("rl_point") is not None:
                    real_entry = row
                    break
            except json.JSONDecodeError:
                continue

    if real_entry is None:
        pytest.skip("No trained-model entry with rl_point in audit log")

    real_prediction = real_entry["prediction"]
    real_odds = real_entry["odds"]

    # Reconstruct minimal game dict (audit log stores teams as "Away @ Home" string).
    game_str = real_entry.get("game", "Unknown @ Unknown")
    parts = game_str.split(" @ ", 1)
    away_team = parts[0] if len(parts) == 2 else "Unknown"
    home_team = parts[1] if len(parts) == 2 else "Unknown"

    game = {
        "game_id": real_entry["game_id"],
        "date":    real_entry["date"],
        "home_team": home_team,
        "away_team": away_team,
        "home_pitcher": "TBD",
        "away_pitcher": "TBD",
        "status": "Preview",
    }

    conn = init_db(tmp_path / "real_integration.db")
    wrote = insert_pregame_snapshot(conn, game, real_prediction, real_odds, "EARLY")
    assert wrote is True, "Expected a new row to be written from real data"

    db_row = conn.execute("SELECT * FROM pregame_snapshots LIMIT 1").fetchone()
    assert db_row is not None

    # The three frozen-market fields must not be silently NULL.
    assert db_row["total_line"] is not None, \
        "total_line stored as NULL — key mismatch in insert_pregame_snapshot"
    assert db_row["rl_point"] is not None, \
        "rl_point stored as NULL — key mismatch in insert_pregame_snapshot"
    assert db_row["rl_price"] is not None, \
        "rl_price stored as NULL — key mismatch in insert_pregame_snapshot"

    # Values must round-trip exactly.
    assert db_row["total_line"] == real_prediction["total_line"], \
        f"total_line mismatch: DB={db_row['total_line']!r} vs pred={real_prediction['total_line']!r}"
    assert db_row["rl_point"] == real_prediction["rl_point"], \
        f"rl_point mismatch: DB={db_row['rl_point']!r} vs pred={real_prediction['rl_point']!r}"
    assert db_row["rl_price"] == real_prediction["rl_price"], \
        f"rl_price mismatch: DB={db_row['rl_price']!r} vs pred={real_prediction['rl_price']!r}"

    # Spot-check a few more fields while we have a real row.
    assert db_row["model_version"] == real_prediction["model_version"]
    assert db_row["game_date"] == str(real_entry["date"])[:10]
    assert db_row["total_over_price"] == (real_odds.get("total_market") or {}).get("over_price")

    conn.close()


# ---------------------------------------------------------------------------
# Task 02 — Snapshot write in fetch_all_games (EARLY stage)
# ---------------------------------------------------------------------------

def test_early_snapshot_one_row(conn):
    """New prediction → exactly one EARLY row in pregame_snapshots."""
    result = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    assert result is True
    count = conn.execute("SELECT COUNT(*) FROM pregame_snapshots").fetchone()[0]
    assert count == 1


def test_early_snapshot_idempotent(conn):
    """Same prediction written twice → still exactly one row (UNIQUE guard)."""
    insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    result = insert_pregame_snapshot(conn, GAME, PREDICTION, ODDS, "EARLY")
    assert result is False
    count = conn.execute("SELECT COUNT(*) FROM pregame_snapshots").fetchone()[0]
    assert count == 1


def test_snapshot_exception_is_swallowed():
    """
    Calls the real bot._safe_snapshot_write with a patched store that raises.
    The function must return normally — no exception must escape.
    Also asserts the connection's finally-close ran despite the insert failure.
    """
    import bot
    import store
    from unittest.mock import MagicMock, patch

    mock_conn = MagicMock()
    with patch.object(store, "init_db", return_value=mock_conn), \
         patch.object(store, "insert_pregame_snapshot", side_effect=RuntimeError("disk full")):
        bot._safe_snapshot_write(GAME, PREDICTION, ODDS, "EARLY")  # must not raise

    mock_conn.close.assert_called_once()  # finally: conn.close() always fires
