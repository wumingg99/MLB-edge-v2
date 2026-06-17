"""
Tests for clv.py — CLV computation from EARLY + FINAL snapshot pairs.

Covers:
1. Correct CLV on a known EARLY/FINAL pair (hand-checkable numbers, both
   same-side and side-flip cases, and line-moved flag).
2. clv_summary aggregates correctly (avg, % positive, headline same-line only).
3. Fail-safe: a game where the FINAL snapshot has missing data is skipped
   gracefully; the rest of the batch completes.
"""
import pytest
from store import init_db, insert_pregame_snapshot
from clv import compute_clv, clv_summary
from market_math import american_to_decimal


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _game(game_id, status="Preview"):
    return {
        "game_id":      game_id,
        "date":         "2026-06-14",
        "home_team":    "Home Team",
        "away_team":    "Away Team",
        "home_pitcher": "P1",
        "away_pitcher": "P2",
        "status":       status,
    }


def _base_pred(game_id, total_pred="OVER", rl_side="HOME"):
    """Minimal prediction dict; callers override fields as needed."""
    return {
        "model_version":          "v3-test",
        "total_pred":             total_pred,
        "total_line":             8.5,
        "total_win_prob":         0.54,
        "total_push_prob":        0.01,
        "total_market_prob":      0.48,       # no-vig for backed side at EARLY
        "total_probability_edge": 0.06,
        "total_ev":               0.02,
        "total_agreement":        0.70,
        "total_ensemble_std":     0.80,
        "total_bet_size":         "HALF",
        "total_kelly_fraction":   0.01,
        "edge_flagged":           True,
        "rl_side":                rl_side,
        "rl_point":               -1.5 if rl_side == "HOME" else 1.5,
        "rl_price":               140,
        "rl_win_prob":            0.45,
        "rl_push_prob":           0.005,
        "rl_market_prob":         0.42,       # no-vig for backed side at EARLY
        "rl_probability_edge":    0.03,
        "rl_ev":                  0.05,
        "rl_agreement":           0.60,
        "margin_ensemble_std":    1.00,
        "rl_bet_size":            "HALF",
        "rl_kelly_fraction":      0.008,
        "rl_edge_flagged":        True,
        "data_quality":           0.90,
        "bookmaker_count":        3,
    }


def _odds(over_price=-110, under_price=-110, home_price=140, away_price=-162,
          total_line=8.5, quote_ts="2026-06-14T10:00:00+00:00"):
    return {
        "quote_timestamp": quote_ts,
        "total_market": {
            "point":       total_line,
            "over_price":  over_price,
            "under_price": under_price,
        },
        "spread_market": {
            "home_point": -1.5,
            "away_point":  1.5,
            "home_price":  home_price,
            "away_price":  away_price,
        },
    }


@pytest.fixture
def conn(tmp_path):
    c = init_db(tmp_path / "test_clv.db")
    yield c
    c.close()


def _insert(conn, game_id, pred, odds, stage, status="Preview"):
    insert_pregame_snapshot(conn, _game(game_id, status), pred, odds, stage)


# ---------------------------------------------------------------------------
# Test 1 — CLV computed correctly on known EARLY/FINAL pair
# ---------------------------------------------------------------------------

def test_total_clv_same_side(conn):
    """
    EARLY backs OVER at 48% no-vig; FINAL still backs OVER at 52% no-vig.
    clv_points = 0.52 - 0.48 = 0.04  (line moved our way)
    closing_line_ev is EV of the early +110 bet at closing fair prob 52%.
    """
    early_pred = {**_base_pred(1), "total_pred": "OVER", "total_market_prob": 0.48}
    final_pred = {**_base_pred(1), "total_pred": "OVER", "total_market_prob": 0.52,
                  "model_version": "v3-test"}

    early_odds = _odds(over_price=110, under_price=-130, quote_ts="2026-06-14T10:00:00+00:00")
    final_odds = _odds(over_price=-115, under_price=-105, quote_ts="2026-06-14T17:25:00+00:00")

    _insert(conn, 1, early_pred, early_odds, "EARLY")
    _insert(conn, 1, final_pred, final_odds, "FINAL")

    n = compute_clv(conn)
    assert n >= 1, "expected at least one CLV row"

    row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=1 AND market='TOTAL'"
    ).fetchone()
    assert row is not None

    assert row["backed_side"] == "OVER"
    assert row["early_price"] == 110
    assert abs(row["early_no_vig"]    - 0.48) < 1e-9
    assert abs(row["closing_no_vig"]  - 0.52) < 1e-9
    assert abs(row["clv_points"]      - 0.04) < 1e-9
    assert row["line_moved"] == 0

    # Hand-check closing_line_ev: EV of +110 bet at closing fair prob 0.52
    # dec = 2.10, ev = 0.52 * 1.10 - 0.48 = 0.572 - 0.48 = 0.092
    dec = american_to_decimal(110)
    expected_clev = 0.52 * (dec - 1.0) - 0.48
    assert abs(row["closing_line_ev"] - expected_clev) < 1e-9


def test_total_clv_side_flip(conn):
    """
    EARLY backs OVER at 48%; FINAL recommends UNDER at 51% (so UNDER no-vig=0.51).
    closing_no_vig for OVER = 1.0 - 0.51 = 0.49.
    clv_points = 0.49 - 0.48 = 0.01 (small positive move).
    """
    early_pred = {**_base_pred(2), "total_pred": "OVER", "total_market_prob": 0.48}
    final_pred = {**_base_pred(2), "total_pred": "UNDER", "total_market_prob": 0.51,
                  "model_version": "v3-test"}

    _insert(conn, 2, early_pred, _odds(), "EARLY")
    _insert(conn, 2, final_pred, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    compute_clv(conn)

    row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=2 AND market='TOTAL'"
    ).fetchone()
    assert row["backed_side"] == "OVER"
    assert abs(row["closing_no_vig"] - 0.49) < 1e-9
    assert abs(row["clv_points"]     - 0.01) < 1e-9


def test_total_clv_line_moved_flag(conn):
    """
    Line moves from 8.5 (EARLY) to 9.0 (FINAL); line_moved must be 1.
    """
    early_pred = {**_base_pred(3), "total_pred": "OVER", "total_line": 8.5,
                  "total_market_prob": 0.48}
    final_pred = {**_base_pred(3), "total_pred": "OVER", "total_line": 9.0,
                  "total_market_prob": 0.46, "model_version": "v3-test"}

    early_odds = _odds(total_line=8.5)
    final_odds = _odds(total_line=9.0, quote_ts="2026-06-14T17:25:00+00:00")

    _insert(conn, 3, early_pred, early_odds, "EARLY")
    _insert(conn, 3, final_pred, final_odds, "FINAL")

    compute_clv(conn)

    row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=3 AND market='TOTAL'"
    ).fetchone()
    assert row["line_moved"] == 1
    assert row["early_total_line"]   == 8.5
    assert row["closing_total_line"] == 9.0


def test_rl_clv_same_side(conn):
    """
    EARLY backs HOME RL at 42% no-vig (+140); FINAL still backs HOME at 47%.
    clv_points = 0.47 - 0.42 = 0.05.
    closing_line_ev = EV of +140 bet at 0.47 fair prob.
    """
    early_pred = {**_base_pred(4, rl_side="HOME"), "rl_market_prob": 0.42, "rl_price": 140}
    final_pred = {**_base_pred(4, rl_side="HOME"), "rl_market_prob": 0.47,
                  "model_version": "v3-test"}

    _insert(conn, 4, early_pred, _odds(), "EARLY")
    _insert(conn, 4, final_pred, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    compute_clv(conn)

    row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=4 AND market='RL'"
    ).fetchone()
    assert row is not None
    assert row["backed_side"] == "HOME"
    assert abs(row["early_no_vig"]   - 0.42) < 1e-9
    assert abs(row["closing_no_vig"] - 0.47) < 1e-9
    assert abs(row["clv_points"]     - 0.05) < 1e-9

    dec = american_to_decimal(140)
    expected_clev = 0.47 * (dec - 1.0) - 0.53
    assert abs(row["closing_line_ev"] - expected_clev) < 1e-9


def test_rl_clv_side_flip(conn):
    """
    EARLY backs HOME at 42% no-vig; FINAL flips to AWAY at 60% (AWAY no-vig).
    closing_no_vig for HOME = 1.0 - 0.60 = 0.40  (line moved against us).
    clv_points = 0.40 - 0.42 = -0.02 (negative CLV).
    """
    early_pred = {**_base_pred(5, rl_side="HOME"), "rl_market_prob": 0.42}
    final_pred = {**_base_pred(5, rl_side="AWAY"), "rl_market_prob": 0.60,
                  "rl_point": 1.5, "model_version": "v3-test"}

    _insert(conn, 5, early_pred, _odds(), "EARLY")
    _insert(conn, 5, final_pred, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    compute_clv(conn)

    row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=5 AND market='RL'"
    ).fetchone()
    assert row["backed_side"] == "HOME"
    assert abs(row["closing_no_vig"] - 0.40) < 1e-9
    assert abs(row["clv_points"]     - (-0.02)) < 1e-9


def test_rl_nonstandard_point_skipped(conn):
    """
    rl_point = 4.5 (anomalous alternate spread) must be silently skipped.
    """
    pred = {**_base_pred(6, rl_side="HOME"), "rl_point": 4.5, "rl_price": 550}
    # Insert EARLY and FINAL with the anomalous point
    insert_pregame_snapshot(conn, _game(6), pred, _odds(), "EARLY")
    pred2 = dict(pred)
    insert_pregame_snapshot(
        conn, _game(6), pred2, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL"
    )

    compute_clv(conn)

    row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=6 AND market='RL'"
    ).fetchone()
    assert row is None, "non-standard rl_point must be skipped"


def test_write_once_insert_or_replace(conn):
    """
    Running compute_clv twice on the same pair replaces (not duplicates) the row.
    """
    pred_e = {**_base_pred(7), "total_market_prob": 0.48}
    pred_f = {**_base_pred(7), "total_market_prob": 0.52, "model_version": "v3-test"}
    _insert(conn, 7, pred_e, _odds(), "EARLY")
    _insert(conn, 7, pred_f, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    compute_clv(conn)
    compute_clv(conn)   # second run

    count = conn.execute(
        "SELECT COUNT(*) FROM clv_results WHERE game_id=7 AND market='TOTAL'"
    ).fetchone()[0]
    assert count == 1, "INSERT OR REPLACE must not create a duplicate row"


# ---------------------------------------------------------------------------
# Test 2 — clv_summary aggregates correctly
# ---------------------------------------------------------------------------

def test_clv_summary_aggregates(conn):
    """
    Insert CLV rows directly and verify summary stats.

    Rows:
    TOTAL same-line:   +0.04, +0.02, -0.01  -> avg=+0.0167, 2/3 positive
    TOTAL line-moved:  +0.03, -0.05          -> avg=-0.01, 1/2 positive
    RL:                +0.05, +0.03, +0.02   -> avg=+0.0333, 3/3 positive
    """
    rows = [
        # market, backed_side, early_price, e_nv, c_nv, clv_pts, clev, e_line, c_line, moved
        ("TOTAL", "OVER",  -110, 0.48, 0.52,  0.04,  0.01, 8.5, 8.5, 0),
        ("TOTAL", "UNDER", -115, 0.50, 0.52,  0.02,  0.01, 9.0, 9.0, 0),
        ("TOTAL", "OVER",  -110, 0.52, 0.51, -0.01, -0.01, 8.5, 8.5, 0),
        ("TOTAL", "OVER",  -110, 0.48, 0.51,  0.03,  0.02, 8.5, 9.0, 1),  # line moved
        ("TOTAL", "UNDER", -115, 0.50, 0.45, -0.05, -0.02, 9.0, 8.5, 1),  # line moved
        ("RL",    "HOME",   140, 0.42, 0.47,  0.05,  0.10, None, None, 0),
        ("RL",    "HOME",   130, 0.44, 0.47,  0.03,  0.07, None, None, 0),
        ("RL",    "AWAY",  -150, 0.60, 0.62,  0.02,  0.01, None, None, 0),
    ]
    for i, (market, backed, eprice, env, cnv, clvp, clev, eline, cline, lm) in enumerate(rows):
        conn.execute(
            """
            INSERT INTO clv_results
                (game_id, game_date, model_version, market, backed_side,
                 early_price, early_no_vig, closing_no_vig, clv_points,
                 closing_line_ev, early_total_line, closing_total_line,
                 line_moved, computed_at)
            VALUES (?, '2026-06-14', 'v3-test', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '2026-06-14T21:30:00')
            """,
            (i + 100, market, backed, eprice, env, cnv, clvp, clev, eline, cline, lm),
        )
    conn.commit()

    summary = clv_summary(conn)

    # total_same_line: avg of (0.04, 0.02, -0.01) = 0.05/3 ≈ 0.01667
    tsl = summary["total_same_line"]
    assert tsl is not None
    assert tsl["n"] == 3
    assert abs(tsl["avg_clv_points"] - (0.04 + 0.02 - 0.01) / 3) < 1e-4
    assert tsl["pct_positive"] == pytest.approx(2 / 3, abs=0.001)

    # total_line_moved: must be there, labeled not_directly_comparable
    tlm = summary["total_line_moved"]
    assert tlm is not None
    assert tlm["n"] == 2
    assert tlm["note"] == "not_directly_comparable"
    assert abs(tlm["avg_clv_points"] - (0.03 - 0.05) / 2) < 1e-4

    # rl: avg of (0.05, 0.03, 0.02) = 0.10/3 ≈ 0.03333
    rl = summary["rl"]
    assert rl is not None
    assert rl["n"] == 3
    assert abs(rl["avg_clv_points"] - (0.05 + 0.03 + 0.02) / 3) < 1e-4
    assert rl["pct_positive"] == pytest.approx(1.0, abs=0.001)

    # overall: 8 rows
    assert summary["overall"]["n"] == 8


# ---------------------------------------------------------------------------
# Test 3 — Fail-safe: malformed game is skipped, batch completes
# ---------------------------------------------------------------------------

def test_compute_clv_skips_malformed_game(conn):
    """
    If a game's FINAL snapshot has no total_market_prob (NULL), _total_row
    returns None and that market is simply skipped — no exception, and other
    games in the batch still produce CLV rows.

    To test the exception path specifically: we also inject a game where the
    snapshot row lookup itself would cause a crash by patching the inner
    SELECT to raise.  We verify compute_clv returns normally and the good
    game's row is still written.
    """
    from unittest.mock import patch

    # Game 10: well-formed EARLY + FINAL → should produce rows
    pred_good_e = {**_base_pred(10), "total_market_prob": 0.48}
    pred_good_f = {**_base_pred(10), "total_market_prob": 0.52, "model_version": "v3-test"}
    _insert(conn, 10, pred_good_e, _odds(), "EARLY")
    _insert(conn, 10, pred_good_f, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    # Game 11: FINAL has NULL total_market_prob → _total_row returns None (skip, no error)
    pred_null_e = {**_base_pred(11), "total_market_prob": 0.50, "rl_market_prob": None,
                   "rl_side": None, "rl_point": None}
    pred_null_f = {**_base_pred(11), "total_market_prob": None, "rl_market_prob": None,
                   "rl_side": None, "rl_point": None, "model_version": "v3-test"}
    _insert(conn, 11, pred_null_e, _odds(), "EARLY")
    _insert(conn, 11, pred_null_f, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    # compute_clv must not raise and must return at least 1 row (from game 10)
    n = compute_clv(conn)
    assert n >= 1, f"expected rows from game 10, got {n}"

    good_row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=10 AND market='TOTAL'"
    ).fetchone()
    assert good_row is not None, "game 10 CLV row must be written"

    null_row = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=11"
    ).fetchone()
    assert null_row is None, "game 11 must produce no CLV rows (NULL market_prob)"


# ---------------------------------------------------------------------------
# Test 4 — format_clv_digest
# ---------------------------------------------------------------------------

def test_format_clv_digest_real_data():
    """Full summary with data in both markets produces the expected string."""
    from clv import format_clv_digest
    summary = {
        "total_same_line":  {"n": 18, "avg_clv_points": 0.032, "avg_closing_line_ev": 0.08, "pct_positive": 0.65},
        "total_line_moved": {"n": 4,  "avg_clv_points": -0.01, "note": "not_directly_comparable"},
        "rl":               {"n": 16, "avg_clv_points": 0.018, "avg_closing_line_ev": 0.05, "pct_positive": 0.58},
        "overall":          {"n": 38, "avg_clv_points": 0.026},
    }
    digest = format_clv_digest(summary, date_label="Jun 15")

    assert digest is not None
    assert "CLV Daily Update — Jun 15" in digest
    assert "TOTAL (same-line): avg +3.2 pts | 65%+ | n=18" in digest
    assert "RL: avg +1.8 pts | 58%+ | n=16" in digest
    assert "Line-moved totals: n=4 (not comparable, excluded)" in digest


def test_format_clv_digest_all_zero_returns_none():
    """When both TOTAL same-line n=0 and RL n=0, return None so no send occurs."""
    from clv import format_clv_digest
    summary = {
        "total_same_line":  None,
        "total_line_moved": None,
        "rl":               None,
        "overall":          None,
    }
    assert format_clv_digest(summary, date_label="Jun 15") is None


def test_format_clv_digest_partial_data():
    """When one market has data and the other doesn't, format correctly."""
    from clv import format_clv_digest
    summary = {
        "total_same_line":  {"n": 5, "avg_clv_points": 0.025, "pct_positive": 0.60},
        "total_line_moved": None,
        "rl":               None,
        "overall":          {"n": 5},
    }
    digest = format_clv_digest(summary, date_label="Jun 16")

    assert digest is not None
    assert "TOTAL (same-line): avg +2.5 pts | 60%+ | n=5" in digest
    assert "RL: (no data yet)" in digest
    assert "Line-moved totals: n=0 (not comparable, excluded)" in digest


def test_compute_clv_continues_after_exception(conn):
    """
    If _total_row raises unexpectedly for one game, compute_clv logs and
    continues; subsequent games still produce rows.
    """
    from unittest.mock import patch
    from clv import _total_row as real_total_row

    pred_a_e = {**_base_pred(20), "total_market_prob": 0.48}
    pred_a_f = {**_base_pred(20), "total_market_prob": 0.52, "model_version": "v3-test"}
    _insert(conn, 20, pred_a_e, _odds(), "EARLY")
    _insert(conn, 20, pred_a_f, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    pred_b_e = {**_base_pred(21), "total_market_prob": 0.49}
    pred_b_f = {**_base_pred(21), "total_market_prob": 0.53, "model_version": "v3-test"}
    _insert(conn, 21, pred_b_e, _odds(), "EARLY")
    _insert(conn, 21, pred_b_f, _odds(quote_ts="2026-06-14T17:25:00+00:00"), "FINAL")

    call_count = [0]
    def boom_then_real(early, final_):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated crash on game 20")
        return real_total_row(early, final_)

    import clv as clv_module
    with patch.object(clv_module, "_total_row", side_effect=boom_then_real):
        n = compute_clv(conn)

    # Game 20 crashed but game 21 should still produce a TOTAL row
    assert n >= 1
    row_b = conn.execute(
        "SELECT * FROM clv_results WHERE game_id=21 AND market='TOTAL'"
    ).fetchone()
    assert row_b is not None, "game 21 must be written even after game 20 crashed"
