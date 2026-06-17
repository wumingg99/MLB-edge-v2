"""
Append-only SQLite snapshot store for MLB Edge predictions.

Design invariants (enforced here, not in callers):
- pregame_snapshots is INSERT OR IGNORE — the UNIQUE constraint makes every
  (game_id, game_date, model_version, stage) write-once at the DB level.
- A finished game (status "Final" or "Live") is rejected before the INSERT
  so that the re-prediction bug is structurally impossible even if the
  scheduler calls this path by mistake.
- game_date is always stored as 'YYYY-MM-DD' text — never an ISO timestamp.
- movement_events and user_bets are plain INSERTs; dedup is not their job.
- No UPDATE paths exist anywhere in this module.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path(__file__).with_name("predictions.db")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS pregame_snapshots (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Unique identity: one row per game × date × model version × stage
    game_id                INTEGER NOT NULL,
    game_date              TEXT    NOT NULL,   -- 'YYYY-MM-DD' only, never a timestamp
    model_version          TEXT    NOT NULL,
    stage                  TEXT    NOT NULL,   -- EARLY | PITCHERS | LINEUPS | FINAL

    -- Metadata
    schema_version         TEXT    NOT NULL DEFAULT 'v3',
    captured_at            TEXT    NOT NULL,   -- UTC ISO from odds_entry["quote_timestamp"]
    home_team              TEXT    NOT NULL,
    away_team              TEXT    NOT NULL,

    -- Frozen market snapshot (written once; UNIQUE constraint prevents overwrites)
    total_line             REAL,
    total_over_price       INTEGER,            -- aligns with V3_FIELDS "over_price"
    total_under_price      INTEGER,            -- aligns with V3_FIELDS "under_price"
    rl_side                TEXT,               -- HOME | AWAY
    rl_point               REAL,               -- aligns with V3_FIELDS "rl_point"
    rl_price               INTEGER,            -- aligns with V3_FIELDS "rl_price"
    ml_price               INTEGER,            -- placeholder; moneyline not yet pulled

    -- Total (O/U) model output — names match predict_game() / V3_FIELDS exactly
    total_pred             TEXT,               -- OVER | UNDER | NO BET
    total_win_prob         REAL,
    total_push_prob        REAL,
    total_market_prob      REAL,               -- no-vig consensus probability
    total_probability_edge REAL,
    total_ev               REAL,
    total_agreement        REAL,
    total_ensemble_std     REAL,
    total_bet_size         TEXT,               -- FULL | HALF | SKIP
    total_kelly_fraction   REAL,
    edge_flagged           INTEGER NOT NULL DEFAULT 0,

    -- Run-line model output — names match predict_game() / V3_FIELDS exactly
    rl_win_prob            REAL,
    rl_push_prob           REAL,
    rl_market_prob         REAL,
    rl_probability_edge    REAL,
    rl_ev                  REAL,
    rl_agreement           REAL,
    margin_ensemble_std    REAL,
    rl_bet_size            TEXT,
    rl_kelly_fraction      REAL,
    rl_edge_flagged        INTEGER NOT NULL DEFAULT 0,

    -- Shared model health
    data_quality           REAL,
    bookmaker_count        INTEGER,

    -- Stage-specific context
    home_pitcher           TEXT,
    away_pitcher           TEXT,
    lineup_confirmed       INTEGER NOT NULL DEFAULT 0,

    UNIQUE(game_id, game_date, model_version, stage)
);

CREATE TABLE IF NOT EXISTS movement_events (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id        INTEGER NOT NULL,
    game_date      TEXT    NOT NULL,   -- 'YYYY-MM-DD'
    model_version  TEXT    NOT NULL,
    event_type     TEXT    NOT NULL,   -- EDGE_GAINED | EDGE_LOST | SIDE_FLIP |
                                       -- LINE_MOVE | PRICE_MOVE |
                                       -- PITCHER_CHANGE | LINEUP_DELTA
    magnitude      REAL,
    detail         TEXT,               -- JSON blob
    captured_at    TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS user_bets (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id          INTEGER NOT NULL,
    game_date        TEXT    NOT NULL,   -- 'YYYY-MM-DD'
    side             TEXT    NOT NULL,   -- OVER | UNDER | HOME | AWAY
    point            REAL,
    price            INTEGER NOT NULL,
    stake            REAL    NOT NULL,
    bet_no_vig       REAL,              -- devigged fair value at bet time
    placed_at        TEXT    NOT NULL,  -- UTC ISO
    ticket_id        TEXT,              -- nullable; sportsbook reference for reconciliation
    -- Filled after game closes
    closing_no_vig   REAL,
    clv_points       REAL,
    closing_line_ev  REAL,
    result           TEXT,              -- W | L | P
    pnl              REAL
);

CREATE TABLE IF NOT EXISTS clv_results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,

    game_id             INTEGER NOT NULL,
    game_date           TEXT    NOT NULL,   -- 'YYYY-MM-DD'
    model_version       TEXT    NOT NULL,
    market              TEXT    NOT NULL,   -- TOTAL | RL

    backed_side         TEXT    NOT NULL,   -- OVER | UNDER | HOME | AWAY
    early_price         INTEGER,            -- American; NULL if odds not captured
    early_no_vig        REAL    NOT NULL,   -- consensus no-vig prob at EARLY
    closing_no_vig      REAL    NOT NULL,   -- consensus no-vig for same side at FINAL
    clv_points          REAL    NOT NULL,   -- closing_no_vig - early_no_vig
    closing_line_ev     REAL,               -- EV of early bet vs closing fair prob

    -- Total-only: track whether the line itself moved
    early_total_line    REAL,               -- NULL for RL rows
    closing_total_line  REAL,               -- NULL for RL rows
    line_moved          INTEGER NOT NULL DEFAULT 0,  -- 1 if lines differ; RL always 0

    computed_at         TEXT    NOT NULL,

    UNIQUE(game_id, game_date, model_version, market)
);
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def init_db(db_path=DB_PATH):
    """Create all tables (idempotent). Returns an open connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_utc():
    return datetime.now(timezone.utc).isoformat()


def _date_only(value):
    """Return the first 10 characters of any date/timestamp string."""
    return str(value)[:10]


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def insert_pregame_snapshot(conn, game, prediction, odds_entry, stage):
    """
    Write one append-only pregame snapshot row.

    Returns True if a new row was written, False if skipped.

    Skips when:
    - game["status"] is "Final" or "Live"  (application-level finished-game guard)
    - this (game_id, game_date, model_version, stage) already exists in the DB
      (UNIQUE constraint silently drops the INSERT OR IGNORE)
    """
    if game.get("status") in ("Final", "Live"):
        return False

    if not prediction:
        return False

    odds = odds_entry or {}
    total_market = odds.get("total_market") or {}

    row = {
        "game_id":                game.get("game_id"),
        "game_date":              _date_only(game.get("date", "")),
        "model_version":          prediction.get("model_version") or "unknown",
        "stage":                  stage,
        "schema_version":         "v3",
        "captured_at":            odds.get("quote_timestamp") or _now_utc(),
        "home_team":              game.get("home_team", ""),
        "away_team":              game.get("away_team", ""),
        # Frozen market
        "total_line":             prediction.get("total_line"),
        "total_over_price":       total_market.get("over_price"),
        "total_under_price":      total_market.get("under_price"),
        "rl_side":                prediction.get("rl_side"),
        "rl_point":               prediction.get("rl_point"),
        "rl_price":               prediction.get("rl_price"),
        "ml_price":               None,
        # Total model output
        "total_pred":             prediction.get("total_pred"),
        "total_win_prob":         prediction.get("total_win_prob"),
        "total_push_prob":        prediction.get("total_push_prob"),
        "total_market_prob":      prediction.get("total_market_prob"),
        "total_probability_edge": prediction.get("total_probability_edge"),
        "total_ev":               prediction.get("total_ev"),
        "total_agreement":        prediction.get("total_agreement"),
        "total_ensemble_std":     prediction.get("total_ensemble_std"),
        "total_bet_size":         prediction.get("total_bet_size"),
        "total_kelly_fraction":   prediction.get("total_kelly_fraction"),
        "edge_flagged":           int(bool(prediction.get("edge_flagged"))),
        # Run-line model output
        "rl_win_prob":            prediction.get("rl_win_prob"),
        "rl_push_prob":           prediction.get("rl_push_prob"),
        "rl_market_prob":         prediction.get("rl_market_prob"),
        "rl_probability_edge":    prediction.get("rl_probability_edge"),
        "rl_ev":                  prediction.get("rl_ev"),
        "rl_agreement":           prediction.get("rl_agreement"),
        "margin_ensemble_std":    prediction.get("margin_ensemble_std"),
        "rl_bet_size":            prediction.get("rl_bet_size"),
        "rl_kelly_fraction":      prediction.get("rl_kelly_fraction"),
        "rl_edge_flagged":        int(bool(prediction.get("rl_edge_flagged"))),
        # Shared
        "data_quality":           prediction.get("data_quality"),
        "bookmaker_count":        prediction.get("bookmaker_count"),
        # Stage-specific
        "home_pitcher":           game.get("home_pitcher"),
        "away_pitcher":           game.get("away_pitcher"),
        "lineup_confirmed":       0,
    }

    columns = ", ".join(row)
    placeholders = ", ".join(f":{k}" for k in row)
    cursor = conn.execute(
        f"INSERT OR IGNORE INTO pregame_snapshots ({columns}) VALUES ({placeholders})",
        row,
    )
    conn.commit()
    return cursor.rowcount == 1


def insert_movement_event(
    conn, game_id, game_date, model_version,
    event_type, magnitude=None, detail=None, captured_at=None,
):
    """Append one movement event. No dedup — every event is a real occurrence."""
    conn.execute(
        """
        INSERT INTO movement_events
            (game_id, game_date, model_version, event_type, magnitude, detail, captured_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            _date_only(game_date),
            model_version,
            event_type,
            magnitude,
            json.dumps(detail) if detail is not None else None,
            captured_at or _now_utc(),
        ),
    )
    conn.commit()


def insert_user_bet(
    conn, game_id, game_date, side, price, stake,
    point=None, bet_no_vig=None, placed_at=None, ticket_id=None,
):
    """Record an actual placed bet. No dedup — caller controls when to insert."""
    conn.execute(
        """
        INSERT INTO user_bets
            (game_id, game_date, side, point, price, stake,
             bet_no_vig, placed_at, ticket_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            _date_only(game_date),
            side,
            point,
            price,
            stake,
            bet_no_vig,
            placed_at or _now_utc(),
            ticket_id,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def get_snapshot(conn, game_id, game_date, stage):
    """Return a single pregame snapshot dict, or None if not found."""
    row = conn.execute(
        """
        SELECT * FROM pregame_snapshots
        WHERE game_id = ? AND game_date = ? AND stage = ?
        ORDER BY model_version DESC
        LIMIT 1
        """,
        (game_id, _date_only(game_date), stage),
    ).fetchone()
    return dict(row) if row else None


def latest_stage(conn, game_id, game_date):
    """
    Return the most recently INSERTED snapshot for this game+date.

    Ordered by autoincrement id (insertion order), not by stage name.
    That means 'latest' reflects the last row written to disk — which
    can differ from the logically furthest stage if snapshots are
    backfilled out of order (e.g. a FINAL row inserted before LINEUPS).
    """
    row = conn.execute(
        """
        SELECT * FROM pregame_snapshots
        WHERE game_id = ? AND game_date = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (game_id, _date_only(game_date)),
    ).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Example (not wired into fetch_all_games yet)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile

    _game = {
        "game_id": 777001,
        "date": "2026-06-14",
        "home_team": "Kansas City Royals",
        "away_team": "Texas Rangers",
        "home_pitcher": "Cole Ragans",
        "away_pitcher": "Nathan Eovaldi",
        "status": "Preview",
    }
    _prediction = {
        "model_version": "v3-price-aware-2026-06",
        "total_pred": "OVER",
        "total_line": 9.5,
        "total_win_prob": 0.5501,
        "total_push_prob": 0.0102,
        "total_market_prob": 0.5245,
        "total_probability_edge": 0.0256,
        "total_ev": 0.031,
        "total_agreement": 0.72,
        "total_ensemble_std": 0.85,
        "total_bet_size": "HALF",
        "total_kelly_fraction": 0.015,
        "edge_flagged": True,
        "rl_side": "HOME",
        "rl_point": -1.5,
        "rl_price": -130,
        "rl_win_prob": 0.4800,
        "rl_push_prob": 0.0050,
        "rl_market_prob": 0.5480,
        "rl_probability_edge": -0.0680,
        "rl_ev": -0.022,
        "rl_agreement": 0.58,
        "margin_ensemble_std": 1.10,
        "rl_bet_size": "SKIP",
        "rl_kelly_fraction": 0.0,
        "rl_edge_flagged": False,
        "data_quality": 0.92,
        "bookmaker_count": 3,
    }
    _odds = {
        "quote_timestamp": "2026-06-14T10:30:00+00:00",
        "total_market": {"point": 9.5, "over_price": -115, "under_price": -105},
        "spread_market": {
            "home_point": -1.5, "away_point": 1.5,
            "home_price": -130, "away_price": 110,
        },
    }

    with tempfile.TemporaryDirectory() as _tmp:
        _conn = init_db(Path(_tmp) / "demo.db")

        # First insert: writes the row
        _wrote = insert_pregame_snapshot(_conn, _game, _prediction, _odds, "EARLY")
        _snap = get_snapshot(_conn, 777001, "2026-06-14", "EARLY")
        print(f"insert returned : {_wrote}")
        print(f"game_date stored: {_snap['game_date']!r}  (10-char date string)")
        print(f"total_line      : {_snap['total_line']}, rl_side: {_snap['rl_side']}")
        print(f"edge_flagged    : {_snap['edge_flagged']}, total_ev: {_snap['total_ev']}")

        # Second insert with identical key: silently ignored
        _wrote2 = insert_pregame_snapshot(_conn, _game, _prediction, _odds, "EARLY")
        print(f"second insert   : {_wrote2}  (False = correctly ignored)")

        # Final-game guard
        _final = dict(_game)
        _final["status"] = "Final"
        _wrote3 = insert_pregame_snapshot(_conn, _final, _prediction, _odds, "FINAL")
        print(f"Final-game guard: {_wrote3}  (False = correctly rejected)")

        _conn.close()
    print("Example complete.")
