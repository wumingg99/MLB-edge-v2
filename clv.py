"""
CLV (Closing Line Value) computation.

Reads paired EARLY + FINAL pregame_snapshots and writes one clv_results row
per (game_id, game_date, model_version, market).

Design:
- Uses stored consensus no-vig fields (total_market_prob / rl_market_prob)
  at both EARLY and FINAL.  Both are computed the same way (multi-book
  average), so the comparison is apples-to-apples regardless of which
  bookmaker's price happened to be best on either day.
- If the model's recommended side flipped between EARLY and FINAL, CLV is
  still computed for the committed EARLY-backed side via (1.0 - final_prob).
  The two sides of a de-vigged market always sum to exactly 1.0.
- Total CLV is split by line_moved in clv_summary.  When the total line
  itself shifts (e.g. 8.5 -> 9.0), the EARLY and FINAL market_prob values
  reference different bets and clv_points is not directly comparable.
  The headline CLV number covers same-line games only.
- RL CLV is computed only for the standard ±1.5 run line.  The audit log
  contains two anomalous entries at ±4.5 / ±8.5 (alternate spreads that
  slipped past the market normalizer); those are skipped.
"""
import logging
from datetime import datetime, timezone

from clv_kelly import closing_line_ev as _clev
from store import _date_only, _now_utc

logger = logging.getLogger(__name__)

_RL_STANDARD_POINT = 1.5


# ---------------------------------------------------------------------------
# Per-market row builders (pure; return None when data is insufficient)
# ---------------------------------------------------------------------------

def _total_row(early, final_):
    early_pred    = early.get("total_pred")
    early_no_vig  = early.get("total_market_prob")
    if not early_pred or early_no_vig is None:
        return None

    final_pred   = final_.get("total_pred")
    final_mprob  = final_.get("total_market_prob")
    if final_mprob is None:
        return None

    # Align closing prob to the EARLY-backed side
    closing_no_vig = (
        final_mprob if final_pred == early_pred else 1.0 - final_mprob
    )

    early_price = (
        early.get("total_over_price")
        if early_pred == "OVER"
        else early.get("total_under_price")
    )

    early_line   = early.get("total_line")
    closing_line = final_.get("total_line")
    line_moved   = int(
        early_line is not None
        and closing_line is not None
        and early_line != closing_line
    )

    clv_pts = closing_no_vig - early_no_vig
    clev    = (
        _clev(float(early_price), closing_no_vig)
        if early_price is not None else None
    )

    return {
        "market":             "TOTAL",
        "backed_side":        early_pred,
        "early_price":        early_price,
        "early_no_vig":       early_no_vig,
        "closing_no_vig":     closing_no_vig,
        "clv_points":         clv_pts,
        "closing_line_ev":    clev,
        "early_total_line":   early_line,
        "closing_total_line": closing_line,
        "line_moved":         line_moved,
    }


def _rl_row(early, final_):
    early_side   = early.get("rl_side")
    early_no_vig = early.get("rl_market_prob")
    early_point  = early.get("rl_point")
    if not early_side or early_no_vig is None or early_point is None:
        return None
    if abs(float(early_point)) != _RL_STANDARD_POINT:
        return None

    final_side  = final_.get("rl_side")
    final_mprob = final_.get("rl_market_prob")
    if final_mprob is None:
        return None

    closing_no_vig = (
        final_mprob if final_side == early_side else 1.0 - final_mprob
    )
    early_price = early.get("rl_price")

    clv_pts = closing_no_vig - early_no_vig
    clev    = (
        _clev(float(early_price), closing_no_vig)
        if early_price is not None else None
    )

    return {
        "market":             "RL",
        "backed_side":        early_side,
        "early_price":        early_price,
        "early_no_vig":       early_no_vig,
        "closing_no_vig":     closing_no_vig,
        "clv_points":         clv_pts,
        "closing_line_ev":    clev,
        "early_total_line":   None,
        "closing_total_line": None,
        "line_moved":         0,
    }


# ---------------------------------------------------------------------------
# Write helper
# ---------------------------------------------------------------------------

def _write_clv_row(conn, game_id, game_date, model_version, row, computed_at):
    conn.execute(
        """
        INSERT OR REPLACE INTO clv_results
            (game_id, game_date, model_version, market, backed_side,
             early_price, early_no_vig, closing_no_vig, clv_points,
             closing_line_ev, early_total_line, closing_total_line,
             line_moved, computed_at)
        VALUES
            (:game_id, :game_date, :model_version, :market, :backed_side,
             :early_price, :early_no_vig, :closing_no_vig, :clv_points,
             :closing_line_ev, :early_total_line, :closing_total_line,
             :line_moved, :computed_at)
        """,
        {
            "game_id":       game_id,
            "game_date":     _date_only(game_date),
            "model_version": model_version,
            "computed_at":   computed_at,
            **row,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_clv(conn):
    """
    Find all (game_id, game_date, model_version) triples that have both EARLY
    and FINAL snapshots.  Compute and insert CLV rows for TOTAL and RL markets.

    Fail-safe per game: an exception for one game is logged and skipped;
    the rest of the batch completes normally.

    Returns the number of clv_results rows inserted or replaced.
    """
    pairs = conn.execute(
        """
        SELECT
            e.game_id, e.game_date, e.model_version,
            e.id AS early_id, f.id AS final_id
        FROM pregame_snapshots e
        JOIN pregame_snapshots f
          ON  e.game_id       = f.game_id
          AND e.game_date     = f.game_date
          AND e.model_version = f.model_version
        WHERE e.stage = 'EARLY'
          AND f.stage = 'FINAL'
        ORDER BY e.game_date DESC, e.game_id
        """
    ).fetchall()

    computed_at = _now_utc()
    written = 0

    for pair in pairs:
        game_id       = pair["game_id"]
        game_date     = pair["game_date"]
        model_version = pair["model_version"]
        try:
            early  = dict(conn.execute(
                "SELECT * FROM pregame_snapshots WHERE id = ?",
                (pair["early_id"],),
            ).fetchone())
            final_ = dict(conn.execute(
                "SELECT * FROM pregame_snapshots WHERE id = ?",
                (pair["final_id"],),
            ).fetchone())

            for row_builder in (_total_row, _rl_row):
                row = row_builder(early, final_)
                if row is not None:
                    _write_clv_row(
                        conn, game_id, game_date, model_version,
                        row, computed_at,
                    )
                    written += 1

            conn.commit()
        except Exception as exc:
            logger.error(
                "CLV compute error game_id=%s game_date=%s: %s",
                game_id, game_date, exc,
            )

    return written


def format_clv_digest(summary, date_label=None):
    """
    Format a Telegram-ready CLV digest from clv_summary() output.

    Returns None when there is no data to report (both TOTAL same-line n=0
    and RL n=0) — the caller skips the send in that case.

    date_label overrides the auto-derived date (useful in tests).
    """
    if date_label is None:
        date_label = datetime.now(timezone.utc).strftime("%b %-d")

    tsl = summary.get("total_same_line") or {}
    rl  = summary.get("rl")             or {}
    tlm = summary.get("total_line_moved") or {}

    if not (tsl.get("n") or 0) + (rl.get("n") or 0):
        return None

    def _mkt_line(label, bucket):
        n = bucket.get("n") or 0
        if n == 0:
            return f"{label}: (no data yet)"
        avg = bucket.get("avg_clv_points")
        pct = bucket.get("pct_positive")
        avg_str = f"{avg * 100:+.1f} pts" if avg is not None else "n/a"
        pct_str = f"{pct:.0%}+"           if pct is not None else "n/a"
        return f"{label}: avg {avg_str} | {pct_str} | n={n}"

    lm_n = tlm.get("n") or 0
    return "\n".join([
        f"\U0001f4ca CLV Daily Update — {date_label}",
        "",
        _mkt_line("TOTAL (same-line)", tsl),
        _mkt_line("RL", rl),
        f"Line-moved totals: n={lm_n} (not comparable, excluded)",
    ])


def clv_summary(conn):
    """
    Return CLV statistics grouped by market and, for totals, by line_moved.

    Headline numbers are in total_same_line (where early and closing total
    lines match — directly comparable bets).  Moved-line totals are reported
    separately, labeled not_directly_comparable.

    Returns a dict with keys:
        total_same_line, total_line_moved, rl, overall
    Each value is a dict with keys:
        n, avg_clv_points, avg_closing_line_ev, pct_positive
    total_line_moved also has "note": "not_directly_comparable".
    """
    bucket_rows = conn.execute(
        """
        SELECT market, line_moved,
               COUNT(*) AS n,
               AVG(clv_points)        AS avg_clv_points,
               AVG(closing_line_ev)   AS avg_clev,
               SUM(CASE WHEN clv_points > 0 THEN 1 ELSE 0 END) AS n_positive
        FROM clv_results
        GROUP BY market, line_moved
        ORDER BY market, line_moved
        """
    ).fetchall()

    def _fmt(row, note=None):
        n = row["n"]
        d = {
            "n":                   n,
            "avg_clv_points":      round(row["avg_clv_points"], 5)
                                   if row["avg_clv_points"] is not None else None,
            "avg_closing_line_ev": round(row["avg_clev"], 5)
                                   if row["avg_clev"] is not None else None,
            "pct_positive":        round(row["n_positive"] / n, 3) if n > 0 else None,
        }
        if note:
            d["note"] = note
        return d

    summary = {
        "total_same_line":  None,
        "total_line_moved": None,
        "rl":               None,
        "overall":          None,
    }

    for row in bucket_rows:
        market = row["market"]
        lm     = row["line_moved"]
        if market == "TOTAL" and lm == 0:
            summary["total_same_line"] = _fmt(row)
        elif market == "TOTAL" and lm == 1:
            summary["total_line_moved"] = _fmt(row, note="not_directly_comparable")
        elif market == "RL":
            summary["rl"] = _fmt(row)

    overall = conn.execute(
        """
        SELECT COUNT(*) AS n,
               AVG(clv_points)        AS avg_clv_points,
               AVG(closing_line_ev)   AS avg_clev,
               SUM(CASE WHEN clv_points > 0 THEN 1 ELSE 0 END) AS n_positive
        FROM clv_results
        """
    ).fetchone()
    if overall and overall["n"] > 0:
        summary["overall"] = _fmt(overall)

    return summary
