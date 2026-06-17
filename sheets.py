from datetime import datetime, timedelta
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import pytz
import requests

from model import american_profit, grade_spread, grade_total


load_dotenv()
SHEETS_URL = os.getenv("SHEETS_URL", "")
SHEETS_SECRET = os.getenv("SHEETS_SECRET", "")
SHEET_NAME = os.getenv("SHEET_NAME", "predictions_v2")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Singapore")
tz = pytz.timezone(TIMEZONE)
AUDIT_PATH = Path(__file__).with_name("prediction_audit.jsonl")
RESULT_AUDIT_PATH = Path(__file__).with_name("result_audit.jsonl")

V3_START = 20
V3_FIELDS = (
    "schema_version",
    "quote_timestamp",
    "game_id",
    "model_version",
    "model_ready",
    "rl_edge_flagged",
    "total_price",
    "rl_side",
    "rl_point",
    "rl_price",
    "total_win_prob",
    "total_push_prob",
    "total_market_prob",
    "total_probability_edge",
    "total_ev",
    "rl_win_prob",
    "rl_push_prob",
    "rl_market_prob",
    "rl_probability_edge",
    "rl_ev",
    "total_agreement",
    "rl_agreement",
    "total_ensemble_std",
    "margin_ensemble_std",
    "data_quality",
    "total_bet_size",
    "rl_bet_size",
    "total_kelly_fraction",
    "rl_kelly_fraction",
    "bookmaker_count",
    "home_spread_line",
    "away_spread_line",
    "over_price",
    "under_price",
    "home_spread_price",
    "away_spread_price",
)


def _append_jsonl(path, payload):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _bool(value):
    return value is True or str(value).strip().lower() == "true"


def _float(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _int(value):
    try:
        return int(float(value)) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _game_date(game):
    return str(game.get("date") or datetime.now(tz).date())[:10]


def log_prediction(game, prediction, odds_entry):
    if not prediction:
        return
    game_string = f"{game['away_team']} @ {game['home_team']}"
    snapshot = {
        "logged_at": datetime.now(tz).isoformat(),
        "date": _game_date(game),
        "game": game_string,
        "game_id": game.get("game_id"),
        "odds": odds_entry,
        "prediction": prediction,
    }
    _append_jsonl(AUDIT_PATH, snapshot)
    if not SHEETS_URL:
        return

    legacy = [
        snapshot["date"],
        game_string,
        prediction.get("our_total"),
        prediction.get("total_line"),
        prediction.get("total_gap"),
        prediction.get("total_pred"),
        prediction.get("total_conf"),
        prediction.get("total_votes"),
        prediction.get("rl_pred"),
        prediction.get("rl_conf"),
        prediction.get("rl_votes"),
        prediction.get("edge_flagged"),
        prediction.get("has_real_pitchers"),
        None, None, None, None, None, None, None,
    ]
    v3_values = [
        "v3",
        prediction.get("quote_timestamp") or snapshot["logged_at"],
        game.get("game_id"),
        prediction.get("model_version"),
        prediction.get("model_ready"),
        prediction.get("rl_edge_flagged"),
        prediction.get("total_price"),
        prediction.get("rl_side"),
        prediction.get("rl_point"),
        prediction.get("rl_price"),
        prediction.get("total_win_prob"),
        prediction.get("total_push_prob"),
        prediction.get("total_market_prob"),
        prediction.get("total_probability_edge"),
        prediction.get("total_ev"),
        prediction.get("rl_win_prob"),
        prediction.get("rl_push_prob"),
        prediction.get("rl_market_prob"),
        prediction.get("rl_probability_edge"),
        prediction.get("rl_ev"),
        prediction.get("total_agreement"),
        prediction.get("rl_agreement"),
        prediction.get("total_ensemble_std"),
        prediction.get("margin_ensemble_std"),
        prediction.get("data_quality"),
        prediction.get("total_bet_size"),
        prediction.get("rl_bet_size"),
        prediction.get("total_kelly_fraction"),
        prediction.get("rl_kelly_fraction"),
        prediction.get("bookmaker_count"),
        (odds_entry or {}).get("spread_market", {}).get("home_point"),
        (odds_entry or {}).get("spread_market", {}).get("away_point"),
        (odds_entry or {}).get("total_market", {}).get("over_price"),
        (odds_entry or {}).get("total_market", {}).get("under_price"),
        (odds_entry or {}).get("spread_market", {}).get("home_price"),
        (odds_entry or {}).get("spread_market", {}).get("away_price"),
    ]
    try:
        response = requests.post(
            SHEETS_URL,
            json={
                "secret": SHEETS_SECRET,
                "action": "log_prediction",
                "sheet": SHEET_NAME,
                "row": legacy + v3_values,
            },
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        print(f"Error logging prediction: {exc}")


def get_results_date():
    et_now = datetime.now(pytz.timezone("America/New_York"))
    if et_now.hour < 23:
        return (et_now - timedelta(days=1)).strftime("%Y-%m-%d")
    return et_now.strftime("%Y-%m-%d")


def log_results(date=None):
    target_date = date or get_results_date()
    try:
        response = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "date": target_date,
                "gameType": "R",
                "hydrate": "linescore,team",
            },
            timeout=15,
        )
        response.raise_for_status()
        results = []
        for date_entry in response.json().get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get(
                    "abstractGameState"
                ) != "Final":
                    continue
                home = game["teams"]["home"]
                away = game["teams"]["away"]
                linescore = game.get("linescore", {}).get("teams", {})
                home_score = linescore.get("home", {}).get("runs")
                away_score = linescore.get("away", {}).get("runs")
                if home_score is None or away_score is None:
                    continue
                results.append({
                    "game_id": game.get("gamePk"),
                    "game": (
                        f"{away['team']['name']} @ "
                        f"{home['team']['name']}"
                    ),
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "total_result": int(home_score) + int(away_score),
                    "date": target_date,
                })
        return results
    except Exception as exc:
        print(f"Error fetching results: {exc}")
        return []


def _read_sheet_rows():
    if not SHEETS_URL:
        return []
    response = requests.get(
        SHEETS_URL,
        params={"sheet": SHEET_NAME},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("rows", []) if isinstance(data, dict) else data


def _read_local_snapshots(target_date):
    grouped = {}
    if not AUDIT_PATH.exists():
        return grouped
    with AUDIT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(payload.get("date"))[:10] != target_date:
                continue
            prediction = dict(payload.get("prediction") or {})
            odds = payload.get("odds") or {}
            spread = odds.get("spread_market") or {}
            prediction.update({
                "date": target_date,
                "game": payload.get("game"),
                "game_id": payload.get("game_id"),
                "home_spread_line": spread.get("home_point"),
                "away_spread_line": spread.get("away_point"),
                "over_price": (odds.get("total_market") or {}).get(
                    "over_price"
                ),
                "under_price": (odds.get("total_market") or {}).get(
                    "under_price"
                ),
                "home_spread_price": spread.get("home_price"),
                "away_spread_price": spread.get("away_price"),
            })
            grouped.setdefault(payload.get("game"), []).append(prediction)
    return grouped


def _read_local_results():
    results = {}
    if not RESULT_AUDIT_PATH.exists():
        return results
    with RESULT_AUDIT_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = (
                str(payload.get("date"))[:10],
                payload.get("game"),
            )
            results[key] = payload
    return results


def _parse_row(row):
    if len(row) < 12:
        return None
    parsed = {
        "date": str(row[0])[:10],
        "game": str(row[1]),
        "our_total": _float(row[2]),
        "total_line": _float(row[3]),
        "total_gap": _float(row[4]),
        "total_pred": str(row[5]),
        "total_conf": _float(row[6]) or 0.0,
        "total_votes": _int(row[7]) or 0,
        "rl_pred": str(row[8]),
        "rl_conf": _float(row[9]) or 0.0,
        "rl_votes": _int(row[10]) or 0,
        "edge_flagged": _bool(row[11]),
        "has_real_pitchers": _bool(row[12]) if len(row) > 12 else False,
        "home_score": _int(row[13]) if len(row) > 13 else None,
        "away_score": _int(row[14]) if len(row) > 14 else None,
        "total_result": _int(row[15]) if len(row) > 15 else None,
        "ou_result": str(row[16]) if len(row) > 16 else "",
        "ou_outcome": str(row[17]) if len(row) > 17 else "",
        "rl_result": str(row[18]) if len(row) > 18 else "",
        "rl_outcome": str(row[19]) if len(row) > 19 else "",
    }
    if len(row) > V3_START and str(row[V3_START]).lower() == "v3":
        for offset, field in enumerate(V3_FIELDS):
            index = V3_START + offset
            parsed[field] = row[index] if index < len(row) else None
        parsed.update({
            "model_ready": _bool(parsed.get("model_ready")),
            "rl_edge_flagged": _bool(parsed.get("rl_edge_flagged")),
            "game_id": _int(parsed.get("game_id")),
            "total_price": _int(parsed.get("total_price")),
            "rl_point": _float(parsed.get("rl_point")),
            "rl_price": _int(parsed.get("rl_price")),
            "total_win_prob": _float(parsed.get("total_win_prob")),
            "total_push_prob": _float(parsed.get("total_push_prob")),
            "total_market_prob": _float(parsed.get("total_market_prob")),
            "total_probability_edge": _float(
                parsed.get("total_probability_edge")
            ),
            "total_ev": _float(parsed.get("total_ev")),
            "rl_win_prob": _float(parsed.get("rl_win_prob")),
            "rl_push_prob": _float(parsed.get("rl_push_prob")),
            "rl_market_prob": _float(parsed.get("rl_market_prob")),
            "rl_probability_edge": _float(
                parsed.get("rl_probability_edge")
            ),
            "rl_ev": _float(parsed.get("rl_ev")),
            "total_agreement": _float(parsed.get("total_agreement")),
            "rl_agreement": _float(parsed.get("rl_agreement")),
            "data_quality": _float(parsed.get("data_quality")),
            "home_spread_line": _float(parsed.get("home_spread_line")),
            "away_spread_line": _float(parsed.get("away_spread_line")),
            "over_price": _int(parsed.get("over_price")),
            "under_price": _int(parsed.get("under_price")),
            "home_spread_price": _int(parsed.get("home_spread_price")),
            "away_spread_price": _int(parsed.get("away_spread_price")),
        })
    else:
        parsed["schema_version"] = "v2"
        parsed["quote_timestamp"] = f"{parsed['date']}T00:00:00"
        parsed["rl_edge_flagged"] = False
        parsed["rl_side"] = (
            "HOME" if parsed["rl_pred"].startswith("HOME") else "AWAY"
        )
        try:
            parsed["rl_point"] = float(parsed["rl_pred"].split()[-1])
        except (ValueError, IndexError):
            parsed["rl_point"] = None
    return parsed


def _merge_game_snapshots(snapshots):
    snapshots.sort(key=lambda item: str(item.get("quote_timestamp") or ""))
    latest = dict(snapshots[-1])
    for field in (
        "home_score", "away_score", "total_result",
        "ou_result", "ou_outcome", "rl_result", "rl_outcome",
    ):
        settled = next(
            (
                item.get(field)
                for item in reversed(snapshots)
                if item.get(field) not in (None, "")
            ),
            None,
        )
        if settled is not None:
            latest[field] = settled
    total_bet = next(
        (item for item in snapshots if item.get("edge_flagged")),
        None,
    )
    rl_bet = next(
        (item for item in snapshots if item.get("rl_edge_flagged")),
        None,
    )
    if total_bet:
        for field in (
            "total_pred", "total_line", "total_price", "total_conf",
            "total_win_prob", "total_push_prob", "total_market_prob",
            "total_probability_edge", "total_ev", "total_bet_size",
            "total_kelly_fraction",
        ):
            latest[field] = total_bet.get(field)
        latest["edge_flagged"] = True
        latest["total_bet_quote_timestamp"] = total_bet.get(
            "quote_timestamp"
        )
    if rl_bet:
        for field in (
            "rl_pred", "rl_side", "rl_point", "rl_price", "rl_conf",
            "rl_win_prob", "rl_push_prob", "rl_market_prob",
            "rl_probability_edge", "rl_ev", "rl_bet_size",
            "rl_kelly_fraction",
        ):
            latest[field] = rl_bet.get(field)
        latest["rl_edge_flagged"] = True
        latest["rl_bet_quote_timestamp"] = rl_bet.get("quote_timestamp")

    closing_total = snapshots[-1].get("total_line")
    latest["closing_total_line"] = closing_total
    if total_bet and closing_total is not None:
        if total_bet.get("total_pred") == "OVER":
            latest["closing_total_price"] = snapshots[-1].get("over_price")
            latest["total_clv_points"] = (
                closing_total - total_bet["total_line"]
            )
        else:
            latest["closing_total_price"] = snapshots[-1].get("under_price")
            latest["total_clv_points"] = (
                total_bet["total_line"] - closing_total
            )
    if rl_bet and rl_bet.get("rl_side") == "HOME":
        closing_rl = snapshots[-1].get("home_spread_line")
    elif rl_bet:
        closing_rl = snapshots[-1].get("away_spread_line")
    else:
        closing_rl = None
    latest["closing_rl_point"] = closing_rl
    if rl_bet and closing_rl is not None:
        latest["closing_rl_price"] = (
            snapshots[-1].get("home_spread_price")
            if rl_bet.get("rl_side") == "HOME"
            else snapshots[-1].get("away_spread_price")
        )
        latest["rl_clv_points"] = rl_bet["rl_point"] - closing_rl
    return latest


def get_stored_predictions(date):
    try:
        # Primary: local audit file has ALL prediction snapshots in
        # chronological order so _merge_game_snapshots correctly picks
        # the first edge-flagged prediction (the moment the alert fired).
        # The sheet only stores the last update per game (edge already closed).
        grouped = _read_local_snapshots(date)
        if not grouped:
            # Fallback: Google Sheets
            rows = _read_sheet_rows()
            for row in rows[1:]:
                parsed = _parse_row(row)
                if not parsed or parsed["date"] != date:
                    continue
                grouped.setdefault(parsed["game"], []).append(parsed)
        return {
            game: _merge_game_snapshots(snapshots)
            for game, snapshots in grouped.items()
        }
    except Exception as exc:
        print(f"Error getting stored predictions: {exc}")
        return {}


def update_results_in_sheet(results, date_override=None):
    results_date = date_override or get_results_date()
    predictions = get_stored_predictions(results_date)
    for result in results:
        prediction = predictions.get(result["game"])
        if not prediction:
            continue
        home_margin = result["home_score"] - result["away_score"]
        ou_outcome = (
            grade_total(
                result["total_result"],
                prediction.get("total_line"),
                prediction.get("total_pred"),
            )
            if prediction.get("edge_flagged") else "NO BET"
        )
        rl_outcome = (
            grade_spread(
                home_margin,
                prediction.get("rl_side"),
                prediction.get("rl_point"),
            )
            if prediction.get("rl_edge_flagged") else "NO BET"
        )
        payload = {
            "date": results_date,
            "game": result["game"],
            "game_id": result.get("game_id"),
            "home_score": result["home_score"],
            "away_score": result["away_score"],
            "total_result": result["total_result"],
            "ou_result": prediction.get("total_pred"),
            "ou_correct": ou_outcome,
            "rl_result": prediction.get("rl_pred"),
            "rl_correct": rl_outcome,
            "total_clv_points": prediction.get("total_clv_points"),
            "rl_clv_points": prediction.get("rl_clv_points"),
            "closing_total_line": prediction.get("closing_total_line"),
            "closing_total_price": prediction.get("closing_total_price"),
            "closing_rl_point": prediction.get("closing_rl_point"),
            "closing_rl_price": prediction.get("closing_rl_price"),
        }
        _append_jsonl(RESULT_AUDIT_PATH, payload)
        if not SHEETS_URL:
            continue
        try:
            response = requests.post(
                SHEETS_URL,
                json={
                    "secret": SHEETS_SECRET,
                    "action": "log_result",
                    "sheet": SHEET_NAME,
                    **payload,
                    "correct": rl_outcome,
                },
                timeout=30,
            )
            response.raise_for_status()
        except Exception as exc:
            print(f"Error updating result: {exc}")


def _normalize_outcome(value):
    value = str(value).strip().upper()
    if value in ("WIN", "W", "✅"):
        return "WIN"
    if value in ("LOSS", "L", "❌"):
        return "LOSS"
    if value in ("PUSH", "P"):
        return "PUSH"
    return None


def get_record():
    try:
        # Use local audit snapshots first (preserves flagged status even
        # after the Sheet row gets overwritten by a later closed-edge run).
        grouped = {}
        if AUDIT_PATH.exists():
            with AUDIT_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prediction = dict(payload.get("prediction") or {})
                    odds = payload.get("odds") or {}
                    spread = odds.get("spread_market") or {}
                    prediction.update({
                        "date": str(payload.get("date"))[:10],
                        "game": payload.get("game"),
                        "home_spread_line": spread.get("home_point"),
                        "away_spread_line": spread.get("away_point"),
                        "over_price": (
                            odds.get("total_market") or {}
                        ).get("over_price"),
                        "under_price": (
                            odds.get("total_market") or {}
                        ).get("under_price"),
                        "home_spread_price": spread.get("home_price"),
                        "away_spread_price": spread.get("away_price"),
                    })
                    key = (prediction["date"], prediction["game"])
                    grouped.setdefault(key, []).append(prediction)
        if not grouped:
            rows = _read_sheet_rows()
            for row in rows[1:]:
                parsed = _parse_row(row)
                if parsed:
                    grouped.setdefault(
                        (parsed["date"], parsed["game"]), []
                    ).append(parsed)
        if not grouped and AUDIT_PATH.exists():
            with AUDIT_PATH.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    prediction = dict(payload.get("prediction") or {})
                    odds = payload.get("odds") or {}
                    spread = odds.get("spread_market") or {}
                    prediction.update({
                        "date": str(payload.get("date"))[:10],
                        "game": payload.get("game"),
                        "home_spread_line": spread.get("home_point"),
                        "away_spread_line": spread.get("away_point"),
                        "over_price": (
                            odds.get("total_market") or {}
                        ).get("over_price"),
                        "under_price": (
                            odds.get("total_market") or {}
                        ).get("under_price"),
                        "home_spread_price": spread.get("home_price"),
                        "away_spread_price": spread.get("away_price"),
                    })
                    key = (prediction["date"], prediction["game"])
                    grouped.setdefault(key, []).append(prediction)

        local_results = _read_local_results()

        wins = losses = pushes = 0
        staked = profit = 0.0
        for key, snapshots in grouped.items():
            merged = _merge_game_snapshots(snapshots)
            local_result = local_results.get(key)
            if local_result:
                merged["ou_outcome"] = local_result.get("ou_correct")
                merged["rl_outcome"] = local_result.get("rl_correct")
            for flagged, outcome_key, price_key in (
                (
                    merged.get("edge_flagged"),
                    "ou_outcome",
                    "total_price",
                ),
                (
                    merged.get("rl_edge_flagged"),
                    "rl_outcome",
                    "rl_price",
                ),
            ):
                if not flagged:
                    continue
                outcome = _normalize_outcome(merged.get(outcome_key))
                if not outcome:
                    continue
                price = merged.get(price_key)
                staked += 1.0
                if outcome == "WIN":
                    wins += 1
                    profit += american_profit(price) or 0.0
                elif outcome == "LOSS":
                    losses += 1
                    profit -= 1.0
                else:
                    pushes += 1
        decisions = wins + losses
        return {
            "settled_bets": wins + losses + pushes,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "hit_rate": round(wins / decisions * 100, 1) if decisions else 0.0,
            "roi": round(profit / staked * 100, 1) if staked else 0.0,
            "profit_units": round(profit, 3),
        }
    except Exception as exc:
        print(f"Error getting record: {exc}")
        return None


def grade_user_bets(date, results):
    """Grade pending mlb_v2 user_bets rows for given date using results list."""
    if not SHEETS_URL:
        return []
    from model import grade_total, grade_spread
    try:
        r = requests.get(SHEETS_URL, params={"sheet": "user_bets"}, timeout=30)
        rows = r.json().get("rows", [])
    except Exception as exc:
        print(f"Error reading user_bets: {exc}")
        return []
    if not rows:
        return []
    graded = []
    for row in rows[1:]:
        if len(row) < 6:
            continue
        row_date, sport, home_team, away_team, market, pick = row[0], row[1], row[2], row[3], row[4], row[5]
        existing_result = row[8] if len(row) > 8 else ""
        if sport != "mlb_v2" or str(row_date)[:10] != date or existing_result:
            continue
        match = next((res for res in results
                       if away_team in res["game"] and home_team in res["game"]), None)
        if not match:
            continue
        home_margin = match["home_score"] - match["away_score"]
        try:
            if market.upper() == "OU":
                direction, line = pick.split()
                outcome = grade_total(match["total_result"], float(line), direction.upper())
            else:
                side, point = pick.split()
                outcome = grade_spread(home_margin, side.upper(), float(point))
        except Exception:
            continue
        payload = {
            "secret": SHEETS_SECRET, "action": "update_user_bet", "sheet": "user_bets",
            "date": row_date, "sport": sport, "home_team": home_team, "away_team": away_team,
            "market": market, "pick": pick,
            "result": outcome, "home_score": match["home_score"],
            "away_score": match["away_score"], "total_result": match["total_result"],
            "home_margin": home_margin
        }
        try:
            requests.post(SHEETS_URL, json=payload, timeout=30)
            graded.append((away_team, home_team, market, pick, outcome))
        except Exception as exc:
            print(f"Error grading user bet: {exc}")
    return graded


def get_user_bets(sport="mlb_v2", days=None):
    """Return user_bets rows for given sport. If days is None, return all."""
    if not SHEETS_URL:
        return []
    try:
        r = requests.get(SHEETS_URL, params={"sheet": "user_bets"}, timeout=30)
        rows = r.json().get("rows", [])
    except Exception as exc:
        print(f"Error reading user_bets: {exc}")
        return []
    if not rows:
        return []
    out = [row for row in rows[1:] if len(row) >= 6 and row[1] == sport]
    if days is None:
        return out
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    return [row for row in out if str(row[0])[:10] >= cutoff]


def units_won(hk_odds, stake, result):
    """HK odds: profit = stake * hk_odds on WIN, -stake on LOSS, 0 on PUSH."""
    try:
        hk_odds = float(hk_odds)
        stake = float(stake)
    except (TypeError, ValueError):
        return 0.0
    if result == "WIN":
        return stake * hk_odds
    if result == "LOSS":
        return -stake
    return 0.0
