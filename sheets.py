import requests
from datetime import datetime, timedelta
import pytz
import os
from dotenv import load_dotenv
load_dotenv()

SHEETS_URL = os.getenv("SHEETS_URL", "")
SHEETS_SECRET = os.getenv("SHEETS_SECRET", "")
SHEET_NAME = os.getenv("SHEET_NAME", "predictions_v2")
TIMEZONE = os.getenv("TIMEZONE", "Asia/Singapore")
tz = pytz.timezone(TIMEZONE)

def log_prediction(game, prediction, odds_entry):
    if not SHEETS_URL or not prediction:
        return
    try:
        today = datetime.now(tz).strftime("%Y-%m-%d")
        game_str = f"{game['away_team']} @ {game['home_team']}"
        total = odds_entry.get("total") if odds_entry else None
        row = [
            today, game_str,
            prediction.get("our_total"), total,
            prediction.get("total_gap"),
            prediction.get("total_pred"),
            prediction.get("total_conf"),
            prediction.get("total_votes"),
            prediction.get("rl_pred"),
            prediction.get("rl_conf"),
            prediction.get("rl_votes"),
            prediction.get("edge_flagged"),
            prediction.get("has_real_pitchers"),
            None, None, None, None, None, None, None
        ]
        requests.post(SHEETS_URL, json={
            "secret": SHEETS_SECRET,
            "action": "log_prediction",
            "sheet": SHEET_NAME,
            "row": row
        }, timeout=30)
    except Exception as e:
        print(f"Error logging prediction: {e}")

def get_results_date():
    et_tz = pytz.timezone("America/New_York")
    et_now = datetime.now(et_tz)
    tz_now = datetime.now(tz)
    today_et = et_now.strftime("%Y-%m-%d")
    yesterday_et = (et_now - timedelta(days=1)).strftime("%Y-%m-%d")
    if et_now.hour < 23:
        return yesterday_et
    return today_et

def log_results(date=None):
    if not SHEETS_URL:
        return []
    try:
        et_tz = pytz.timezone("America/New_York")
        et_now = datetime.now(et_tz)
        target_date = date or get_results_date()
        url = "https://statsapi.mlb.com/api/v1/schedule"
        params = {
            "sportId": 1, "date": target_date,
            "gameType": "R",
            "hydrate": "linescore,team"
        }
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        results = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get(
                        "abstractGameState") != "Final":
                    continue
                home = game["teams"]["home"]
                away = game["teams"]["away"]
                linescore = game.get("linescore", {})
                home_score = linescore.get("teams", {}).get(
                    "home", {}).get("runs")
                away_score = linescore.get("teams", {}).get(
                    "away", {}).get("runs")
                if home_score is None or away_score is None:
                    continue
                results.append({
                    "game": f"{away['team']['name']} @ {home['team']['name']}",
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "total_result": int(home_score) + int(away_score),
                    "date": target_date,
                })
        return results
    except Exception as e:
        print(f"Error fetching results: {e}")
        return []

def get_stored_predictions(date):
    if not SHEETS_URL:
        return {}
    try:
        r = requests.get(SHEETS_URL,
            params={"sheet": SHEET_NAME}, timeout=30)
        data = r.json()
        rows = data.get("rows", []) if isinstance(data, dict) else data
        preds = {}
        for row in rows[1:]:
            if len(row) < 12:
                continue
            stored_date = str(row[0])[:10]
            if stored_date != date:
                continue
            game = str(row[1])
            try:
                preds[game] = {
                    "total_pred": str(row[5]),
                    "total_conf": float(row[6] or 0),
                    "total_votes": int(row[7] or 0),
                    "rl_pred": str(row[8]),
                    "rl_conf": float(row[9] or 0),
                    "rl_votes": int(row[10] or 0),
                    "edge_flagged": str(row[11]) == "True" or row[11] is True,
                    "rl_edge_flagged": (float(row[9] or 0) >= 65.0 and
                                        int(row[10] or 0) >= 3),
                    "open_total": float(row[3]) if row[3] else None,
                }
            except Exception:
                continue
        return preds
    except Exception as e:
        print(f"Error getting stored predictions: {e}")
        return {}

def update_results_in_sheet(results, date_override=None):
    if not SHEETS_URL:
        return
    results_date = date_override or get_results_date()
    stored_preds = get_stored_predictions(results_date)

    for result in results:
        pred = stored_preds.get(result["game"])
        if not pred:
            continue
        if not (pred.get("edge_flagged") or pred.get("rl_edge_flagged")):
            continue
        home_score = result["home_score"]
        away_score = result["away_score"]
        total_result = result["total_result"]
        open_total = pred.get("open_total") or 8.5
        ou_result = "OVER" if total_result > open_total else "UNDER"
        ou_correct = "✅" if pred.get("total_pred") == ou_result else "❌"
        rl_pred = pred.get("rl_pred", "")
        home_margin = home_score - away_score
        if rl_pred == "HOME -1.5":
            rl_correct = "✅" if home_margin > 1.5 else "❌"
        elif rl_pred == "HOME +1.5":
            rl_correct = "✅" if home_margin >= -1.5 else "❌"
        elif rl_pred == "AWAY +1.5":
            rl_correct = "✅" if home_margin <= 1.5 else "❌"
        elif rl_pred == "AWAY -1.5":
            rl_correct = "✅" if home_margin < -1.5 else "❌"
        else:
            rl_correct = "❌"
        try:
            r = requests.post(SHEETS_URL, json={
                "secret": SHEETS_SECRET,
                "action": "log_result",
                "sheet": SHEET_NAME,
                "date": results_date,
                "game": result["game"],
                "home_score": home_score,
                "away_score": away_score,
                "total_result": total_result,
                "ou_result": ou_result,
                "ou_correct": ou_correct,
                "rl_result": rl_pred,
                "rl_correct": rl_correct,
                "correct": rl_correct
            }, timeout=30)
            print(f"Result: {result['game']} | "
                  f"RL: {rl_pred} {rl_correct} | "
                  f"O/U: {ou_result} {ou_correct}")
        except Exception as e:
            print(f"Error updating result: {e}")

def get_record():
    if not SHEETS_URL:
        return None
    try:
        r = requests.get(SHEETS_URL,
            params={"sheet": SHEET_NAME}, timeout=30)
        data = r.json()
        rows = data.get("rows", []) if isinstance(data, dict) else data
        if len(rows) <= 1:
            return None
        rl_total = rl_correct_count = 0
        ou_total = ou_correct_count = 0
        rl_flagged_total = rl_flagged_correct = 0
        ou_flagged_total = ou_flagged_correct = 0
        monthly = {}
        for row in rows[1:]:
            if len(row) < 19:
                continue
            date = str(row[0])[:7]
            rl_corr = str(row[19]) if len(row) > 19 else ""
            ou_corr = str(row[17]) if len(row) > 17 else ""
            is_rl_flagged = (float(row[9] or 0) >= 65.0 and
                             int(row[10] or 0) >= 3)
            is_ou_flagged = (str(row[11]) == "True" or row[11] is True)
            if rl_corr in ["✅", "❌"]:
                rl_total += 1
                if rl_corr == "✅":
                    rl_correct_count += 1
                if is_rl_flagged:
                    rl_flagged_total += 1
                    if rl_corr == "✅":
                        rl_flagged_correct += 1
            if ou_corr in ["✅", "❌"]:
                ou_total += 1
                if ou_corr == "✅":
                    ou_correct_count += 1
                if is_ou_flagged:
                    ou_flagged_total += 1
                    if ou_corr == "✅":
                        ou_flagged_correct += 1
            if date not in monthly:
                monthly[date] = {
                    "rl": 0, "rl_correct": 0,
                    "ou": 0, "ou_correct": 0
                }
            if rl_corr in ["✅", "❌"] and is_rl_flagged:
                monthly[date]["rl"] += 1
                if rl_corr == "✅":
                    monthly[date]["rl_correct"] += 1
            if ou_corr in ["✅", "❌"] and is_ou_flagged:
                monthly[date]["ou"] += 1
                if ou_corr == "✅":
                    monthly[date]["ou_correct"] += 1
        return {
            "rl_total": rl_total,
            "rl_correct": rl_correct_count,
            "rl_accuracy": round(
                rl_correct_count / rl_total * 100, 1) if rl_total > 0 else 0,
            "ou_total": ou_total,
            "ou_correct": ou_correct_count,
            "ou_accuracy": round(
                ou_correct_count / ou_total * 100, 1) if ou_total > 0 else 0,
            "rl_flagged_total": rl_flagged_total,
            "rl_flagged_correct": rl_flagged_correct,
            "rl_flagged_accuracy": round(
                rl_flagged_correct / rl_flagged_total * 100, 1
            ) if rl_flagged_total > 0 else 0,
            "ou_flagged_total": ou_flagged_total,
            "ou_flagged_correct": ou_flagged_correct,
            "ou_flagged_accuracy": round(
                ou_flagged_correct / ou_flagged_total * 100, 1
            ) if ou_flagged_total > 0 else 0,
            "total": rl_total + ou_total,
            "monthly": monthly
        }
    except Exception as e:
        print(f"Error getting record: {e}")
        return None
