from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
from datetime import datetime

tz = pytz.timezone("Asia/Singapore")

async def scheduled_brief(app):
    try:
        from bot import send_message, fetch_all_games, format_summary
        from config import ODDS_API_KEY
        now = datetime.now(tz).strftime("%b %d, %Y")
        games, games_data = await fetch_all_games(ODDS_API_KEY)
        if not games_data:
            return
        edge_count = sum(1 for _, p, _ in games_data
                         if p and (p.get("edge_flagged") or
                                   p.get("rl_edge_flagged")))
        if edge_count > 0:
            await send_message(app, format_summary(games_data, now))
        else:
            await send_message(app,
                f"⚾ MLB Edge V2 — {now}\n\n"
                f"{len(games_data)} games — no edges flagged.")
    except Exception as e:
        print(f"V2 Brief error: {e}", flush=True)

async def check_new_lines(app):
    try:
        from bot import send_message, fetch_all_games
        from config import ODDS_API_KEY
        games, games_data = await fetch_all_games(ODDS_API_KEY)
        if not games_data:
            return
        edges = [(g, p, o) for g, p, o in games_data
                 if p and (p.get("edge_flagged") or p.get("rl_edge_flagged"))]
        if edges:
            now = datetime.now(tz).strftime("%b %d, %Y")
            msg = f"🆕 V2 New Edges — {now}\n{len(edges)} edge(s)\n━━━━━━━━━━━━━━━━━━━━\n"
            for game, pred, odds in edges[:5]:
                total = odds.get("total") if odds else "N/A"
                msg += (f"⚾ {game['away_team'].split()[-1]} @ "
                        f"{game['home_team'].split()[-1]}\n"
                        f"   O/U: {total} | RL: {pred.get('rl_pred')} "
                        f"{pred.get('rl_conf')}%\n")
            msg += "\nRun /v2_brief for full predictions"
            await send_message(app, msg)
    except Exception as e:
        print(f"V2 New lines error: {e}", flush=True)

async def evening_results(app):
    try:
        from bot import send_message
        from sheets import log_results, update_results_in_sheet, get_results_date, get_stored_predictions
        from config import ODDS_API_KEY
        now = datetime.now(tz).strftime("%b %d, %Y")
        results_date = get_results_date()
        results = log_results()
        if not results:
            return
        update_results_in_sheet(results, date_override=results_date)
        stored_preds = get_stored_predictions(results_date)
        flagged = [(r, stored_preds.get(r["game"]))
                   for r in results
                   if stored_preds.get(r["game"]) and
                   (stored_preds[r["game"]].get("edge_flagged") or
                    stored_preds[r["game"]].get("rl_edge_flagged"))]
        if not flagged:
            return
        correct = 0
        for r, pred in flagged:
            rl_pred = pred.get("rl_pred", "")
            margin = r["home_score"] - r["away_score"]
            if rl_pred == "HOME -1.5" and margin > 1.5:
                correct += 1
            elif rl_pred == "HOME +1.5" and margin >= -1.5:
                correct += 1
            elif rl_pred == "AWAY +1.5" and margin <= 1.5:
                correct += 1
            elif rl_pred == "AWAY -1.5" and margin < -1.5:
                correct += 1
        msg = f"🌙 V2 Results — {now}\n━━━━━━━━━━━━━━━━━━━━\n"
        for r, pred in flagged:
            rl_pred = pred.get("rl_pred", "")
            margin = r["home_score"] - r["away_score"]
            ok = ((rl_pred == "HOME -1.5" and margin > 1.5) or
                  (rl_pred == "HOME +1.5" and margin >= -1.5) or
                  (rl_pred == "AWAY +1.5" and margin <= 1.5) or
                  (rl_pred == "AWAY -1.5" and margin < -1.5))
            emoji = "✅" if ok else "❌"
            msg += (f"{emoji} {r['game']}\n"
                    f"   {r['away_score']}-{r['home_score']} | "
                    f"RL: {rl_pred}\n")
        msg += (f"━━━━━━━━━━━━━━━━━━━━\n"
                f"V2 RL: {correct}/{len(flagged)} correct\n"
                f"Run /v2_record for full stats")
        await send_message(app, msg)
    except Exception as e:
        print(f"V2 Evening results error: {e}", flush=True)

async def nightly_retrain(app):
    try:
        from bot import send_message
        import os
        if os.path.exists("models.pkl"):
            age = (datetime.now(tz).timestamp() -
                   os.path.getmtime("models.pkl")) / 86400
            if age > 7:
                from historical import train_on_historical
                train_on_historical()
                await send_message(app, "🧠 V2 Model retrained")
    except Exception as e:
        print(f"V2 Retrain error: {e}", flush=True)

def setup_scheduler(app):
    scheduler = AsyncIOScheduler(timezone=tz)

    # Daily briefs — same times as v1
    for hour, name in [(23, "11PM"), (1, "1AM"),
                       (6, "6AM"), (10, "10AM")]:
        scheduler.add_job(scheduled_brief, CronTrigger(
            hour=hour, minute=5, timezone=tz),
            args=[app], id=f"v2_brief_{hour}",
            name=f"V2 Brief {name}")

    # New lines check every 3 hours (offset by 30min from v1)
    for hour in [0, 3, 6, 9, 12, 15, 18, 21]:
        scheduler.add_job(check_new_lines, CronTrigger(
            hour=hour, minute=30, timezone=tz),
            args=[app], id=f"v2_lines_{hour}",
            name=f"V2 Lines {hour}:30")

    # Evening results 8:15PM SGT
    scheduler.add_job(evening_results, CronTrigger(
        hour=20, minute=15, timezone=tz),
        args=[app], id="v2_evening_results",
        name="V2 Evening Results")

    # Nightly retrain 3AM SGT
    scheduler.add_job(nightly_retrain, CronTrigger(
        hour=3, minute=0, timezone=tz),
        args=[app], id="v2_nightly_retrain",
        name="V2 Nightly Retrain")

    return scheduler
