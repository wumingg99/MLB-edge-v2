from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz


tz = pytz.timezone("Asia/Singapore")


async def scheduled_brief(app):
    try:
        from bot import fetch_all_games, format_summary, send_message
        from config import ODDS_API_KEY

        _, games_data = await fetch_all_games(ODDS_API_KEY)
        if games_data:
            now = datetime.now(tz).strftime("%b %d, %Y")
            await send_message(app, format_summary(games_data, now))
    except Exception as exc:
        print(f"V3 brief error: {exc}", flush=True)


async def check_new_lines(app):
    try:
        from bot import fetch_all_games, format_summary, send_message
        from config import ODDS_API_KEY

        _, games_data = await fetch_all_games(
            ODDS_API_KEY, force_refresh=True
        )
        edges = [
            item
            for item in games_data
            if item[1]
            and (
                item[1].get("edge_flagged")
                or item[1].get("rl_edge_flagged")
            )
        ]
        if edges:
            now = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
            await send_message(app, format_summary(games_data, now))
    except Exception as exc:
        print(f"V3 line refresh error: {exc}", flush=True)


async def evening_results(app):
    try:
        from bot import send_message
        from model import grade_spread, grade_total
        from sheets import (
            get_results_date,
            get_stored_predictions,
            log_results,
            update_results_in_sheet,
        )

        results_date = get_results_date()
        results = log_results()
        if not results:
            return
        update_results_in_sheet(results, date_override=results_date)
        predictions = get_stored_predictions(results_date)
        lines = [f"V3 Results - {results_date}"]
        settled = 0
        for result in results:
            prediction = predictions.get(result["game"])
            if not prediction:
                continue
            if prediction.get("edge_flagged"):
                outcome = grade_total(
                    result["total_result"],
                    prediction.get("total_line"),
                    prediction.get("total_pred"),
                )
                lines.append(
                    f"{result['game']} O/U "
                    f"{prediction.get('total_pred')}: {outcome}"
                )
                settled += 1
            if prediction.get("rl_edge_flagged"):
                outcome = grade_spread(
                    result["home_score"] - result["away_score"],
                    prediction.get("rl_side"),
                    prediction.get("rl_point"),
                )
                lines.append(
                    f"{result['game']} RL "
                    f"{prediction.get('rl_pred')}: {outcome}"
                )
                settled += 1
        if settled:
            await send_message(app, "\n".join(lines))
    except Exception as exc:
        print(f"V3 results error: {exc}", flush=True)


def setup_scheduler(app):
    scheduler = AsyncIOScheduler(timezone=tz)
    for hour in (23, 1, 6, 10):
        scheduler.add_job(
            scheduled_brief,
            CronTrigger(hour=hour, minute=5, timezone=tz),
            args=[app],
            id=f"v3_brief_{hour}",
        )
    for hour in (0, 3, 6, 9, 12, 15, 18, 21):
        scheduler.add_job(
            check_new_lines,
            CronTrigger(hour=hour, minute=30, timezone=tz),
            args=[app],
            id=f"v3_lines_{hour}",
        )
    scheduler.add_job(
        evening_results,
        CronTrigger(hour=20, minute=15, timezone=tz),
        args=[app],
        id="v3_evening_results",
    )
    return scheduler
