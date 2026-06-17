from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
import pytz


tz = pytz.timezone("Asia/Singapore")

# Minutes before first pitch to capture the closing line.
CLOSE_OFFSET_MIN = 35
# Games whose capture targets fall within this window share one API refresh.
COALESCE_WINDOW_MIN = 15


_BRIEF_PURPOSE = {
    23: ("\U0001F3AF", "RL Action Window"),
    1:  ("\U0001F4B0", "Totals Action Window"),
    6:  ("\U0001F440", "Watch \u2014 Line Check"),
    10: ("\U0001F440", "Watch \u2014 Final Check"),
}


async def scheduled_brief(app):
    try:
        from bot import fetch_all_games, format_summary, send_message
        from config import ODDS_API_KEY

        _, games_data = await fetch_all_games(ODDS_API_KEY)
        if games_data:
            sgt_hour = datetime.now(tz).hour
            emoji, purpose = _BRIEF_PURPOSE.get(
                sgt_hour, ("\U0001F319", "Snapshot")
            )
            et_date = datetime.now(pytz.timezone("America/New_York")).strftime("%b %d")
            now = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
            header = f"{emoji} {purpose} ({et_date} ET games) \u2014 posted {now}"
            await send_message(app, format_summary(games_data, header))
    except Exception as exc:
        print(f"V3 brief error: {exc}", flush=True)


async def check_new_lines(app):
    try:
        from bot import fetch_all_games, format_summary, send_message
        from config import ODDS_API_KEY

        _, games_data = await fetch_all_games(
            ODDS_API_KEY, force_refresh=True
        )
        new_edges = [
            item
            for item in games_data
            if item[1]
            and (
                item[1].get("edge_flagged")
                or item[1].get("rl_edge_flagged")
            )
            and item[1].get("_newly_logged")
        ]
        if new_edges:
            now = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
            header = f"\U0001F514 New Edge Alert \u2014 posted {now}"
            await send_message(app, format_summary(games_data, header))
    except Exception as exc:
        print(f"V3 line refresh error: {exc}", flush=True)


_results_graded_dates: set = set()  # dates already auto-graded


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
        now_str = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
        lines = [f"V3 Results - {results_date} ET ({now_str} posted)"]
        settled = 0

        def _emoji(outcome):
            o = str(outcome).upper()
            if o == "WIN":
                return "\u2705"
            if o == "LOSS":
                return "\u274C"
            if o == "PUSH":
                return "\u27A1\uFE0F"
            return ""

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
                    f"{_emoji(outcome)} {result['game']} O/U "
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
                    f"{_emoji(outcome)} {result['game']} RL "
                    f"{prediction.get('rl_pred')}: {outcome}"
                )
                settled += 1
        if settled:
            await send_message(app, "\n".join(lines))
    except Exception as exc:
        print(f"V3 results error: {exc}", flush=True)



async def check_all_games_final(app):
    """
    Polls MLB API every 30 min between noon-14:30 SGT (midnight-2:30am EDT).
    Fires evening_results immediately once all games reach a terminal status.
    A daily flag prevents double-firing with the 20:15 SGT cron.
    """
    try:
        import requests
        from sheets import get_results_date
        grade_date = get_results_date()
        if grade_date in _results_graded_dates:
            return
        r = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": grade_date},
            timeout=10,
        )
        data = r.json()
        games = []
        for date_entry in data.get("dates", []):
            games.extend(date_entry.get("games", []))
        if not games:
            return
        terminal = {"Final", "Cancelled", "Postponed", "Suspended"}
        non_final = [
            g for g in games
            if g.get("status", {}).get("abstractGameState", "") not in terminal
        ]
        if non_final:
            return
        print(
            f"[auto-grade] All {len(games)} games Final for {grade_date} — grading now",
            flush=True,
        )
        _results_graded_dates.add(grade_date)
        await evening_results(app)
    except Exception as exc:
        print(f"[auto-grade] Error: {exc}", flush=True)


async def _guarded_evening_results(app):
    """
    20:15 SGT cron wrapper. Skips if check_all_games_final already
    graded results for today, preventing duplicate Telegram messages.
    """
    try:
        from sheets import get_results_date
        grade_date = get_results_date()
        if grade_date in _results_graded_dates:
            print(f"[evening_results] Already graded {grade_date} — skipping cron", flush=True)
            return
        _results_graded_dates.add(grade_date)
        await evening_results(app)
    except Exception as exc:
        print(f"[guarded_evening_results] Error: {exc}", flush=True)


async def run_clv_computation(app):
    """
    Daily CLV computation — runs after most West-Coast games have closed.
    Additive and fail-safe; never affects other jobs.
    """
    try:
        from store import init_db, DB_PATH
        from clv import compute_clv, clv_summary, format_clv_digest

        conn = init_db(DB_PATH)
        try:
            n = compute_clv(conn)
            summary = clv_summary(conn)
            total_sl = summary.get("total_same_line") or {}
            rl       = summary.get("rl") or {}
            total_lm = summary.get("total_line_moved") or {}
            avg_t  = total_sl.get("avg_clv_points")
            pct_t  = total_sl.get("pct_positive")
            avg_r  = rl.get("avg_clv_points")
            pct_r  = rl.get("pct_positive")
            print(
                f"V3 CLV: {n} row(s) written | "
                f"TOTAL same-line n={total_sl.get('n', 0)} "
                + (f"avg={avg_t:+.3f} {pct_t:.0%}+" if avg_t is not None else "(no data)")
                + f" | TOTAL line-moved n={total_lm.get('n', 0)} (not comparable)"
                + f" | RL n={rl.get('n', 0)} "
                + (f"avg={avg_r:+.3f} {pct_r:.0%}+" if avg_r is not None else "(no data)"),
                flush=True,
            )
            digest = format_clv_digest(summary)
            if digest:
                try:
                    from bot import send_message
                    await send_message(app, digest)
                except Exception as exc:
                    print(f"V2 CLV digest send error: {exc}", flush=True)
        finally:
            conn.close()
    except Exception as exc:
        print(f"V3 CLV error: {exc}", flush=True)


def _coalesce_games(games, close_offset_min=None, coalesce_window_min=None):
    """
    Group games into clusters so nearby capture targets share one API refresh.

    For each game, target = game_time_utc - close_offset_min.
    Games whose targets fall within coalesce_window_min of the cluster's
    earliest target are grouped together.  The cluster fires at that earliest
    target (min), guaranteeing no game in the cluster slips past its own window.

    Returns [(fire_time_utc, [game_id, ...]), ...] sorted by fire_time.
    Games with no parseable game_time_utc are silently skipped.
    """
    if close_offset_min is None:
        close_offset_min = CLOSE_OFFSET_MIN
    if coalesce_window_min is None:
        coalesce_window_min = COALESCE_WINDOW_MIN

    targets = []
    for game in games:
        raw = game.get("game_time_utc")
        if not raw:
            continue
        try:
            fp = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            target = fp - timedelta(minutes=close_offset_min)
            targets.append((target, game.get("game_id")))
        except (ValueError, TypeError):
            continue

    if not targets:
        return []

    targets.sort(key=lambda t: t[0])

    clusters = []
    cluster_fire, first_id = targets[0]
    cluster_ids = [first_id]
    for target, game_id in targets[1:]:
        if (target - cluster_fire).total_seconds() <= coalesce_window_min * 60:
            cluster_ids.append(game_id)
        else:
            clusters.append((cluster_fire, cluster_ids))
            cluster_fire, cluster_ids = target, [game_id]
    clusters.append((cluster_fire, cluster_ids))
    return clusters


async def _run_closing_cluster(app, game_date, game_ids):
    """
    One-off APScheduler job: refresh odds once for this cluster, then write a
    FINAL snapshot for every game in game_ids that is still in Preview status.
    Any error is logged and swallowed — never affects other scheduled jobs.
    """
    try:
        from bot import fetch_all_games, _safe_snapshot_write
        from config import ODDS_API_KEY

        _, games_data = await fetch_all_games(ODDS_API_KEY, force_refresh=True)
        by_game_id = {
            item[0].get("game_id"): item
            for item in games_data
            if item[0]
        }
        written = 0
        for gid in game_ids:
            item = by_game_id.get(gid)
            if not item:
                continue
            game, prediction, odds_entry = item
            if game.get("status") == "Preview":
                _safe_snapshot_write(game, prediction, odds_entry, "FINAL")
                written += 1
        print(
            f"V3: Closing cluster done — {written} FINAL snapshot(s) for {game_date}",
            flush=True,
        )
    except Exception as exc:
        print(f"V3: Closing cluster error ({game_date}): {exc}", flush=True)


async def schedule_closing_captures(scheduler, app):
    """
    Register one-off DateTrigger jobs for today's closing captures.
    Called once at startup (via 90-s one-shot) and once daily at 10:30 SGT.
    replace_existing=True makes mid-day restarts safe — a double-registration
    is a no-op.  Clusters whose fire_time has already passed are skipped.
    """
    from data import get_todays_date, get_todays_games

    game_date, _ = get_todays_date()
    games = get_todays_games()
    clusters = _coalesce_games(games)
    now_utc = datetime.now(timezone.utc)
    scheduled = 0
    for fire_time, game_ids in clusters:
        if (fire_time - now_utc).total_seconds() < 60:
            continue  # past or too close; skip
        job_id = f"close_{game_date}_{fire_time.strftime('%H%M%S')}"
        scheduler.add_job(
            _run_closing_cluster,
            DateTrigger(run_date=fire_time, timezone=timezone.utc),
            args=[app, game_date, game_ids],
            id=job_id,
            replace_existing=True,
        )
        scheduled += 1
    print(
        f"V3: Scheduled {scheduled} closing-capture cluster(s) for {game_date}",
        flush=True,
    )


def setup_scheduler(app):
    scheduler = AsyncIOScheduler(
        timezone=tz,
        job_defaults={
            "misfire_grace_time": 3600,
            "coalesce": True,
        }
    )
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
        _guarded_evening_results,
        CronTrigger(hour=20, minute=15, timezone=tz),
        args=[app],
        id="v3_evening_results",
    )
    # Daily: compute CLV from EARLY+FINAL pairs after West-Coast games have closed.
    scheduler.add_job(
        run_clv_computation,
        CronTrigger(hour=21, minute=30, timezone=tz),
        args=[app],
        id="v3_clv_daily",
    )
    # Auto-grade: check every 30 min noon-14:30 SGT; fire results once all games Final.
    scheduler.add_job(
        check_all_games_final,
        CronTrigger(hour="12-14", minute="0,30", timezone=tz),
        args=[app],
        id="v3_auto_grade",
    )
    # Daily: re-register that day's closing-capture cluster jobs at 10:30 SGT.
    scheduler.add_job(
        schedule_closing_captures,
        CronTrigger(hour=10, minute=30, timezone=tz),
        args=[scheduler, app],
        id="v3_schedule_closers",
    )
    # Startup one-shot: runs 90 s after start so the scheduler is fully running.
    # Handles mid-day restarts — today's cluster jobs are re-registered without
    # waiting for the 10:30 SGT cron.  replace_existing on cluster jobs makes
    # a double-registration harmless.
    scheduler.add_job(
        schedule_closing_captures,
        DateTrigger(
            run_date=datetime.now(timezone.utc) + timedelta(seconds=90),
            timezone=timezone.utc,
        ),
        args=[scheduler, app],
        id="v3_startup_closers",
        replace_existing=True,
    )
    return scheduler
