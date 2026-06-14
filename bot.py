import json
import logging
from datetime import datetime

import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from config import ODDS_API_KEY, TELEGRAM_CHAT_ID, TELEGRAM_TOKEN, TIMEZONE
from data import (
    clear_cache,
    get_cached_games_data,
    preload_all_data,
)
from model import grade_spread, grade_total


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
tz = pytz.timezone(TIMEZONE)

# Durable dedup: in-memory fast path, falls back to disk on cache miss.
_logged_keys: set = set()


def _already_logged(game_id, quote_ts):
    """Return True if (game_id, quote_ts) was already written to the audit log.

    Checks the in-memory set first. On a miss, scans prediction_audit.jsonl so
    the check survives a process restart. Populates the cache on a disk hit.
    """
    key = f"{game_id}:{quote_ts}"
    if key in _logged_keys:
        return True
    try:
        from sheets import AUDIT_PATH
        if not AUDIT_PATH.exists():
            return False
        with AUDIT_PATH.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (row.get("game_id") == game_id
                        and (row.get("prediction") or {}).get("quote_timestamp") == quote_ts):
                    _logged_keys.add(key)
                    return True
    except Exception:
        pass
    return False


async def send_message(app, text):
    try:
        for start in range(0, len(text), 4096):
            await app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text[start:start + 4096],
            )
    except Exception as exc:
        logger.error("Send message error: %s", exc)


async def fetch_all_games(api_key=None, force_refresh=False):
    from model import predict_game

    cached = get_cached_games_data()
    if not cached or force_refresh:
        cached = preload_all_data(
            api_key or ODDS_API_KEY,
            force_odds_refresh=force_refresh,
        )
    if not cached:
        return [], []

    games_data = []
    for game, context, odds_entry in cached:
        total = odds_entry.get("total") if odds_entry else None
        run_line = odds_entry.get("run_line") if odds_entry else None
        prediction = predict_game(
            context,
            total=total,
            run_line=run_line,
            odds_entry=odds_entry,
        )
        games_data.append((game, prediction, odds_entry))

    try:
        from sheets import log_prediction

        for game, prediction, odds_entry in games_data:
            if not prediction:
                continue
            game_id = game.get("game_id")
            quote_time = prediction.get("quote_timestamp") or "no-quote"
            is_new = not _already_logged(game_id, quote_time)
            if is_new:
                log_prediction(game, prediction, odds_entry)
                _logged_keys.add(f"{game_id}:{quote_time}")
            prediction["_newly_logged"] = is_new
    except Exception as exc:
        logger.error("Prediction logging error: %s", exc)

    return [game for game, _, _ in cached], games_data


def _price(value):
    if value is None:
        return "no price"
    return f"+{value}" if value > 0 else str(value)


def _pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _ev(value):
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def format_summary(games_data, now):
    from data import _cache
    day_label = "tomorrow" if _cache.get("showing_next_day") else "today"
    edge_count = sum(
        1
        for _, prediction, _ in games_data
        if prediction
        and (prediction.get("edge_flagged") or prediction.get("rl_edge_flagged"))
    )
    model_ready = any(
        prediction and prediction.get("model_ready")
        for _, prediction, _ in games_data
    )
    message = (
        f"⚾ MLB Edge V2 — {now}\n"
        f"{len(games_data)} games {day_label} | {edge_count} bet(s)\n"
    )
    if not model_ready:
        message += "⚠️ MODEL NOT TRAINED — all markets SKIP\n"
    tbd_count = sum(
        1
        for _, prediction, _ in games_data
        if prediction and not prediction.get("has_real_pitchers")
    )
    if tbd_count:
        message += f"⚠️ {tbd_count} game(s) have unconfirmed starters\n"
    if edge_count == 0:
        return message + "\nNo price-positive edges today."
    message += "\n💰 Price-Positive Edges:\n" + "━" * 20 + "\n"
    for game, prediction, _ in games_data:
        if not prediction or not (
            prediction.get("edge_flagged") or prediction.get("rl_edge_flagged")
        ):
            continue
        home = game["home_team"].split()[-1]
        away = game["away_team"].split()[-1]
        message += f"\n⚡ {away} @ {home}\n"
        if prediction.get("edge_flagged"):
            size = prediction.get("total_bet_size", "")
            emoji = "✅🔥" if size == "FULL" else "✅"
            message += (
                f"  O/U {prediction['total_pred']} {prediction.get('total_line')} "
                f"({_price(prediction.get('total_price'))}) | "
                f"EV {_ev(prediction.get('total_ev'))} {emoji} {size}\n"
            )
        if prediction.get("rl_edge_flagged"):
            size = prediction.get("rl_bet_size", "")
            emoji = "✅🔥" if size == "FULL" else "✅"
            message += (
                f"  RL {prediction['rl_pred']} "
                f"({_price(prediction.get('rl_price'))}) | "
                f"EV {_ev(prediction.get('rl_ev'))} {emoji} {size}\n"
            )
    return message + "\nUse /v2_edge for full details."


def _format_detail(game, prediction):
    start = game.get("start_time_sgt", "")
    if start:
        try:
            start = datetime.fromisoformat(start).strftime("%b %d %I:%M %p SGT")
        except ValueError:
            pass

    header = (
        f"⚡ {game['away_team']} @ {game['home_team']}\n"
        f"🏟 {game.get('venue', '')} | 🕐 {start}\n"
        f"🎯 {game.get('away_pitcher', 'TBD')} vs {game.get('home_pitcher', 'TBD')}\n"
        f"📊 Data quality: {_pct(prediction.get('data_quality'))} | "
        f"Model: {prediction.get('model_version')}\n"
    )

    expected = (
        f"\nExpected: total {prediction.get('our_total')} | "
        f"margin {prediction.get('our_home_margin'):+.2f}\n"
    )

    ou_bet = prediction.get("edge_flagged")
    ou_size = prediction.get("total_bet_size", "SKIP")
    if ou_size == "FULL":
        ou_emoji = "✅🔥"
    elif ou_bet:
        ou_emoji = "✅"
    else:
        ou_emoji = "⏭"
    ou_block = (
        f"\nO/U {prediction.get('total_pred')} {prediction.get('total_line')} "
        f"({_price(prediction.get('total_price'))})\n"
        f"  Win {_pct(prediction.get('total_win_prob'))} | "
        f"Market {_pct(prediction.get('total_market_prob'))} | "
        f"Edge {_pct(prediction.get('total_probability_edge'))}\n"
        f"  EV {_ev(prediction.get('total_ev'))} | "
        f"Agreement {_pct(prediction.get('total_agreement'))} | "
        f"{ou_emoji} {'BET ' + ou_size if ou_bet else 'SKIP'}\n"
    )

    rl_bet = prediction.get("rl_edge_flagged")
    rl_size = prediction.get("rl_bet_size", "SKIP")
    if rl_size == "FULL":
        rl_emoji = "✅🔥"
    elif rl_bet:
        rl_emoji = "✅"
    else:
        rl_emoji = "⏭"
    rl_block = (
        f"\nRL {prediction.get('rl_pred')} "
        f"({_price(prediction.get('rl_price'))})\n"
        f"  Win {_pct(prediction.get('rl_win_prob'))} | "
        f"Market {_pct(prediction.get('rl_market_prob'))} | "
        f"Edge {_pct(prediction.get('rl_probability_edge'))}\n"
        f"  EV {_ev(prediction.get('rl_ev'))} | "
        f"Agreement {_pct(prediction.get('rl_agreement'))} | "
        f"{rl_emoji} {'BET ' + rl_size if rl_bet else 'SKIP'}\n"
    )

    footer = (
        f"\nModel std: total {prediction.get('total_ensemble_std')} | "
        f"margin {prediction.get('margin_ensemble_std')}\n"
        + "━" * 20
    )

    return header + expected + ou_block + rl_block + footer




async def cmd_v2_bets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    del context
    from sheets import get_user_bets, units_won
    from collections import defaultdict
    from datetime import datetime, timedelta

    all_rows = get_user_bets("mlb_v2", days=None)
    if not all_rows:
        await update.message.reply_text("No bets logged yet in user_bets.")
        return

    today_str = datetime.now(tz).strftime("%Y-%m-%d")
    week_cutoff = (datetime.now(tz) - timedelta(days=7)).strftime("%Y-%m-%d")

    def summarize(rows):
        w = l = p = 0
        units = 0.0
        for row in rows:
            result = row[8] if len(row) > 8 else ""
            if not result:
                continue
            price = row[6] if len(row) > 6 else 0
            stake = row[7] if len(row) > 7 else 1
            u = units_won(price, stake, result)
            units += u
            if result == "WIN":
                w += 1
            elif result == "LOSS":
                l += 1
            elif result == "PUSH":
                p += 1
        return w, l, p, units

    daily = [r for r in all_rows if str(r[0])[:10] == today_str]
    weekly = [r for r in all_rows if str(r[0])[:10] >= week_cutoff]
    lifetime = all_rows

    # Recent detail (last 3 days)
    detail_cutoff = (datetime.now(tz) - timedelta(days=3)).strftime("%Y-%m-%d")
    detail_rows = [r for r in all_rows if str(r[0])[:10] >= detail_cutoff]
    by_date = defaultdict(list)
    for row in detail_rows:
        by_date[str(row[0])[:10]].append(row)

    lines = ["\U0001F4CA V2 Bet Tracker"]
    for date in sorted(by_date.keys(), reverse=True):
        lines.append(f"\n{date}:")
        day_total = 0.0
        for row in by_date[date]:
            home, away, market, pick = row[2], row[3], row[4], row[5]
            price = row[6] if len(row) > 6 else ""
            stake = row[7] if len(row) > 7 else 1
            result = row[8] if len(row) > 8 else ""
            label = f"{away} @ {home} | {market} {pick} ({price})"
            if not result:
                lines.append(f"  {label} | {stake}u | PENDING")
            else:
                u = units_won(price, stake, result)
                day_total += u
                sign = "+" if u >= 0 else ""
                lines.append(f"  {label} | {stake}u | {result} {sign}{u:.2f}u")
        if day_total:
            sign = "+" if day_total >= 0 else ""
            lines.append(f"  Day total: {sign}{day_total:.2f}u")

    dw, dl, dp, du = summarize(daily)
    ww, wl, wp, wu = summarize(weekly)
    lw, ll, lp, lu = summarize(lifetime)

    def fmt(label, w, l, p, u):
        sign = "+" if u >= 0 else ""
        return f"{label}: {w}W-{l}L-{p}P | {sign}{u:.2f}u"

    lines.append("")
    lines.append(fmt("\U0001F4C5 Daily", dw, dl, dp, du))
    lines.append(fmt("\U0001F4C6 Weekly", ww, wl, wp, wu))
    lines.append(fmt("\U0001F4CA Lifetime", lw, ll, lp, lu))

    await update.message.reply_text("\n".join(lines))


async def cmd_v2_brief(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    await update.message.reply_text("⏳ Loading V2 edges...")
    _, games_data = await fetch_all_games(ODDS_API_KEY)
    if not games_data:
        await update.message.reply_text("No MLB games available.")
        return
    now = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
    await update.message.reply_text(format_summary(games_data, now))


async def cmd_v2_edge(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    await update.message.reply_text("⏳ Fetching V2 edges...")
    _, games_data = await fetch_all_games(ODDS_API_KEY)
    edges = [
        (game, prediction)
        for game, prediction, _ in games_data
        if prediction
        and (
            prediction.get("edge_flagged")
            or prediction.get("rl_edge_flagged")
        )
    ]
    if not edges:
        model_ready = any(
            prediction and prediction.get("model_ready")
            for _, prediction, _ in games_data
        )
        reason = (
            "⚠️ No trained V2 model is installed."
            if not model_ready
            else "No markets clear the EV and uncertainty filters."
        )
        await update.message.reply_text(reason)
        return
    message = "⚾ MLB Edge V2 — full details\n\n"
    message += "\n\n".join(
        _format_detail(game, prediction)
        for game, prediction in edges
    )
    for start in range(0, len(message), 4096):
        await update.message.reply_text(message[start:start + 4096])


async def cmd_v2_refresh(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    await update.message.reply_text("Refreshing game data and prices...")
    clear_cache()
    games, games_data = await fetch_all_games(
        ODDS_API_KEY, force_refresh=True
    )
    edge_count = sum(
        1
        for _, prediction, _ in games_data
        if prediction
        and (
            prediction.get("edge_flagged")
            or prediction.get("rl_edge_flagged")
        )
    )
    await update.message.reply_text(
        f"✅ V2 refreshed: {len(games)} games, {edge_count} bets."
    )


async def cmd_v2_results(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    await update.message.reply_text("⏳ Fetching V2 results...")
    try:
        from sheets import (
            get_results_date,
            get_stored_predictions,
            log_results,
            update_results_in_sheet,
        )

        results_date = get_results_date()
        results = log_results()
        if not results:
            await update.message.reply_text("No final results available.")
            return
        update_results_in_sheet(results, date_override=results_date)
        from sheets import grade_user_bets
        grade_user_bets(results_date, results)
        predictions = get_stored_predictions(results_date)
        lines = [f"⚾ MLB Edge V2 — Results {results_date}"]
        for result in results:
            prediction = predictions.get(result["game"])
            if not prediction or not (
                prediction.get("edge_flagged")
                or prediction.get("rl_edge_flagged")
            ):
                continue
            lines.append(
                f"\n{result['game']}\n"
                f"  score {result['away_score']}-{result['home_score']}"
            )
            if prediction.get("edge_flagged"):
                outcome = grade_total(
                    result["total_result"],
                    prediction.get("total_line"),
                    prediction.get("total_pred"),
                )
                lines.append(f"  O/U {prediction.get('total_pred')}: {outcome}")
            if prediction.get("rl_edge_flagged"):
                outcome = grade_spread(
                    result["home_score"] - result["away_score"],
                    prediction.get("rl_side"),
                    prediction.get("rl_point"),
                )
                lines.append(f"  RL {prediction.get('rl_pred')}: {outcome}")
        await update.message.reply_text("\n".join(lines))
    except Exception as exc:
        await update.message.reply_text(f"Result error: {exc}")


async def cmd_v2_record(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    try:
        from sheets import get_record

        record = get_record()
        if not record or not record.get("settled_bets"):
            await update.message.reply_text("No settled V2 bets yet.")
            return
        message = (
            "⚾ MLB Edge V2 Record\n"
            f"Settled bets: {record['settled_bets']}\n"
            f"Pushes: {record['pushes']}\n"
            f"W-L: {record['wins']}-{record['losses']}\n"
            f"Hit rate: {record['hit_rate']}%\n"
            f"ROI: {record['roi']}%"
        )
        await update.message.reply_text(message)
    except Exception as exc:
        await update.message.reply_text(f"Record error: {exc}")


async def cmd_v2_status(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    from model import load_models

    bundle = load_models()
    cached = get_cached_games_data()
    if bundle:
        model_line = (
            f"Model {bundle['version']} | "
            f"test through {bundle.get('test_end')}"
        )
        metrics = bundle.get("metrics", {})
        metric_line = (
            f"Test MAE: total {metrics.get('total_mae', 0):.3f}, "
            f"margin {metrics.get('margin_mae', 0):.3f}"
        )
    else:
        model_line = "MODEL NOT TRAINED - betting disabled"
        metric_line = "Run: python historical.py"
    await update.message.reply_text(
        "⚾ MLB Edge V2\n"
        f"{datetime.now(tz).strftime('%b %d %Y %H:%M SGT')}\n"
        f"Games loaded: {len(cached)}\n"
        f"{model_line}\n"
        f"{metric_line}"
    )


def main():
    print("Starting MLB Edge V2 Bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("v2_brief", cmd_v2_brief))
    app.add_handler(CommandHandler("v2_bets", cmd_v2_bets))
    app.add_handler(CommandHandler("v2_edge", cmd_v2_edge))
    app.add_handler(CommandHandler("v2_refresh", cmd_v2_refresh))
    app.add_handler(CommandHandler("v2_results", cmd_v2_results))
    app.add_handler(CommandHandler("v2_record", cmd_v2_record))
    app.add_handler(CommandHandler("v2_status", cmd_v2_status))

    from scheduler import setup_scheduler

    scheduler = setup_scheduler(app)

    async def post_init(application):
        del application
        scheduler.start()
        print("V2 scheduler started")
        import threading

        thread = threading.Thread(
            target=preload_all_data,
            args=(ODDS_API_KEY,),
            daemon=True,
        )
        thread.start()

    app.post_init = post_init
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
