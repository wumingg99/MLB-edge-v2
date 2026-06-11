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

        if not hasattr(fetch_all_games, "_logged"):
            fetch_all_games._logged = set()
        for game, prediction, odds_entry in games_data:
            if not prediction:
                continue
            quote_time = prediction.get("quote_timestamp") or "no-quote"
            key = f"{game.get('game_id')}:{quote_time}"
            if key not in fetch_all_games._logged:
                log_prediction(game, prediction, odds_entry)
                fetch_all_games._logged.add(key)
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
        and (
            prediction.get("edge_flagged")
            or prediction.get("rl_edge_flagged")
        )
    )
    model_ready = any(
        prediction and prediction.get("model_ready")
        for _, prediction, _ in games_data
    )
    message = (
        f"MLB Edge V3 - {now}\n"
        f"{len(games_data)} games {day_label} | {edge_count} bets\n"
    )
    if not model_ready:
        message += "MODEL NOT TRAINED - all markets forced to SKIP\n"
    tbd_count = sum(
        1
        for _, prediction, _ in games_data
        if prediction and not prediction.get("has_real_pitchers")
    )
    if tbd_count:
        message += f"{tbd_count} games have unconfirmed starters\n"
    if edge_count == 0:
        return message + "\nNo price-positive edges."

    message += "\nPrice-positive edges:\n"
    for game, prediction, _ in games_data:
        if not prediction or not (
            prediction.get("edge_flagged")
            or prediction.get("rl_edge_flagged")
        ):
            continue
        message += f"\n{game['away_team']} @ {game['home_team']}\n"
        if prediction.get("edge_flagged"):
            message += (
                f"  O/U: {prediction['total_pred']} "
                f"{prediction.get('total_line')} "
                f"{_price(prediction.get('total_price'))} | "
                f"EV {_ev(prediction.get('total_ev'))} | "
                f"{prediction.get('total_bet_size')}\n"
            )
        if prediction.get("rl_edge_flagged"):
            message += (
                f"  RL: {prediction['rl_pred']} "
                f"{_price(prediction.get('rl_price'))} | "
                f"EV {_ev(prediction.get('rl_ev'))} | "
                f"{prediction.get('rl_bet_size')}\n"
            )
    return message + "\nUse /v2_edge for full details."


def _format_detail(game, prediction):
    start = game.get("start_time_sgt", "")
    if start:
        try:
            start = datetime.fromisoformat(start).strftime(
                "%b %d %I:%M %p SGT"
            )
        except ValueError:
            pass
    lines = [
        f"{game['away_team']} @ {game['home_team']}",
        f"{game.get('venue', '')} | {start}",
        (
            f"{game.get('away_pitcher', 'TBD')} vs "
            f"{game.get('home_pitcher', 'TBD')}"
        ),
        (
            f"Model: {prediction.get('model_version')} | "
            f"data quality {prediction.get('data_quality')}"
        ),
        (
            f"Expected total {prediction.get('our_total')} | "
            f"home margin {prediction.get('our_home_margin'):+.2f}"
        ),
        (
            f"O/U: {prediction.get('total_pred')} "
            f"{prediction.get('total_line')} "
            f"{_price(prediction.get('total_price'))}"
        ),
        (
            f"  win {_pct(prediction.get('total_win_prob'))} | "
            f"push {_pct(prediction.get('total_push_prob'))} | "
            f"market {_pct(prediction.get('total_market_prob'))}"
        ),
        (
            f"  edge {_pct(prediction.get('total_probability_edge'))} | "
            f"EV {_ev(prediction.get('total_ev'))} | "
            f"agreement {_pct(prediction.get('total_agreement'))} | "
            f"{'BET' if prediction.get('edge_flagged') else 'SKIP'}"
        ),
        (
            f"RL: {prediction.get('rl_pred')} "
            f"{_price(prediction.get('rl_price'))}"
        ),
        (
            f"  win {_pct(prediction.get('rl_win_prob'))} | "
            f"push {_pct(prediction.get('rl_push_prob'))} | "
            f"market {_pct(prediction.get('rl_market_prob'))}"
        ),
        (
            f"  edge {_pct(prediction.get('rl_probability_edge'))} | "
            f"EV {_ev(prediction.get('rl_ev'))} | "
            f"agreement {_pct(prediction.get('rl_agreement'))} | "
            f"{'BET' if prediction.get('rl_edge_flagged') else 'SKIP'}"
        ),
        (
            f"Model spread: total "
            f"{prediction.get('total_ensemble_std')} | margin "
            f"{prediction.get('margin_ensemble_std')}"
        ),
    ]
    return "\n".join(lines)


async def cmd_v2_brief(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    await update.message.reply_text("Loading V3 markets...")
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
    await update.message.reply_text("Fetching V3 price-positive edges...")
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
            "No trained V3 model is installed."
            if not model_ready
            else "No markets clear the EV and uncertainty filters."
        )
        await update.message.reply_text(reason)
        return
    message = "MLB Edge V3 - full details\n\n"
    message += "\n\n".join(
        _format_detail(game, prediction)
        for game, prediction in edges
    )
    await update.message.reply_text(message)


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
        f"V3 refreshed: {len(games)} games, {edge_count} bets."
    )


async def cmd_v2_results(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    del context
    await update.message.reply_text("Fetching V3 results...")
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
        predictions = get_stored_predictions(results_date)
        lines = [f"V3 Results - {results_date}"]
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
            await update.message.reply_text("No settled V3 bets yet.")
            return
        message = (
            "MLB Edge V3 Record\n"
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
        "MLB Edge V3\n"
        f"{datetime.now(tz).strftime('%b %d %Y %H:%M SGT')}\n"
        f"Games loaded: {len(cached)}\n"
        f"{model_line}\n"
        f"{metric_line}"
    )


def main():
    print("Starting MLB Edge V3 Bot...")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("v2_brief", cmd_v2_brief))
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
        print("V3 scheduler started")
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
