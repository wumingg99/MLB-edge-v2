import logging
from datetime import datetime
import pytz
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from config import (TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, TIMEZONE,
                    ODDS_API_KEY, MIN_CONFIDENCE, MIN_MODELS_AGREE,
                    RL_MIN_CONFIDENCE, EDGE_THRESHOLD,
                    BET_CONFIDENCE, FULL_BET_CONFIDENCE)
from data import preload_all_data, get_cached_games_data, clear_cache

logging.basicConfig(
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
    level=logging.INFO)
logger = logging.getLogger(__name__)

tz = pytz.timezone(TIMEZONE)

async def send_message(app, text):
    try:
        max_len = 4096
        if len(text) <= max_len:
            await app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, text=text)
        else:
            for i in range(0, len(text), max_len):
                await app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=text[i:i+max_len])
    except Exception as e:
        logger.error(f"Send message error: {e}")

async def fetch_all_games(api_key=None):
    from model import predict_game
    cached = get_cached_games_data()
    if not cached:
        cached = preload_all_data(api_key or ODDS_API_KEY)
    if not cached:
        return [], []
    games_data = []
    for game, context, odds_entry in cached:
        total = odds_entry.get("total") if odds_entry else None
        run_line = odds_entry.get("run_line") if odds_entry else None
        if total and total > 18.0:
            total = None
        prediction = predict_game(context, total, run_line)
        games_data.append((game, prediction, odds_entry))
    # Log predictions to Sheets
    try:
        from sheets import log_prediction
        if not hasattr(fetch_all_games, "_logged"):
            fetch_all_games._logged = set()
        for game, prediction, odds_entry in games_data:
            if prediction and (prediction.get("edge_flagged") or
                               prediction.get("rl_edge_flagged")):
                game_key = f"{game['away_team']} @ {game['home_team']}"
                if game_key not in fetch_all_games._logged:
                    log_prediction(game, prediction, odds_entry)
                    fetch_all_games._logged.add(game_key)
    except Exception as e:
        print(f"Prediction logging error: {e}")
    return [g for g, c, o in cached], games_data

def format_summary(games_data, now):
    from data import _cache
    showing_next = _cache.get("showing_next_day", False)
    day_label = "tomorrow" if showing_next else "today"
    edge_count = sum(1 for _, p, _ in games_data
                     if p and (p.get("edge_flagged") or
                               p.get("rl_edge_flagged")))
    total_games = len(games_data)
    tbd_count = sum(1 for _, p, _ in games_data
                    if p and not p.get("has_real_pitchers"))

    msg = f"⚾ MLB Edge V2 — {now}\n"
    msg += f"{total_games} games {day_label} | {edge_count} edge(s) flagged\n"
    if tbd_count > 0:
        msg += f"⚠️ {tbd_count} game(s) have TBD pitchers\n"
    msg += "\n"

    if edge_count == 0:
        msg += "No edges flagged — check back later.\n"
        return msg

    msg += "Today's edges:\n\n"
    for game, pred, odds in games_data:
        if not pred:
            continue
        if not (pred.get("edge_flagged") or pred.get("rl_edge_flagged")):
            continue

        home = game["home_team"].split()[-1]
        away = game["away_team"].split()[-1]
        total = odds.get("total") if odds else "N/A"

        ou_skip = (pred["total_conf"] < MIN_CONFIDENCE or
                   abs(pred["total_gap"]) < EDGE_THRESHOLD or
                   not pred.get("mc_agrees") or
                   not pred.get("poisson_agrees"))
        rl_skip = (pred["rl_conf"] < RL_MIN_CONFIDENCE or
                   pred["rl_votes"] < MIN_MODELS_AGREE)

        ou_flag = "⏭" if ou_skip else "✅"
        rl_flag = "⏭" if rl_skip else "✅"
        pitcher_flag = "⚠️" if not pred.get("has_real_pitchers") else "⚡"

        # Bet size indicator
        bet_size = pred.get("bet_size", "SKIP")
        size_emoji = "🔥" if bet_size == "FULL" else "✅" if bet_size == "HALF" else "⏭"

        msg += (f"{pitcher_flag} {away} @ {home}\n"
                f"   O/U: {pred['total_pred']} {total} "
                f"({pred['total_votes']}/{pred['total_models']}, "
                f"{pred['total_conf']}%)  {ou_flag}\n"
                f"   RL: {pred['rl_pred']} "
                f"({pred['rl_votes']}/{pred['rl_models']}, "
                f"{pred['rl_conf']}%)  {rl_flag} {size_emoji}\n")

    msg += "\nType /v2_edge for full details"
    return msg

async def cmd_v2_brief(update: Update,
                       context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Loading V2 edges...")
    now = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
    games, games_data = await fetch_all_games(ODDS_API_KEY)
    if not games:
        await update.message.reply_text("No MLB games today.")
        return
    await update.message.reply_text(format_summary(games_data, now))

async def cmd_v2_edge(update: Update,
                      context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching V2 edges...")
    now = datetime.now(tz).strftime("%b %d, %Y %H:%M SGT")
    games, games_data = await fetch_all_games(ODDS_API_KEY)
    if not games:
        await update.message.reply_text("No MLB games today.")
        return

    edges = [(g, p, o) for g, p, o in games_data
             if p and (p.get("edge_flagged") or p.get("rl_edge_flagged"))]
    if not edges:
        await update.message.reply_text(
            f"⚾ MLB Edge V2 — {now}\n\nNo edges flagged today.")
        return

    msg = f"⚾ MLB Edge V2 — {now}\n"
    msg += f"{len(edges)} edge(s) flagged\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

    for game, pred, odds in edges:
        home = game["home_team"]
        away = game["away_team"]
        hp = game.get("home_pitcher", "TBD")
        ap = game.get("away_pitcher", "TBD")
        venue = game.get("venue", "")
        total = odds.get("total") if odds else "N/A"
        start = game.get("start_time_sgt", "")
        if start:
            try:
                dt = datetime.fromisoformat(start)
                start = dt.strftime("%b %d %I:%M %p SGT")
            except Exception:
                pass

        ou_skip = (pred["total_conf"] < MIN_CONFIDENCE or
                   abs(pred["total_gap"]) < EDGE_THRESHOLD or
                   not pred.get("mc_agrees") or
                   not pred.get("poisson_agrees"))
        rl_skip = (pred["rl_conf"] < RL_MIN_CONFIDENCE or
                   pred["rl_votes"] < MIN_MODELS_AGREE)
        ou_flag = "⏭" if ou_skip else "✅"
        rl_flag = "⏭" if rl_skip else "✅"

        bet_size = pred.get("bet_size", "SKIP")
        size_str = ("🔥 FULL BET" if bet_size == "FULL"
                    else "✅ HALF BET" if bet_size == "HALF"
                    else "⏭ SKIP")

        # Convergence indicators
        mc_ok = "✅" if pred.get("mc_agrees") else "❌"
        poi_ok = "✅" if pred.get("poisson_agrees") else "❌"
        std_ok = "✅" if pred.get("mc_std", 99) <= 4.0 else "❌"
        diff_ok = "✅" if pred.get("mc_formula_diff", 99) <= 1.5 else "❌"

        msg += f"⚡ {away} @ {home}\n"
        msg += f"🏟 {venue}\n"
        msg += f"🎯 {ap} vs {hp}\n"
        if start:
            msg += f"🕐 {start}\n"
        msg += (f"Our total: {pred['our_total']} | "
                f"Open: {total} (gap: {pred['total_gap']:+.1f})\n")
        msg += (f"O/U: {pred['total_pred']} — "
                f"{pred['total_conf']}% — "
                f"{pred['total_votes']}/{pred['total_models']}  {ou_flag}\n")
        msg += (f"RL: {pred['rl_pred']} — "
                f"{pred['rl_conf']}% — "
                f"{pred['rl_votes']}/{pred['rl_models']}  {rl_flag}\n")
        msg += f"Bet size: {size_str}\n"
        msg += (f"Convergence: MC {mc_ok} | "
                f"Poisson {poi_ok} | "
                f"Std {std_ok} | "
                f"Diff {diff_ok}\n")
        msg += (f"MC avg: {pred['mc_avg_total']} | "
                f"MC over: {pred['mc_over_prob']*100:.0f}% | "
                f"Std: {pred['mc_std']}\n")
        msg += (f"Win prob: Home {pred['home_win_prob']*100:.0f}% / "
                f"Away {pred['away_win_prob']*100:.0f}%\n")
        msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

    await update.message.reply_text(msg)

async def cmd_v2_refresh(update: Update,
                         context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Refreshing V2 data...")
    clear_cache()
    if hasattr(fetch_all_games, "_logged"):
        fetch_all_games._logged.clear()
    games, games_data = await fetch_all_games(ODDS_API_KEY)
    edge_count = sum(1 for _, p, _ in games_data
                     if p and (p.get("edge_flagged") or
                               p.get("rl_edge_flagged")))
    await update.message.reply_text(
        f"✅ V2 Refreshed — {len(games_data)} games, "
        f"{edge_count} edge(s)")

async def cmd_v2_results(update: Update,
                         context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching V2 results...")
    try:
        from sheets import log_results, update_results_in_sheet, get_results_date
        results_date = get_results_date()
        results = log_results()
        if not results:
            await update.message.reply_text(
                "No final results yet — games may still be in progress.")
            return
        update_results_in_sheet(results, date_override=results_date)
        stored_preds = __import__("sheets").get_stored_predictions(
            results_date)
        flagged = []
        other = []
        for r in results:
            pred = stored_preds.get(r["game"])
            is_flagged = pred and (pred.get("edge_flagged") or
                                   pred.get("rl_edge_flagged"))
            if is_flagged:
                flagged.append((r, pred))
            else:
                other.append(r)

        msg = f"⚾ V2 Results — {results_date}\n━━━━━━━━━━━━━━━━━━━━\n"
        if flagged:
            msg += f"📍 Flagged edges ({len(flagged)}):\n"
            for r, pred in flagged:
                home_score = r["home_score"]
                away_score = r["away_score"]
                total_result = r["total_result"]
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
                ou_tag = " ← BET" if pred.get("edge_flagged") else ""
                rl_tag = " ← BET" if pred.get("rl_edge_flagged") else ""
                msg += (f"\n{r['game']}\n"
                        f"   Score: {away_score}-{home_score} "
                        f"(total: {total_result})\n"
                        f"   RL: {rl_pred} {rl_correct}{rl_tag}\n"
                        f"   O/U: {pred.get('total_pred')} → "
                        f"{ou_result} {ou_correct}{ou_tag}\n")
        if other:
            msg += f"\nOther games ({len(other)}):\n"
            for r in other[:8]:
                msg += f"• {r['game']} — {r['away_score']}-{r['home_score']}\n"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def cmd_v2_record(update: Update,
                        context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Fetching V2 record...")
    try:
        from sheets import get_record
        record = get_record()
        if not record or record.get("total", 0) == 0:
            await update.message.reply_text(
                "No V2 results yet.\n"
                "Run /v2_results after games finish.")
            return
        msg = "📊 MLB Edge V2 Record\n━━━━━━━━━━━━━━━━━━━━\n"
        msg += "\n🎯 Flagged bets (actionable):\n"
        msg += (f"  RL: {record['rl_flagged_correct']}/"
                f"{record['rl_flagged_total']} "
                f"({record['rl_flagged_accuracy']}%)\n")
        msg += (f"  O/U: {record['ou_flagged_correct']}/"
                f"{record['ou_flagged_total']} "
                f"({record['ou_flagged_accuracy']}%)\n")
        msg += "\n📊 All games (model validation):\n"
        msg += (f"  RL: {record['rl_correct']}/{record['rl_total']} "
                f"({record['rl_accuracy']}%)\n")
        msg += (f"  O/U: {record['ou_correct']}/{record['ou_total']} "
                f"({record['ou_accuracy']}%)\n")
        if record.get("monthly"):
            msg += "\nMonthly (flagged RL):\n"
            for month, data in sorted(
                    record["monthly"].items(), reverse=True)[:6]:
                rl = data.get("rl", 0)
                rl_c = data.get("rl_correct", 0)
                acc = round(rl_c / rl * 100, 1) if rl > 0 else 0
                msg += f"  {month}: {rl_c}/{rl} ({acc}%)\n"
            msg += "\nMonthly (flagged O/U):\n"
            for month, data in sorted(
                    record["monthly"].items(), reverse=True)[:6]:
                ou = data.get("ou", 0)
                ou_c = data.get("ou_correct", 0)
                acc = round(ou_c / ou * 100, 1) if ou > 0 else 0
                msg += f"  {month}: {ou_c}/{ou} ({acc}%)\n"
        await update.message.reply_text(msg)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def cmd_v2_status(update: Update,
                        context: ContextTypes.DEFAULT_TYPE):
    now = datetime.now(tz).strftime("%b %d %Y %H:%M SGT")
    cached = get_cached_games_data()
    msg = (f"⚾ MLB Edge V2\n\n"
           f"🕐 {now}\n"
           f"📊 Games loaded: {len(cached)}\n"
           f"✅ V2 Bot is live\n\n"
           f"V2 improvements:\n"
           f"→ 5 models (LR+RF+XGB+LGB+NN)\n"
           f"→ Calibrated probabilities\n"
           f"→ Convergence filter\n"
           f"→ Poisson O/U model\n"
           f"→ Recent form features\n"
           f"→ MC std dev check\n\n"
           f"Commands:\n"
           f"/v2_brief — today's edges\n"
           f"/v2_edge — full edge details\n"
           f"/v2_refresh — refresh data\n"
           f"/v2_results — log results\n"
           f"/v2_record — win/loss record")
    await update.message.reply_text(msg)

def main():
    print("Starting MLB Edge V2 Bot...")
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
        scheduler.start()
        print("V2 Scheduler started")
        import threading
        t = threading.Thread(target=preload_all_data, args=(ODDS_API_KEY,), daemon=True)
        t.start()
        print("MLB Edge V2 is live — preloading in background")

    app.post_init = post_init
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
