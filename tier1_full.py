"""
Tier 1 Full Season — 2025 MLB Pinnacle line movement + all prices.
Saves complete data for Tier 3 model input (no more proxy prices).
Resumes from saved progress if interrupted.
"""
import json, requests, time, statistics
from datetime import date, timedelta
from pathlib import Path

KEY    = "3acae5ce6c892d55adac98d9542e76f0"
OUTPUT = Path("/root/mlb-edge-v2/tier1_full_2025.json")

def fetch(date_str, utc_time):
    r = requests.get(
        "https://api.the-odds-api.com/v4/historical/sports/baseball_mlb/odds",
        params={
            "apiKey": KEY, "regions": "eu", "markets": "totals,spreads",
            "bookmakers": "pinnacle", "oddsFormat": "american",
            "date": f"{date_str}T{utc_time}",
        },
        timeout=15
    )
    remaining = r.headers.get("x-requests-remaining", "?")
    used      = r.headers.get("x-requests-used", "?")
    if r.status_code != 200:
        return {}, remaining, used
    games = r.json().get("data", [])
    out = {}
    for g in games:
        away, home = g.get("away_team",""), g.get("home_team","")
        key = f"{away} @ {home}"
        entry = {"commence": g.get("commence_time","")}
        for bm in g.get("bookmakers",[]):
            for mkt in bm.get("markets",[]):
                oc = mkt.get("outcomes",[])
                if mkt["key"] == "totals":
                    ov = next((o for o in oc if o["name"]=="Over"),  None)
                    un = next((o for o in oc if o["name"]=="Under"), None)
                    if ov:
                        entry["total"]       = ov["point"]
                        entry["over_price"]  = ov["price"]
                        entry["under_price"] = un["price"] if un else None
                elif mkt["key"] == "spreads":
                    h = next((o for o in oc if o["name"]==home), None)
                    a = next((o for o in oc if o["name"]==away), None)
                    if h:
                        entry["home_rl"]       = h["point"]
                        entry["home_rl_price"] = h["price"]
                        entry["away_rl_price"] = a["price"] if a else None
        out[key] = entry
    return out, remaining, used

def nv(p1, p2):
    def ip(p): return 100/(p+100) if p>0 else abs(p)/(abs(p)+100)
    a, b = ip(p1), ip(p2)
    return a/(a+b) if (a+b) else 0.5

# Full 2025 MLB regular season
start_d, end_d = date(2025, 3, 27), date(2025, 9, 28)
all_dates = []
d = start_d
while d <= end_d:
    all_dates.append(d.isoformat())
    d += timedelta(days=1)

# Load saved progress
saved = {}
if OUTPUT.exists():
    saved = json.loads(OUTPUT.read_text())
    print(f"Resuming: {len(saved)} dates already cached\n")

new_fetches = 0
for i, date_str in enumerate(all_dates):
    if date_str in saved:
        continue

    # Opening: 15:00 UTC = 11am EDT
    open_odds, rem, used = fetch(date_str, "15:00:00Z")
    time.sleep(0.4)

    if not open_odds:
        saved[date_str] = {"games": 0}
        OUTPUT.write_text(json.dumps(saved))
        continue

    # Closing: 22:30 UTC = 6:30pm EDT
    close_odds, rem, used = fetch(date_str, "22:30:00Z")
    time.sleep(0.4)

    rows = []
    for gkey, od in open_odds.items():
        if gkey not in close_odds:
            continue
        cd  = close_odds[gkey]
        row = {"game": gkey, "commence": od.get("commence","")}

        # Total — point + prices (opening & closing)
        if od.get("total") and cd.get("total"):
            row["total_open"]        = od["total"]
            row["total_close"]       = cd["total"]
            row["total_move"]        = round(cd["total"] - od["total"], 2)
            row["over_price"]        = od.get("over_price")    # ← SAVED NOW
            row["under_price"]       = od.get("under_price")   # ← SAVED NOW
            row["close_over_price"]  = cd.get("over_price")
            row["close_under_price"] = cd.get("under_price")

        # Run line — point + prices (opening & closing)
        ohp = od.get("home_rl_price"); oap = od.get("away_rl_price")
        chp = cd.get("home_rl_price"); cap = cd.get("away_rl_price")
        if od.get("home_rl") is not None:
            row["home_rl"]             = od["home_rl"]           # ← SAVED NOW
            row["away_rl"]             = od.get("away_rl") or (-od["home_rl"] if od["home_rl"] else None)
            row["home_rl_price"]       = ohp                     # ← SAVED NOW
            row["away_rl_price"]       = oap                     # ← SAVED NOW
            row["close_home_rl_price"] = chp
            row["close_away_rl_price"] = cap
        if ohp and oap and chp and cap:
            onv  = nv(ohp, oap)
            cnv  = nv(chp, cap)
            row["rl_open_nv"]  = round(onv,  4)
            row["rl_close_nv"] = round(cnv,  4)
            row["rl_abs_move"] = round(abs(cnv - onv) * 100, 2)

        rows.append(row)

    saved[date_str] = {"games": len(rows), "data": rows}
    new_fetches += 2
    OUTPUT.write_text(json.dumps(saved))

    if i % 20 == 0 or not rows:
        pct = i / len(all_dates) * 100
        print(f"[{pct:5.1f}%] {date_str} | {len(rows):2d} games | credits rem: {rem} used: {used}")

print(f"\nDone. {new_fetches} API calls. Data → {OUTPUT}")

# Quick summary
total_moves, rl_moves = [], []
game_count = 0
for rec in saved.values():
    for row in rec.get("data", []):
        game_count += 1
        if "total_move" in row:  total_moves.append(row["total_move"])
        if "rl_abs_move" in row: rl_moves.append(row["rl_abs_move"])

print(f"\n{game_count} game-day pairs | {len(total_moves)} totals | {len(rl_moves)} run lines")
if total_moves:
    print(f"Total move: mean abs={statistics.mean([abs(m) for m in total_moves]):.3f}pts")
if rl_moves:
    print(f"RL move:    mean abs={statistics.mean(rl_moves):.2f}pp")
