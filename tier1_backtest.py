"""
Tier 1 Backtest: 2025 MLB Pinnacle Line Movement Analysis
Answers: does Pinnacle move enough between open and close to make CLV meaningful?
Saves progress to JSON so you can resume if interrupted.
"""
import json, requests, time, statistics
from datetime import date, timedelta
from pathlib import Path

KEY = "3acae5ce6c892d55adac98d9542e76f0"
OUTPUT = Path("/root/mlb-edge-v2/tier1_2025.json")

def fetch_snapshot(date_str, utc_time):
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
        commence  = g.get("commence_time","")
        key = f"{away} @ {home}"
        entry = {"commence": commence}
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
                    h = next((o for o in oc if o["name"]==home),  None)
                    a = next((o for o in oc if o["name"]==away),  None)
                    if h:
                        entry["home_rl"]       = h["point"]
                        entry["home_rl_price"] = h["price"]
                        entry["away_rl_price"] = a["price"] if a else None
        out[key] = entry
    return out, remaining, used

def nv_prob(p1, p2):
    def ip(p): return 100/(p+100) if p>0 else abs(p)/(abs(p)+100)
    a, b = ip(p1), ip(p2)
    return a/(a+b) if (a+b) > 0 else 0.5

# ── Dates: 2025 MLB regular season ──────────────────────────────
start_d, end_d = date(2025, 3, 27), date(2025, 9, 28)
all_dates = []
d = start_d
while d <= end_d:
    all_dates.append(d.isoformat())
    d += timedelta(days=1)

# ── Load saved progress ──────────────────────────────────────────
saved = {}
if OUTPUT.exists():
    saved = json.loads(OUTPUT.read_text())
    print(f"Resuming: {len(saved)} dates already cached\n")

# ── Fetch ────────────────────────────────────────────────────────
new_fetches = 0
for i, date_str in enumerate(all_dates):
    if date_str in saved:
        continue

    # Opening line: 15:00 UTC = 11am EDT (lines typically posted by now)
    open_odds, rem, used = fetch_snapshot(date_str, "15:00:00Z")
    time.sleep(0.4)

    if not open_odds:
        saved[date_str] = {"games": 0}
        continue

    # Closing line: 22:30 UTC = 6:30pm EDT (30 min before typical 7pm first pitch)
    close_odds, rem, used = fetch_snapshot(date_str, "22:30:00Z")
    time.sleep(0.4)

    rows = []
    for gkey, od in open_odds.items():
        if gkey not in close_odds:
            continue
        cd   = close_odds[gkey]
        row  = {"game": gkey, "commence": od.get("commence","")}

        # Total movement
        if od.get("total") and cd.get("total"):
            row["total_open"]  = od["total"]
            row["total_close"] = cd["total"]
            row["total_move"]  = round(cd["total"] - od["total"], 2)

        # Run-line no-vig movement
        ohp = od.get("home_rl_price"); oap = od.get("away_rl_price")
        chp = cd.get("home_rl_price"); cap = cd.get("away_rl_price")
        if ohp and oap and chp and cap:
            open_nv  = nv_prob(ohp, oap)
            close_nv = nv_prob(chp, cap)
            row["rl_open_nv"]  = round(open_nv,  4)
            row["rl_close_nv"] = round(close_nv, 4)
            row["rl_abs_move"] = round(abs(close_nv - open_nv) * 100, 2)

        rows.append(row)

    saved[date_str] = {"games": len(rows), "data": rows}
    new_fetches += 2
    OUTPUT.write_text(json.dumps(saved))

    if (i % 15 == 0) or not open_odds:
        pct = i / len(all_dates) * 100
        print(f"  [{pct:5.1f}%] {date_str} | {len(rows):2d} games | credits rem: {rem} used: {used}")

print(f"\nFetch complete. {new_fetches} API calls made.\n")

# ── Analysis ─────────────────────────────────────────────────────
total_moves, rl_moves = [], []
game_count = 0

for date_str, rec in saved.items():
    for row in rec.get("data", []):
        game_count += 1
        if "total_move" in row:
            total_moves.append(row["total_move"])
        if "rl_abs_move" in row:
            rl_moves.append(row["rl_abs_move"])

print(f"{'='*62}")
print(f"  TIER 1: 2025 MLB PINNACLE LINE MOVEMENT  ({game_count} game/day pairs)")
print(f"{'='*62}")

if total_moves:
    abs_m = [abs(m) for m in total_moves]
    flat  = sum(1 for m in total_moves if m == 0)
    up    = sum(1 for m in total_moves if m > 0)
    down  = sum(1 for m in total_moves if m < 0)
    print(f"\nTOTALS  (n={len(total_moves)})")
    print(f"  Mean abs movement : {statistics.mean(abs_m):6.3f} pts")
    print(f"  Median abs movement:{statistics.median(abs_m):6.3f} pts")
    print(f"  Std dev            :{statistics.stdev(total_moves):6.3f} pts")
    print(f"  Direction → up / flat / down : {up/len(total_moves):.1%} / {flat/len(total_moves):.1%} / {down/len(total_moves):.1%}")
    print(f"  Moved >0.0 pts     : {sum(1 for m in abs_m if m>0.0)/len(abs_m):6.1%}")
    print(f"  Moved ≥0.5 pts     : {sum(1 for m in abs_m if m>=0.49)/len(abs_m):6.1%}")
    print(f"  Moved ≥1.0 pts     : {sum(1 for m in abs_m if m>=0.99)/len(abs_m):6.1%}")
    print(f"  Moved ≥2.0 pts     : {sum(1 for m in abs_m if m>=1.99)/len(abs_m):6.1%}")
    print(f"\n  ▶ A model beating close by avg +0.3pts would be top-tier.")
    print(f"  ▶ Noise floor ≈ {statistics.stdev(total_moves)/len(total_moves)**0.5:.3f} pts at n=1 season")

if rl_moves:
    print(f"\nRUN LINES  (n={len(rl_moves)})")
    print(f"  Mean abs NV move   : {statistics.mean(rl_moves):6.2f} pp")
    print(f"  Median abs NV move : {statistics.median(rl_moves):6.2f} pp")
    print(f"  Moved >1 pp        : {sum(1 for m in rl_moves if m>1)/len(rl_moves):6.1%}")
    print(f"  Moved >3 pp        : {sum(1 for m in rl_moves if m>3)/len(rl_moves):6.1%}")
    print(f"  Moved >5 pp        : {sum(1 for m in rl_moves if m>5)/len(rl_moves):6.1%}")

print(f"\nData saved → {OUTPUT}")
print(f"Tier 3 go/no-go: if mean abs total move ≥ 0.25 pts → proceed.")
