"""
Build umpire run factors from 2024-2025 completed games.
Computes each HP umpire's average total runs vs league average.
Outputs umpire_factors.json: {umpire_id: {"name": ..., "factor": ..., "n": ...}}
"""
import json, requests, time
from collections import defaultdict
from pathlib import Path

OUT = Path("/root/mlb-edge-v3/umpire_factors.json")

def fetch_season(season):
    r = requests.get("https://statsapi.mlb.com/api/v1/schedule", params={
        "sportId": 1, "season": season, "gameType": "R",
        "hydrate": "officials,linescore",
    }, timeout=30)
    r.raise_for_status()
    return r.json()

umpire_games = defaultdict(list)  # umpire_id -> list of total_runs
umpire_names = {}

for season in [2024, 2025]:
    print(f"Fetching {season}...")
    data = fetch_season(season)
    season_totals = []
    for date_entry in data.get("dates", []):
        for game in date_entry.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            ls = game.get("linescore", {}).get("teams", {})
            home_runs = ls.get("home", {}).get("runs")
            away_runs = ls.get("away", {}).get("runs")
            if home_runs is None or away_runs is None:
                continue
            total = home_runs + away_runs
            season_totals.append(total)
            for official in game.get("officials", []):
                if official.get("officialType") == "Home Plate":
                    uid = official["official"]["id"]
                    name = official["official"]["fullName"]
                    umpire_games[uid].append(total)
                    umpire_names[uid] = name
    league_avg = sum(season_totals) / len(season_totals) if season_totals else 9.0
    print(f"  {season}: {len(season_totals)} games, league avg {league_avg:.2f} runs")
    time.sleep(0.5)

# Compute factors (min 20 games for reliability)
all_runs = [r for runs in umpire_games.values() for r in runs]
overall_avg = sum(all_runs) / len(all_runs) if all_runs else 9.0
print(f"\nOverall avg: {overall_avg:.3f} runs/game")

factors = {}
for uid, runs in umpire_games.items():
    if len(runs) < 20:
        continue
    avg = sum(runs) / len(runs)
    factor = avg / overall_avg  # 1.0 = neutral, >1 = more runs
    factors[uid] = {
        "name": umpire_names[uid],
        "factor": round(factor, 4),
        "avg_runs": round(avg, 3),
        "n": len(runs),
    }

# Sort by factor descending for readability
sorted_f = dict(sorted(factors.items(), key=lambda x: x[1]["factor"], reverse=True))
OUT.write_text(json.dumps(sorted_f, indent=2))

print(f"\n{len(factors)} umpires with ≥20 games:")
print(f"{'Name':<25} {'n':>5}  {'Avg':>6}  {'Factor':>8}")
print("-" * 50)
for uid, d in sorted_f.items():
    bar = "▲" * min(int((d['factor']-1)*20+1), 4) if d['factor'] > 1.02 else \
          "▼" * min(int((1-d['factor'])*20+1), 4) if d['factor'] < 0.98 else " "
    print(f"  {d['name']:<23} n={d['n']:3d}  {d['avg_runs']:5.2f}  {d['factor']:.4f} {bar}")

print(f"\nSaved → {OUT}")
