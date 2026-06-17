"""
Fetch 2025 MLB final scores and match to tier3 predictions.
Outputs calibration dataset: model_prob → actual_outcome.
"""
import json, requests, statistics
from pathlib import Path
from sklearn.isotonic import IsotonicRegression
import numpy as np

MLB = "https://statsapi.mlb.com/api/v1"

# Load tier1 dates for date range
tier1 = json.loads(Path("tier1_full_2025.json").read_text())
dates = sorted(d for d in tier1 if tier1[d].get("games", 0) > 0)

# ── 1. Fetch all 2025 scores ──────────────────────────────────────
SCORES_CACHE = Path("tier3_scores_2025.json")
if SCORES_CACHE.exists():
    scores = json.loads(SCORES_CACHE.read_text())
    print(f"Scores loaded from cache: {len(scores)} games")
else:
    print("Fetching 2025 game scores from MLB API...")
    r = requests.get(f"{MLB}/schedule", params={
        "sportId": 1, "gameType": "R",
        "startDate": dates[0], "endDate": dates[-1],
        "hydrate": "linescore",
    }, timeout=30)
    r.raise_for_status()

    scores = {}  # key: "date||Away @ Home" → {home_score, away_score, total}
    for de in r.json().get("dates", []):
        gd = de["date"]
        for g in de.get("games", []):
            state = g.get("status", {}).get("abstractGameState", "")
            if state != "Final":
                continue
            ht = g.get("teams", {}).get("home", {})
            at = g.get("teams", {}).get("away", {})
            hs = ht.get("score")
            as_ = at.get("score")
            hn = ht.get("team", {}).get("name", "")
            an = at.get("team", {}).get("name", "")
            if hs is None or as_ is None:
                continue
            total = hs + as_
            key = f"{gd}||{an} @ {hn}"
            scores[key] = {
                "home_score": hs, "away_score": as_,
                "total_runs": total,
                "home_win": hs > as_,
            }

    SCORES_CACHE.write_text(json.dumps(scores))
    print(f"  {len(scores)} final games cached")

# ── 2. Match scores to predictions ───────────────────────────────
NAME_MAP = {"Oakland Athletics": "Athletics"}
def norm(s):
    for o, n in NAME_MAP.items(): s = s.replace(o, n)
    return s

preds = json.loads(Path("tier3_predictions.json").read_text())
print(f"\nMatching scores to {len(preds)} predictions...")

matched = 0
calib_data = []  # (model_prob, actual_outcome, conf_bucket, market)

for p in preds:
    key = f"{p['date']}||{norm(p['game'])}"
    sc  = scores.get(key)
    if not sc:
        continue
    matched += 1
    total_line = p.get("total_open")
    if not total_line:
        continue

    actual_over = sc["total_runs"] > total_line   # True=OVER hit
    actual_push = sc["total_runs"] == total_line  # push (rare)
    if actual_push:
        continue

    # Total picks
    if p.get("edge_flagged") and p.get("total_pred") and p.get("total_conf"):
        conf    = p["total_conf"] / 100.0
        picked  = p["total_pred"]  # "OVER" or "UNDER"
        outcome = 1 if (picked == "OVER" and actual_over) or \
                       (picked == "UNDER" and not actual_over) else 0
        calib_data.append({
            "market": "total", "model_prob": conf,
            "outcome": outcome, "pred": picked,
            "conf_pct": p["total_conf"],
        })

print(f"  Score matches: {matched}/{len(preds)}")
print(f"  Calibration samples (totals): {len(calib_data)}")

# ── 3. Win rate by confidence bucket ─────────────────────────────
from collections import defaultdict
buckets = defaultdict(list)
for d in calib_data:
    buckets[d["conf_pct"]].append(d["outcome"])

print(f"\n{'='*55}")
print("CALIBRATION TABLE — Total picks")
print(f"{'='*55}")
print(f"{'Conf':>6}  {'n':>5}  {'Win%':>6}  {'vs Expected':>12}")
for conf in sorted(buckets):
    wins = buckets[conf]
    wr   = statistics.mean(wins)
    print(f"  {int(conf):3d}%   n={len(wins):3d}   {wr:.1%}   {'↑ better' if wr > conf/100 else '↓ worse'}")

# ── 4. Fit isotonic calibration ───────────────────────────────────
if len(calib_data) >= 20:
    X = np.array([d["model_prob"] for d in calib_data])
    y = np.array([d["outcome"]    for d in calib_data])
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(X, y)

    print(f"\nIsotonic calibration fitted on n={len(calib_data)} picks")
    print("Calibrated probabilities at key confidence levels:")
    for p_test in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        cal = ir.predict([p_test])[0]
        print(f"  Model {p_test:.0%} → calibrated {cal:.1%}")

    # Save calibrator
    import pickle
    with open("calibrator_totals_2025.pkl", "wb") as f:
        pickle.dump(ir, f)
    print("\nCalibrator saved → calibrator_totals_2025.pkl")
    print("Wire into V3 model_combine.py to replace calibrator=None")
else:
    print(f"\nOnly {len(calib_data)} samples — need ≥20 to fit calibration")
