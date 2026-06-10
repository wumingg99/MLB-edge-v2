import requests
import numpy as np
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
LEAGUE_AVG_TOTAL = 8.8
SEASON_WEIGHTS = {2022: 0.4, 2023: 0.6, 2024: 0.85, 2025: 1.0, 2026: 2.0}

def get_season_games(season):
    print(f"  Fetching MLB {season} schedule...", flush=True)
    games = []
    try:
        start = f"{season}-03-01"
        end = f"{season}-11-30"
        url = f"{MLB_API_BASE}/schedule"
        params = {
            "sportId": 1, "startDate": start, "endDate": end,
            "gameType": "R",
            "hydrate": "team,probablePitcher,linescore,venue"
        }
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get(
                    "abstractGameState", "")
                if status != "Final":
                    continue
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                linescore = game.get("linescore", {})
                home_score = linescore.get("teams", {}).get(
                    "home", {}).get("runs")
                away_score = linescore.get("teams", {}).get(
                    "away", {}).get("runs")
                if home_score is None or away_score is None:
                    continue
                home_team = home.get("team", {})
                away_team = away.get("team", {})
                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})
                date_str = date_entry.get("date", "")
                month = int(date_str[5:7]) if len(date_str) >= 7 else 6
                games.append({
                    "game_id": game.get("gamePk"),
                    "date": date_str,
                    "season": season,
                    "month": month,
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "home_team": home_team.get("name", ""),
                    "away_team": away_team.get("name", ""),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "has_pitchers": bool(home_pitcher and away_pitcher),
                    "venue": game.get("venue", {}).get("name", ""),
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                    "total": int(home_score) + int(away_score),
                })
        print(f"  Found {len(games)} games for {season}")
    except Exception as e:
        print(f"  Error fetching {season}: {e}")
    return games

def get_pitcher_stats_historical(pitcher_id, season):
    if not pitcher_id:
        return None
    try:
        url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
        params = {"stats": "season", "group": "pitching",
                  "season": season, "gameType": "R"}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        for sg in data.get("stats", []):
            splits = sg.get("splits", [])
            if splits:
                s = splits[0].get("stat", {})
                return {
                    "xfip": float(s.get("xfip", 4.50) or 4.50),
                    "fip": float(s.get("fip", 4.50) or 4.50),
                    "era": float(s.get("era", 4.50) or 4.50),
                    "k9": float(s.get("strikeoutsPer9Inn", 8.0) or 8.0),
                    "bb9": float(s.get("walksPer9Inn", 3.0) or 3.0),
                }
    except Exception:
        pass
    return None

def get_team_stats_historical(team_id, season):
    if not team_id:
        return None
    try:
        url = f"{MLB_API_BASE}/teams/{team_id}/stats"
        params = {"stats": "season", "group": "hitting,pitching",
                  "season": season, "gameType": "R"}
        r = requests.get(url, params=params, timeout=8)
        data = r.json()
        result = {}
        for sg in data.get("stats", []):
            group = sg.get("group", {}).get("displayName", "")
            splits = sg.get("splits", [])
            if not splits:
                continue
            s = splits[0].get("stat", {})
            if group == "hitting":
                result["ops"] = float(s.get("ops", 0.720) or 0.720)
                result["rpg"] = float(s.get("runs", 0) or 0) / max(
                    float(s.get("gamesPlayed", 1) or 1), 1)
            elif group == "pitching":
                result["team_era"] = float(s.get("era", 4.50) or 4.50)
                result["team_k9"] = float(
                    s.get("strikeoutsPer9Inn", 8.0) or 8.0)
        return result if result else None
    except Exception:
        return None

def default_pitcher():
    return {"xfip": 4.50, "fip": 4.50, "era": 4.50,
            "k9": 8.0, "bb9": 3.0}

def default_team():
    return {"ops": 0.720, "rpg": 4.5, "team_era": 4.50, "team_k9": 8.0}

def get_park_factors():
    return {
        "Coors Field": 1.35, "Fenway Park": 1.10,
        "Globe Life Field": 1.05, "Great American Ball Park": 1.12,
        "Wrigley Field": 1.08, "Oracle Park": 0.88,
        "Tropicana Field": 0.94, "Petco Park": 0.92,
        "T-Mobile Park": 0.91, "Dodger Stadium": 0.95,
    }

def build_features_historical(home_stats, away_stats, home_p, away_p,
                               park_factor, month, run_line, total):
    hs = home_stats or default_team()
    as_ = away_stats or default_team()
    hp = home_p or default_pitcher()
    ap = away_p or default_pitcher()

    home_rpg = float(hs.get("rpg", 4.5))
    away_rpg = float(as_.get("rpg", 4.5))
    home_ops = float(hs.get("ops", 0.720))
    away_ops = float(as_.get("ops", 0.720))
    home_era = float(hs.get("team_era", 4.50))
    away_era = float(as_.get("team_era", 4.50))
    xfip_h = float(hp.get("xfip", 4.50))
    xfip_a = float(ap.get("xfip", 4.50))
    k9_h = float(hp.get("k9", 8.0))
    k9_a = float(ap.get("k9", 8.0))
    bb9_h = float(hp.get("bb9", 3.0))
    bb9_a = float(ap.get("bb9", 3.0))

    # For historical, use season stats as both season and recent
    # (we don't have game logs for historical)
    home_recent_rpg = home_rpg
    away_recent_rpg = away_rpg
    home_recent_era = xfip_h
    away_recent_era = xfip_a

    xfip_comb = (xfip_h + xfip_a) / 2
    fip_comb = (float(hp.get("fip", 4.50)) +
                float(ap.get("fip", 4.50))) / 2
    k9_comb = (k9_h + k9_a) / 2
    bb9_comb = (bb9_h + bb9_a) / 2
    kbb = k9_comb / max(bb9_comb, 0.1)
    ops_comb = (home_ops + away_ops) / 2
    rpg_comb = (home_rpg + away_rpg) / 2
    bullpen = 4.2

    env_f = park_factor
    impl_season = (home_rpg + away_rpg) * env_f
    impl_recent = (home_recent_rpg + away_recent_rpg) * env_f
    implied = 0.6 * impl_recent + 0.4 * impl_season
    implied = max(3.0, min(implied, 20.0))

    vegas = total if total else LEAGUE_AVG_TOTAL
    total_gap = implied - vegas
    rl_norm = (run_line or -1.5) / 2
    home_str = (home_rpg - home_era)
    away_str = (away_rpg - away_era)
    str_diff = home_str - away_str
    fatigue = (1.0 if month <= 6 else 1.05 if month <= 8 else 1.10)

    return [
        xfip_comb, fip_comb, k9_comb, bb9_comb, kbb, xfip_h - xfip_a,
        (xfip_h + xfip_a) / 2, 0.0, k9_h, k9_a,
        ops_comb, rpg_comb, home_rpg - away_rpg, home_ops - away_ops,
        home_era - away_era, bullpen,
        rpg_comb, 0.0, home_rpg, away_rpg,
        0.500, 0.500,
        park_factor, park_factor, 1.0, 0.0, env_f,
        implied, total_gap, vegas, vegas - LEAGUE_AVG_TOTAL,
        rl_norm, str_diff, 0.0,
        0.0, 0.0, 0.0, 0.0,
        fatigue,
    ]

def train_on_historical():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, brier_score_loss
    from sklearn.utils.class_weight import compute_class_weight
    from xgboost import XGBClassifier
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    print("Building MLB v2 historical dataset...")
    print("Seasons: 2022-2026 | Features: 38 | Models: 5\n")

    X, y_total, y_runline, weights = [], [], [], []
    pitcher_cache = {}
    team_cache = {}
    park_factors = get_park_factors()
    total_games = total_skipped = 0

    for season in [2022, 2023, 2024, 2025, 2026]:
        season_weight = SEASON_WEIGHTS.get(season, 1.0)
        games = get_season_games(season)
        if not games:
            continue

        print(f"Processing {season} — {len(games)} games "
              f"(weight: {season_weight}x)...", flush=True)

        for game in games:
            try:
                home_id = game["home_team_id"]
                away_id = game["away_team_id"]
                home_p_id = game["home_pitcher_id"]
                away_p_id = game["away_pitcher_id"]

                ht_key = f"{home_id}_{season}"
                at_key = f"{away_id}_{season}"
                hp_key = f"{home_p_id}_{season}"
                ap_key = f"{away_p_id}_{season}"

                if ht_key not in team_cache:
                    team_cache[ht_key] = get_team_stats_historical(
                        home_id, season)
                if at_key not in team_cache:
                    team_cache[at_key] = get_team_stats_historical(
                        away_id, season)
                if hp_key not in pitcher_cache:
                    pitcher_cache[hp_key] = get_pitcher_stats_historical(
                        home_p_id, season)
                if ap_key not in pitcher_cache:
                    pitcher_cache[ap_key] = get_pitcher_stats_historical(
                        away_p_id, season)

                park = park_factors.get(game["venue"], 1.0)
                features = build_features_historical(
                    team_cache[ht_key], team_cache[at_key],
                    pitcher_cache[hp_key], pitcher_cache[ap_key],
                    park, game["month"], -1.5, LEAGUE_AVG_TOTAL)

                actual_total = game["total"]
                goes_over = 1 if actual_total > LEAGUE_AVG_TOTAL else 0
                home_margin = game["home_score"] - game["away_score"]
                covers_rl = 1 if home_margin > 1.5 else 0

                X.append(features)
                y_total.append(goes_over)
                y_runline.append(covers_rl)
                weights.append(season_weight)
                total_games += 1

            except Exception:
                total_skipped += 1
                continue

        print(f"  {season}: {len([g for g in games])} games processed",
              flush=True)

    print(f"\nTotal dataset: {total_games} games, {total_skipped} skipped")

    if len(X) < 500:
        print("Not enough data — using simulated training")
        from model import train_models
        train_models()
        return False

    X = np.array(X)
    y_total = np.array(y_total)
    y_runline = np.array(y_runline)
    sample_weights = np.array(weights)

    print(f"\nTraining v2 on {len(X)} real games...")
    print("Models: LR + RF + XGB + LGB + NN + Calibration\n")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, yt_train, yt_test, sw_train, _ = train_test_split(
        X_scaled, y_total, sample_weights,
        test_size=0.2, random_state=42)
    _, _, yr_train, yr_test, _, _ = train_test_split(
        X_scaled, y_runline, sample_weights,
        test_size=0.2, random_state=42)

    total_cw = dict(zip(np.unique(yt_train),
        compute_class_weight("balanced",
                             classes=np.unique(yt_train),
                             y=yt_train)))
    rl_cw = dict(zip(np.unique(yr_train),
        compute_class_weight("balanced",
                             classes=np.unique(yr_train),
                             y=yr_train)))

    models_total = {}
    models_runline = {}

    print("Training Logistic Regression...")
    lr_t = LogisticRegression(max_iter=2000, C=0.5,
                               class_weight=total_cw, random_state=42)
    lr_t.fit(X_train, yt_train, sample_weight=sw_train)
    models_total["lr"] = lr_t
    print(f"  LR Total: {accuracy_score(yt_test, lr_t.predict(X_test)):.3f}")
    lr_r = LogisticRegression(max_iter=2000, C=0.5,
                               class_weight=rl_cw, random_state=42)
    lr_r.fit(X_train, yr_train, sample_weight=sw_train)
    models_runline["lr"] = lr_r
    print(f"  LR RunLine: {accuracy_score(yr_test, lr_r.predict(X_test)):.3f}")

    print("Training Random Forest...")
    rf_t = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=20,
                                   class_weight=total_cw, random_state=42)
    rf_t.fit(X_train, yt_train, sample_weight=sw_train)
    models_total["rf"] = rf_t
    print(f"  RF Total: {accuracy_score(yt_test, rf_t.predict(X_test)):.3f}")
    rf_r = RandomForestClassifier(n_estimators=300, max_depth=8,
                                   min_samples_leaf=20,
                                   class_weight=rl_cw, random_state=42)
    rf_r.fit(X_train, yr_train, sample_weight=sw_train)
    models_runline["rf"] = rf_r
    print(f"  RF RunLine: {accuracy_score(yr_test, rf_r.predict(X_test)):.3f}")

    print("Training XGBoost...")
    sps_t = sum(yt_train == 0) / max(sum(yt_train == 1), 1)
    xgb_t = XGBClassifier(n_estimators=400, max_depth=5,
                           learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8,
                           scale_pos_weight=sps_t, random_state=42,
                           eval_metric="logloss", verbosity=0)
    xgb_t.fit(X_train, yt_train, sample_weight=sw_train)
    models_total["xgb"] = xgb_t
    print(f"  XGB Total: {accuracy_score(yt_test, xgb_t.predict(X_test)):.3f}")
    sps_r = sum(yr_train == 0) / max(sum(yr_train == 1), 1)
    xgb_r = XGBClassifier(n_estimators=400, max_depth=5,
                           learning_rate=0.03, subsample=0.8,
                           colsample_bytree=0.8,
                           scale_pos_weight=sps_r, random_state=42,
                           eval_metric="logloss", verbosity=0)
    xgb_r.fit(X_train, yr_train, sample_weight=sw_train)
    models_runline["xgb"] = xgb_r
    print(f"  XGB RunLine: {accuracy_score(yr_test, xgb_r.predict(X_test)):.3f}")

    if has_lgb:
        print("Training LightGBM...")
        import lightgbm as lgb
        lgb_t = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                                    num_leaves=31, random_state=42,
                                    verbose=-1, class_weight=total_cw)
        lgb_t.fit(X_train, yt_train, sample_weight=sw_train)
        models_total["lgb"] = lgb_t
        print(f"  LGB Total: {accuracy_score(yt_test, lgb_t.predict(X_test)):.3f}")
        lgb_r = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03,
                                    num_leaves=31, random_state=42,
                                    verbose=-1, class_weight=rl_cw)
        lgb_r.fit(X_train, yr_train, sample_weight=sw_train)
        models_runline["lgb"] = lgb_r
        print(f"  LGB RunLine: {accuracy_score(yr_test, lgb_r.predict(X_test)):.3f}")

    print("Training Neural Network...")
    nn_t = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                         max_iter=3000, learning_rate_init=0.001,
                         early_stopping=True, validation_fraction=0.1,
                         random_state=42)
    nn_t.fit(X_train, yt_train)
    models_total["nn"] = nn_t
    print(f"  NN Total: {accuracy_score(yt_test, nn_t.predict(X_test)):.3f}")
    nn_r = MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                         max_iter=3000, learning_rate_init=0.001,
                         early_stopping=True, validation_fraction=0.1,
                         random_state=42)
    nn_r.fit(X_train, yr_train)
    models_runline["nn"] = nn_r
    print(f"  NN RunLine: {accuracy_score(yr_test, nn_r.predict(X_test)):.3f}")

    # Calibration using isotonic regression
    print("\nCalibrating probabilities (isotonic regression)...")
    calibrators_total = {}
    calibrators_runline = {}
    for name, model in models_total.items():
        try:
            probs = model.predict_proba(X_test)[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(probs, yt_test)
            calibrators_total[name] = cal
        except Exception:
            pass
    for name, model in models_runline.items():
        try:
            probs = model.predict_proba(X_test)[:, 1]
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(probs, yr_test)
            calibrators_runline[name] = cal
        except Exception:
            pass
    print(f"  Calibrated {len(calibrators_total)} total models")
    print(f"  Calibrated {len(calibrators_runline)} runline models")

    with open("models.pkl", "wb") as f:
        pickle.dump({
            "models_total": models_total,
            "models_runline": models_runline,
            "scaler": scaler,
            "calibrators_total": calibrators_total,
            "calibrators_runline": calibrators_runline,
            "version": "v2",
            "games_count": len(X),
            "n_models": len(models_total),
        }, f)

    print(f"\n✅ V2 Training complete — {len(X)} real games")
    print(f"Models: {list(models_total.keys())}")
    print("models.pkl saved")
    return True

if __name__ == "__main__":
    train_on_historical()
