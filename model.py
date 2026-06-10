import numpy as np
import pickle
import os
from datetime import datetime
import pytz

LEAGUE_AVG_TOTAL = 8.8

def build_features(context, total, run_line):
    try:
        hs = context.get("home_stats") or {}
        as_ = context.get("away_stats") or {}
        hf = context.get("home_form") or {}
        af = context.get("away_form") or {}
        hp = context.get("home_pitcher") or {}
        ap = context.get("away_pitcher") or {}
        hpr = context.get("home_pitcher_recent") or {}
        apr = context.get("away_pitcher_recent") or {}
        hb = context.get("home_bullpen") or {}
        ab = context.get("away_bullpen") or {}
        park_factor = float(context.get("park_factor", 1.0))
        park_hr = float(context.get("park_hr_factor", 1.0))
        weather = context.get("weather") or {}
        weather_factor = float(weather.get("weather_factor", 1.0))
        is_dome = bool(weather.get("is_dome", False))

        # Season stats
        home_rpg = float(hs.get("rpg", 4.5))
        away_rpg = float(as_.get("rpg", 4.5))
        home_ops = float(hs.get("ops", 0.720))
        away_ops = float(as_.get("ops", 0.720))
        home_era = float(hs.get("team_era", 4.50))
        away_era = float(as_.get("team_era", 4.50))

        # Recent form (last 10 games) — KEY v2 feature
        home_recent_rpg = float(hf.get("recent_rpg", home_rpg))
        away_recent_rpg = float(af.get("recent_rpg", away_rpg))
        home_recent_ops = float(hf.get("recent_ops", home_ops))
        away_recent_ops = float(af.get("recent_ops", away_ops))
        home_form_trend = float(hf.get("form_trend", 0))
        away_form_trend = float(af.get("form_trend", 0))
        home_recent_win_pct = float(hf.get("recent_win_pct", 0.500))
        away_recent_win_pct = float(af.get("recent_win_pct", 0.500))

        # Pitcher season stats
        home_xfip = float(hp.get("xfip", 4.50))
        away_xfip = float(ap.get("xfip", 4.50))
        home_fip = float(hp.get("fip", 4.50))
        away_fip = float(ap.get("fip", 4.50))
        home_k9 = float(hp.get("k9", 8.0))
        away_k9 = float(ap.get("k9", 8.0))
        home_bb9 = float(hp.get("bb9", 3.0))
        away_bb9 = float(ap.get("bb9", 3.0))

        # Pitcher recent form (last 3 starts) — KEY v2 feature
        home_recent_era = float(hpr.get("recent_era", home_xfip))
        away_recent_era = float(apr.get("recent_era", away_xfip))
        home_pitcher_trend = float(hpr.get("recent_trend", 0))
        away_pitcher_trend = float(apr.get("recent_trend", 0))
        home_recent_k9 = float(hpr.get("recent_k9", home_k9))
        away_recent_k9 = float(apr.get("recent_k9", away_k9))

        # Bullpen
        home_bullpen_era = float(hb.get("bullpen_era", 4.50))
        away_bullpen_era = float(ab.get("bullpen_era", 4.50))

        # Combined metrics
        xfip_combined = (home_xfip + away_xfip) / 2
        fip_combined = (home_fip + away_fip) / 2
        k9_combined = (home_k9 + away_k9) / 2
        bb9_combined = (home_bb9 + away_bb9) / 2
        kbb_ratio = k9_combined / max(bb9_combined, 0.1)
        ops_combined = (home_ops + away_ops) / 2
        rpg_combined = (home_rpg + away_rpg) / 2
        bullpen_combined = (home_bullpen_era + away_bullpen_era) / 2

        # Recent form combined
        recent_rpg_combined = (home_recent_rpg + away_recent_rpg) / 2
        recent_era_combined = (home_recent_era + away_recent_era) / 2
        form_momentum = (home_form_trend + away_form_trend) / 2
        pitcher_momentum = (home_pitcher_trend + away_pitcher_trend) / 2

        # Environment
        env_factor = park_factor * weather_factor
        dome_factor = 0.0 if is_dome else 1.0

        # Implied total using recent form (v2 uses recent, not season)
        implied_total_season = (home_rpg + away_rpg) * env_factor
        implied_total_recent = (home_recent_rpg + away_recent_rpg) * env_factor
        # Blend season and recent (60% recent, 40% season)
        implied_total = (0.6 * implied_total_recent +
                         0.4 * implied_total_season)
        implied_total = max(3.0, min(implied_total, 20.0))

        vegas_line = total if total else LEAGUE_AVG_TOTAL
        total_gap = implied_total - vegas_line

        # Run line context
        home_strength = (home_rpg - home_era +
                         (home_recent_win_pct - 0.5) * 2)
        away_strength = (away_rpg - away_era +
                         (away_recent_win_pct - 0.5) * 2)
        strength_diff = home_strength - away_strength
        run_line_norm = (run_line or (-1.5 if strength_diff > 0 else 1.5)) / 2

        # Season timing
        month = datetime.now(pytz.timezone("Asia/Singapore")).month
        fatigue = (1.0 if month <= 6 else
                   1.05 if month <= 8 else 1.10)

        # 38 features (v1 had 30)
        return [
            # Pitcher season stats (6)
            xfip_combined, fip_combined, k9_combined,
            bb9_combined, kbb_ratio, (home_xfip - away_xfip),

            # Pitcher recent form (4) — NEW in v2
            recent_era_combined, pitcher_momentum,
            home_recent_k9, away_recent_k9,

            # Team season stats (6)
            ops_combined, rpg_combined,
            (home_rpg - away_rpg), (home_ops - away_ops),
            (home_era - away_era), bullpen_combined,

            # Team recent form (6) — NEW in v2
            recent_rpg_combined, form_momentum,
            home_recent_rpg, away_recent_rpg,
            home_recent_win_pct, away_recent_win_pct,

            # Environment (5)
            park_factor, park_hr, weather_factor,
            dome_factor, env_factor,

            # Total context (4)
            implied_total, total_gap,
            vegas_line, (vegas_line - LEAGUE_AVG_TOTAL),

            # Run line context (3)
            run_line_norm, strength_diff,
            home_recent_win_pct - away_recent_win_pct,

            # Advanced metrics (4)
            (home_recent_rpg - home_rpg),   # home form vs season
            (away_recent_rpg - away_rpg),   # away form vs season
            (home_recent_era - home_xfip),  # pitcher form vs season
            (away_recent_era - away_xfip),  # pitcher form vs season

            # Timing (1)
            fatigue,
        ]
    except Exception as e:
        print(f"Feature error: {e}")
        return None

def poisson_total_prob(context, total):
    """Poisson regression for O/U — more statistically correct than logistic."""
    try:
        hs = context.get("home_stats") or {}
        as_ = context.get("away_stats") or {}
        hf = context.get("home_form") or {}
        af = context.get("away_form") or {}
        park_factor = float(context.get("park_factor", 1.0))
        weather_factor = float(context.get("weather") or {}).get(
            "weather_factor", 1.0)

        home_rpg = float(hs.get("rpg", 4.5))
        away_rpg = float(as_.get("rpg", 4.5))
        home_recent = float(hf.get("recent_rpg", home_rpg))
        away_recent = float(af.get("recent_rpg", away_rpg))

        # Blend season and recent
        home_exp = (0.6 * home_recent + 0.4 * home_rpg) * park_factor * weather_factor
        away_exp = (0.6 * away_recent + 0.4 * away_rpg) * park_factor * weather_factor

        home_exp = max(1.0, min(home_exp, 15.0))
        away_exp = max(1.0, min(away_exp, 15.0))

        # Poisson probability that total > vegas line
        from scipy.stats import nbinom
        vegas = total if total else LEAGUE_AVG_TOTAL
        # Negative binomial handles baseball overdispersion better than Poisson
        # Dispersion parameter r=5 calibrated for baseball run scoring
        def nb_pmf(k, mu, r=5.0):
            p = r / (r + mu)
            return nbinom.pmf(k, r, p)
        over_prob = 0.0
        for h in range(0, 25):
            for a in range(0, 25):
                if h + a > vegas:
                    over_prob += nb_pmf(h, home_exp) * nb_pmf(a, away_exp)
        return round(over_prob, 3)
    except Exception as e:
        print(f"Poisson error: {e}")
        return 0.5

def calibrate_probability(raw_prob, calibrator=None):
    """Apply isotonic regression calibration if available."""
    if calibrator is None:
        return raw_prob
    try:
        return float(calibrator.predict([[raw_prob]])[0])
    except Exception:
        return raw_prob

def train_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from xgboost import XGBClassifier
    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False
        print("LightGBM not available — using 4 models")

    print("Generating simulated training data for v2...")
    np.random.seed(42)
    X, y_total, y_runline = [], [], []

    for _ in range(2000):
        # Season stats
        home_rpg = np.random.normal(4.5, 0.8)
        away_rpg = np.random.normal(4.5, 0.8)
        home_ops = np.random.normal(0.720, 0.05)
        away_ops = np.random.normal(0.720, 0.05)
        home_era = np.random.normal(4.50, 0.8)
        away_era = np.random.normal(4.50, 0.8)
        xfip_h = np.random.normal(4.2, 0.8)
        xfip_a = np.random.normal(4.2, 0.8)
        k9_h = np.random.normal(8.5, 1.5)
        k9_a = np.random.normal(8.5, 1.5)
        bb9_h = np.random.normal(3.0, 0.8)
        bb9_a = np.random.normal(3.0, 0.8)

        # Recent form — correlated with season but with noise
        home_recent_rpg = home_rpg + np.random.normal(0, 0.5)
        away_recent_rpg = away_rpg + np.random.normal(0, 0.5)
        home_recent_era = xfip_h + np.random.normal(0, 0.5)
        away_recent_era = xfip_a + np.random.normal(0, 0.5)
        home_form_trend = np.random.normal(0, 0.3)
        away_form_trend = np.random.normal(0, 0.3)
        home_recent_win = np.random.uniform(0.3, 0.7)
        away_recent_win = np.random.uniform(0.3, 0.7)
        home_recent_k9 = k9_h + np.random.normal(0, 0.5)
        away_recent_k9 = k9_a + np.random.normal(0, 0.5)
        pitcher_trend_h = np.random.normal(0, 0.3)
        pitcher_trend_a = np.random.normal(0, 0.3)

        # Environment
        park_f = np.random.choice([0.88, 0.92, 0.95, 1.0, 1.05, 1.10, 1.35])
        weather_f = np.random.uniform(0.95, 1.08)
        is_dome = np.random.choice([0.0, 1.0], p=[0.7, 0.3])
        env_f = park_f * weather_f

        # Implied total
        impl_season = (home_rpg + away_rpg) * env_f
        impl_recent = (home_recent_rpg + away_recent_rpg) * env_f
        implied = 0.6 * impl_recent + 0.4 * impl_season
        implied = max(3.0, min(implied, 20.0))

        vegas = implied + np.random.uniform(-1.5, 1.5)
        total_gap = implied - vegas
        run_line = np.random.choice([-1.5, 1.5])
        strength_diff = ((home_rpg - home_era) - (away_rpg - away_era) +
                         (home_recent_win - away_recent_win) * 2)

        xfip_comb = (xfip_h + xfip_a) / 2
        fip_comb = (xfip_h * 0.9 + xfip_a * 0.9) / 2
        k9_comb = (k9_h + k9_a) / 2
        bb9_comb = (bb9_h + bb9_a) / 2
        kbb = k9_comb / max(bb9_comb, 0.1)
        ops_comb = (home_ops + away_ops) / 2
        rpg_comb = (home_rpg + away_rpg) / 2
        bullpen = np.random.normal(4.2, 0.5)
        recent_rpg_comb = (home_recent_rpg + away_recent_rpg) / 2
        recent_era_comb = (home_recent_era + away_recent_era) / 2
        form_mom = (home_form_trend + away_form_trend) / 2
        pitcher_mom = (pitcher_trend_h + pitcher_trend_a) / 2
        fatigue = 1.0

        features = [
            xfip_comb, fip_comb, k9_comb, bb9_comb, kbb, xfip_h - xfip_a,
            recent_era_comb, pitcher_mom, home_recent_k9, away_recent_k9,
            ops_comb, rpg_comb, home_rpg - away_rpg, home_ops - away_ops,
            home_era - away_era, bullpen,
            recent_rpg_comb, form_mom, home_recent_rpg, away_recent_rpg,
            home_recent_win, away_recent_win,
            park_f, park_f, weather_f, is_dome, env_f,
            implied, total_gap, vegas, vegas - LEAGUE_AVG_TOTAL,
            run_line / 2, strength_diff, home_recent_win - away_recent_win,
            home_recent_rpg - home_rpg, away_recent_rpg - away_rpg,
            home_recent_era - xfip_h, away_recent_era - xfip_a,
            fatigue,
        ]

        noise = np.random.normal(0, 1.5)
        actual = implied + noise
        goes_over = 1 if actual > vegas else 0
        rl_margin = strength_diff * 0.5 + np.random.normal(0, 2.0)
        home_is_fav = run_line < 0
        covers_rl = 1 if (rl_margin > 1.5 if home_is_fav else rl_margin < -1.5) else 0

        X.append(features)
        y_total.append(goes_over)
        y_runline.append(covers_rl)

    X = np.array(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, yt_train, yt_test = train_test_split(
        X_scaled, y_total, test_size=0.2, random_state=42)
    _, _, yr_train, yr_test = train_test_split(
        X_scaled, y_runline, test_size=0.2, random_state=42)

    models_total = {}
    models_runline = {}

    # LR
    lr_t = LogisticRegression(max_iter=1000, random_state=42)
    lr_t.fit(X_train, yt_train)
    models_total["lr"] = lr_t
    lr_r = LogisticRegression(max_iter=1000, random_state=42)
    lr_r.fit(X_train, yr_train)
    models_runline["lr"] = lr_r

    # RF
    rf_t = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_t.fit(X_train, yt_train)
    models_total["rf"] = rf_t
    rf_r = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_r.fit(X_train, yr_train)
    models_runline["rf"] = rf_r

    # XGB
    xgb_t = XGBClassifier(n_estimators=100, random_state=42,
                           eval_metric="logloss", verbosity=0)
    xgb_t.fit(X_train, yt_train)
    models_total["xgb"] = xgb_t
    xgb_r = XGBClassifier(n_estimators=100, random_state=42,
                           eval_metric="logloss", verbosity=0)
    xgb_r.fit(X_train, yr_train)
    models_runline["xgb"] = xgb_r

    # LightGBM (v2 new model)
    if has_lgb:
        lgb_t = lgb.LGBMClassifier(n_estimators=100, random_state=42,
                                    verbose=-1)
        lgb_t.fit(X_train, yt_train)
        models_total["lgb"] = lgb_t
        lgb_r = lgb.LGBMClassifier(n_estimators=100, random_state=42,
                                    verbose=-1)
        lgb_r.fit(X_train, yr_train)
        models_runline["lgb"] = lgb_r

    # NN
    nn_t = MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                         max_iter=2000, random_state=42)
    nn_t.fit(X_train, yt_train)
    models_total["nn"] = nn_t
    nn_r = MLPClassifier(hidden_layer_sizes=(64, 32, 16),
                         max_iter=2000, random_state=42)
    nn_r.fit(X_train, yr_train)
    models_runline["nn"] = nn_r

    with open("models.pkl", "wb") as f:
        pickle.dump({
            "models_total": models_total,
            "models_runline": models_runline,
            "scaler": scaler,
            "version": "v2",
            "n_models": len(models_total),
        }, f)
    print(f"V2 simulated models saved ({len(models_total)} models)")
    return models_total, models_runline, scaler

def load_models():
    if os.path.exists("models.pkl"):
        with open("models.pkl", "rb") as f:
            data = pickle.load(f)
        if "models_runline" in data:
            return (data["models_total"], data["models_runline"],
                    data["scaler"],
                    data.get("calibrators_total"),
                    data.get("calibrators_runline"))
    return None, None, None, None, None

def ensemble_predict(models, X_scaled, calibrators=None):
    weights = {"lr": 1, "rf": 2, "xgb": 3, "lgb": 3, "nn": 2}
    weighted_prob = total_weight = 0
    yes_votes = no_votes = 0
    raw_probs = []
    for name, model in models.items():
        prob = model.predict_proba(X_scaled)[0][1]
        raw_probs.append(prob)
        # Apply calibration if available
        if calibrators and name in calibrators:
            prob = calibrate_probability(prob, calibrators[name])
        w = weights.get(name, 1)
        weighted_prob += prob * w
        total_weight += w
        if prob > 0.5:
            yes_votes += 1
        else:
            no_votes += 1
    avg_prob = weighted_prob / total_weight
    return avg_prob, yes_votes, no_votes, len(models)

def monte_carlo_simulate(context, total, n=10000):
    np.random.seed(None)
    hs = context.get("home_stats") or {}
    as_ = context.get("away_stats") or {}
    hf = context.get("home_form") or {}
    af = context.get("away_form") or {}
    park_factor = float(context.get("park_factor", 1.0))
    weather_factor = float(
        (context.get("weather") or {}).get("weather_factor", 1.0))

    home_rpg = float(hs.get("rpg", 4.5))
    away_rpg = float(as_.get("rpg", 4.5))
    home_recent = float(hf.get("recent_rpg", home_rpg))
    away_recent = float(af.get("recent_rpg", away_rpg))

    # Blend season and recent
    home_exp = (0.6 * home_recent + 0.4 * home_rpg) * park_factor * weather_factor
    away_exp = (0.6 * away_recent + 0.4 * away_rpg) * park_factor * weather_factor
    home_exp = max(1.0, min(home_exp, 15.0))
    away_exp = max(1.0, min(away_exp, 15.0))

    home_scores = np.random.poisson(home_exp, n)
    away_scores = np.random.poisson(away_exp, n)
    totals = home_scores + away_scores

    vegas = total if total else LEAGUE_AVG_TOTAL
    home_wins = np.sum(home_scores > away_scores)
    ties = np.sum(home_scores == away_scores)
    home_wins_final = home_wins + int(ties * 0.54)

    mc_std = float(np.std(totals))

    return {
        "home_win_prob": round(home_wins_final / n, 3),
        "away_win_prob": round(1 - home_wins_final / n, 3),
        "over_prob": round(np.sum(totals > vegas) / n, 3),
        "simulated_avg_total": round(float(np.mean(totals)), 1),
        "mc_std": round(mc_std, 2),
    }

def predict_game(context, total, run_line):
    models_total, models_runline, scaler, cal_t, cal_r = load_models()
    if models_total is None:
        train_models()
        models_total, models_runline, scaler, cal_t, cal_r = load_models()

    features = build_features(context, total, run_line)
    if features is None:
        return None

    f = np.array(features).reshape(1, -1)
    try:
        f_scaled = scaler.transform(f)
    except Exception:
        train_models()
        models_total, models_runline, scaler, cal_t, cal_r = load_models()
        f_scaled = scaler.transform(f)

    # Ensemble predictions with calibration
    total_prob, total_yes, total_no, total_count = ensemble_predict(
        models_total, f_scaled, cal_t)
    rl_prob, rl_yes, rl_no, rl_count = ensemble_predict(
        models_runline, f_scaled, cal_r)

    # Monte Carlo simulation
    mc = monte_carlo_simulate(context, total)
    mc_std = mc.get("mc_std", 3.0)

    # Poisson O/U probability
    poisson_prob = poisson_total_prob(context, total)

    # Implied total
    hs = context.get("home_stats") or {}
    as_ = context.get("away_stats") or {}
    hf = context.get("home_form") or {}
    af = context.get("away_form") or {}
    park_factor = float(context.get("park_factor", 1.0))
    weather_factor = float(
        (context.get("weather") or {}).get("weather_factor", 1.0))

    home_rpg = float(hs.get("rpg", 4.5))
    away_rpg = float(as_.get("rpg", 4.5))
    home_recent = float(hf.get("recent_rpg", home_rpg))
    away_recent = float(af.get("recent_rpg", away_rpg))

    impl_season = (home_rpg + away_rpg) * park_factor * weather_factor
    impl_recent = (home_recent + away_recent) * park_factor * weather_factor
    implied_total = 0.6 * impl_recent + 0.4 * impl_season
    implied_total = max(3.0, min(implied_total, 20.0))
    our_total = round(implied_total, 1)
    vegas_line = total if total else LEAGUE_AVG_TOTAL
    total_gap = round(our_total - vegas_line, 1)

    # MC vs formula agreement check
    mc_formula_diff = abs(implied_total - mc["simulated_avg_total"])

    # Blend total_prob with poisson for better O/U
    blended_over_prob = (total_prob * 0.5 + poisson_prob * 0.3 +
                         mc["over_prob"] * 0.2)

    if total_gap > 0:
        total_pred = "OVER"
        total_votes = total_yes
        total_conf = round(blended_over_prob * 100, 1)
    else:
        total_pred = "UNDER"
        total_votes = total_no
        total_conf = round((1 - blended_over_prob) * 100, 1)

    # Run line direction
    home_is_fav = (run_line or -1.5) < 0
    if rl_prob > 0.5:
        rl_pred = "HOME -1.5" if home_is_fav else "HOME +1.5"
        rl_votes = rl_yes
    else:
        rl_pred = "AWAY +1.5" if home_is_fav else "AWAY -1.5"
        rl_votes = rl_no
    rl_conf = round(max(rl_prob, 1 - rl_prob) * 100, 1)

    from config import (MIN_CONFIDENCE, RL_MIN_CONFIDENCE, MIN_MODELS_AGREE,
                        EDGE_THRESHOLD, MC_STD_MAX, MC_FORMULA_MAX_DIFF,
                        BET_CONFIDENCE, FULL_BET_CONFIDENCE)

    # V2 CONVERGENCE FILTER — all signals must agree
    mc_agrees_with_pred = (
        (total_pred == "OVER" and mc["over_prob"] > 0.50) or
        (total_pred == "UNDER" and mc["over_prob"] < 0.50)
    )
    poisson_agrees = (
        (total_pred == "OVER" and poisson_prob > 0.50) or
        (total_pred == "UNDER" and poisson_prob < 0.50)
    )

    # O/U edge — all three signals must converge
    edge_flagged = (
        abs(total_gap) >= EDGE_THRESHOLD and
        total_votes >= MIN_MODELS_AGREE and
        total_conf >= MIN_CONFIDENCE and
        mc_agrees_with_pred and          # MC agrees
        poisson_agrees and               # Poisson agrees
        mc_std <= MC_STD_MAX and         # Low variance game
        mc_formula_diff <= MC_FORMULA_MAX_DIFF and  # Formula ≈ MC
        context.get("has_real_pitchers", False)      # Confirmed starters
    )

    # RL edge
    rl_edge_flagged = (
        rl_votes >= MIN_MODELS_AGREE and
        rl_conf >= RL_MIN_CONFIDENCE and
        mc_std <= MC_STD_MAX             # Low variance game
    )

    # Confidence bands for bet sizing
    if rl_conf >= FULL_BET_CONFIDENCE:
        bet_size = "FULL"
    elif rl_conf >= BET_CONFIDENCE:
        bet_size = "HALF"
    else:
        bet_size = "SKIP"

    return {
        "our_total": our_total,
        "total_gap": total_gap,
        "total_pred": total_pred,
        "total_conf": total_conf,
        "total_votes": total_votes,
        "total_models": total_count,
        "rl_pred": rl_pred,
        "rl_conf": rl_conf,
        "rl_votes": rl_votes,
        "rl_models": rl_count,
        "home_win_prob": round(mc["home_win_prob"], 3),
        "away_win_prob": round(mc["away_win_prob"], 3),
        "mc_avg_total": mc["simulated_avg_total"],
        "mc_over_prob": mc["over_prob"],
        "mc_std": mc_std,
        "mc_formula_diff": round(mc_formula_diff, 2),
        "poisson_over_prob": poisson_prob,
        "mc_agrees": mc_agrees_with_pred,
        "poisson_agrees": poisson_agrees,
        "edge_flagged": edge_flagged,
        "rl_edge_flagged": rl_edge_flagged,
        "bet_size": bet_size,
        "has_real_pitchers": context.get("has_real_pitchers", False),
    }
