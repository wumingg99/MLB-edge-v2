import math
import os
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np


MODEL_VERSION = "v3-price-aware-2026-06"
MODEL_PATH = Path(__file__).with_name("models.pkl")
LEAGUE_AVG_RPG = 4.5
LEAGUE_AVG_RA9 = 4.5
LEAGUE_AVG_OPS = 0.720
FIP_CONSTANT = 3.10
LEAGUE_HR_PER_FLY_BALL = 0.12

FEATURE_NAMES = (
    "home_rpg", "away_rpg", "home_ra9", "away_ra9",
    "home_ops", "away_ops", "home_win_pct", "away_win_pct",
    "home_recent_rpg", "away_recent_rpg",
    "home_recent_ra9", "away_recent_ra9",
    "home_recent_ops", "away_recent_ops",
    "home_recent_win_pct", "away_recent_win_pct",
    "home_ema_rpg", "away_ema_rpg", "home_ema_ra9", "away_ema_ra9",
    "home_pitcher_era", "away_pitcher_era",
    "home_pitcher_fip", "away_pitcher_fip",
    "home_pitcher_xfip", "away_pitcher_xfip",
    "home_pitcher_k9", "away_pitcher_k9",
    "home_pitcher_bb9", "away_pitcher_bb9",
    "home_pitcher_whip", "away_pitcher_whip",
    "home_pitcher_ip", "away_pitcher_ip",
    "home_recent_pitcher_era", "away_recent_pitcher_era",
    "home_recent_pitcher_k9", "away_recent_pitcher_k9",
    "home_recent_pitcher_bb9", "away_recent_pitcher_bb9",
    "home_bullpen_era", "away_bullpen_era",
    "home_bullpen_workload", "away_bullpen_workload",
    "park_run_factor", "park_hr_factor", "weather_factor", "is_open_air",
    "home_rest_days", "away_rest_days",
    "month_sin", "month_cos", "season_progress", "data_quality",
)


def _safe_float(value, default):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float(default)
    except (TypeError, ValueError):
        return float(default)


def parse_innings(value):
    """Convert baseball innings notation (12.1 == 12 1/3) to innings."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text:
        return 0.0
    if "." not in text:
        return _safe_float(text, 0.0)
    whole, partial = text.split(".", 1)
    outs = int(partial[:1]) if partial[:1].isdigit() else 0
    if outs not in (0, 1, 2):
        return _safe_float(text, 0.0)
    return int(whole or 0) + outs / 3.0


def compute_pitching_metrics(stat):
    """Compute rate statistics MLB Stats API does not supply directly."""
    stat = stat or {}
    outs = _safe_float(stat.get("outs"), 0.0)
    innings = outs / 3.0 if outs > 0 else parse_innings(
        stat.get("inningsPitched")
    )
    innings = max(innings, 0.0)
    earned_runs = _safe_float(stat.get("earnedRuns"), 0.0)
    strikeouts = _safe_float(
        stat.get("strikeOuts", stat.get("strikeouts")), 0.0
    )
    walks = _safe_float(stat.get("baseOnBalls"), 0.0)
    hit_batters = _safe_float(stat.get("hitBatsmen"), 0.0)
    home_runs = _safe_float(stat.get("homeRuns"), 0.0)
    hits = _safe_float(stat.get("hits"), 0.0)
    air_outs = _safe_float(stat.get("airOuts"), 0.0)

    if innings <= 0:
        return {
            "era": 4.50, "fip": 4.50, "xfip": 4.50,
            "k9": 8.0, "bb9": 3.0, "whip": 1.30,
            "ip": 0.0, "sample_size": 0.0,
        }

    era = 9.0 * earned_runs / innings
    fip = (
        13.0 * home_runs
        + 3.0 * (walks + hit_batters)
        - 2.0 * strikeouts
    ) / innings + FIP_CONSTANT
    expected_home_runs = (
        air_outs * LEAGUE_HR_PER_FLY_BALL
        if air_outs > 0 else home_runs
    )
    xfip = (
        13.0 * expected_home_runs
        + 3.0 * (walks + hit_batters)
        - 2.0 * strikeouts
    ) / innings + FIP_CONSTANT
    return {
        "era": float(np.clip(era, 0.0, 15.0)),
        "fip": float(np.clip(fip, 0.0, 15.0)),
        "xfip": float(np.clip(xfip, 0.0, 15.0)),
        "k9": float(np.clip(9.0 * strikeouts / innings, 0.0, 25.0)),
        "bb9": float(np.clip(9.0 * walks / innings, 0.0, 15.0)),
        "whip": float(np.clip((walks + hits) / innings, 0.0, 5.0)),
        "ip": float(innings),
        "sample_size": float(min(innings / 50.0, 1.0)),
    }


def american_to_implied(price):
    price = _safe_float(price, 0.0)
    if price == 0:
        return None
    return 100.0 / (price + 100.0) if price > 0 else -price / (-price + 100.0)


def american_profit(price):
    price = _safe_float(price, 0.0)
    if price == 0:
        return None
    return price / 100.0 if price > 0 else 100.0 / -price


def no_vig_probabilities(price_a, price_b):
    implied_a = american_to_implied(price_a)
    implied_b = american_to_implied(price_b)
    if implied_a is None or implied_b is None:
        return None, None
    total = implied_a + implied_b
    return (
        (implied_a / total, implied_b / total)
        if total > 0 else (None, None)
    )


def expected_value(p_win, p_loss, price):
    profit = american_profit(price)
    return None if profit is None else float(p_win * profit - p_loss)


def quarter_kelly(p_win, p_loss, price, fraction=0.25, cap=0.02):
    profit = american_profit(price)
    if profit is None or profit <= 0:
        return 0.0
    full_kelly = (profit * p_win - p_loss) / profit
    return float(np.clip(full_kelly * fraction, 0.0, cap))


def grade_total(actual_total, line, side):
    actual_total = _safe_float(actual_total, 0.0)
    line = _safe_float(line, 0.0)
    if math.isclose(actual_total, line, abs_tol=1e-9):
        return "PUSH"
    if side == "OVER":
        return "WIN" if actual_total > line else "LOSS"
    if side == "UNDER":
        return "WIN" if actual_total < line else "LOSS"
    return "INVALID"


def grade_spread(home_margin, side, point):
    home_margin = _safe_float(home_margin, 0.0)
    point = _safe_float(point, 0.0)
    adjusted = home_margin + point if side == "HOME" else -home_margin + point
    if side not in ("HOME", "AWAY"):
        return "INVALID"
    if math.isclose(adjusted, 0.0, abs_tol=1e-9):
        return "PUSH"
    return "WIN" if adjusted > 0 else "LOSS"


def _context_date(context):
    raw = context.get("as_of") or context.get("game_date")
    if isinstance(raw, datetime):
        return raw
    if raw:
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.utcnow()


def _derive_data_quality(context):
    if context.get("data_quality") is not None:
        return float(np.clip(_safe_float(context["data_quality"], 0.0), 0, 1))
    hp = context.get("home_pitcher") or {}
    ap = context.get("away_pitcher") or {}
    hs = context.get("home_stats") or {}
    away_stats = context.get("away_stats") or {}
    components = [
        1.0 if context.get("has_real_pitchers") else 0.0,
        min(_safe_float(hp.get("ip"), 0.0) / 30.0, 1.0),
        min(_safe_float(ap.get("ip"), 0.0) / 30.0, 1.0),
        min(_safe_float(hs.get("games"), 0.0) / 20.0, 1.0),
        min(_safe_float(away_stats.get("games"), 0.0) / 20.0, 1.0),
        1.0 if context.get("weather") else 0.5,
    ]
    return float(np.mean(components))


def build_features(context, total=None, run_line=None):
    """Build line-independent features available before first pitch."""
    del total, run_line
    try:
        hs = context.get("home_stats") or {}
        away_stats = context.get("away_stats") or {}
        hf = context.get("home_form") or {}
        af = context.get("away_form") or {}
        hp = context.get("home_pitcher") or {}
        ap = context.get("away_pitcher") or {}
        hpr = context.get("home_pitcher_recent") or {}
        apr = context.get("away_pitcher_recent") or {}
        hb = context.get("home_bullpen") or {}
        ab = context.get("away_bullpen") or {}
        weather = context.get("weather") or {}

        as_of = _context_date(context)
        month_angle = 2.0 * math.pi * (as_of.month - 1) / 12.0
        season_start = datetime(as_of.year, 3, 20, tzinfo=as_of.tzinfo)
        season_end = datetime(as_of.year, 10, 1, tzinfo=as_of.tzinfo)
        season_days = max((season_end - season_start).days, 1)
        progress = np.clip((as_of - season_start).days / season_days, 0, 1)
        quality = _derive_data_quality(context)
        home_ra9 = _safe_float(
            hs.get("ra9", hs.get("team_era")), LEAGUE_AVG_RA9
        )
        away_ra9 = _safe_float(
            away_stats.get("ra9", away_stats.get("team_era")),
            LEAGUE_AVG_RA9,
        )

        values = [
            _safe_float(hs.get("rpg"), LEAGUE_AVG_RPG),
            _safe_float(away_stats.get("rpg"), LEAGUE_AVG_RPG),
            home_ra9, away_ra9,
            _safe_float(hs.get("ops"), LEAGUE_AVG_OPS),
            _safe_float(away_stats.get("ops"), LEAGUE_AVG_OPS),
            _safe_float(hs.get("win_pct"), 0.5),
            _safe_float(away_stats.get("win_pct"), 0.5),
            _safe_float(hf.get("recent_rpg"), LEAGUE_AVG_RPG),
            _safe_float(af.get("recent_rpg"), LEAGUE_AVG_RPG),
            _safe_float(hf.get("recent_ra9"), home_ra9),
            _safe_float(af.get("recent_ra9"), away_ra9),
            _safe_float(hf.get("recent_ops"), hs.get("ops", LEAGUE_AVG_OPS)),
            _safe_float(
                af.get("recent_ops"), away_stats.get("ops", LEAGUE_AVG_OPS)
            ),
            _safe_float(hf.get("recent_win_pct"), 0.5),
            _safe_float(af.get("recent_win_pct"), 0.5),
            _safe_float(hf.get("ema_rpg"), hs.get("rpg", LEAGUE_AVG_RPG)),
            _safe_float(
                af.get("ema_rpg"), away_stats.get("rpg", LEAGUE_AVG_RPG)
            ),
            _safe_float(hf.get("ema_ra9"), home_ra9),
            _safe_float(af.get("ema_ra9"), away_ra9),
            _safe_float(hp.get("era"), 4.5),
            _safe_float(ap.get("era"), 4.5),
            _safe_float(hp.get("fip"), 4.5),
            _safe_float(ap.get("fip"), 4.5),
            _safe_float(hp.get("xfip"), 4.5),
            _safe_float(ap.get("xfip"), 4.5),
            _safe_float(hp.get("k9"), 8.0),
            _safe_float(ap.get("k9"), 8.0),
            _safe_float(hp.get("bb9"), 3.0),
            _safe_float(ap.get("bb9"), 3.0),
            _safe_float(hp.get("whip"), 1.3),
            _safe_float(ap.get("whip"), 1.3),
            min(_safe_float(hp.get("ip"), 0.0), 250.0),
            min(_safe_float(ap.get("ip"), 0.0), 250.0),
            _safe_float(hpr.get("recent_era"), hp.get("era", 4.5)),
            _safe_float(apr.get("recent_era"), ap.get("era", 4.5)),
            _safe_float(hpr.get("recent_k9"), hp.get("k9", 8.0)),
            _safe_float(apr.get("recent_k9"), ap.get("k9", 8.0)),
            _safe_float(hpr.get("recent_bb9"), hp.get("bb9", 3.0)),
            _safe_float(apr.get("recent_bb9"), ap.get("bb9", 3.0)),
            _safe_float(hb.get("bullpen_era"), 4.5),
            _safe_float(ab.get("bullpen_era"), 4.5),
            _safe_float(hb.get("bullpen_workload"), 0.0),
            _safe_float(ab.get("bullpen_workload"), 0.0),
            _safe_float(context.get("park_factor"), 1.0),
            _safe_float(context.get("park_hr_factor"), 1.0),
            _safe_float(weather.get("weather_factor"), 1.0),
            0.0 if bool(weather.get("is_dome")) else 1.0,
            min(_safe_float(context.get("home_rest_days"), 3.0), 10.0),
            min(_safe_float(context.get("away_rest_days"), 3.0), 10.0),
            math.sin(month_angle), math.cos(month_angle),
            float(progress), quality,
        ]
        if len(values) != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature schema mismatch: {len(values)} != {len(FEATURE_NAMES)}"
            )
        return values
    except Exception as exc:
        print(f"Feature error: {exc}")
        return None


def _make_regressors(random_state=42):
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "ridge": Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=12.0))]
        ),
        "rf": RandomForestRegressor(
            n_estimators=350, max_depth=9, min_samples_leaf=15,
            max_features=0.75, n_jobs=-1, random_state=random_state,
        ),
        "extra": ExtraTreesRegressor(
            n_estimators=350, max_depth=10, min_samples_leaf=12,
            max_features=0.8, n_jobs=-1, random_state=random_state + 1,
        ),
        "hist": HistGradientBoostingRegressor(
            max_iter=250, learning_rate=0.035, max_leaf_nodes=20,
            min_samples_leaf=25, l2_regularization=2.0,
            random_state=random_state,
        ),
        "gbr": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.025, max_depth=2,
            min_samples_leaf=20, loss="huber", random_state=random_state,
        ),
    }


def _fit_regressor(model, X, y, sample_weight):
    try:
        if hasattr(model, "named_steps"):
            model.fit(X, y, model__sample_weight=sample_weight)
        else:
            model.fit(X, y, sample_weight=sample_weight)
    except TypeError:
        model.fit(X, y)


def _prediction_matrix(models, X):
    return np.column_stack(
        [np.asarray(model.predict(X), dtype=float) for model in models.values()]
    )


def _weights_from_calibration(predictions, actual):
    rmse = np.sqrt(np.mean((predictions - actual[:, None]) ** 2, axis=0))
    inverse = 1.0 / np.maximum(rmse, 0.10)
    return inverse / inverse.sum()


def _ensemble(predictions, weights):
    return np.asarray(predictions, dtype=float) @ np.asarray(weights, dtype=float)


def _market_test_metrics(
    records,
    total_pred,
    margin_pred,
    residual_pairs,
):
    total_brier_values = []
    total_roi_values = []
    spread_brier_values = []
    spread_roi_values = []
    residual_total = residual_pairs[:, 0]
    residual_margin = residual_pairs[:, 1]
    for idx, record in enumerate(records):
        market = record.get("market") or {}
        line = market.get("total_line")
        over_price = market.get("over_price")
        under_price = market.get("under_price")
        if line is not None and over_price is not None and under_price is not None:
            total_samples = np.rint(
                np.clip(total_pred[idx] + residual_total, 0, 30)
            )
            p_over = float(np.mean(total_samples > float(line)))
            p_under = float(np.mean(total_samples < float(line)))
            if record["actual_total"] != float(line):
                actual_over = float(record["actual_total"] > float(line))
                conditional_over = p_over / max(p_over + p_under, 1e-9)
                total_brier_values.append(
                    (conditional_over - actual_over) ** 2
                )
            ev_over = expected_value(p_over, p_under, over_price)
            ev_under = expected_value(p_under, p_over, under_price)
            side = (
                "OVER"
                if (ev_over if ev_over is not None else -99)
                >= (ev_under if ev_under is not None else -99)
                else "UNDER"
            )
            price = over_price if side == "OVER" else under_price
            result = grade_total(record["actual_total"], line, side)
            profit = american_profit(price) or 0.0
            total_roi_values.append(
                profit
                if result == "WIN"
                else -1.0
                if result == "LOSS"
                else 0.0
            )

        home_spread = market.get("home_spread")
        away_spread = market.get("away_spread")
        home_price = market.get("home_price")
        away_price = market.get("away_price")
        if (
            home_spread is None
            or away_spread is None
            or home_price is None
            or away_price is None
        ):
            continue
        margin_samples = np.rint(
            np.clip(margin_pred[idx] + residual_margin, -20, 20)
        )
        home_adjusted = margin_samples + float(home_spread)
        away_adjusted = -margin_samples + float(away_spread)
        p_home = float(np.mean(home_adjusted > 0))
        p_home_loss = float(np.mean(home_adjusted < 0))
        p_away = float(np.mean(away_adjusted > 0))
        p_away_loss = float(np.mean(away_adjusted < 0))
        actual_home_adjusted = (
            record["home_margin"] + float(home_spread)
        )
        if not math.isclose(actual_home_adjusted, 0.0, abs_tol=1e-9):
            actual_home_cover = float(actual_home_adjusted > 0)
            conditional_home = p_home / max(p_home + p_home_loss, 1e-9)
            spread_brier_values.append(
                (conditional_home - actual_home_cover) ** 2
            )
        ev_home = expected_value(p_home, p_home_loss, home_price)
        ev_away = expected_value(p_away, p_away_loss, away_price)
        if (
            (ev_home if ev_home is not None else -99)
            >= (ev_away if ev_away is not None else -99)
        ):
            side, point, price = "HOME", home_spread, home_price
        else:
            side, point, price = "AWAY", away_spread, away_price
        result = grade_spread(record["home_margin"], side, point)
        profit = american_profit(price) or 0.0
        spread_roi_values.append(
            profit
            if result == "WIN"
            else -1.0
            if result == "LOSS"
            else 0.0
        )
    return {
        "total_market_rows": len(total_roi_values),
        "total_brier": (
            float(np.mean(total_brier_values))
            if total_brier_values else None
        ),
        "total_naive_selection_roi": (
            float(np.mean(total_roi_values))
            if total_roi_values else None
        ),
        "spread_market_rows": len(spread_roi_values),
        "spread_brier": (
            float(np.mean(spread_brier_values))
            if spread_brier_values else None
        ),
        "spread_naive_selection_roi": (
            float(np.mean(spread_roi_values))
            if spread_roi_values else None
        ),
    }


def train_models(records, model_path=MODEL_PATH):
    """Train only on chronological, pregame feature records."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error

    if records is None:
        raise ValueError("Historical records are required; synthetic training is disabled")
    clean = [
        row for row in records
        if row.get("features") is not None
        and len(row["features"]) == len(FEATURE_NAMES)
        and row.get("date")
        and row.get("actual_total") is not None
        and row.get("home_margin") is not None
    ]
    clean.sort(key=lambda row: str(row["date"]))
    if len(clean) < 500:
        raise ValueError(
            f"At least 500 chronological games are required, got {len(clean)}"
        )

    X = np.asarray([row["features"] for row in clean], dtype=float)
    y_total = np.asarray([row["actual_total"] for row in clean], dtype=float)
    y_margin = np.asarray([row["home_margin"] for row in clean], dtype=float)
    sample_weight = np.asarray(
        [max(_safe_float(row.get("sample_weight"), 1.0), 0.05) for row in clean]
    )
    train_end = max(int(len(clean) * 0.65), 1)
    calibration_end = max(int(len(clean) * 0.80), train_end + 1)
    calibration_end = min(calibration_end, len(clean) - 1)
    train_slice = slice(0, train_end)
    calibration_slice = slice(train_end, calibration_end)
    test_slice = slice(calibration_end, len(clean))

    total_models = _make_regressors(42)
    margin_models = _make_regressors(142)
    for model in total_models.values():
        _fit_regressor(
            model, X[train_slice], y_total[train_slice],
            sample_weight[train_slice],
        )
    for model in margin_models.values():
        _fit_regressor(
            model, X[train_slice], y_margin[train_slice],
            sample_weight[train_slice],
        )

    total_cal_matrix = _prediction_matrix(total_models, X[calibration_slice])
    margin_cal_matrix = _prediction_matrix(margin_models, X[calibration_slice])
    total_weights = _weights_from_calibration(
        total_cal_matrix, y_total[calibration_slice]
    )
    margin_weights = _weights_from_calibration(
        margin_cal_matrix, y_margin[calibration_slice]
    )
    total_cal_pred = _ensemble(total_cal_matrix, total_weights)
    margin_cal_pred = _ensemble(margin_cal_matrix, margin_weights)
    residual_pairs = np.column_stack([
        y_total[calibration_slice] - total_cal_pred,
        y_margin[calibration_slice] - margin_cal_pred,
    ])

    total_test_matrix = _prediction_matrix(total_models, X[test_slice])
    margin_test_matrix = _prediction_matrix(margin_models, X[test_slice])
    total_test_pred = _ensemble(total_test_matrix, total_weights)
    margin_test_pred = _ensemble(margin_test_matrix, margin_weights)
    metrics = {
        "total_mae": float(mean_absolute_error(
            y_total[test_slice], total_test_pred
        )),
        "total_rmse": float(mean_squared_error(
            y_total[test_slice], total_test_pred
        ) ** 0.5),
        "margin_mae": float(mean_absolute_error(
            y_margin[test_slice], margin_test_pred
        )),
        "margin_rmse": float(mean_squared_error(
            y_margin[test_slice], margin_test_pred
        ) ** 0.5),
    }
    metrics.update(_market_test_metrics(
        clean[calibration_end:],
        total_test_pred,
        margin_test_pred,
        residual_pairs,
    ))

    bundle = {
        "version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "models_total": total_models,
        "models_margin": margin_models,
        "weights_total": dict(zip(total_models.keys(), total_weights)),
        "weights_margin": dict(zip(margin_models.keys(), margin_weights)),
        "residual_pairs": residual_pairs,
        "train_rows": train_end,
        "calibration_rows": calibration_end - train_end,
        "test_rows": len(clean) - calibration_end,
        "train_start": str(clean[0]["date"]),
        "train_end": str(clean[train_end - 1]["date"]),
        "calibration_end": str(clean[calibration_end - 1]["date"]),
        "test_end": str(clean[-1]["date"]),
        "metrics": metrics,
    }
    model_path = Path(model_path)
    temporary_path = model_path.with_suffix(".tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(bundle, handle)
    os.replace(temporary_path, model_path)
    return bundle


_MODEL_BUNDLE_CACHE = {"bundle": None, "mtime": None, "path": None}


def load_models(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        mtime = model_path.stat().st_mtime
        cache = _MODEL_BUNDLE_CACHE
        if (cache["bundle"] is not None
                and cache["mtime"] == mtime
                and cache["path"] == str(model_path)):
            return cache["bundle"]
        with model_path.open("rb") as handle:
            bundle = pickle.load(handle)
        if (
            bundle.get("version") != MODEL_VERSION
            or tuple(bundle.get("feature_names", ())) != FEATURE_NAMES
            or "models_margin" not in bundle
            or "residual_pairs" not in bundle
        ):
            return None
        cache["bundle"] = bundle
        cache["mtime"] = mtime
        cache["path"] = str(model_path)
        return bundle
    except Exception as exc:
        print(f"Model load error: {exc}")
        return None


def _baseline_means(context):
    hs = context.get("home_stats") or {}
    away_stats = context.get("away_stats") or {}
    hf = context.get("home_form") or {}
    af = context.get("away_form") or {}
    park = _safe_float(context.get("park_factor"), 1.0)
    weather = _safe_float(
        (context.get("weather") or {}).get("weather_factor"), 1.0
    )
    home_for = (
        0.65 * _safe_float(hf.get("ema_rpg"), hs.get("rpg", LEAGUE_AVG_RPG))
        + 0.35 * _safe_float(hs.get("rpg"), LEAGUE_AVG_RPG)
    )
    away_for = (
        0.65 * _safe_float(
            af.get("ema_rpg"), away_stats.get("rpg", LEAGUE_AVG_RPG)
        )
        + 0.35 * _safe_float(away_stats.get("rpg"), LEAGUE_AVG_RPG)
    )
    total_mean = np.clip((home_for + away_for) * park * weather, 3.0, 18.0)
    home_ra = _safe_float(
        hf.get("ema_ra9"), hs.get("ra9", hs.get("team_era", 4.5))
    )
    away_ra = _safe_float(
        af.get("ema_ra9"),
        away_stats.get("ra9", away_stats.get("team_era", 4.5)),
    )
    margin_mean = np.clip(
        ((home_for - away_for) + (away_ra - home_ra)) / 2.0 + 0.15,
        -8.0, 8.0,
    )
    return float(total_mean), float(margin_mean)


def _predict_bundle(bundle, features):
    X = np.asarray(features, dtype=float).reshape(1, -1)
    total_names = list(bundle["models_total"])
    margin_names = list(bundle["models_margin"])
    total_predictions = np.asarray([
        float(bundle["models_total"][name].predict(X)[0])
        for name in total_names
    ])
    margin_predictions = np.asarray([
        float(bundle["models_margin"][name].predict(X)[0])
        for name in margin_names
    ])
    total_weights = np.asarray([
        bundle["weights_total"][name] for name in total_names
    ])
    margin_weights = np.asarray([
        bundle["weights_margin"][name] for name in margin_names
    ])
    total_mean = float(total_predictions @ total_weights)
    margin_mean = float(margin_predictions @ margin_weights)
    residual_pairs = np.asarray(bundle["residual_pairs"], dtype=float)
    return {
        "total_mean": total_mean,
        "margin_mean": margin_mean,
        "total_predictions": total_predictions,
        "margin_predictions": margin_predictions,
        "total_samples": np.rint(np.clip(
            total_mean + residual_pairs[:, 0], 0.0, 30.0
        )),
        "margin_samples": np.rint(np.clip(
            margin_mean + residual_pairs[:, 1], -20.0, 20.0
        )),
    }


def _extract_markets(total, run_line, odds_entry):
    odds_entry = odds_entry or {}
    total_market = dict(odds_entry.get("total_market") or {})
    spread_market = dict(odds_entry.get("spread_market") or {})
    if total_market.get("point") is None:
        total_market["point"] = (
            total if total is not None else odds_entry.get("total")
        )
    total_market.setdefault("over_price", odds_entry.get("over_price"))
    total_market.setdefault("under_price", odds_entry.get("under_price"))
    if spread_market.get("home_point") is None:
        spread_market["home_point"] = (
            run_line if run_line is not None else odds_entry.get("run_line")
        )
    if (
        spread_market.get("away_point") is None
        and spread_market.get("home_point") is not None
    ):
        spread_market["away_point"] = -float(spread_market["home_point"])
    spread_market.setdefault("home_price", odds_entry.get("home_price"))
    spread_market.setdefault("away_price", odds_entry.get("away_price"))
    return total_market, spread_market


def _probability_parts(wins, pushes):
    p_win = float(np.mean(wins))
    p_push = float(np.mean(pushes))
    p_loss = max(0.0, 1.0 - p_win - p_push)
    conditional = p_win / max(p_win + p_loss, 1e-9)
    return p_win, p_loss, p_push, conditional


def _select_total(total_samples, model_predictions, market):
    line = market.get("point")
    if line is None:
        return None
    line = float(line)
    over = _probability_parts(total_samples > line, total_samples == line)
    under = _probability_parts(total_samples < line, total_samples == line)
    over_price = market.get("over_price")
    under_price = market.get("under_price")
    consensus_over = market.get("consensus_over_prob")
    consensus_under = market.get("consensus_under_prob")
    if consensus_over is None or consensus_under is None:
        consensus_over, consensus_under = no_vig_probabilities(
            over_price, under_price
        )
    candidates = []
    for side, parts, price, market_prob in (
        ("OVER", over, over_price, consensus_over),
        ("UNDER", under, under_price, consensus_under),
    ):
        p_win, p_loss, p_push, conditional = parts
        ev = expected_value(p_win, p_loss, price)
        edge = conditional - market_prob if market_prob is not None else None
        agreement = float(
            np.mean(model_predictions > line)
            if side == "OVER" else np.mean(model_predictions < line)
        )
        candidates.append({
            "side": side, "line": line, "price": price,
            "p_win": p_win, "p_loss": p_loss, "p_push": p_push,
            "conditional_prob": conditional, "market_prob": market_prob,
            "probability_edge": edge, "ev": ev, "agreement": agreement,
        })
    return max(
        candidates,
        key=lambda item: item["ev"] if item["ev"] is not None else -99,
    )


def _select_spread(margin_samples, model_predictions, market):
    home_point = market.get("home_point")
    away_point = market.get("away_point")
    if home_point is None or away_point is None:
        return None
    home_point = float(home_point)
    away_point = float(away_point)
    home_adjusted = margin_samples + home_point
    away_adjusted = -margin_samples + away_point
    home = _probability_parts(home_adjusted > 0, home_adjusted == 0)
    away = _probability_parts(away_adjusted > 0, away_adjusted == 0)
    home_price = market.get("home_price")
    away_price = market.get("away_price")
    consensus_home = market.get("consensus_home_prob")
    consensus_away = market.get("consensus_away_prob")
    if consensus_home is None or consensus_away is None:
        consensus_home, consensus_away = no_vig_probabilities(
            home_price, away_price
        )
    candidates = []
    for side, point, parts, price, market_prob in (
        ("HOME", home_point, home, home_price, consensus_home),
        ("AWAY", away_point, away, away_price, consensus_away),
    ):
        p_win, p_loss, p_push, conditional = parts
        ev = expected_value(p_win, p_loss, price)
        edge = conditional - market_prob if market_prob is not None else None
        agreement = float(
            np.mean(model_predictions + home_point > 0)
            if side == "HOME"
            else np.mean(-model_predictions + away_point > 0)
        )
        candidates.append({
            "side": side, "point": point, "price": price,
            "p_win": p_win, "p_loss": p_loss, "p_push": p_push,
            "conditional_prob": conditional, "market_prob": market_prob,
            "probability_edge": edge, "ev": ev, "agreement": agreement,
        })
    return max(
        candidates,
        key=lambda item: item["ev"] if item["ev"] is not None else -99,
    )


def _bet_size(selection, flagged):
    if not flagged or not selection:
        return "SKIP", 0.0
    from config import KELLY_FRACTION, MAX_BET_FRACTION
    fraction = quarter_kelly(
        selection["p_win"], selection["p_loss"], selection["price"],
        KELLY_FRACTION, MAX_BET_FRACTION,
    )
    return (
        ("FULL", fraction)
        if fraction >= MAX_BET_FRACTION * 0.75
        else ("HALF", fraction)
    )


def predict_game(context, total=None, run_line=None, odds_entry=None):
    from config import (
        MAX_ENSEMBLE_STD_MARGIN,
        MAX_ENSEMBLE_STD_TOTAL,
        MIN_DATA_QUALITY,
        MIN_EXPECTED_VALUE,
        MIN_MODEL_AGREEMENT,
        MIN_PROBABILITY_EDGE,
    )

    features = build_features(context)
    if features is None:
        return None
    bundle = load_models()
    model_ready = bundle is not None
    if model_ready:
        prediction = _predict_bundle(bundle, features)
        model_version = bundle["version"]
        model_metrics = bundle.get("metrics", {})
    else:
        total_mean, margin_mean = _baseline_means(context)
        prediction = {
            "total_mean": total_mean,
            "margin_mean": margin_mean,
            "total_predictions": np.asarray([total_mean]),
            "margin_predictions": np.asarray([margin_mean]),
            "total_samples": np.asarray([]),
            "margin_samples": np.asarray([]),
        }
        model_version = "UNTRAINED"
        model_metrics = {}

    total_market, spread_market = _extract_markets(
        total, run_line, odds_entry
    )
    total_selection = (
        _select_total(
            prediction["total_samples"],
            prediction["total_predictions"],
            total_market,
        ) if model_ready else None
    )
    spread_selection = (
        _select_spread(
            prediction["margin_samples"],
            prediction["margin_predictions"],
            spread_market,
        ) if model_ready else None
    )
    total_ensemble_std = float(np.std(prediction["total_predictions"]))
    margin_ensemble_std = float(np.std(prediction["margin_predictions"]))
    data_quality = _derive_data_quality(context)
    pitchers_ready = bool(context.get("has_real_pitchers"))

    def is_actionable(selection, ensemble_std, max_std):
        return bool(
            model_ready and selection
            and selection.get("price") is not None
            and selection.get("ev") is not None
            and selection["ev"] >= MIN_EXPECTED_VALUE
            and selection.get("probability_edge") is not None
            and selection["probability_edge"] >= MIN_PROBABILITY_EDGE
            and selection["agreement"] >= MIN_MODEL_AGREEMENT
            and ensemble_std <= max_std
            and data_quality >= MIN_DATA_QUALITY
            and pitchers_ready
        )

    edge_flagged = is_actionable(
        total_selection, total_ensemble_std, MAX_ENSEMBLE_STD_TOTAL
    )
    rl_edge_flagged = is_actionable(
        spread_selection, margin_ensemble_std, MAX_ENSEMBLE_STD_MARGIN
    )
    total_bet_size, total_kelly = _bet_size(total_selection, edge_flagged)
    rl_bet_size, rl_kelly = _bet_size(spread_selection, rl_edge_flagged)
    total_line = total_market.get("point")
    home_win_prob = (
        float(
            np.mean(prediction["margin_samples"] > 0)
            + 0.5 * np.mean(prediction["margin_samples"] == 0)
        ) if model_ready else None
    )
    total_pred = total_selection["side"] if total_selection else "NO BET"
    total_conf = (
        round(total_selection["conditional_prob"] * 100, 1)
        if total_selection else 0.0
    )
    total_votes = (
        int(round(
            total_selection["agreement"]
            * len(prediction["total_predictions"])
        )) if total_selection else 0
    )
    if spread_selection:
        rl_pred = f"{spread_selection['side']} {spread_selection['point']:+g}"
        rl_conf = round(spread_selection["conditional_prob"] * 100, 1)
        rl_votes = int(round(
            spread_selection["agreement"]
            * len(prediction["margin_predictions"])
        ))
    else:
        rl_pred, rl_conf, rl_votes = "NO BET", 0.0, 0
    total_gap = (
        prediction["total_mean"] - float(total_line)
        if total_line is not None else None
    )
    predictive_total_std = (
        float(np.std(prediction["total_samples"])) if model_ready else None
    )

    return {
        "model_ready": model_ready,
        "model_version": model_version,
        "model_metrics": model_metrics,
        "feature_count": len(features),
        "data_quality": round(data_quality, 3),
        "our_total": round(prediction["total_mean"], 2),
        "our_home_margin": round(prediction["margin_mean"], 2),
        "total_gap": round(total_gap, 2) if total_gap is not None else None,
        "total_pred": total_pred,
        "total_line": float(total_line) if total_line is not None else None,
        "total_price": total_selection.get("price") if total_selection else None,
        "total_conf": total_conf,
        "total_win_prob": (
            round(total_selection["p_win"], 4) if total_selection else None
        ),
        "total_push_prob": (
            round(total_selection["p_push"], 4) if total_selection else None
        ),
        "total_market_prob": (
            round(total_selection["market_prob"], 4)
            if total_selection and total_selection["market_prob"] is not None
            else None
        ),
        "total_probability_edge": (
            round(total_selection["probability_edge"], 4)
            if total_selection
            and total_selection["probability_edge"] is not None else None
        ),
        "total_ev": (
            round(total_selection["ev"], 4)
            if total_selection and total_selection["ev"] is not None else None
        ),
        "total_votes": total_votes,
        "total_models": (
            len(prediction["total_predictions"]) if model_ready else 0
        ),
        "total_agreement": (
            round(total_selection["agreement"], 3)
            if total_selection else None
        ),
        "total_ensemble_std": round(total_ensemble_std, 3),
        "predictive_total_std": (
            round(predictive_total_std, 3)
            if predictive_total_std is not None else None
        ),
        "total_bet_size": total_bet_size,
        "total_kelly_fraction": round(total_kelly, 4),
        "rl_pred": rl_pred,
        "rl_side": spread_selection.get("side") if spread_selection else None,
        "rl_point": (
            float(spread_selection["point"]) if spread_selection else None
        ),
        "rl_price": spread_selection.get("price") if spread_selection else None,
        "rl_conf": rl_conf,
        "rl_win_prob": (
            round(spread_selection["p_win"], 4)
            if spread_selection else None
        ),
        "rl_push_prob": (
            round(spread_selection["p_push"], 4)
            if spread_selection else None
        ),
        "rl_market_prob": (
            round(spread_selection["market_prob"], 4)
            if spread_selection
            and spread_selection["market_prob"] is not None else None
        ),
        "rl_probability_edge": (
            round(spread_selection["probability_edge"], 4)
            if spread_selection
            and spread_selection["probability_edge"] is not None else None
        ),
        "rl_ev": (
            round(spread_selection["ev"], 4)
            if spread_selection and spread_selection["ev"] is not None else None
        ),
        "rl_votes": rl_votes,
        "rl_models": (
            len(prediction["margin_predictions"]) if model_ready else 0
        ),
        "rl_agreement": (
            round(spread_selection["agreement"], 3)
            if spread_selection else None
        ),
        "margin_ensemble_std": round(margin_ensemble_std, 3),
        "rl_bet_size": rl_bet_size,
        "rl_kelly_fraction": round(rl_kelly, 4),
        "home_win_prob": (
            round(home_win_prob, 4) if home_win_prob is not None else None
        ),
        "away_win_prob": (
            round(1.0 - home_win_prob, 4)
            if home_win_prob is not None else None
        ),
        "edge_flagged": edge_flagged,
        "rl_edge_flagged": rl_edge_flagged,
        "bet_size": (
            "FULL" if "FULL" in (total_bet_size, rl_bet_size)
            else "HALF" if "HALF" in (total_bet_size, rl_bet_size)
            else "SKIP"
        ),
        "has_real_pitchers": pitchers_ready,
        "quote_timestamp": (odds_entry or {}).get("quote_timestamp"),
        "bookmaker_count": (odds_entry or {}).get("bookmaker_count", 0),
    }
