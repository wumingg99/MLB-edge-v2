from collections import defaultdict
from datetime import datetime, timedelta, timezone
import math
import statistics

import pytz
import requests

from model import compute_pitching_metrics, no_vig_probabilities, parse_innings


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
TIMEZONE = "Asia/Singapore"
ODDS_CACHE_SECONDS = 300

_cache = {}
_games_data_cache = []


def _now_utc():
    return datetime.now(timezone.utc)


def _safe_float(value, default=0.0):
    try:
        parsed = float(value)
        return parsed if math.isfinite(parsed) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _request_json(url, params=None, timeout=15):
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def get_todays_date():
    et_now = datetime.now(pytz.timezone("America/New_York"))
    if et_now.hour >= 23:
        return (et_now + timedelta(days=1)).strftime("%Y-%m-%d"), True
    return et_now.strftime("%Y-%m-%d"), False


def get_todays_games(force_refresh=False):
    today, showing_next = get_todays_date()
    _cache["showing_next_day"] = showing_next
    if (
        not force_refresh
        and "games" in _cache
        and _cache.get("games_date") == today
    ):
        return _cache["games"]
    try:
        data = _request_json(
            f"{MLB_API_BASE}/schedule",
            {
                "sportId": 1,
                "date": today,
                "hydrate": "team,probablePitcher,venue,linescore",
            },
        )
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get(
                    "abstractGameState", ""
                )
                # Only include pregame games — Live/Final games
                # have unreliable in-game/closing odds that
                # corrupt predictions for already-played games
                if status != "Preview":
                    continue
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                home_team = home.get("team", {})
                away_team = away.get("team", {})
                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})
                game_time_utc = game.get("gameDate", "")
                game_time_sgt = ""
                if game_time_utc:
                    try:
                        parsed = datetime.fromisoformat(
                            game_time_utc.replace("Z", "+00:00")
                        )
                        game_time_sgt = parsed.astimezone(
                            pytz.timezone(TIMEZONE)
                        ).strftime("%Y-%m-%dT%H:%M:%S%z")
                    except ValueError:
                        pass
                games.append({
                    "game_id": game.get("gamePk"),
                    "date": today,
                    "game_time_utc": game_time_utc,
                    "home_team": home_team.get("name", "Unknown"),
                    "away_team": away_team.get("name", "Unknown"),
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "home_pitcher": home_pitcher.get("fullName", "TBD"),
                    "away_pitcher": away_pitcher.get("fullName", "TBD"),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "has_real_pitchers": bool(
                        home_pitcher.get("id") and away_pitcher.get("id")
                    ),
                    "venue": game.get("venue", {}).get("name", "Unknown"),
                    "start_time_sgt": game_time_sgt,
                    "status": status,
                })
        _cache["games"] = games
        _cache["games_date"] = today
        return games
    except Exception as exc:
        print(f"Error fetching games: {exc}")
        return []


def _default_pitcher_stats():
    return {
        "era": 4.50, "fip": 4.50, "xfip": 4.50,
        "k9": 8.0, "bb9": 3.0, "whip": 1.30,
        "ip": 0.0, "sample_size": 0.0,
    }


def get_pitcher_stats(pitcher_id, season=None):
    if not pitcher_id:
        return _default_pitcher_stats()
    season = season or datetime.now().year
    cache_key = f"pitcher_{pitcher_id}_{season}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_pitcher_stats()
    try:
        data = _request_json(
            f"{MLB_API_BASE}/people/{pitcher_id}/stats",
            {
                "stats": "season",
                "group": "pitching",
                "season": season,
                "gameType": "R",
            },
            timeout=10,
        )
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                result = compute_pitching_metrics(
                    splits[0].get("stat", {})
                )
                break
    except Exception as exc:
        print(f"Pitcher stats error {pitcher_id}: {exc}")
    _cache[cache_key] = result
    return result


def _sum_pitching_splits(splits):
    totals = defaultdict(float)
    fields = (
        "outs", "earnedRuns", "strikeOuts", "baseOnBalls",
        "hitBatsmen", "homeRuns", "hits", "airOuts",
    )
    for split in splits:
        stat = split.get("stat", {})
        for field in fields:
            totals[field] += _safe_float(stat.get(field), 0.0)
        if not stat.get("outs"):
            totals["outs"] += parse_innings(
                stat.get("inningsPitched")
            ) * 3.0
    return dict(totals)


def get_recent_pitcher_stats(pitcher_id, last_n=3, season=None):
    if not pitcher_id:
        return {
            "recent_era": 4.50, "recent_k9": 8.0,
            "recent_bb9": 3.0, "recent_ip": 0.0,
        }
    season = season or datetime.now().year
    cache_key = f"pitcher_recent_{pitcher_id}_{season}_{last_n}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = {
        "recent_era": 4.50, "recent_k9": 8.0,
        "recent_bb9": 3.0, "recent_ip": 0.0,
    }
    try:
        data = _request_json(
            f"{MLB_API_BASE}/people/{pitcher_id}/stats",
            {
                "stats": "gameLog",
                "group": "pitching",
                "season": season,
                "gameType": "R",
            },
            timeout=10,
        )
        for stat_group in data.get("stats", []):
            splits = sorted(
                stat_group.get("splits", []),
                key=lambda item: item.get("date", ""),
            )[-last_n:]
            if not splits:
                continue
            metrics = compute_pitching_metrics(_sum_pitching_splits(splits))
            result = {
                "recent_era": metrics["era"],
                "recent_k9": metrics["k9"],
                "recent_bb9": metrics["bb9"],
                "recent_ip": metrics["ip"],
            }
            break
    except Exception as exc:
        print(f"Recent pitcher stats error {pitcher_id}: {exc}")
    _cache[cache_key] = result
    return result


def _default_team_stats():
    return {
        "ops": 0.720, "rpg": 4.5, "ra9": 4.5,
        "team_era": 4.50, "win_pct": 0.500, "games": 0,
    }


def get_team_stats(team_id, season=None):
    if not team_id:
        return _default_team_stats()
    season = season or datetime.now().year
    cache_key = f"team_{team_id}_{season}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_team_stats()
    try:
        data = _request_json(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            {
                "stats": "season",
                "group": "hitting,pitching",
                "season": season,
                "gameType": "R",
            },
            timeout=10,
        )
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            splits = stat_group.get("splits", [])
            if not splits:
                continue
            stat = splits[0].get("stat", {})
            games = max(_safe_float(stat.get("gamesPlayed"), 0.0), 0.0)
            if group == "hitting":
                result["ops"] = _safe_float(stat.get("ops"), 0.720)
                result["rpg"] = (
                    _safe_float(stat.get("runs"), 0.0) / games
                    if games > 0 else 4.5
                )
                result["games"] = int(games)
            elif group == "pitching":
                result["team_era"] = _safe_float(stat.get("era"), 4.50)
                result["ra9"] = _safe_float(
                    stat.get("runsScoredPer9"), result["team_era"]
                )
        form = get_recent_team_form(team_id, last_n=200, season=season)
        result["win_pct"] = form.get("season_win_pct", 0.5)
    except Exception as exc:
        print(f"Team stats error {team_id}: {exc}")
    _cache[cache_key] = result
    return result


def _batting_ops(stats):
    at_bats = sum(_safe_float(s.get("atBats"), 0) for s in stats)
    hits = sum(_safe_float(s.get("hits"), 0) for s in stats)
    doubles = sum(_safe_float(s.get("doubles"), 0) for s in stats)
    triples = sum(_safe_float(s.get("triples"), 0) for s in stats)
    home_runs = sum(_safe_float(s.get("homeRuns"), 0) for s in stats)
    walks = sum(_safe_float(s.get("baseOnBalls"), 0) for s in stats)
    hit_by_pitch = sum(_safe_float(s.get("hitByPitch"), 0) for s in stats)
    sacrifice_flies = sum(_safe_float(s.get("sacFlies"), 0) for s in stats)
    singles = max(hits - doubles - triples - home_runs, 0.0)
    obp_den = at_bats + walks + hit_by_pitch + sacrifice_flies
    obp = (
        (hits + walks + hit_by_pitch) / obp_den
        if obp_den > 0 else 0.320
    )
    slg = (
        (singles + 2 * doubles + 3 * triples + 4 * home_runs) / at_bats
        if at_bats > 0 else 0.400
    )
    return float(np_clip(obp + slg, 0.4, 1.4))


def np_clip(value, low, high):
    return max(low, min(float(value), high))


def _ema(values, alpha=0.25, default=4.5):
    if not values:
        return default
    current = float(values[0])
    for value in values[1:]:
        current = alpha * float(value) + (1.0 - alpha) * current
    return current


def _default_form():
    return {
        "recent_rpg": 4.5, "recent_ra9": 4.5,
        "recent_ops": 0.720, "recent_win_pct": 0.500,
        "season_win_pct": 0.500,
        "ema_rpg": 4.5, "ema_ra9": 4.5,
        "games": 0,
    }


def get_recent_team_form(team_id, last_n=10, season=None):
    if not team_id:
        return _default_form()
    season = season or datetime.now().year
    cache_key = f"team_form_{team_id}_{season}_{last_n}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_form()
    try:
        data = _request_json(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            {
                "stats": "gameLog",
                "group": "hitting,pitching",
                "season": season,
                "gameType": "R",
            },
            timeout=10,
        )
        hitting_splits = []
        pitching_splits = []
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            splits = sorted(
                stat_group.get("splits", []),
                key=lambda item: item.get("date", ""),
            )
            if group == "hitting":
                hitting_splits = splits
            elif group == "pitching":
                pitching_splits = splits
        recent_hitting = hitting_splits[-last_n:]
        recent_pitching = pitching_splits[-last_n:]
        runs_for = [
            _safe_float(split.get("stat", {}).get("runs"), 0.0)
            for split in recent_hitting
        ]
        runs_against = [
            _safe_float(split.get("stat", {}).get("runs"), 0.0)
            for split in recent_pitching
        ]
        all_runs_for = [
            _safe_float(split.get("stat", {}).get("runs"), 0.0)
            for split in hitting_splits
        ]
        all_runs_against = [
            _safe_float(split.get("stat", {}).get("runs"), 0.0)
            for split in pitching_splits
        ]
        wins = sum(1 for split in recent_hitting if split.get("isWin"))
        season_wins = sum(1 for split in hitting_splits if split.get("isWin"))
        if recent_hitting:
            result = {
                "recent_rpg": statistics.fmean(runs_for),
                "recent_ra9": (
                    statistics.fmean(runs_against)
                    if runs_against else 4.5
                ),
                "recent_ops": _batting_ops([
                    split.get("stat", {}) for split in recent_hitting
                ]),
                "recent_win_pct": wins / len(recent_hitting),
                "season_win_pct": (
                    season_wins / len(hitting_splits)
                    if hitting_splits else 0.5
                ),
                "ema_rpg": _ema(all_runs_for[-30:], default=4.5),
                "ema_ra9": _ema(all_runs_against[-30:], default=4.5),
                "games": len(hitting_splits),
            }
    except Exception as exc:
        print(f"Recent form error {team_id}: {exc}")
    _cache[cache_key] = result
    return result


def _aggregate_reliever_splits(splits):
    weighted_earned_runs = 0.0
    innings = 0.0
    for split in splits:
        stat = split.get("stat", {})
        games_pitched = _safe_float(stat.get("gamesPitched"), 0.0)
        games_started = _safe_float(stat.get("gamesStarted"), 0.0)
        if games_pitched <= 0 or games_started / games_pitched >= 0.5:
            continue
        player_innings = parse_innings(stat.get("inningsPitched"))
        innings += player_innings
        weighted_earned_runs += _safe_float(stat.get("earnedRuns"), 0.0)
    era = 9.0 * weighted_earned_runs / innings if innings > 0 else 4.5
    return float(np_clip(era, 0.0, 15.0)), innings


def get_bullpen_stats(team_id, season=None):
    if not team_id:
        return {"bullpen_era": 4.50, "bullpen_workload": 0.0}
    season = season or datetime.now().year
    cache_key = f"bullpen_{team_id}_{season}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = {"bullpen_era": 4.50, "bullpen_workload": 0.0}
    try:
        season_data = _request_json(
            f"{MLB_API_BASE}/stats",
            {
                "stats": "season",
                "group": "pitching",
                "teamId": team_id,
                "season": season,
                "gameType": "R",
                "playerPool": "ALL",
            },
            timeout=12,
        )
        season_splits = season_data.get("stats", [{}])[0].get("splits", [])
        result["bullpen_era"], _ = _aggregate_reliever_splits(season_splits)

        today = datetime.now().date()
        recent_data = _request_json(
            f"{MLB_API_BASE}/stats",
            {
                "stats": "byDateRange",
                "group": "pitching",
                "teamId": team_id,
                "startDate": (today - timedelta(days=3)).isoformat(),
                "endDate": today.isoformat(),
                "gameType": "R",
                "playerPool": "ALL",
            },
            timeout=12,
        )
        recent_splits = recent_data.get("stats", [{}])[0].get("splits", [])
        _, workload = _aggregate_reliever_splits(recent_splits)
        result["bullpen_workload"] = round(workload, 2)
    except Exception as exc:
        print(f"Bullpen error {team_id}: {exc}")
    _cache[cache_key] = result
    return result


def get_park_factors():
    """
    2025 MLB park factors — 5-year weighted average, multiplier format.
    1.00 = neutral. Keyed to exact MLB Stats API venue names.
    """
    return {
        # Extreme hitter parks
        "Coors Field":                   {"runs": 1.20, "hr": 1.12},  # COL altitude
        "Great American Ball Park":      {"runs": 1.10, "hr": 1.15},  # CIN
        # Hitter-friendly
        "American Family Field":         {"runs": 1.06, "hr": 1.10},  # MIL retractable
        "Citizens Bank Park":            {"runs": 1.05, "hr": 1.08},  # PHI
        "Rate Field":                    {"runs": 1.04, "hr": 1.08},  # CWS
        "Yankee Stadium":                {"runs": 1.04, "hr": 1.14},  # NYY
        "Fenway Park":                   {"runs": 1.04, "hr": 0.97},  # BOS
        "Oriole Park at Camden Yards":   {"runs": 1.02, "hr": 1.05},  # BAL
        "Daikin Park":                   {"runs": 1.02, "hr": 1.04},  # HOU retractable
        "Minute Maid Park":              {"runs": 1.02, "hr": 1.04},  # HOU legacy name
        "Wrigley Field":                 {"runs": 1.02, "hr": 1.02},  # CHC wind-dependent
        # Near-neutral
        "Nationals Park":                {"runs": 1.01, "hr": 1.03},  # WSH
        "Chase Field":                   {"runs": 1.01, "hr": 0.99},  # ARI retractable
        "Rogers Centre":                 {"runs": 1.00, "hr": 1.02},  # TOR dome
        "Truist Park":                   {"runs": 1.00, "hr": 1.00},  # ATL
        "Progressive Field":             {"runs": 0.98, "hr": 0.95},  # CLE
        "Globe Life Field":              {"runs": 0.97, "hr": 1.00},  # TEX dome
        "Busch Stadium":                 {"runs": 0.97, "hr": 0.93},  # STL
        "PNC Park":                      {"runs": 0.97, "hr": 0.93},  # PIT
        "Kauffman Stadium":              {"runs": 0.97, "hr": 0.90},  # KC
        "Angel Stadium":                 {"runs": 0.97, "hr": 0.92},  # LAA
        # Pitcher-friendly
        "Dodger Stadium":                {"runs": 0.96, "hr": 0.88},  # LAD
        "Comerica Park":                 {"runs": 0.96, "hr": 0.88},  # DET
        "Target Field":                  {"runs": 0.96, "hr": 0.92},  # MIN
        "Tropicana Field":               {"runs": 0.95, "hr": 0.93},  # TB dome
        "Citi Field":                    {"runs": 0.95, "hr": 0.89},  # NYM
        "loanDepot park":                {"runs": 0.95, "hr": 0.93},  # MIA dome
        "T-Mobile Park":                 {"runs": 0.93, "hr": 0.89},  # SEA
        "Petco Park":                    {"runs": 0.92, "hr": 0.86},  # SD
        "Oracle Park":                   {"runs": 0.91, "hr": 0.78},  # SF
        # Temporary / unknown → neutral
        "Sutter Health Park":            {"runs": 1.00, "hr": 1.00},  # ATH Sacramento
        "Journey Bank Ballpark":         {"runs": 1.00, "hr": 1.00},
        "Bristol Motor Speedway":        {"runs": 1.00, "hr": 1.00},
        "George M. Steinbrenner Field":  {"runs": 1.00, "hr": 1.00},
    }

VENUE_COORDS = {
    "Fenway Park":                   (42.3467, -71.0972),
    "Yankee Stadium":                (40.8296, -73.9262),
    "Oriole Park at Camden Yards":   (39.2838, -76.6218),
    "Citizens Bank Park":            (39.9061, -75.1665),
    "Citi Field":                    (40.7571, -73.8458),
    "Nationals Park":                (38.8730, -77.0074),
    "Truist Park":                   (33.8908, -84.4679),
    "loanDepot park":                (25.7781, -80.2197),
    "American Family Field":         (43.0280, -87.9712),
    "Wrigley Field":                 (41.9484, -87.6553),
    "Rate Field":                    (41.8299, -87.6338),
    "Busch Stadium":                 (38.6226, -90.1928),
    "PNC Park":                      (40.4469, -80.0057),
    "Great American Ball Park":      (39.0979, -84.5082),
    "Progressive Field":             (41.4962, -81.6852),
    "Comerica Park":                 (42.3390, -83.0485),
    "Kauffman Stadium":              (39.0517, -94.4803),
    "Target Field":                  (44.9817, -93.2781),
    "Dodger Stadium":                (34.0739, -118.2400),
    "Angel Stadium":                 (33.8003, -117.8827),
    "Oracle Park":                   (37.7786, -122.3893),
    "Petco Park":                    (32.7073, -117.1566),
    "Coors Field":                   (39.7559, -104.9942),
    "T-Mobile Park":                 (47.5914, -122.3325),
    "Sutter Health Park":            (38.5897, -121.5001),
}
def get_weather(venue, game_time_utc):
    cache_key = f"weather_{venue}_{game_time_utc}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = {
        "temp": 72.0, "wind_speed": 5.0, "weather_factor": 1.0,
        "is_dome": False, "available": False,
    }
    domes = (
        "Tropicana Field", "Globe Life Field", "Rogers Centre",
        "Chase Field", "Minute Maid Park", "American Family Field",
        "loanDepot park", "Daikin Park",
    )
    if any(name.lower() in venue.lower() for name in domes):
        result.update({"is_dome": True, "available": True})
        _cache[cache_key] = result
        return result
    coords = VENUE_COORDS.get(venue)
    if not coords:
        _cache[cache_key] = result
        return result
    try:
        lat, lon = coords
        data = _request_json(
            "https://api.open-meteo.com/v1/forecast",
            {
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,windspeed_10m",
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "forecast_days": 3,
                "timezone": "UTC",
            },
            timeout=8,
        )
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        target = datetime.fromisoformat(
            str(game_time_utc).replace("Z", "+00:00")
        )
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        indices = []
        for idx, value in enumerate(times):
            parsed = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
            indices.append((abs((parsed - target).total_seconds()), idx))
        if indices:
            idx = min(indices)[1]
            result["temp"] = _safe_float(
                hourly.get("temperature_2m", [72])[idx], 72
            )
            result["wind_speed"] = _safe_float(
                hourly.get("windspeed_10m", [5])[idx], 5
            )
            factor = 1.0
            if result["temp"] > 85:
                factor += 0.04
            elif result["temp"] < 50:
                factor -= 0.04
            # Wind direction is not available in this feed, so do not assume
            # that strong wind always helps hitters.
            result["weather_factor"] = round(factor, 3)
            result["available"] = True
    except Exception as exc:
        print(f"Weather error: {exc}")
    _cache[cache_key] = result
    return result


def _normalize_team(name):
    return "".join(char for char in str(name).lower() if char.isalnum())


def _best_price(values):
    valid = [int(value) for value in values if value is not None]
    return max(valid) if valid else None


def _modal_point(offers):
    counts = defaultdict(int)
    for offer in offers:
        counts[float(offer["point"])] += 1
    if not counts:
        return None
    return max(counts, key=lambda point: (counts[point], -abs(point)))


def _build_total_market(offers):
    point = _modal_point(offers)
    if point is None:
        return {}
    selected = [offer for offer in offers if float(offer["point"]) == point]
    paired = [
        offer for offer in selected
        if offer.get("over_price") is not None
        and offer.get("under_price") is not None
    ]
    consensus = [
        no_vig_probabilities(
            offer["over_price"], offer["under_price"]
        ) for offer in paired
    ]
    return {
        "point": point,
        "over_price": _best_price(
            offer.get("over_price") for offer in selected
        ),
        "under_price": _best_price(
            offer.get("under_price") for offer in selected
        ),
        "consensus_over_prob": (
            statistics.fmean(pair[0] for pair in consensus)
            if consensus else None
        ),
        "consensus_under_prob": (
            statistics.fmean(pair[1] for pair in consensus)
            if consensus else None
        ),
        "books": len(paired),
    }


def _build_spread_market(offers):
    point = _modal_point(offers)
    if point is None:
        return {}
    selected = [
        offer for offer in offers if float(offer["point"]) == point
    ]
    paired = [
        offer for offer in selected
        if offer.get("home_price") is not None
        and offer.get("away_price") is not None
    ]
    consensus = [
        no_vig_probabilities(
            offer["home_price"], offer["away_price"]
        ) for offer in paired
    ]
    return {
        "home_point": point,
        "away_point": -point,
        "home_price": _best_price(
            offer.get("home_price") for offer in selected
        ),
        "away_price": _best_price(
            offer.get("away_price") for offer in selected
        ),
        "consensus_home_prob": (
            statistics.fmean(pair[0] for pair in consensus)
            if consensus else None
        ),
        "consensus_away_prob": (
            statistics.fmean(pair[1] for pair in consensus)
            if consensus else None
        ),
        "books": len(paired),
    }


def get_odds(api_key, force_refresh=False):
    cached_at = _cache.get("odds_cached_at")
    if (
        not force_refresh
        and cached_at
        and (_now_utc() - cached_at).total_seconds() < ODDS_CACHE_SECONDS
    ):
        return _cache.get("odds", [])
    if not api_key:
        return []
    odds = []
    try:
        data = _request_json(
            "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
            {
                "apiKey": api_key,
                "regions": "eu",
                "markets": "totals,spreads",
                "oddsFormat": "american",
                "bookmakers": "pinnacle",
            },
            timeout=15,
        )
        quote_timestamp = _now_utc().isoformat()
        for game in data:
            total_offers = []
            spread_offers = []
            home_name = game.get("home_team", "")
            away_name = game.get("away_team", "")
            for bookmaker in game.get("bookmakers", []):
                bookmaker_key = bookmaker.get("key")
                for market in bookmaker.get("markets", []):
                    outcomes = market.get("outcomes", [])
                    if market.get("key") == "totals":
                        over = next(
                            (item for item in outcomes if item.get("name") == "Over"),
                            None,
                        )
                        under = next(
                            (item for item in outcomes if item.get("name") == "Under"),
                            None,
                        )
                        if over and under and over.get("point") == under.get("point"):
                            total_offers.append({
                                "book": bookmaker_key,
                                "point": over.get("point"),
                                "over_price": over.get("price"),
                                "under_price": under.get("price"),
                            })
                    elif market.get("key") == "spreads":
                        home = next(
                            (
                                item for item in outcomes
                                if _normalize_team(item.get("name"))
                                == _normalize_team(home_name)
                            ),
                            None,
                        )
                        away = next(
                            (
                                item for item in outcomes
                                if _normalize_team(item.get("name"))
                                == _normalize_team(away_name)
                            ),
                            None,
                        )
                        if home and away:
                            spread_offers.append({
                                "book": bookmaker_key,
                                "point": home.get("point"),
                                "home_price": home.get("price"),
                                "away_price": away.get("price"),
                            })
            total_market = _build_total_market(total_offers)
            spread_market = _build_spread_market(spread_offers)
            odds.append({
                "event_id": game.get("id"),
                "commence_time": game.get("commence_time"),
                "home_team": home_name,
                "away_team": away_name,
                "total_market": total_market,
                "spread_market": spread_market,
                "total": total_market.get("point"),
                "run_line": spread_market.get("home_point"),
                "over_price": total_market.get("over_price"),
                "under_price": total_market.get("under_price"),
                "home_price": spread_market.get("home_price"),
                "away_price": spread_market.get("away_price"),
                "quote_timestamp": quote_timestamp,
                "bookmaker_count": max(
                    total_market.get("books", 0),
                    spread_market.get("books", 0),
                ),
            })
    except Exception as exc:
        print(f"Odds error: {exc}")
    _cache["odds"] = odds
    _cache["odds_cached_at"] = _now_utc()
    return odds


def get_team_rest_days(team_id, game_date):
    if not team_id or not game_date:
        return 3.0
    cache_key = f"rest_{team_id}_{game_date}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = 3.0
    try:
        target = datetime.fromisoformat(str(game_date)[:10]).date()
        data = _request_json(
            f"{MLB_API_BASE}/schedule",
            {
                "sportId": 1,
                "teamId": team_id,
                "startDate": (target - timedelta(days=10)).isoformat(),
                "endDate": (target - timedelta(days=1)).isoformat(),
                "gameType": "R",
            },
            timeout=10,
        )
        dates = [
            datetime.fromisoformat(item["date"]).date()
            for item in data.get("dates", [])
            if item.get("games")
        ]
        if dates:
            result = float(min(max((target - max(dates)).days, 0), 10))
    except Exception as exc:
        print(f"Rest days error {team_id}: {exc}")
    _cache[cache_key] = result
    return result


def build_game_context(game, api_key=None):
    del api_key
    home_id = game.get("home_team_id")
    away_id = game.get("away_team_id")
    home_pitcher_id = game.get("home_pitcher_id")
    away_pitcher_id = game.get("away_pitcher_id")
    venue = game.get("venue", "")
    game_date = game.get("date", "")

    home_stats = get_team_stats(home_id)
    away_stats = get_team_stats(away_id)
    home_form = get_recent_team_form(home_id)
    away_form = get_recent_team_form(away_id)
    home_pitcher = get_pitcher_stats(home_pitcher_id)
    away_pitcher = get_pitcher_stats(away_pitcher_id)
    home_pitcher_recent = get_recent_pitcher_stats(home_pitcher_id)
    away_pitcher_recent = get_recent_pitcher_stats(away_pitcher_id)
    home_bullpen = get_bullpen_stats(home_id)
    away_bullpen = get_bullpen_stats(away_id)
    park = get_park_factors().get(venue, {"runs": 1.0, "hr": 1.0})
    weather = get_weather(venue, game.get("game_time_utc"))
    home_rest = get_team_rest_days(home_id, game_date)
    away_rest = get_team_rest_days(away_id, game_date)
    quality_parts = [
        1.0 if game.get("has_real_pitchers") else 0.0,
        min(home_pitcher.get("ip", 0.0) / 30.0, 1.0),
        min(away_pitcher.get("ip", 0.0) / 30.0, 1.0),
        min(home_stats.get("games", 0) / 20.0, 1.0),
        min(away_stats.get("games", 0) / 20.0, 1.0),
        1.0 if weather.get("available") else 0.5,
    ]
    return {
        "home_stats": home_stats,
        "away_stats": away_stats,
        "home_form": home_form,
        "away_form": away_form,
        "home_pitcher": home_pitcher,
        "away_pitcher": away_pitcher,
        "home_pitcher_recent": home_pitcher_recent,
        "away_pitcher_recent": away_pitcher_recent,
        "home_bullpen": home_bullpen,
        "away_bullpen": away_bullpen,
        "park_factor": park["runs"],
        "park_hr_factor": park["hr"],
        "weather": weather,
        "home_rest_days": home_rest,
        "away_rest_days": away_rest,
        "has_real_pitchers": game.get("has_real_pitchers", False),
        "data_quality": statistics.fmean(quality_parts),
        "venue": venue,
        "as_of": game.get("game_time_utc") or game_date,
        "game_date": game_date,
    }


def clear_cache():
    global _games_data_cache
    _cache.clear()
    _games_data_cache = []


def preload_all_data(api_key, force_odds_refresh=False):
    global _games_data_cache
    print("V3: Preloading leak-safe features and market prices...")
    games = get_todays_games()
    if not games:
        print("No games today")
        return []
    odds_list = get_odds(api_key, force_refresh=force_odds_refresh)
    odds_by_matchup = {
        (
            _normalize_team(entry.get("home_team")),
            _normalize_team(entry.get("away_team")),
        ): entry
        for entry in odds_list
    }
    games_data = []
    for game in games:
        print(f"Loading: {game['away_team']} @ {game['home_team']}")
        context = build_game_context(game, api_key)
        matchup = (
            _normalize_team(game["home_team"]),
            _normalize_team(game["away_team"]),
        )
        games_data.append(
            (game, context, odds_by_matchup.get(matchup))
        )
    _games_data_cache = games_data
    print(f"V3: Preloaded {len(games_data)} games")
    return games_data


def get_cached_games_data():
    return _games_data_cache
