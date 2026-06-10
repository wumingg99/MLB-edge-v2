import requests
from datetime import datetime, timedelta
import pytz
import statistics

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
TIMEZONE = "Asia/Singapore"

_cache = {}
_games_data_cache = []

def get_todays_date():
    et_tz = pytz.timezone("America/New_York")
    et_now = datetime.now(et_tz)
    if et_now.hour >= 23:
        return (et_now + timedelta(days=1)).strftime("%Y-%m-%d"), True
    return et_now.strftime("%Y-%m-%d"), False

def get_todays_games():
    today, showing_next = get_todays_date()
    _cache["showing_next_day"] = showing_next
    if "games" in _cache and _cache.get("games_date") == today:
        return _cache["games"]
    try:
        url = f"{MLB_API_BASE}/schedule"
        params = {"sportId": 1, "date": today,
                  "hydrate": "team,probablePitcher,venue,linescore"}
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        games = []
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                status = game.get("status", {}).get("abstractGameState", "")
                if status in ["Final", "Cancelled"]:
                    continue
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                home_team = home.get("team", {})
                away_team = away.get("team", {})
                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})
                tz = pytz.timezone(TIMEZONE)
                game_time_sgt = ""
                game_time_str = game.get("gameDate", "")
                if game_time_str:
                    try:
                        utc_dt = datetime.fromisoformat(
                            game_time_str.replace("Z", "+00:00"))
                        sgt_dt = utc_dt.astimezone(tz)
                        game_time_sgt = sgt_dt.strftime("%Y-%m-%dT%H:%M:%S")
                    except Exception:
                        pass
                games.append({
                    "game_id": game.get("gamePk"),
                    "date": today,
                    "home_team": home_team.get("name", "Unknown"),
                    "away_team": away_team.get("name", "Unknown"),
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "home_pitcher": home_pitcher.get("fullName", "TBD"),
                    "away_pitcher": away_pitcher.get("fullName", "TBD"),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "has_real_pitchers": bool(home_pitcher and away_pitcher),
                    "venue": game.get("venue", {}).get("name", "Unknown"),
                    "start_time_sgt": game_time_sgt,
                    "status": status,
                })
        _cache["games"] = games
        _cache["games_date"] = today
        return games
    except Exception as e:
        print(f"Error fetching games: {e}")
        return []

def get_pitcher_stats(pitcher_id):
    if not pitcher_id:
        return _default_pitcher_stats()
    cache_key = f"pitcher_{pitcher_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_pitcher_stats()
    try:
        url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
        params = {"stats": "season,lastXGames", "group": "pitching",
                  "season": datetime.now().year, "gameType": "R",
                  "sitCodes": "vl,vr"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for stat_group in data.get("stats", []):
            stat_type = stat_group.get("type", {}).get("displayName", "")
            splits = stat_group.get("splits", [])
            if not splits:
                continue
            s = splits[0].get("stat", {})
            if stat_type == "statsSingleSeason":
                result["era"] = float(s.get("era", 4.50) or 4.50)
                result["fip"] = float(s.get("fip", 4.50) or 4.50)
                result["xfip"] = float(s.get("xfip", 4.50) or 4.50)
                result["k9"] = float(s.get("strikeoutsPer9Inn", 8.0) or 8.0)
                result["bb9"] = float(s.get("walksPer9Inn", 3.0) or 3.0)
                result["kbb"] = float(s.get("strikeoutWalkRatio", 2.5) or 2.5)
                result["whip"] = float(s.get("whip", 1.30) or 1.30)
                result["ip"] = float(s.get("inningsPitched", 0) or 0)
    except Exception as e:
        print(f"Pitcher stats error {pitcher_id}: {e}")
    _cache[cache_key] = result
    return result

def get_recent_pitcher_stats(pitcher_id, last_n=3):
    """Get pitcher stats from last N starts."""
    if not pitcher_id:
        return _default_pitcher_stats()
    cache_key = f"pitcher_recent_{pitcher_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_pitcher_stats()
    try:
        url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
        params = {"stats": "gameLog", "group": "pitching",
                  "season": datetime.now().year, "gameType": "R"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            if not splits:
                continue
            recent = splits[-last_n:]
            if not recent:
                continue
            eras = []
            k9s = []
            bb9s = []
            for start in recent:
                s = start.get("stat", {})
                era = float(s.get("era", 4.50) or 4.50)
                k9 = float(s.get("strikeoutsPer9Inn", 8.0) or 8.0)
                bb9 = float(s.get("walksPer9Inn", 3.0) or 3.0)
                eras.append(era)
                k9s.append(k9)
                bb9s.append(bb9)
            if eras:
                result["recent_era"] = sum(eras) / len(eras)
                result["recent_k9"] = sum(k9s) / len(k9s)
                result["recent_bb9"] = sum(bb9s) / len(bb9s)
                result["recent_trend"] = eras[-1] - eras[0] if len(eras) > 1 else 0
    except Exception as e:
        print(f"Recent pitcher stats error {pitcher_id}: {e}")
    _cache[cache_key] = result
    return result

def _default_pitcher_stats():
    return {
        "era": 4.50, "fip": 4.50, "xfip": 4.50,
        "k9": 8.0, "bb9": 3.0, "kbb": 2.5, "whip": 1.30,
        "ip": 0, "recent_era": 4.50, "recent_k9": 8.0,
        "recent_bb9": 3.0, "recent_trend": 0,
    }

def get_team_stats(team_id):
    if not team_id:
        return _default_team_stats()
    cache_key = f"team_{team_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_team_stats()
    try:
        season = datetime.now().year
        url = f"{MLB_API_BASE}/teams/{team_id}/stats"
        params = {"stats": "season", "group": "hitting,pitching",
                  "season": season, "gameType": "R"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            splits = stat_group.get("splits", [])
            if not splits:
                continue
            s = splits[0].get("stat", {})
            if group == "hitting":
                result["ops"] = float(s.get("ops", 0.720) or 0.720)
                result["avg"] = float(s.get("avg", 0.250) or 0.250)
                result["rpg"] = float(s.get("runs", 0) or 0) / max(
                    float(s.get("gamesPlayed", 1) or 1), 1)
                result["hrpg"] = float(s.get("homeRuns", 0) or 0) / max(
                    float(s.get("gamesPlayed", 1) or 1), 1)
            elif group == "pitching":
                result["team_era"] = float(s.get("era", 4.50) or 4.50)
                result["team_whip"] = float(s.get("whip", 1.30) or 1.30)
                result["team_k9"] = float(s.get("strikeoutsPer9Inn", 8.0) or 8.0)
    except Exception as e:
        print(f"Team stats error {team_id}: {e}")
    _cache[cache_key] = result
    return result

def get_recent_team_form(team_id, last_n=10):
    """Get team performance over last N games — KEY v2 feature."""
    if not team_id:
        return _default_form()
    cache_key = f"team_form_{team_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = _default_form()
    try:
        season = datetime.now().year
        url = f"{MLB_API_BASE}/teams/{team_id}/stats"
        params = {"stats": "gameLog", "group": "hitting,pitching",
                  "season": season, "gameType": "R"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            splits = stat_group.get("splits", [])
            if not splits:
                continue
            recent = splits[-last_n:]
            if group == "hitting":
                runs = [float(s.get("stat", {}).get("runs", 0) or 0)
                        for s in recent]
                ops_vals = [float(s.get("stat", {}).get("ops", 0.720) or 0.720)
                            for s in recent]
                if runs:
                    result["recent_rpg"] = sum(runs) / len(runs)
                    result["recent_ops"] = sum(ops_vals) / len(ops_vals)
                    result["form_trend"] = (
                        sum(runs[-3:]) / 3 - sum(runs[:3]) / 3
                        if len(runs) >= 6 else 0)
                    result["recent_wins"] = sum(
                        1 for s in recent
                        if s.get("isWin", False))
                    result["recent_win_pct"] = (
                        result["recent_wins"] / len(recent))
    except Exception as e:
        print(f"Recent form error {team_id}: {e}")
    _cache[cache_key] = result
    return result

def _default_form():
    return {
        "recent_rpg": 4.5, "recent_ops": 0.720,
        "form_trend": 0, "recent_wins": 5,
        "recent_win_pct": 0.500,
    }

def _default_team_stats():
    return {
        "ops": 0.720, "avg": 0.250, "rpg": 4.5,
        "hrpg": 1.0, "team_era": 4.50,
        "team_whip": 1.30, "team_k9": 8.0,
    }

def get_bullpen_stats(team_id):
    if not team_id:
        return {"bullpen_era": 4.50, "bullpen_fip": 4.50, "bullpen_workload": 0}
    cache_key = f"bullpen_{team_id}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = {"bullpen_era": 4.50, "bullpen_fip": 4.50, "bullpen_workload": 0}
    try:
        season = datetime.now().year
        url = f"{MLB_API_BASE}/teams/{team_id}/stats"
        params = {"stats": "season", "group": "pitching",
                  "season": season, "gameType": "R",
                  "playerPool": "qualifier"}
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        for stat_group in data.get("stats", []):
            splits = stat_group.get("splits", [])
            bullpen_splits = [s for s in splits
                              if s.get("stat", {}).get("gamesStarted", 1) == 0]
            if bullpen_splits:
                eras = [float(s.get("stat", {}).get("era", 4.50) or 4.50)
                        for s in bullpen_splits[:8]]
                result["bullpen_era"] = sum(eras) / len(eras) if eras else 4.50
    except Exception as e:
        print(f"Bullpen error {team_id}: {e}")
    _cache[cache_key] = result
    return result

def get_park_factors():
    return {
        "Coors Field": {"runs": 1.35, "hr": 1.40},
        "Fenway Park": {"runs": 1.10, "hr": 0.95},
        "Globe Life Field": {"runs": 1.05, "hr": 1.10},
        "Great American Ball Park": {"runs": 1.12, "hr": 1.20},
        "Wrigley Field": {"runs": 1.08, "hr": 1.05},
        "Oracle Park": {"runs": 0.88, "hr": 0.75},
        "Tropicana Field": {"runs": 0.94, "hr": 0.92},
        "Petco Park": {"runs": 0.92, "hr": 0.85},
        "T-Mobile Park": {"runs": 0.91, "hr": 0.88},
        "Dodger Stadium": {"runs": 0.95, "hr": 0.98},
        "Minute Maid Park": {"runs": 1.02, "hr": 1.05},
    }

def get_weather(venue, game_date):
    cache_key = f"weather_{venue}_{game_date}"
    if cache_key in _cache:
        return _cache[cache_key]
    result = {"temp": 72, "wind_speed": 5,
              "wind_direction": "calm", "condition": "clear",
              "weather_factor": 1.0, "is_dome": False}
    domes = ["Tropicana Field", "Globe Life Field", "Rogers Centre",
             "Chase Field", "Minute Maid Park", "American Family Field",
             "loanDepot park", "Daikin Park"]
    if any(dome.lower() in venue.lower() for dome in domes):
        result["is_dome"] = True
        result["weather_factor"] = 1.0
        _cache[cache_key] = result
        return result
    try:
        venue_coords = {
            "Fenway Park": (42.3467, -71.0972),
            "Yankee Stadium": (40.8296, -73.9262),
            "Wrigley Field": (41.9484, -87.6553),
            "Dodger Stadium": (34.0739, -118.2400),
            "Oracle Park": (37.7786, -122.3893),
            "Coors Field": (39.7559, -104.9942),
            "Great American Ball Park": (39.0979, -84.5082),
            "Petco Park": (32.7073, -117.1566),
            "T-Mobile Park": (47.5914, -122.3325),
        }
        coords = venue_coords.get(venue)
        if coords:
            lat, lon = coords
            url = (f"https://api.open-meteo.com/v1/forecast"
                   f"?latitude={lat}&longitude={lon}"
                   f"&hourly=temperature_2m,windspeed_10m"
                   f"&temperature_unit=fahrenheit"
                   f"&windspeed_unit=mph&forecast_days=2")
            r = requests.get(url, timeout=5)
            weather_data = r.json()
            hourly = weather_data.get("hourly", {})
            temps = hourly.get("temperature_2m", [72])[12:20]
            winds = hourly.get("windspeed_10m", [5])[12:20]
            if temps:
                result["temp"] = sum(temps) / len(temps)
            if winds:
                result["wind_speed"] = sum(winds) / len(winds)
            temp = result["temp"]
            wind = result["wind_speed"]
            factor = 1.0
            if temp > 85:
                factor += 0.05
            elif temp < 50:
                factor -= 0.05
            if wind > 15:
                factor += 0.05
            elif wind > 10:
                factor += 0.02
            result["weather_factor"] = round(factor, 3)
    except Exception as e:
        print(f"Weather error: {e}")
    _cache[cache_key] = result
    return result

def get_odds(api_key):
    if "odds" in _cache:
        return _cache["odds"]
    odds = []
    try:
        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "totals,spreads",
            "oddsFormat": "american",
            "bookmakers": "draftkings,fanduel,betmgm"
        }
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        seen = set()
        for game in data:
            key = f"{game.get('home_team')}_{game.get('away_team')}"
            if key in seen:
                continue
            seen.add(key)
            entry = {
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "total": None,
                "run_line": None,
            }
            for bookmaker in game.get("bookmakers", [])[:1]:
                for market in bookmaker.get("markets", []):
                    if market["key"] == "totals":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == "Over":
                                entry["total"] = outcome["point"]
                    if market["key"] == "spreads":
                        for outcome in market["outcomes"]:
                            if outcome["name"] == game["home_team"]:
                                entry["run_line"] = outcome["point"]
            odds.append(entry)
    except Exception as e:
        print(f"Odds error: {e}")
    _cache["odds"] = odds
    return odds

def build_game_context(game, api_key=None):
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
    park_factors = get_park_factors()
    park = park_factors.get(venue, {"runs": 1.0, "hr": 1.0})
    weather = get_weather(venue, game_date)

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
        "has_real_pitchers": game.get("has_real_pitchers", False),
        "venue": venue,
    }

def clear_cache():
    global _games_data_cache
    _cache.clear()
    _games_data_cache = []

def preload_all_data(api_key):
    global _games_data_cache
    print("V2: Preloading game data with enhanced features...")
    games = get_todays_games()
    if not games:
        print("No games today")
        return []
    odds_list = get_odds(api_key)
    games_data = []
    for game in games:
        print(f"Loading: {game['away_team']} @ {game['home_team']}")
        context = build_game_context(game, api_key)
        odds_entry = next((
            o for o in odds_list
            if (game["home_team"].lower()[:8] in
                (o.get("home_team") or "").lower() or
                game["away_team"].lower()[:8] in
                (o.get("away_team") or "").lower())
        ), None)
        games_data.append((game, context, odds_entry))
    _games_data_cache = games_data
    print(f"V2: Preloaded {len(games_data)} games")
    return games_data

def get_cached_games_data():
    return _games_data_cache
