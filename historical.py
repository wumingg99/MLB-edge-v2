from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path

import requests

from data import get_park_factors
from model import (
    LEAGUE_AVG_OPS,
    LEAGUE_AVG_RA9,
    LEAGUE_AVG_RPG,
    build_features,
    compute_pitching_metrics,
    parse_innings,
    train_models,
)


MLB_API_BASE = "https://statsapi.mlb.com/api/v1"
CACHE_DIR = Path(__file__).with_name("cache")
DATASET_PATH = Path(__file__).with_name("historical_dataset.jsonl")
DEFAULT_ODDS_PATH = Path(__file__).with_name("historical_odds.csv")
SEASONS = tuple(range(2022, datetime.now().year + 1))
SEASON_WEIGHTS = {
    2022: 0.55,
    2023: 0.70,
    2024: 0.85,
    2025: 1.00,
    2026: 1.10,
}

BATTING_FIELDS = (
    "atBats", "hits", "doubles", "triples", "homeRuns",
    "baseOnBalls", "hitByPitch", "sacFlies",
)
PITCHING_FIELDS = (
    "outs", "earnedRuns", "strikeOuts", "baseOnBalls",
    "hitBatsmen", "homeRuns", "hits", "airOuts",
)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _cache_path(url, params):
    CACHE_DIR.mkdir(exist_ok=True)
    payload = json.dumps([url, params], sort_keys=True).encode("utf-8")
    return CACHE_DIR / f"{hashlib.sha256(payload).hexdigest()}.json"


def _cached_request(url, params, timeout=30):
    path = _cache_path(url, params)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    response = requests.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    os.replace(temporary, path)
    return data


def get_season_games(season):
    print(f"  Fetching MLB {season} schedule...", flush=True)
    games = []
    try:
        data = _cached_request(
            f"{MLB_API_BASE}/schedule",
            {
                "sportId": 1,
                "startDate": f"{season}-03-01",
                "endDate": f"{season}-11-30",
                "gameType": "R",
                "hydrate": "team,probablePitcher,linescore,venue",
            },
        )
        for date_entry in data.get("dates", []):
            for game in date_entry.get("games", []):
                if game.get("status", {}).get(
                    "abstractGameState"
                ) != "Final":
                    continue
                home = game.get("teams", {}).get("home", {})
                away = game.get("teams", {}).get("away", {})
                linescore = game.get("linescore", {}).get("teams", {})
                home_score = linescore.get("home", {}).get("runs")
                away_score = linescore.get("away", {}).get("runs")
                if home_score is None or away_score is None:
                    continue
                home_team = home.get("team", {})
                away_team = away.get("team", {})
                home_pitcher = home.get("probablePitcher", {})
                away_pitcher = away.get("probablePitcher", {})
                games.append({
                    "game_id": int(game.get("gamePk")),
                    "date": date_entry.get("date"),
                    "game_time": game.get(
                        "gameDate", f"{date_entry.get('date')}T12:00:00Z"
                    ),
                    "season": season,
                    "home_team_id": home_team.get("id"),
                    "away_team_id": away_team.get("id"),
                    "home_team": home_team.get("name", ""),
                    "away_team": away_team.get("name", ""),
                    "home_pitcher_id": home_pitcher.get("id"),
                    "away_pitcher_id": away_pitcher.get("id"),
                    "venue": game.get("venue", {}).get("name", ""),
                    "home_score": int(home_score),
                    "away_score": int(away_score),
                })
        games.sort(key=lambda item: (item["game_time"], item["game_id"]))
        print(f"  Found {len(games)} final games for {season}")
    except Exception as exc:
        print(f"  Schedule error for {season}: {exc}")
    return games


def _game_id(split):
    game = split.get("game") or {}
    value = game.get("gamePk") or game.get("pk")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def get_team_game_logs(team_id, season):
    result = defaultdict(dict)
    try:
        data = _cached_request(
            f"{MLB_API_BASE}/teams/{team_id}/stats",
            {
                "stats": "gameLog",
                "group": "hitting,pitching",
                "season": season,
                "gameType": "R",
            },
        )
        for stat_group in data.get("stats", []):
            group = stat_group.get("group", {}).get("displayName", "")
            for split in stat_group.get("splits", []):
                game_id = _game_id(split)
                if game_id is None:
                    continue
                if group == "hitting":
                    result[game_id]["batting"] = split.get("stat", {})
                    result[game_id]["is_win"] = bool(split.get("isWin"))
                elif group == "pitching":
                    result[game_id]["pitching"] = split.get("stat", {})
        return dict(result)
    except Exception as exc:
        print(f"  Team log error {team_id}/{season}: {exc}")
        return {}


def get_pitcher_game_logs(pitcher_id, season):
    if not pitcher_id:
        return {}
    try:
        data = _cached_request(
            f"{MLB_API_BASE}/people/{pitcher_id}/stats",
            {
                "stats": "gameLog",
                "group": "pitching",
                "season": season,
                "gameType": "R",
            },
        )
        result = {}
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                game_id = _game_id(split)
                if game_id is not None:
                    result[game_id] = split.get("stat", {})
        return result
    except Exception as exc:
        print(f"  Pitcher log error {pitcher_id}/{season}: {exc}")
        return {}


def _add_fields(target, stat, fields):
    for field in fields:
        target[field] += _safe_float(stat.get(field), 0.0)
    if "outs" in fields and not stat.get("outs"):
        target["outs"] += parse_innings(stat.get("inningsPitched")) * 3.0


def _ops(raw):
    at_bats = raw["atBats"]
    hits = raw["hits"]
    doubles = raw["doubles"]
    triples = raw["triples"]
    home_runs = raw["homeRuns"]
    walks = raw["baseOnBalls"]
    hit_by_pitch = raw["hitByPitch"]
    sacrifice_flies = raw["sacFlies"]
    singles = max(hits - doubles - triples - home_runs, 0.0)
    obp_denominator = at_bats + walks + hit_by_pitch + sacrifice_flies
    obp = (
        (hits + walks + hit_by_pitch) / obp_denominator
        if obp_denominator > 0 else 0.320
    )
    slugging = (
        (singles + 2 * doubles + 3 * triples + 4 * home_runs) / at_bats
        if at_bats > 0 else 0.400
    )
    return max(0.4, min(obp + slugging, 1.4))


def _shrunk(value_total, observations, prior_mean, prior_observations):
    return (
        value_total + prior_mean * prior_observations
    ) / max(observations + prior_observations, 1.0)


class TeamState:
    def __init__(self):
        self.games = 0
        self.wins = 0
        self.runs_for = 0.0
        self.runs_against = 0.0
        self.batting = defaultdict(float)
        self.recent = deque(maxlen=10)
        self.ema_rpg = LEAGUE_AVG_RPG
        self.ema_ra9 = LEAGUE_AVG_RA9
        self.last_game_date = None
        self.bullpen_ip = 0.0
        self.bullpen_er = 0.0
        self.bullpen_recent = deque()

    def snapshot(self, game_date):
        prior_games = 15.0
        season_rpg = _shrunk(
            self.runs_for, self.games, LEAGUE_AVG_RPG, prior_games
        )
        season_ra9 = _shrunk(
            self.runs_against, self.games, LEAGUE_AVG_RA9, prior_games
        )
        season_win_pct = _shrunk(
            self.wins, self.games, 0.5, prior_games
        )
        raw_ops = _ops(self.batting) if self.games else LEAGUE_AVG_OPS
        plate_appearances = (
            self.batting["atBats"]
            + self.batting["baseOnBalls"]
            + self.batting["hitByPitch"]
        )
        ops_weight = min(plate_appearances / 600.0, 1.0)
        season_ops = (
            ops_weight * raw_ops + (1.0 - ops_weight) * LEAGUE_AVG_OPS
        )
        recent_games = list(self.recent)
        recent_count = len(recent_games)
        recent_prior = 5.0
        recent_rpg = _shrunk(
            sum(item["runs_for"] for item in recent_games),
            recent_count,
            LEAGUE_AVG_RPG,
            recent_prior,
        )
        recent_ra9 = _shrunk(
            sum(item["runs_against"] for item in recent_games),
            recent_count,
            LEAGUE_AVG_RA9,
            recent_prior,
        )
        recent_win_pct = _shrunk(
            sum(item["win"] for item in recent_games),
            recent_count,
            0.5,
            recent_prior,
        )
        recent_batting = defaultdict(float)
        for item in recent_games:
            _add_fields(recent_batting, item["batting"], BATTING_FIELDS)
        recent_ops_raw = (
            _ops(recent_batting) if recent_games else LEAGUE_AVG_OPS
        )
        recent_ops = (
            recent_count / (recent_count + recent_prior) * recent_ops_raw
            + recent_prior / (recent_count + recent_prior) * LEAGUE_AVG_OPS
        )
        bullpen_era = (
            9.0 * (self.bullpen_er + 15.0)
            / max(self.bullpen_ip + 30.0, 1.0)
        )
        cutoff = game_date - timedelta(days=3)
        workload = sum(
            innings
            for appearance_date, innings in self.bullpen_recent
            if appearance_date >= cutoff
        )
        rest_days = (
            min(max((game_date - self.last_game_date).days, 0), 10)
            if self.last_game_date else 3
        )
        return {
            "stats": {
                "rpg": season_rpg,
                "ra9": season_ra9,
                "team_era": season_ra9,
                "ops": season_ops,
                "win_pct": season_win_pct,
                "games": self.games,
            },
            "form": {
                "recent_rpg": recent_rpg,
                "recent_ra9": recent_ra9,
                "recent_ops": recent_ops,
                "recent_win_pct": recent_win_pct,
                "ema_rpg": self.ema_rpg,
                "ema_ra9": self.ema_ra9,
                "games": self.games,
            },
            "bullpen": {
                "bullpen_era": max(0.0, min(bullpen_era, 15.0)),
                "bullpen_workload": workload,
            },
            "rest_days": rest_days,
        }

    def update(
        self,
        game_date,
        runs_for,
        runs_against,
        batting_stat,
        team_pitching_stat,
        starter_stat,
    ):
        batting_stat = batting_stat or {}
        team_pitching_stat = team_pitching_stat or {}
        starter_stat = starter_stat or {}
        self.games += 1
        self.wins += int(runs_for > runs_against)
        self.runs_for += runs_for
        self.runs_against += runs_against
        _add_fields(self.batting, batting_stat, BATTING_FIELDS)
        self.recent.append({
            "runs_for": runs_for,
            "runs_against": runs_against,
            "win": int(runs_for > runs_against),
            "batting": batting_stat,
        })
        alpha = 0.25
        self.ema_rpg = alpha * runs_for + (1.0 - alpha) * self.ema_rpg
        self.ema_ra9 = (
            alpha * runs_against + (1.0 - alpha) * self.ema_ra9
        )
        team_ip = parse_innings(team_pitching_stat.get("inningsPitched"))
        starter_ip = parse_innings(starter_stat.get("inningsPitched"))
        bullpen_ip = max(team_ip - starter_ip, 0.0)
        bullpen_er = max(
            _safe_float(team_pitching_stat.get("earnedRuns"), runs_against)
            - _safe_float(starter_stat.get("earnedRuns"), 0.0),
            0.0,
        )
        self.bullpen_ip += bullpen_ip
        self.bullpen_er += bullpen_er
        self.bullpen_recent.append((game_date, bullpen_ip))
        cutoff = game_date - timedelta(days=7)
        while self.bullpen_recent and self.bullpen_recent[0][0] < cutoff:
            self.bullpen_recent.popleft()
        self.last_game_date = game_date


class PitcherState:
    def __init__(self):
        self.raw = defaultdict(float)
        self.recent = deque(maxlen=3)

    @staticmethod
    def _shrink(metrics):
        weight = min(metrics["ip"] / 50.0, 1.0)
        return {
            "era": weight * metrics["era"] + (1.0 - weight) * 4.5,
            "fip": weight * metrics["fip"] + (1.0 - weight) * 4.5,
            "xfip": weight * metrics["xfip"] + (1.0 - weight) * 4.5,
            "k9": weight * metrics["k9"] + (1.0 - weight) * 8.0,
            "bb9": weight * metrics["bb9"] + (1.0 - weight) * 3.0,
            "whip": weight * metrics["whip"] + (1.0 - weight) * 1.3,
            "ip": metrics["ip"],
            "sample_size": weight,
        }

    def snapshot(self):
        season = self._shrink(compute_pitching_metrics(self.raw))
        recent_raw = defaultdict(float)
        for stat in self.recent:
            _add_fields(recent_raw, stat, PITCHING_FIELDS)
        recent = self._shrink(compute_pitching_metrics(recent_raw))
        return {
            "season": season,
            "recent": {
                "recent_era": recent["era"],
                "recent_k9": recent["k9"],
                "recent_bb9": recent["bb9"],
                "recent_ip": recent["ip"],
            },
        }

    def update(self, stat):
        if not stat:
            return
        _add_fields(self.raw, stat, PITCHING_FIELDS)
        self.recent.append(stat)


def load_historical_odds(path=None):
    """Load real timestamped quotes; no synthetic market line is fabricated."""
    path = Path(
        path
        or os.getenv("HISTORICAL_ODDS_FILE", "")
        or DEFAULT_ODDS_PATH
    )
    if not path.exists():
        print(
            "Historical odds file not found. Score models will train, "
            "but price/ROI test metrics will remain unavailable."
        )
        return {}
    result = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                quote_time = row.get("quote_time")
                if not quote_time:
                    continue
                market = {
                    "total_line": float(row["total_line"])
                    if row.get("total_line") else None,
                    "over_price": int(float(row["over_price"]))
                    if row.get("over_price") else None,
                    "under_price": int(float(row["under_price"]))
                    if row.get("under_price") else None,
                    "home_spread": float(row["home_spread"])
                    if row.get("home_spread") else None,
                    "home_price": int(float(row["home_price"]))
                    if row.get("home_price") else None,
                    "away_spread": float(row["away_spread"])
                    if row.get("away_spread") else None,
                    "away_price": int(float(row["away_price"]))
                    if row.get("away_price") else None,
                    "bookmaker": row.get("bookmaker"),
                    "quote_time": quote_time,
                }
                if row.get("game_id"):
                    result[("game_id", int(row["game_id"]))].append(market)
                matchup_key = (
                    "matchup",
                    row.get("date", "")[:10],
                    row.get("away_team", "").strip().lower(),
                    row.get("home_team", "").strip().lower(),
                )
                result[matchup_key].append(market)
            except (TypeError, ValueError, KeyError):
                continue
    print(f"Loaded {len(result)} historical market keys from {path.name}")
    return result


def _market_for_game(odds, game):
    candidates = odds.get(("game_id", game["game_id"])) or odds.get((
        "matchup",
        game["date"],
        game["away_team"].strip().lower(),
        game["home_team"].strip().lower(),
    ), [])
    try:
        game_time = datetime.fromisoformat(
            str(game["game_time"]).replace("Z", "+00:00")
        )
    except (TypeError, ValueError):
        return None
    valid = []
    for market in candidates:
        try:
            quote_time = datetime.fromisoformat(
                str(market["quote_time"]).replace("Z", "+00:00")
            )
        except (TypeError, ValueError):
            continue
        if quote_time <= game_time:
            valid.append((quote_time, market))
    if not valid:
        return None
    return max(valid, key=lambda item: item[0])[1]


def _snapshot_quality(home, away, home_pitcher, away_pitcher):
    values = [
        min(home["stats"]["games"] / 20.0, 1.0),
        min(away["stats"]["games"] / 20.0, 1.0),
        min(home_pitcher["season"]["ip"] / 30.0, 1.0),
        min(away_pitcher["season"]["ip"] / 30.0, 1.0),
        1.0,
        0.5,
    ]
    return sum(values) / len(values)


def build_historical_dataset(
    seasons=SEASONS,
    odds_path=None,
    dataset_path=DATASET_PATH,
):
    odds = load_historical_odds(odds_path)
    park_factors = get_park_factors()
    records = []

    for season in seasons:
        games = get_season_games(season)
        if not games:
            continue
        team_ids = sorted({
            game["home_team_id"] for game in games
        } | {
            game["away_team_id"] for game in games
        })
        print(f"  Loading {len(team_ids)} team game logs...", flush=True)
        team_logs = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(get_team_game_logs, team_id, season): team_id
                for team_id in team_ids
            }
            for future in as_completed(futures):
                team_id = futures[future]
                team_logs[team_id] = future.result()

        pitcher_ids = sorted({
            pitcher_id
            for game in games
            for pitcher_id in (
                game["home_pitcher_id"],
                game["away_pitcher_id"],
            )
            if pitcher_id
        })
        print(
            f"  Loading {len(pitcher_ids)} starter game logs...",
            flush=True,
        )
        pitcher_log_cache = {}
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    get_pitcher_game_logs, pitcher_id, season
                ): pitcher_id
                for pitcher_id in pitcher_ids
            }
            for completed, future in enumerate(
                as_completed(futures), start=1
            ):
                pitcher_id = futures[future]
                pitcher_log_cache[pitcher_id] = future.result()
                if completed % 100 == 0:
                    print(
                        f"    starter logs: {completed}/{len(pitcher_ids)}",
                        flush=True,
                    )
        team_states = defaultdict(TeamState)
        pitcher_states = defaultdict(PitcherState)
        processed = 0

        for game in games:
            game_date = datetime.fromisoformat(game["date"]).date()
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]
            home_pitcher_id = game["home_pitcher_id"]
            away_pitcher_id = game["away_pitcher_id"]
            if not home_pitcher_id or not away_pitcher_id:
                continue

            home_team_snapshot = team_states[home_id].snapshot(game_date)
            away_team_snapshot = team_states[away_id].snapshot(game_date)
            home_pitcher_snapshot = pitcher_states[
                home_pitcher_id
            ].snapshot()
            away_pitcher_snapshot = pitcher_states[
                away_pitcher_id
            ].snapshot()
            park = park_factors.get(
                game["venue"], {"runs": 1.0, "hr": 1.0}
            )
            quality = _snapshot_quality(
                home_team_snapshot,
                away_team_snapshot,
                home_pitcher_snapshot,
                away_pitcher_snapshot,
            )
            context = {
                "home_stats": home_team_snapshot["stats"],
                "away_stats": away_team_snapshot["stats"],
                "home_form": home_team_snapshot["form"],
                "away_form": away_team_snapshot["form"],
                "home_pitcher": home_pitcher_snapshot["season"],
                "away_pitcher": away_pitcher_snapshot["season"],
                "home_pitcher_recent": home_pitcher_snapshot["recent"],
                "away_pitcher_recent": away_pitcher_snapshot["recent"],
                "home_bullpen": home_team_snapshot["bullpen"],
                "away_bullpen": away_team_snapshot["bullpen"],
                "park_factor": park["runs"],
                "park_hr_factor": park["hr"],
                "weather": {
                    "weather_factor": 1.0,
                    "is_dome": False,
                    "available": False,
                },
                "home_rest_days": home_team_snapshot["rest_days"],
                "away_rest_days": away_team_snapshot["rest_days"],
                "has_real_pitchers": True,
                "data_quality": quality,
                "venue": game["venue"],
                "as_of": game["game_time"],
            }
            features = build_features(context)
            if features is not None:
                records.append({
                    "game_id": game["game_id"],
                    "date": game["game_time"],
                    "season": season,
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "features": features,
                    "actual_total": game["home_score"] + game["away_score"],
                    "home_margin": game["home_score"] - game["away_score"],
                    "market": _market_for_game(odds, game),
                    "sample_weight": SEASON_WEIGHTS.get(season, 1.0),
                })

            home_team_game = team_logs.get(home_id, {}).get(
                game["game_id"], {}
            )
            away_team_game = team_logs.get(away_id, {}).get(
                game["game_id"], {}
            )
            home_starter_stat = pitcher_log_cache[
                home_pitcher_id
            ].get(game["game_id"], {})
            away_starter_stat = pitcher_log_cache[
                away_pitcher_id
            ].get(game["game_id"], {})
            team_states[home_id].update(
                game_date,
                game["home_score"],
                game["away_score"],
                home_team_game.get("batting"),
                home_team_game.get("pitching"),
                home_starter_stat,
            )
            team_states[away_id].update(
                game_date,
                game["away_score"],
                game["home_score"],
                away_team_game.get("batting"),
                away_team_game.get("pitching"),
                away_starter_stat,
            )
            pitcher_states[home_pitcher_id].update(home_starter_stat)
            pitcher_states[away_pitcher_id].update(away_starter_stat)
            processed += 1
            if processed % 250 == 0:
                print(
                    f"    {season}: {processed}/{len(games)} games",
                    flush=True,
                )
        print(f"  {season}: built {processed} pregame snapshots")

    dataset_path = Path(dataset_path)
    temporary = dataset_path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
    os.replace(temporary, dataset_path)
    print(f"Saved {len(records)} rows to {dataset_path.name}")
    return records


def load_historical_dataset(path=DATASET_PATH):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def train_on_historical(rebuild=True, odds_path=None):
    print("Building MLB V3 leak-safe historical dataset...")
    records = (
        build_historical_dataset(odds_path=odds_path)
        if rebuild else load_historical_dataset()
    )
    if len(records) < 500:
        raise RuntimeError(
            f"Only {len(records)} usable games were built; model not replaced"
        )
    bundle = train_models(records)
    print("\nV3 training complete")
    print(
        f"Rows: train={bundle['train_rows']} "
        f"calibration={bundle['calibration_rows']} "
        f"test={bundle['test_rows']}"
    )
    print(json.dumps(bundle["metrics"], indent=2))
    return bundle


if __name__ == "__main__":
    train_on_historical(rebuild=True)
