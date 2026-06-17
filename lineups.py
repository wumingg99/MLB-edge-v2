"""
lineups.py — Fetch season wOBA for confirmed lineup players.

In-session cache: player stats fetched once per player per process restart.
Season stats don't change mid-game so this is safe and efficient.
"""
from __future__ import annotations
import requests
from lineup_strength import LG_WOBA

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
_woba_cache: dict[int, float] = {}


def _compute_woba(stat: dict) -> float:
    """Compute wOBA from MLB Stats API season hitting stat dict."""
    hits    = stat.get("hits", 0) or 0
    doubles = stat.get("doubles", 0) or 0
    triples = stat.get("triples", 0) or 0
    hr      = stat.get("homeRuns", 0) or 0
    singles = max(0, hits - doubles - triples - hr)
    bb      = stat.get("baseOnBalls", 0) or 0
    ibb     = stat.get("intentionalWalks", 0) or 0
    ubb     = max(0, bb - ibb)
    hbp     = stat.get("hitByPitch", 0) or 0
    ab      = stat.get("atBats", 0) or 0
    sf      = stat.get("sacFlies", 0) or 0
    denom   = ab + ubb + hbp + sf
    if denom < 50:
        return LG_WOBA  # too few PAs — fall back to league average
    numer = (0.690 * ubb + 0.722 * hbp + 0.888 * singles +
             1.271 * doubles + 1.616 * triples + 2.101 * hr)
    return round(numer / denom, 4)


def get_player_woba(player_id: int, season: int = 2026) -> float:
    """Fetch season wOBA for one player. In-session cached."""
    if player_id in _woba_cache:
        return _woba_cache[player_id]
    try:
        r = requests.get(
            f"{MLB_STATS_BASE}/people/{player_id}/stats",
            params={"stats": "season", "group": "hitting", "season": season},
            timeout=5,
        )
        splits = r.json().get("stats", [{}])[0].get("splits", [])
        woba = _compute_woba(splits[0].get("stat", {})) if splits else LG_WOBA
    except Exception:
        woba = LG_WOBA
    _woba_cache[player_id] = woba
    return woba


def get_lineup_wobas(player_ids: list[int], season: int = 2026) -> list[float] | None:
    """
    Return 9 wOBAs in batting order for a confirmed lineup.
    Slots without a confirmed player default to LG_WOBA.
    Returns None if player_ids is empty.
    """
    if not player_ids:
        return None
    wobas = [get_player_woba(pid, season) for pid in player_ids[:9]]
    while len(wobas) < 9:
        wobas.append(LG_WOBA)
    return wobas[:9]
