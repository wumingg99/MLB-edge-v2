"""
Fetch 2025 historical weather for all MLB venues using Open-Meteo (free, no key).
One API call per venue for the full season. Caches to tier3_weather_2025.json.
"""
import json, requests, time
from pathlib import Path

WEATHER_CACHE = Path("/root/mlb-edge-v2/tier3_weather_2025.json")
START, END = "2025-03-27", "2025-09-28"

VENUE_COORDS = {
    "Fenway Park":                   (42.3467, -71.0972,  "America/New_York"),
    "Yankee Stadium":                (40.8296, -73.9262,  "America/New_York"),
    "Oriole Park at Camden Yards":   (39.2838, -76.6218,  "America/New_York"),
    "Citizens Bank Park":            (39.9061, -75.1665,  "America/New_York"),
    "Citi Field":                    (40.7571, -73.8458,  "America/New_York"),
    "Nationals Park":                (38.8730, -77.0074,  "America/New_York"),
    "Truist Park":                   (33.8908, -84.4679,  "America/New_York"),
    "loanDepot park":                (25.7781, -80.2197,  "America/New_York"),
    "PNC Park":                      (40.4469, -80.0057,  "America/New_York"),
    "Great American Ball Park":      (39.0979, -84.5082,  "America/New_York"),
    "Progressive Field":             (41.4962, -81.6852,  "America/New_York"),
    "Comerica Park":                 (42.3390, -83.0485,  "America/New_York"),
    "Wrigley Field":                 (41.9484, -87.6553,  "America/Chicago"),
    "Rate Field":                    (41.8299, -87.6338,  "America/Chicago"),
    "Busch Stadium":                 (38.6226, -90.1928,  "America/Chicago"),
    "Kauffman Stadium":              (39.0517, -94.4803,  "America/Chicago"),
    "Target Field":                  (44.9817, -93.2781,  "America/Chicago"),
    "Coors Field":                   (39.7559, -104.9942, "America/Denver"),
    "Dodger Stadium":                (34.0739, -118.2400, "America/Los_Angeles"),
    "Angel Stadium":                 (33.8003, -117.8827, "America/Los_Angeles"),
    "Oracle Park":                   (37.7786, -122.3893, "America/Los_Angeles"),
    "Petco Park":                    (32.7073, -117.1566, "America/Los_Angeles"),
    "T-Mobile Park":                 (47.5914, -122.3325, "America/Los_Angeles"),
    "Sutter Health Park":            (38.5897, -121.5001, "America/Los_Angeles"),
    "Rogers Centre":                 (43.6414, -79.3894,  "America/Toronto"),
}

DOMES = {
    "Tropicana Field", "Globe Life Field", "Chase Field",
    "Minute Maid Park", "Daikin Park", "American Family Field",
    "loanDepot park", "Rogers Centre",
}

def weather_factor(temp_f, wind_mph, is_dome):
    if is_dome:
        return 1.0
    temp_eff  = max(0, 72 - temp_f)  * 0.003   # -0.3% per °F below 72
    wind_eff  = min(wind_mph * 0.001, 0.02)     # +0.1% per mph, cap +2%
    return round(max(0.85, min(1.0 - temp_eff + wind_eff, 1.15)), 4)

# Load cache
cached = json.loads(WEATHER_CACHE.read_text()) if WEATHER_CACHE.exists() else {}
print(f"Cached venues: {len(cached)}")

for venue, (lat, lon, tz) in VENUE_COORDS.items():
    if venue in cached:
        continue
    is_dome = venue in DOMES
    try:
        r = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude": lat, "longitude": lon,
                "start_date": START, "end_date": END,
                "hourly": "temperature_2m,windspeed_10m",
                "timezone": tz,
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
            },
            timeout=20
        )
        d = r.json()
        # Store as {date: {hour: {temp, wind, factor}}}
        times  = d["hourly"]["time"]          # ["2025-03-27T00:00", ...]
        temps  = d["hourly"]["temperature_2m"]
        winds  = d["hourly"]["windspeed_10m"]
        by_date = {}
        for t, temp, wind in zip(times, temps, winds):
            date_str, hour_str = t.split("T")
            hour = int(hour_str[:2])
            if date_str not in by_date:
                by_date[date_str] = {}
            by_date[date_str][str(hour)] = {
                "temp_f": round(temp, 1) if temp is not None else 72.0,
                "wind_mph": round(wind, 1) if wind is not None else 5.0,
                "factor": weather_factor(
                    temp if temp is not None else 72.0,
                    wind if wind is not None else 5.0,
                    is_dome
                ),
                "is_dome": is_dome,
            }
        cached[venue] = by_date
        WEATHER_CACHE.write_text(json.dumps(cached))
        print(f"  ✓ {venue}: {len(by_date)} dates")
        time.sleep(0.3)
    except Exception as e:
        print(f"  ✗ {venue}: {e}")

print(f"\nDone. {len(cached)} venues cached → {WEATHER_CACHE}")
