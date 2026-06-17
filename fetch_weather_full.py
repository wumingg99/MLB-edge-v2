"""Fetch historical weather for all training seasons using Open-Meteo (free)."""
import json, requests, time
from pathlib import Path

CACHE = Path("/root/mlb-edge-v2/tier3_weather_full.json")
SEASONS = [(2022,2023), (2023,2024), (2024,2025), (2025,2026)]  # start-end pairs

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
    "Guaranteed Rate Field":         (41.8299, -87.6338,  "America/Chicago"),
    "Busch Stadium":                 (38.6226, -90.1928,  "America/Chicago"),
    "Kauffman Stadium":              (39.0517, -94.4803,  "America/Chicago"),
    "Target Field":                  (44.9817, -93.2781,  "America/Chicago"),
    "Minute Maid Park":              (29.7573, -95.3555,  "America/Chicago"),
    "Daikin Park":                   (29.7573, -95.3555,  "America/Chicago"),
    "Coors Field":                   (39.7559, -104.9942, "America/Denver"),
    "Dodger Stadium":                (34.0739, -118.2400, "America/Los_Angeles"),
    "Angel Stadium":                 (33.8003, -117.8827, "America/Los_Angeles"),
    "Oracle Park":                   (37.7786, -122.3893, "America/Los_Angeles"),
    "Petco Park":                    (32.7073, -117.1566, "America/Los_Angeles"),
    "T-Mobile Park":                 (47.5914, -122.3325, "America/Los_Angeles"),
    "Sutter Health Park":            (38.5897, -121.5001, "America/Los_Angeles"),
    "Oakland Coliseum":              (37.7516, -122.2005, "America/Los_Angeles"),
    "RingCentral Coliseum":          (37.7516, -122.2005, "America/Los_Angeles"),
    "Rogers Centre":                 (43.6414, -79.3894,  "America/Toronto"),
    "American Family Field":         (43.0280, -87.9712,  "America/Chicago"),
    "Chase Field":                   (33.4455, -112.0667, "America/Phoenix"),
}

DOMES = {
    "Tropicana Field","Globe Life Field","Chase Field","Minute Maid Park",
    "Daikin Park","American Family Field","loanDepot park","Rogers Centre",
    "Marlins Park","Roof Field",
}

def weather_factor(temp_f, wind_mph, is_dome):
    if is_dome: return 1.0
    temp_eff = max(0, 72 - temp_f) * 0.003
    wind_eff = min(wind_mph * 0.001, 0.02)
    return round(max(0.85, min(1.0 - temp_eff + wind_eff, 1.15)), 4)

saved = json.loads(CACHE.read_text()) if CACHE.exists() else {}

for venue, (lat, lon, tz) in VENUE_COORDS.items():
    is_dome = venue in DOMES
    for start, end in SEASONS:
        cache_key = f"{venue}::{start}"
        if cache_key in saved:
            continue
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params={
                    "latitude": lat, "longitude": lon,
                    "start_date": f"{start}-03-01",
                    "end_date": f"{end}-03-01",
                    "hourly": "temperature_2m,windspeed_10m",
                    "timezone": tz,
                    "temperature_unit": "fahrenheit",
                    "windspeed_unit": "mph",
                },
                timeout=30
            )
            d = r.json()
            times = d["hourly"]["time"]
            temps = d["hourly"]["temperature_2m"]
            winds = d["hourly"]["windspeed_10m"]
            by_date = {}
            for t, temp, wind in zip(times, temps, winds):
                date_str, hour_str = t.split("T")
                hour = int(hour_str[:2])
                if date_str not in by_date:
                    by_date[date_str] = {}
                tf = temp if temp is not None else 72.0
                wf = wind if wind is not None else 5.0
                by_date[date_str][str(hour)] = {
                    "temp_f": round(tf, 1),
                    "wind_mph": round(wf, 1),
                    "factor": weather_factor(tf, wf, is_dome),
                    "is_dome": is_dome,
                }
            saved[cache_key] = by_date
            CACHE.write_text(json.dumps(saved))
            print(f"  ✓ {venue} {start}: {len(by_date)} dates")
            time.sleep(0.25)
        except Exception as e:
            print(f"  ✗ {venue} {start}: {e}")

print(f"\nDone. {len(saved)} venue-season combinations cached.")
