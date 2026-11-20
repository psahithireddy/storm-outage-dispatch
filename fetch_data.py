"""Phase 3a: download real weather (actual + historical forecast) and OSM power assets.

Study area: Hyderabad region. Data lands in data/ as JSON.
"""
import json
import subprocess
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "data"
BBOX = (17.2, 78.2, 17.6, 78.7)        # south, west, north, east
GRID = 4                               # GRID x GRID weather points
START, END = "2024-08-25", "2024-09-10"
HOURLY = "wind_gusts_10m,wind_speed_10m,precipitation"
UA = "outage-dispatch-study/0.1"


def get_json(url, data=None):
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def grid_points():
    s, w, n, e = BBOX
    lats = np.linspace(s, n, GRID)
    lons = np.linspace(w, e, GRID)
    return [(round(float(la), 4), round(float(lo), 4)) for la in lats for lo in lons]


def fetch_weather(base, name, variables):
    pts = grid_points()
    q = urllib.parse.urlencode({
        "latitude": ",".join(str(p[0]) for p in pts),
        "longitude": ",".join(str(p[1]) for p in pts),
        "start_date": START, "end_date": END,
        "hourly": ",".join(variables), "timezone": "Asia/Kolkata",
    })
    res = get_json(f"{base}?{q}")
    res = res if isinstance(res, list) else [res]
    out = {"points": pts, "series": []}
    for r in res:
        out["series"].append({"time": r["hourly"]["time"],
                              **{k: r["hourly"][k] for k in variables}})
    (OUT / f"weather_{name}.json").write_text(json.dumps(out))
    print(f"{name}: {len(res)} points x {len(res[0]['hourly']['time'])} hours")


def fetch_power():
    s, w, n, e = BBOX
    box = f"({s},{w},{n},{e})"
    query = (f"[out:json][timeout:80];("
             f'node["power"="substation"]{box};way["power"="substation"]{box};'
             f'way["power"="line"]{box};way["power"="minor_line"]{box};'
             f");out center geom;")
    # curl: Python 3.9's bundled SSL fails the TLS handshake with overpass-api.de
    raw = subprocess.run(
        ["curl", "-s", "-m", "120", "-A", UA, "--data-urlencode", f"data={query}",
         "https://overpass-api.de/api/interpreter"],
        check=True, capture_output=True, text=True).stdout
    res = json.loads(raw)
    (OUT / "power_osm.json").write_text(raw)
    kinds = {}
    for el in res["elements"]:
        k = (el["type"], el.get("tags", {}).get("power"))
        kinds[k] = kinds.get(k, 0) + 1
    print("power:", kinds)


def main():
    OUT.mkdir(exist_ok=True)
    base_vars = HOURLY.split(",")
    fetch_weather("https://archive-api.open-meteo.com/v1/archive", "actual", base_vars)
    time.sleep(2)
    # true lead-time forecasts: what the model said 1 and 2 days before each hour
    lead_vars = [f"{v}_previous_day{d}" for v in base_vars for d in (1, 2)]
    fetch_weather("https://previous-runs-api.open-meteo.com/v1/forecast", "forecast", lead_vars)
    time.sleep(2)
    fetch_power()


if __name__ == "__main__":
    main()
