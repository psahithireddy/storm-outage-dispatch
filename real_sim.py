"""Phase 3a: replay a real storm window through the dispatch simulator and export for the web app.

Real: weather (Open-Meteo), substation/line geography (OpenStreetMap).
Assumed: customers per zone, vulnerability, fault rule, repair times, crews.
Faults are SYNTHETIC: drawn from a documented hazard rule driven by real weather.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import prepos
import sim

ROOT = Path(__file__).parent
WINDOW_START = "2024-08-31T00:00"
WINDOW_HOURS = 72                   # faults can only start in this storm window
WEATHER_HOURS = 240                 # weather kept longer so late restorations still show real weather
N_ZONES = 24
CUSTOMERS_PER_SUBSTATION = 2500     # assumption
TARGET_FAULTS = 90                  # assumption: calibrates the hazard scale
WIND_THRESHOLD_KMH = 35.0           # below this, gusts add no fault risk
FAULT_ZONE_SHARE = (0.05, 0.3)      # assumption: share of a zone's customers hit by one fault
N_REALIZATIONS = 150
CREW_OPTIONS = (2, 3, 4, 6, 10, 14)
DEFAULT_CREWS = 6
LAT0, LON0 = 17.4, 78.45
N_DEPOTS = 3


def to_xy(lat, lon):
    return ((lon - LON0) * 111.32 * np.cos(np.radians(LAT0)), (lat - LAT0) * 110.57)


def load_power():
    d = json.load(open(ROOT / "data/power_osm.json"))
    subs, lines = [], []
    for e in d["elements"]:
        p = e.get("tags", {}).get("power")
        if p == "substation":
            if e["type"] == "node":
                subs.append((e["lat"], e["lon"]))
            elif "center" in e:
                subs.append((e["center"]["lat"], e["center"]["lon"]))
            elif "geometry" in e:
                g = e["geometry"]
                subs.append((np.mean([p["lat"] for p in g]), np.mean([p["lon"] for p in g])))
        elif p in ("line", "minor_line") and "geometry" in e:
            lines.append([[round(g["lat"], 4), round(g["lon"], 4)] for g in e["geometry"]])
    return np.array(subs), lines


def kmeans(pts, k, rng, iters=50):
    cent = pts[rng.choice(len(pts), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((pts[:, None] - cent[None]) ** 2).sum(-1), axis=1)
        new = np.array([pts[lab == j].mean(0) if (lab == j).any() else cent[j] for j in range(k)])
        if np.allclose(new, cent):
            break
        cent = new
    return lab, cent


def load_weather():
    def arr(d, key):
        return np.array([[0.0 if v is None else v for v in s[key]] for s in d["series"]], float)
    a = json.load(open(ROOT / "data/weather_actual.json"))
    f = json.load(open(ROOT / "data/weather_forecast.json"))
    times = a["series"][0]["time"]
    i0 = times.index(WINDOW_START)
    sl = slice(i0, i0 + WEATHER_HOURS)
    return {
        "points": a["points"],
        "times": times[sl],
        "gust": arr(a, "wind_gusts_10m")[:, sl],
        "rain": arr(a, "precipitation")[:, sl],
        "gust_f1": arr(f, "wind_gusts_10m_previous_day1")[:, sl],
        "rain_f1": arr(f, "precipitation_previous_day1")[:, sl],
    }


def hazard(gust, rain):
    wind = np.maximum(0.0, gust - WIND_THRESHOLD_KMH) / 25.0
    return wind ** 2 + 0.3 * rain / 10.0


def build_zones(rng):
    subs, lines = load_power()
    lab, cent = kmeans(subs, N_ZONES, rng)
    n_sub = np.bincount(lab, minlength=N_ZONES)
    cust = (n_sub * CUSTOMERS_PER_SUBSTATION * rng.lognormal(0, 0.3, N_ZONES)).astype(int)
    critical = np.zeros(N_ZONES, bool)
    critical[rng.choice(N_ZONES, 4, replace=False)] = True
    vuln = rng.lognormal(0, 0.5, N_ZONES)          # tree cover / asset age placeholder
    return subs, lines, cent, n_sub, cust, critical, vuln


def make_faults(rng, zones_xy, cust, critical, expo, haz_zone, scale):
    """haz_zone: (zones, hours) hazard. Returns sorted sim.Fault list."""
    faults = []
    lam = scale * expo[:, None] * haz_zone
    counts = rng.poisson(lam)
    for z, h in zip(*np.nonzero(counts)):
        for _ in range(counts[z, h]):
            t = h + rng.uniform()
            frac = rng.uniform(*FAULT_ZONE_SHARE)
            rep = sim.REPAIR_MEDIAN_H * np.exp(rng.normal(0.0, sim.REPAIR_SIGMA))
            faults.append(sim.Fault(int(z), float(t), float(zones_xy[z, 0]), float(zones_xy[z, 1]),
                                    max(1, int(cust[z] * frac)), float(rep), bool(critical[z])))
    faults.sort(key=lambda f: f.t)
    return faults


def setup():
    """Shared scenario: zones, depots, real weather, hazard scale and simulated storms."""
    rng = np.random.default_rng(7)
    subs, lines, cent, n_sub, cust, critical, vuln = build_zones(rng)
    wx = load_weather()

    zones_xy = np.array([to_xy(la, lo) for la, lo in cent])
    depots = zones_xy[rng.choice(N_ZONES, N_DEPOTS, replace=False)]

    # nearest weather grid point per zone
    gp = np.array(wx["points"])
    nearest = np.argmin(((cent[:, None] - gp[None]) ** 2).sum(-1), axis=1)
    haz_zone = hazard(wx["gust"][nearest], wx["rain"][nearest])[:, :WINDOW_HOURS]
    expo = n_sub * vuln
    scale = TARGET_FAULTS / (expo[:, None] * haz_zone).sum()

    realizations = [make_faults(rng, zones_xy, cust, critical, expo, haz_zone, scale)
                    for _ in range(N_REALIZATIONS)]
    realizations = [f for f in realizations if f]
    return SimpleNamespace(subs=subs, lines=lines, cent=cent, n_sub=n_sub, cust=cust,
                           critical=critical, vuln=vuln, wx=wx, zones_xy=zones_xy,
                           depots=depots, total_cust=int(cust.sum()), nearest=nearest,
                           haz_zone=haz_zone, expo=expo, scale=scale,
                           realizations=realizations)


def from_xy(x, y):
    return [round(LAT0 + y / 110.57, 5), round(LON0 + x / (111.32 * np.cos(np.radians(LAT0))), 5)]


def build_planners(c):
    """Positioning setups: None = crews stay at depots; others reposition idle crews."""
    consequence = c.cust + sim.CRITICAL_BONUS * c.critical
    hz_actual = hazard(c.wx["gust"][c.nearest], c.wx["rain"][c.nearest])[:, :WINDOW_HOURS]
    hz_fcst = hazard(c.wx["gust_f1"][c.nearest], c.wx["rain_f1"][c.nearest])[:, :WINDOW_HOURS]
    flat = np.ones_like(hz_actual) * hz_actual.mean()
    risk = lambda hz: c.scale * c.expo[:, None] * hz
    return {
        "depot": None,
        "flat": prepos.Planner(c.zones_xy, risk(flat), consequence),
        "forecast": prepos.Planner(c.zones_xy, risk(hz_fcst), consequence),
        "oracle": prepos.Planner(c.zones_xy, risk(hz_actual), consequence),
    }


def main():
    c = setup()
    subs, lines, cent, n_sub = c.subs, c.lines, c.cent, c.n_sub
    cust, critical, vuln, wx = c.cust, c.critical, c.vuln, c.wx
    zones_xy, depots, total_cust = c.zones_xy, c.depots, c.total_cust
    nearest, realizations = c.nearest, c.realizations
    names = list(sim.STRATEGIES)
    fault_counts = [len(f) for f in realizations]
    mean_n = np.mean(fault_counts)

    # representative realization: fault count closest to the mean, has a critical fault
    order = np.argsort([abs(len(f) - mean_n) for f in realizations])
    rep = next(realizations[k] for k in order if any(f.critical for f in realizations[k]))
    out_faults = [{"zone": f.zone, "t": round(f.t, 2), "lat": float(cent[f.zone][0]),
                   "lon": float(cent[f.zone][1]), "customers": f.customers,
                   "critical": f.critical, "repair_h": round(f.repair_h, 2)} for f in rep]

    planners = build_planners(c)
    modes = {"depot": planners["depot"], "prepos": planners["forecast"]}
    by_crews = {}
    for crews in CREW_OPTIONS:
        sim.N_CREWS = crews
        entry, arrs = {}, {}
        for mode, planner in modes.items():
            agg = {n: {"saidi": [], "saifi": [], "crit_h": [], "mean_h": [], "travel_h": []} for n in names}
            for faults in realizations:
                for n in names:
                    m = sim.simulate(faults, depots, sim.STRATEGIES[n], planner=planner)
                    agg[n]["saidi"].append(m["customer_minutes"] / total_cust)
                    agg[n]["saifi"].append(m["customers_interrupted"] / total_cust)
                    agg[n]["crit_h"].append(m["critical_restore_h"])
                    agg[n]["mean_h"].append(m["mean_restore_h"])
                    agg[n]["travel_h"].append(m["mean_travel_h"])
            summary = {n: {k: [round(float(np.nanmean(v)), 3), round(float(np.nanstd(v)), 3)]
                           for k, v in agg[n].items()} for n in names}
            strategies = {}
            for n in names:
                tracks = []
                m, restore = sim.simulate(rep, depots, sim.STRATEGIES[n], detail=True,
                                          planner=planner, tracks=tracks)
                arrs[mode, n] = np.array([[f.t, restore[id(f)], f.customers] for f in rep])
                strategies[n] = {
                    "restore": [round(restore[id(f)], 2) for f in rep],
                    "saidi": round(m["customer_minutes"] / total_cust, 2),
                    "saifi": round(m["customers_interrupted"] / total_cust, 4),
                    "crit_h": round(m["critical_restore_h"], 2),
                    "mean_h": round(m["mean_restore_h"], 2),
                    "tracks": [[[round(t, 2)] + from_xy(x, y) for t, x, y in tr] for tr in tracks],
                }
            entry[mode] = {"summary": summary, "storm": strategies}
        # timeline runs until the last fault in this storm is restored (capped by weather data)
        horizon = int(min(WEATHER_HOURS, np.ceil(max(a[:, 1].max() for a in arrs.values()))))
        hours = np.arange(horizon + 1)
        for mode in modes:
            for n in names:
                a = arrs[mode, n]
                entry[mode]["storm"][n]["customers_out"] = [
                    int(a[(a[:, 0] <= h) & (a[:, 1] > h), 2].sum()) for h in hours]
        entry["horizon"] = horizon
        by_crews[str(crews)] = entry
    sim.N_CREWS = DEFAULT_CREWS

    zones_out = [{"lat": float(cent[z][0]), "lon": float(cent[z][1]), "customers": int(cust[z]),
                  "substations": int(n_sub[z]), "critical": bool(critical[z]),
                  "vuln": round(float(vuln[z]), 2), "grid": int(nearest[z])} for z in range(N_ZONES)]
    data = {
        "meta": {"window_start": WINDOW_START, "hours": WINDOW_HOURS, "weather_hours": WEATHER_HOURS, "crew_options": list(CREW_OPTIONS),
                 "default_crews": DEFAULT_CREWS,
                 "realizations": len(fault_counts), "mean_faults": round(float(mean_n), 1),
                 "total_customers": total_cust, "target_faults": TARGET_FAULTS},
        "times": wx["times"],
        "grid": wx["points"],
        "weather": {k: np.round(wx[k], 1).tolist() for k in ("gust", "rain", "gust_f1", "rain_f1")},
        "zones": zones_out,
        "substations": [[round(float(a), 4), round(float(b), 4)] for a, b in subs],
        "lines": lines,
        "depots": [[LAT0 + y / 110.57, LON0 + x / (111.32 * np.cos(np.radians(LAT0)))]
                   for x, y in depots],
        "faults": out_faults,
        "by_crews": by_crews,
    }
    web = ROOT / "web"
    web.mkdir(exist_ok=True)
    (web / "data.js").write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n")

    print(f"{len(fault_counts)} realizations, mean faults {mean_n:.1f}, customers {total_cust:,}")
    print(f"{'crews':>5s} {'rule':9s} {'SAIDI depot':>12s} {'SAIDI prepos':>13s} {'change':>7s}")
    for crews in CREW_OPTIONS:
        for n in names:
            d = by_crews[str(crews)]["depot"]["summary"][n]["saidi"][0]
            p = by_crews[str(crews)]["prepos"]["summary"][n]["saidi"][0]
            print(f"{crews:5d} {n:9s} {d:12.1f} {p:13.1f} {(p / d - 1) * 100:6.1f}%")
    print("representative storm:", len(out_faults), "faults")


if __name__ == "__main__":
    main()
