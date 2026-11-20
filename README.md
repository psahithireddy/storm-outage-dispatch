# Storm outage prediction and crew dispatch

A prototype for the TISC challenge (Department of Energy, Government of Telangana):
predict where storms will hurt the power network, position repair crews, and dispatch them
so power comes back faster.

**Live demo:** https://psahithireddy.github.io/storm-outage-dispatch/
Replay the 31 Aug – 2 Sep 2024 monsoon spell over Hyderabad, change the number of crews,
switch dispatch rules, and turn crew pre-positioning on and off. The page shows SAIDI and SAIFI.

## Read this first
- **Real:** hourly wind and rain and day-ahead forecasts (Open-Meteo), and substations and power
  lines (OpenStreetMap).
- **Simulated or assumed:** the faults, customers per zone, repair times, travel speed, crew counts
  and which zones are critical loads. There are no real DISCOM outage records in this project.
- **Not validated:** nothing here has been tested against real fault data or in a pilot.
  The numbers show how the dispatch logic behaves, not what a real utility would see.

## Main findings (simulation, ~88 faults per storm, 150 random fault draws)
- Priority dispatch cuts SAIDI by 37% / 35% / 29% against nearest-fault dispatch with 2 / 3 / 4 crews.
  With 10 or more crews there is no queue and the dispatch rule makes no difference.
- Moving idle crews toward high-risk zones saves about 3–4% of SAIDI with 10–14 crews and almost
  nothing with 6 or fewer.
- Day-ahead weather forecasts added nothing over staging by customers and critical loads in this
  test, because the wind was nearly uniform across the area.
- SAIFI does not change with dispatch. Lowering it needs prediction and prevention, which are planned
  and not built. See `PLAN.md`.

## Run it
Python 3.9+ with `numpy`.

```bash
python3 fetch_data.py      # download weather and OpenStreetMap data into data/
python3 run_phase1.py      # dispatch rules on synthetic storms
python3 real_sim.py        # replay the real storm and write web/data.js
python3 phase2.py          # pre-positioning experiment
python3 -m http.server 8765 --directory web
```

## Files
| File | What it does |
|---|---|
| `sim.py` | Dispatch simulator: crews, three dispatch rules, optional pre-positioning |
| `prepos.py` | Greedy planner that sends idle crews toward high-risk, high-customer zones |
| `real_sim.py` | Builds zones from real substations, replays real weather, exports `web/data.js` |
| `fetch_data.py` | Downloads weather and OpenStreetMap data |
| `phase2.py`, `run_phase1.py`, `sweep_load.py` | Experiments |
| `web/` | The interactive page (Leaflet map, charts) |

## Data credits
Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0). Map data and power infrastructure:
© [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors (ODbL).
