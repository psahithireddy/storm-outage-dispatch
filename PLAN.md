# Storm Outage Prediction & Crew Dispatch — Project Plan

TISC challenge (Dept. of Energy, Telangana). Goal: solve the problem first; the pitch deck is a write-up of the results.

## Core question
Does predicting weather-driven faults and pre-positioning crews reduce customer-minutes interrupted (SAIDI) and customers interrupted (SAIFI) compared with reactive dispatch?

## Principles
- Solution first, deck last.
- Keep prediction claims and dispatch claims separate (simulated faults test dispatch, not prediction).
- Label every assumption (repair times, crew counts, road delays, synthetic data).
- Baseline before complexity: each layer must beat the one before it.

## Phases

### Phase 0 — Setup
- Project structure, Python environment, version notes.
- Pick study area: Hyderabad region + a few districts (small enough to iterate fast).
- Done when: folder runs a hello-world pipeline.

### Phase 1 — Dispatch simulator (first, on made-up data)
- Synthetic zones: customers, critical-load weight, feeder hierarchy, depots, travel-time matrix.
- Fault generator over time from a per-zone risk score.
- Strategies: (a) reactive nearest/first-come baseline, (b) priority rule = (customers + critical weight) / (travel + repair time).
- Metrics: customer-minutes interrupted, customers interrupted, critical-load restoration time, crew travel distance.
- Done when: comparison table of (a) vs (b) over many random storms.

### Phase 2 — Pre-positioning
- Greedy facility-location placement of crews by expected weighted response time; keep a reserve.
- Done when: pre-positioned + priority beats priority-only, or we know when it does not (forecast uncertainty sweep).

### Phase 3 — Real data foundation
- Weather: Open-Meteo historical (wind, gusts, rain) for the study area.
- Geography: OpenStreetMap power lines/substations via Overpass; road times via OSRM.
- Context: tree cover (Global Forest Watch), population (WorldPop).
- Synthetic outage generator with a documented rule (risk rises with gusts, rain, vegetation, asset age, plus noise).
- Done when: map of study area with zones, assets and a weather time series.

### Phase 4 — Prediction model
- Baseline: threshold rule, then logistic regression, then gradient boosting.
- Real-data validation: US EAGLE-I outage + weather data to show weather predicts outages (clearly labeled as US data).
- Done when: model beats baseline on a held-out storm; results recorded with honest caveats.

### Phase 5 — Agent + dashboard
- Agent loop: ingest forecast, re-score risk, propose crew moves only when expected gain exceeds move cost, explain in plain language, dispatcher approves.
- Dashboard: risk map, ranked zones, crew panel, proposed moves.
- Done when: replay of a storm shows proposals updating over time.

### Phase 6 — Evaluation write-up
- Compare reactive vs priority vs pre-positioned vs agent on SAIDI/SAIFI proxies.
- Sensitivity to assumptions (repair time, crew count, forecast error).
- Done when: results and limitations written down plainly.

### Phase 7 — Pitch deck (.pptx, <= 20 MB)
Follow TISC guidelines: problem + audience, how it works (screenshots/demo), backed claims with labeled assumptions, opportunity + next steps, one idea per slide.
- Optional: demo video (up to two links; test in a private window).
- Check Submission tab for deadline, slide count, format.

## Open items
- Submission deadline (not in guidelines PDF; check Submission tab).
- Study area choice.
- Language/stack: default Python (pandas, scikit-learn/LightGBM, OR-Tools optional) + simple web dashboard.

## Risks
- No real DISCOM fault data: mitigate with synthetic generator + US validation + pilot ask.
- Overbuilding: stop each phase at its "done when".

## Status and findings (updated after Phase 2)

Done: Phase 0, 1, 2 and Phase 3a (real weather + OSM geography, website in web/).
Not done: real-data prediction model (Phase 4), agent loop (Phase 5), write-up and deck.

Simulator fix found during Phase 2: a new fault used to go to the crew free longest, not the
nearest free crew. Fixed in sim.py; earlier SAIDI numbers at high crew counts were pessimistic.

Real storm replay (31 Aug-2 Sep 2024, Hyderabad region, 150 simulated fault draws, ~88 faults each):
- Dispatch rule matters only when crews are scarce: priority vs nearest cuts SAIDI ~37% at 2 crews,
  ~35% at 3, ~29% at 4, ~11% at 6, and ~0% at 10+. Critical loads recover far faster at <= 6 crews.
- Pre-positioning idle crews matters only when crews are plentiful: SAIDI -3.5% at 10 crews,
  -4.5% at 14; about 0-1% at <= 6 crews (crews are never idle).
- The day-ahead weather forecast adds nothing over staging by customers + critical loads
  (forecast ~= flat ~= oracle). Weather is nearly uniform across the ~45 km area at 12 km grid
  spacing, so risk differences come from vulnerability and consequence, not weather.
- Forecast under-predicts gusts (peak 52 vs 62 km/h; bias about -6 km/h; rain correlation 0.42).
- SAIFI is identical for every dispatch rule: dispatch shortens outages, it cannot prevent them.

Caveats: faults, customers per zone, vulnerability, repair times, crews are assumptions;
weather "actual" is reanalysis; one storm window only.
