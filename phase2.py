"""Phase 2: does pre-positioning crews help, and how much of that comes from the forecast?

All setups use the same priority dispatch rule and the same simulated storms:
  depot    crews start at depots and stay where their last repair ended
  flat     idle crews reposition by customers + critical loads only (no weather)
  forecast idle crews reposition using the real day-ahead weather forecast
  oracle   idle crews reposition using the actual weather (upper bound)
"""
import numpy as np

import real_sim as rs
import sim

CREWS = (2, 3, 4, 6, 10, 14)


def main():
    c = rs.setup()
    setups = rs.build_planners(c)
    hz_actual = rs.hazard(c.wx["gust"][c.nearest], c.wx["rain"][c.nearest])[:, :rs.WINDOW_HOURS]
    hz_fcst = rs.hazard(c.wx["gust_f1"][c.nearest], c.wx["rain_f1"][c.nearest])[:, :rs.WINDOW_HOURS]
    risk = lambda hz: c.scale * c.expo[:, None] * hz
    exp_f, exp_a = risk(hz_fcst).sum(), risk(hz_actual).sum()
    print(f"{len(c.realizations)} storms, {c.total_cust:,} customers. "
          f"Forecast expects {exp_f:.0f} faults vs {exp_a:.0f} from actual weather.\n")

    print(f"{'crews':>5s} {'setup':9s} {'SAIDI':>7s} {'vs depot':>9s} {'critical h':>11s} "
          f"{'travel h':>9s} {'beats depot':>12s}")
    for crews in CREWS:
        sim.N_CREWS = crews
        res = {}
        for name, planner in setups.items():
            res[name] = [sim.simulate(f, c.depots, sim.pick_priority, planner=planner)
                         for f in c.realizations]
        base = np.array([m["customer_minutes"] for m in res["depot"]])
        for name in setups:
            cm = np.array([m["customer_minutes"] for m in res[name]])
            print(f"{crews:5d} {name:9s} {cm.mean() / c.total_cust:7.1f} "
                  f"{(1 - cm.mean() / base.mean()) * 100:8.1f}% "
                  f"{np.nanmean([m['critical_restore_h'] for m in res[name]]):11.2f} "
                  f"{np.mean([m['mean_travel_h'] for m in res[name]]):9.2f} "
                  f"{(np.mean(cm < base) * 100 if name != 'depot' else float('nan')):11.0f}%")
        print()


if __name__ == "__main__":
    main()
