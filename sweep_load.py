"""Sweep crew count to find a realistic load, and check whether the rule ranking holds."""
import numpy as np

import sim

N_STORMS = 150


def run(n_crews):
    sim.N_CREWS = n_crews
    rng = np.random.default_rng(42)
    zones, depots = sim.make_network(rng)
    out = {n: [] for n in sim.STRATEGIES}
    for _ in range(N_STORMS):
        faults = sim.make_storm(rng, zones)
        if len(faults) < 5:
            continue
        for name, picker in sim.STRATEGIES.items():
            out[name].append(sim.simulate(faults, depots, picker))
    return out


def main():
    print(f"{'crews':>5s} {'rule':9s} {'cust-min (M)':>12s} {'mean h':>7s} {'crit h':>7s} {'vs nearest':>11s}")
    for n_crews in (6, 10, 14, 18, 24):
        out = run(n_crews)
        near = np.array([r["customer_minutes"] for r in out["nearest"]])
        for name, rows in out.items():
            cm = np.array([r["customer_minutes"] for r in rows])
            print(f"{n_crews:5d} {name:9s} {cm.mean()/1e6:12.2f} "
                  f"{np.mean([r['mean_restore_h'] for r in rows]):7.2f} "
                  f"{np.nanmean([r['critical_restore_h'] for r in rows]):7.2f} "
                  f"{(1 - cm.mean()/near.mean())*100:10.1f}%")
        print()


if __name__ == "__main__":
    main()
