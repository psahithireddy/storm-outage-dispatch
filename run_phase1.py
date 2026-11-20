"""Phase 1: compare dispatch rules over many random synthetic storms."""
import numpy as np

import sim

N_STORMS = 300
NAMES = list(sim.STRATEGIES)


def main():
    rng = np.random.default_rng(42)
    zones, depots = sim.make_network(rng)

    rows = {n: [] for n in NAMES}
    n_faults = []
    for _ in range(N_STORMS):
        faults = sim.make_storm(rng, zones)
        if len(faults) < 5:
            continue
        n_faults.append(len(faults))
        for name in NAMES:
            rows[name].append(sim.simulate(faults, depots, sim.STRATEGIES[name]))

    print(f"{len(n_faults)} storms, mean faults/storm {np.mean(n_faults):.1f}, "
          f"{sim.N_CREWS} crews\n")
    cols = ["customer_minutes", "mean_restore_h", "critical_restore_h"]
    print(f"{'strategy':10s} {'cust-min (M)':>13s} {'mean restore h':>15s} {'critical restore h':>19s}")
    for name in NAMES:
        cm = np.mean([r["customer_minutes"] for r in rows[name]]) / 1e6
        mr = np.mean([r["mean_restore_h"] for r in rows[name]])
        cr = np.nanmean([r["critical_restore_h"] for r in rows[name]])
        print(f"{name:10s} {cm:13.2f} {mr:15.2f} {cr:19.2f}")

    base = np.array([r["customer_minutes"] for r in rows["fifo"]])
    print("\nCustomer-minutes vs fifo baseline (paired, per storm):")
    for name in NAMES[1:]:
        cur = np.array([r["customer_minutes"] for r in rows[name]])
        red = 1 - cur / base
        print(f"{name:10s} mean reduction {red.mean()*100:5.1f}%   "
              f"better in {np.mean(cur < base)*100:4.0f}% of storms")


if __name__ == "__main__":
    main()
