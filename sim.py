"""Phase 1 storm outage / crew dispatch simulator (synthetic data).

All parameters below are ASSUMPTIONS for a made-up service area, not real
DISCOM figures. They exist to test dispatch logic, not to predict real faults.
"""
from dataclasses import dataclass

import numpy as np

AREA_KM = 60.0
N_ZONES = 30
N_CRITICAL = 4          # zones holding hospitals / water pumping etc.
N_CREWS = 14                  # calibrated by sweep_load.py: ~6 h mean restoration
N_DEPOTS = 3
STORM_HOURS = 12.0
FRONT_SPEED_KMH = 8.0         # storm front moves west -> east
TRAVEL_SPEED_KMH = 25.0       # slowed by rain / debris
REPAIR_MEDIAN_H = 2.5
REPAIR_SIGMA = 0.4
CRITICAL_BONUS = 5000.0       # customer-equivalents added to a critical fault's score
FAULT_RATE_SCALE = 3.0        # tunes faults per storm


@dataclass
class Zone:
    x: float
    y: float
    customers: int
    critical: bool
    vulnerability: float      # tree cover / asset age proxy


@dataclass
class Fault:
    zone: int
    t: float                  # report time (h)
    x: float
    y: float
    customers: int            # customers affected by this fault
    repair_h: float
    critical: bool


def make_network(rng):
    xy = rng.uniform(0, AREA_KM, size=(N_ZONES, 2))
    customers = rng.integers(800, 15000, size=N_ZONES)
    critical = np.zeros(N_ZONES, dtype=bool)
    critical[rng.choice(N_ZONES, N_CRITICAL, replace=False)] = True
    vuln = rng.lognormal(mean=0.0, sigma=0.5, size=N_ZONES)
    zones = [Zone(xy[i, 0], xy[i, 1], int(customers[i]), bool(critical[i]), float(vuln[i]))
             for i in range(N_ZONES)]
    depots = rng.uniform(0, AREA_KM, size=(N_DEPOTS, 2))
    return zones, depots


def make_storm(rng, zones):
    """Sample faults for one storm. Returns faults sorted by report time."""
    track_y = rng.uniform(10, AREA_KM - 10)
    severity = rng.uniform(0.6, 1.6)
    faults = []
    for i, z in enumerate(zones):
        intensity = severity * np.exp(-((z.y - track_y) ** 2) / (2 * 18.0 ** 2))
        n = rng.poisson(FAULT_RATE_SCALE * z.vulnerability * intensity)
        arrival = z.x / FRONT_SPEED_KMH
        for _ in range(n):
            t = max(0.0, rng.normal(arrival, 1.5))
            frac = rng.uniform(0.1, 0.6)
            repair = REPAIR_MEDIAN_H * np.exp(rng.normal(0.0, REPAIR_SIGMA))
            faults.append(Fault(i, t, z.x, z.y, max(1, int(z.customers * frac)),
                                float(repair), z.critical))
    faults.sort(key=lambda f: f.t)
    return faults


def travel_h(ax, ay, bx, by):
    return float(np.hypot(ax - bx, ay - by)) / TRAVEL_SPEED_KMH


# --- dispatch rules: given the free crews' positions, pick (fault index, crew index) ---

def _nearest_crew(f, crews):
    return min(range(len(crews)), key=lambda i: travel_h(crews[i][0], crews[i][1], f.x, f.y))


def pick_fifo(pending, crews):
    k = min(range(len(pending)), key=lambda j: pending[j].t)
    return k, _nearest_crew(pending[k], crews)


def pick_nearest(pending, crews):
    return min(((k, i) for k in range(len(pending)) for i in range(len(crews))),
               key=lambda ki: travel_h(crews[ki[1]][0], crews[ki[1]][1],
                                       pending[ki[0]].x, pending[ki[0]].y))


def pick_priority(pending, crews):
    best = None
    for k, f in enumerate(pending):
        i = _nearest_crew(f, crews)
        weight = f.customers + (CRITICAL_BONUS if f.critical else 0.0)
        score = weight / (travel_h(crews[i][0], crews[i][1], f.x, f.y) + f.repair_h)
        if best is None or score > best[0]:
            best = (score, k, i)
    return best[1], best[2]


STRATEGIES = {"fifo": pick_fifo, "nearest": pick_nearest, "priority": pick_priority}


def simulate(faults, depots, picker, crew_starts=None, detail=False, planner=None, tracks=None):
    """Dispatch crews greedily. Each crew stays where its last repair ended.

    crew_starts: optional (N_CREWS, 2) array; default is crews spread over depots.
    detail: also return {id(fault): restoration time (h)}.
    planner: optional callable (t, crew, positions) -> (x, y) or None. While no fault
        is waiting, idle crews drive toward the point it returns (pre-positioning).
    tracks: optional list that is filled with one [(t, x, y), ...] waypoint list per crew.
    """
    if crew_starts is None:
        crew_starts = np.array([depots[i % len(depots)] for i in range(N_CREWS)])
    pos = [list(p) for p in crew_starts]
    free_at = [0.0] * N_CREWS
    if tracks is not None:
        tracks.extend([[(0.0, p[0], p[1])] for p in pos])

    queue = list(faults)          # sorted by report time
    pending = []
    restore = {}                  # id(fault) -> restoration time
    next_idx = 0
    travel_total = 0.0

    while next_idx < len(queue) or pending:
        if planner is not None and not pending and next_idx < len(queue):
            t_next = queue[next_idx].t
            for k in sorted(range(N_CREWS), key=lambda j: free_at[j]):
                if free_at[k] >= t_next:
                    continue
                tgt = planner(free_at[k], k, pos)
                if tgt is not None:
                    d = float(np.hypot(tgt[0] - pos[k][0], tgt[1] - pos[k][1]))
                    if d > 0:
                        frac = min(1.0, (t_next - free_at[k]) * TRAVEL_SPEED_KMH / d)
                        pos[k] = [pos[k][0] + frac * (tgt[0] - pos[k][0]),
                                  pos[k][1] + frac * (tgt[1] - pos[k][1])]
                if tracks is not None:
                    tracks[k].append((t_next, pos[k][0], pos[k][1]))
                free_at[k] = t_next
        c = min(range(N_CREWS), key=lambda k: free_at[k])
        t = free_at[c]
        if not pending and next_idx < len(queue):
            t = max(t, queue[next_idx].t)
        while next_idx < len(queue) and queue[next_idx].t <= t:
            pending.append(queue[next_idx])
            next_idx += 1
        if not pending:
            continue
        free = [j for j in range(N_CREWS) if free_at[j] <= t]
        k, i = picker(pending, [pos[j] for j in free])
        c = free[i]
        f = pending.pop(k)
        tr = travel_h(pos[c][0], pos[c][1], f.x, f.y)
        travel_total += tr
        done = t + tr + f.repair_h
        if tracks is not None:
            tracks[c] += [(t, pos[c][0], pos[c][1]), (t + tr, f.x, f.y), (done, f.x, f.y)]
        restore[id(f)] = done
        pos[c] = [f.x, f.y]
        free_at[c] = done

    m = metrics(faults, restore)
    m["mean_travel_h"] = travel_total / len(faults)
    return (m, restore) if detail else m


def metrics(faults, restore):
    if not faults:
        return None
    dur = np.array([restore[id(f)] - f.t for f in faults])
    cust = np.array([f.customers for f in faults])
    crit = np.array([f.critical for f in faults])
    return {
        "faults": len(faults),
        "customer_minutes": float((cust * dur).sum() * 60.0),   # SAIDI numerator
        "customers_interrupted": int(cust.sum()),               # SAIFI numerator
        "mean_restore_h": float(dur.mean()),
        "critical_restore_h": float(dur[crit].mean()) if crit.any() else float("nan"),
    }
