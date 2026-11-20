"""Phase 2: pre-positioning planner for idle crews.

An idle crew drives to the zone centroid that most reduces the expected,
consequence-weighted distance to upcoming faults, given where the other crews are.
This is a greedy one-crew-at-a-time version of the p-median facility-location problem.
"""
import numpy as np

import sim


class Planner:
    def __init__(self, zones_xy, risk, consequence, lookahead_h=6, min_gain=0.01):
        """risk: (zones, hours) expected faults per hour as the planner believes them.
        consequence: (zones,) customer-equivalents lost per fault in each zone.
        """
        self.xy = zones_xy
        self.w = risk * consequence[:, None]
        self.L = lookahead_h
        self.min_gain = min_gain
        d = zones_xy[:, None, :] - zones_xy[None, :, :]
        self.D = np.hypot(d[..., 0], d[..., 1])          # zone-to-zone km

    def __call__(self, t, k, pos):
        h0 = int(t)
        w = self.w[:, h0:h0 + self.L].sum(axis=1)
        if w.sum() <= 0:
            return None
        others = np.array([p for j, p in enumerate(pos) if j != k])
        d_other = np.hypot(self.xy[:, None, 0] - others[None, :, 0],
                           self.xy[:, None, 1] - others[None, :, 1]).min(axis=1)
        d_here = np.hypot(self.xy[:, 0] - pos[k][0], self.xy[:, 1] - pos[k][1])
        current = (w * np.minimum(d_other, d_here)).sum()
        cost = (w[:, None] * np.minimum(d_other[:, None], self.D)).sum(axis=0)
        best = int(np.argmin(cost))
        if cost[best] < current * (1 - self.min_gain):
            return self.xy[best]
        return None
