"""ParticipationScheduler (R2, FR 17, 19, 23, Alg. 2).

The R x K participation matrix is drawn once from its own random stream, before
round 1, and shared by every strategy, so identical participation holds by
construction. One uniform draw per (r, k) in fixed order makes the matrices for
p1 < p2 with the same seed *nested*: every client present at p2 is present at p1.
"""
from __future__ import annotations

import numpy as np


def build_schedule(K: int, R: int, p: float, g_part: np.random.Generator) -> np.ndarray:
    u = g_part.random((R, K))          # row-major: r = 1..R, k = 1..K
    return u >= p                      # P[r, k] = 1  <=>  client k participates in round r+1


class ParticipationScheduler:
    def __init__(self, P: np.ndarray):
        self.P = P.astype(bool)

    def participants(self, r: int) -> list[int]:
        """S_r for 1-based round r, in increasing client id."""
        return np.flatnonzero(self.P[r - 1]).tolist()

    def dropped(self, r: int) -> list[int]:
        return np.flatnonzero(~self.P[r - 1]).tolist()
