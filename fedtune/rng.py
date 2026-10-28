"""RngService: independent named random streams from one master seed (FR 3).

Each purpose (``partition``, ``participation``, ``init``, ``client/k/r``,
``eval`` ...) gets its own ``numpy.random.Generator`` derived from
``SeedSequence(master, spawn_key=hash(name))``. Adding a new consumer never
shifts another stream, so partitions and participation stay bitwise identical.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _key(name: str) -> tuple[int, ...]:
    d = hashlib.sha256(name.encode()).digest()
    return tuple(int.from_bytes(d[i:i + 4], "little") for i in range(0, 16, 4))


class RngService:
    def __init__(self, master_seed: int):
        self.master_seed = int(master_seed)
        self._live: dict[str, np.random.Generator] = {}

    def stream(self, name: str) -> np.random.Generator:
        """Stateful stream: repeated calls with the same name continue it."""
        if name not in self._live:
            self._live[name] = self.fresh(name)
        return self._live[name]

    def fresh(self, name: str) -> np.random.Generator:
        """A new generator for ``name``, always starting from the same state."""
        ss = np.random.SeedSequence(self.master_seed, spawn_key=_key(name))
        return np.random.Generator(np.random.PCG64(ss))

    def int_seed(self, name: str) -> int:
        return int(self.fresh(name).integers(0, 2**31 - 1))

    # checkpoint support -----------------------------------------------------
    def state(self) -> dict:
        return {k: g.bit_generator.state for k, g in self._live.items()}

    def restore(self, state: dict) -> None:
        for k, st in state.items():
            g = self.stream(k)
            g.bit_generator.state = st
