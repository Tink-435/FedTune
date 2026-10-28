"""ArtifactStore and CheckpointManager (R6, FR 27, 34, Section 2.1.7).

Run directory layout::

    runs/<run_id>/
      config.resolved.yaml    resolved configuration + config_hash
      partition_report.json   per-client sizes, label/domain histograms, H
      participation.npy       R x K boolean schedule (written before round 1)
      summary_log.jsonl       per-client summary fields revealed (FR 12)
      rounds.jsonl            one JSON record per round (append-only)
      checkpoints/ckpt_<r>.pt atomic; last two kept
      run_summary.json        status = completed | failed

Round records are appended and fsync'd; checkpoints and JSON files are written to a
temporary file and renamed, so a crash never leaves a half-written file and the
read-only dashboard can tail safely.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch


def _atomic_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class ArtifactStore:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "checkpoints").mkdir(exist_ok=True)
        self.rounds_path = self.dir / "rounds.jsonl"
        self.summary_log_path = self.dir / "summary_log.jsonl"

    # JSON / numpy -----------------------------------------------------------
    def write_json(self, name: str, obj) -> None:
        _atomic_bytes(self.dir / name, json.dumps(obj, indent=2, default=_jsonable).encode())

    def read_json(self, name: str):
        p = self.dir / name
        return json.loads(p.read_text()) if p.exists() else None

    def write_participation(self, P: np.ndarray) -> None:
        tmp = self.dir / "participation.tmp.npy"
        np.save(tmp, P)
        os.replace(tmp, self.dir / "participation.npy")

    def read_participation(self) -> np.ndarray:
        return np.load(self.dir / "participation.npy")

    # append-only logs -----------------------------------------------------------
    def _append(self, path: Path, rec: dict) -> int:
        with open(path, "a") as f:
            f.write(json.dumps(rec, default=_jsonable) + "\n")
            f.flush()
            os.fsync(f.fileno())
            return f.tell()

    def append_round(self, rec: dict) -> int:
        return self._append(self.rounds_path, rec)

    def append_summary(self, rec: dict) -> int:
        return self._append(self.summary_log_path, rec)

    def rounds_offset(self) -> int:
        return self.rounds_path.stat().st_size if self.rounds_path.exists() else 0

    def truncate_rounds(self, offset: int) -> None:
        """On resume: drop any round written after the checkpoint, so no round appears twice."""
        if self.rounds_path.exists():
            with open(self.rounds_path, "r+b") as f:
                f.truncate(offset)

    def read_rounds(self) -> list[dict]:
        if not self.rounds_path.exists():
            return []
        out = []
        for line in self.rounds_path.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:   # a line still being written by a live run
                break
        return out


class CheckpointManager:
    def __init__(self, store: ArtifactStore, keep: int = 2):
        self.dir = store.dir / "checkpoints"
        self.keep = keep

    def save(self, rnd: int, state: dict) -> Path:
        import io
        buf = io.BytesIO()
        torch.save(state, buf)
        path = self.dir / f"ckpt_{rnd:04d}.pt"
        _atomic_bytes(path, buf.getvalue())
        for old in sorted(self.dir.glob("ckpt_*.pt"))[:-self.keep]:
            old.unlink()
        return path

    def latest(self) -> dict | None:
        cks = sorted(self.dir.glob("ckpt_*.pt"))
        return torch.load(cks[-1], weights_only=False) if cks else None


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
