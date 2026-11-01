#!/usr/bin/env python3
"""Download the XNLI files FedTune needs into data/xnli/ (validation.parquet, test.parquet).

The loader (fedtune/data/loader.py) reads the Hugging Face `facebook/xnli` config `all_languages`
(one row = the same example in 15 languages). The files are fetched from a PINNED commit, so every
teammate gets byte-identical data. Needs internet; no extra packages.

    python tools/get_xnli.py            # from the repo root; skips files that are already valid
    python tools/get_xnli.py --check    # only verify, never download

Prints each file's SHA-256: all six teammates must see the same two values.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

COMMIT = "079130b655a6e1d3be8d969c01a6fef3c6a0c36c"
BASE = f"https://huggingface.co/datasets/facebook/xnli/resolve/{COMMIT}/all_languages"
FILES = {  # local name -> (remote name, expected rows)
    "validation.parquet": ("validation-00000-of-00001.parquet", 2490),
    "test.parquet": ("test-00000-of-00001.parquet", 5010),
}
KNOWN_SHA256 = {  # published by Hugging Face for the pinned file
    "test.parquet": "599e3a0191403f19cbe802afdf69841152000b41eaed725e4f463d432c0ffb49",
}
OUT = Path(__file__).resolve().parents[1] / "data" / "xnli"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def valid(p: Path, rows: int) -> bool:
    try:
        import pyarrow.parquet as pq
        return p.exists() and pq.ParquetFile(p).metadata.num_rows == rows
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for local, (remote, rows) in FILES.items():
        p = OUT / local
        if not valid(p, rows):
            if a.check:
                print(f"MISSING/INVALID: {p}")
                return 1
            print(f"downloading {remote} ...", flush=True)
            tmp = p.with_suffix(".part")
            urllib.request.urlretrieve(f"{BASE}/{remote}", tmp)
            tmp.replace(p)
            if not valid(p, rows):
                print(f"ERROR: {p} does not hold {rows} rows")
                return 1
        digest = sha256(p)
        want = KNOWN_SHA256.get(local)
        if want and digest != want:
            print(f"ERROR: {local} sha256 {digest} != expected {want}")
            return 1
        print(f"OK  {local}  rows={rows}  sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
