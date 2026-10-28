#!/usr/bin/env python3
"""Print the code fingerprint of a FedTune checkout.

The fingerprint is a short SHA-256 over every ``fedtune/**/*.py`` and ``configs/*.yaml`` file
(relative path + bytes). It is the same value the Kaggle runner notebook prints, so machines that run
parts of one experiment can prove they run identical code. Files outside those two places (tools,
tests, notebooks, results, docs) do not change it.

    python tools/fingerprint.py              # run from the repository root
    python tools/fingerprint.py path/to/checkout
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def fingerprint(root: Path) -> str:
    files = sorted([*(root / "fedtune").rglob("*.py"), *(root / "configs").glob("*.yaml")])
    if not files:
        raise SystemExit(f"{root}: no fedtune/**/*.py or configs/*.yaml found; is this a FedTune checkout?")
    digest = hashlib.sha256()
    for f in files:
        digest.update(f.relative_to(root).as_posix().encode())   # posix separators: same value on Windows
        digest.update(f.read_bytes())
    return digest.hexdigest()[:16]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    print(fingerprint(Path(args[0] if args else ".").resolve()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
