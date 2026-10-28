"""Dataset loading (FR 4) from the local ``data/`` folder — no network needed.

A ``TextDataset`` holds texts (single strings or (premise, hypothesis) pairs),
integer labels and the partitioning attribute (topic / genre / language).

Bundled datasets (see ``data/README.md`` for sources):
  * ``ag_news`` - data/ag_news/{train,test}.csv    single text -> 4 topics      (attribute = topic)
  * ``mnli``    - data/mnli/*.parquet              sentence pair -> 3 NLI labels (attribute = genre)
  * ``xnli``    - data/xnli/*.parquet              sentence pair -> 3 NLI labels (attribute = language)
  * any ``*.csv`` / ``*.jsonl`` path with the configured text/label/attribute fields
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
NLI_LABELS = ["entailment", "neutral", "contradiction"]


@dataclass
class TextDataset:
    texts: list            # list[str] or list[tuple[str, str]]
    labels: np.ndarray     # int64, shape (N,)
    attributes: np.ndarray # str,   shape (N,)
    label_names: list[str]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def subset(self, idx) -> "TextDataset":
        idx = np.asarray(idx, dtype=np.int64)
        return TextDataset([self.texts[i] for i in idx], self.labels[idx], self.attributes[idx], self.label_names)


def _missing(path: Path) -> FileNotFoundError:
    return FileNotFoundError(f"{path} not found - the dataset files belong in the project's data/ folder (see data/README.md)")


# --------------------------------------------------------------------------- AG News (CSV: class 1-4, title, description)
def _ag_news(root: Path) -> TextDataset:
    names = (root / "classes.txt").read_text().split("\n")
    names = [n.strip() for n in names if n.strip()]
    texts, labels = [], []
    for split in ("train.csv", "test.csv"):
        p = root / split
        if not p.exists():
            raise _missing(p)
        with open(p, newline="", encoding="utf-8") as f:
            for cls, title, desc in csv.reader(f):
                texts.append(f"{title} {desc}".replace("\\", " "))
                labels.append(int(cls) - 1)
    labels = np.array(labels, dtype=np.int64)
    return TextDataset(texts, labels, np.array([names[y] for y in labels]), names)


# --------------------------------------------------------------------------- MNLI (Parquet: premise, hypothesis, genre, label)
def _mnli(root: Path) -> TextDataset:
    import pyarrow.parquet as pq
    texts, labels, attrs = [], [], []
    for split in ("train", "validation_matched", "validation_mismatched"):
        p = root / f"{split}.parquet"
        if not p.exists():
            raise _missing(p)
        t = pq.read_table(p, columns=["premise", "hypothesis", "genre", "label"]).to_pydict()
        for pr, hy, g, y in zip(t["premise"], t["hypothesis"], t["genre"], t["label"]):
            if int(y) >= 0:
                texts.append((pr, hy)); labels.append(int(y)); attrs.append(g)
    return TextDataset(texts, np.array(labels, dtype=np.int64), np.array(attrs), NLI_LABELS)


# --------------------------------------------------------------------------- XNLI (Parquet, all_languages: one row = 15 translations)
def _xnli(root: Path, languages: list[str] | None) -> TextDataset:
    import pyarrow.parquet as pq
    texts, labels, attrs = [], [], []
    for split in ("validation", "test"):
        p = root / f"{split}.parquet"
        if not p.exists():
            raise _missing(p)
        for row in pq.read_table(p).to_pylist():
            hyp = dict(zip(row["hypothesis"]["language"], row["hypothesis"]["translation"]))
            for lg, prem in row["premise"].items():
                if prem is None or lg not in hyp or (languages and lg not in languages):
                    continue
                texts.append((prem, hyp[lg])); labels.append(int(row["label"])); attrs.append(lg)
    return TextDataset(texts, np.array(labels, dtype=np.int64), np.array(attrs), NLI_LABELS)


# --------------------------------------------------------------------------- local CSV / JSONL
def _from_file(path: Path, text_fields: list[str], label_field: str, attr_field: str | None) -> TextDataset:
    if path.suffix == ".csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    else:
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    raw_labels = [str(r[label_field]) for r in rows]
    names = sorted(set(raw_labels))
    idx = {n: i for i, n in enumerate(names)}
    texts = [r[text_fields[0]] if len(text_fields) == 1 else (r[text_fields[0]], r[text_fields[1]]) for r in rows]
    attrs = [str(r[attr_field]) if attr_field else lab for r, lab in zip(rows, raw_labels)]
    return TextDataset(texts, np.array([idx[l] for l in raw_labels], dtype=np.int64), np.array(attrs), names)


def load_dataset_from_config(dcfg: dict, seed: int) -> TextDataset:
    name = dcfg["dataset"]
    if name == "ag_news":
        data = _ag_news(DATA_DIR / "ag_news")
    elif name == "mnli":
        data = _mnli(DATA_DIR / "mnli")
    elif name == "xnli":
        data = _xnli(DATA_DIR / "xnli", dcfg.get("languages"))
    else:
        p = Path(name)
        if p.suffix not in (".csv", ".jsonl"):
            raise ValueError(f"unknown dataset {name!r}: use ag_news, mnli, xnli or a .csv/.jsonl path")
        if not p.exists():
            raise _missing(p)
        data = _from_file(p, dcfg["text_fields"], dcfg["label_field"], dcfg["attribute_field"])
    if len(data) > dcfg["max_examples"]:          # seeded subsample, same for every strategy
        data = data.subset(np.sort(np.random.default_rng(seed).choice(len(data), dcfg["max_examples"], replace=False)))
    return data
