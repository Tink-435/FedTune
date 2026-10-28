"""The client/coordinator boundary (R1, NFR 6, Table 3).

Only two immutable message types cross it. Raw text, token ids and per-example
labels cannot be expressed: every payload field is a tensor, a number or a
short tuple of numbers, and ``__post_init__`` rejects anything else. Bytes are
measured on the serialised form (safetensors for tensors, JSON otherwise).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import torch
from safetensors.torch import save as st_save

# module name -> {"A": tensor (r x d_in), "B": tensor (d_out x r)}
Adapter = Mapping[str, Mapping[str, torch.Tensor]]
Head = Mapping[str, torch.Tensor]


class BoundaryViolation(TypeError):
    pass


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _check_numeric_tuple(name: str, v) -> None:
    if v is None:
        return
    if isinstance(v, torch.Tensor):
        if not (v.is_floating_point() and v.dim() == 1):
            raise BoundaryViolation(f"{name}: summary tensors must be 1-D float")
        return
    if not isinstance(v, tuple) or not all(_is_number(x) for x in v):
        raise BoundaryViolation(f"{name}: must be a tuple of numbers or a 1-D float tensor")


def _check_adapter(name: str, adapter) -> None:
    for mod, ab in adapter.items():
        if not isinstance(mod, str) or set(ab) != {"A", "B"}:
            raise BoundaryViolation(f"{name}: module {mod!r} must hold exactly A and B")
        for t in ab.values():
            if not isinstance(t, torch.Tensor) or not t.is_floating_point():
                raise BoundaryViolation(f"{name}: adapter factors must be float tensors")


def _check_head(name: str, head) -> None:
    for k, t in head.items():
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            raise BoundaryViolation(f"{name}: head entry {k!r} must be a float tensor")


def _freeze(adapter, head):
    a = MappingProxyType({m: MappingProxyType(dict(ab)) for m, ab in adapter.items()})
    return a, MappingProxyType(dict(head))


def _thaw(v):
    if isinstance(v, MappingProxyType):
        return {k: _thaw(x) for k, x in v.items()}
    return v


class _Picklable:
    """Messages are immutable (mappingproxy); pickle them (checkpoints) as plain dicts and re-validate."""

    def __reduce__(self):
        from dataclasses import fields
        return (_rebuild, (type(self), {f.name: _thaw(getattr(self, f.name)) for f in fields(self)}))


def _rebuild(cls, kw):
    return cls(**kw)


@dataclass(frozen=True)
class GlobalBroadcast(_Picklable):
    round: int
    adapter: Adapter
    head: Head
    hparams: Mapping[str, float]             # lr, batch_size, local_epochs, ... (numbers only)
    client_seed: int
    codebook: torch.Tensor | None = None     # Q x D centres; sent once in round 1 (diversity strategies)
    needs_summary: bool = False

    def __post_init__(self):
        _check_adapter("GlobalBroadcast.adapter", self.adapter)
        _check_head("GlobalBroadcast.head", self.head)
        if not all(isinstance(k, str) and (_is_number(v) or v is None) for k, v in self.hparams.items()):
            raise BoundaryViolation("GlobalBroadcast.hparams must map names to numbers")
        if self.codebook is not None and not isinstance(self.codebook, torch.Tensor):
            raise BoundaryViolation("GlobalBroadcast.codebook must be a tensor")
        a, h = _freeze(self.adapter, self.head)
        object.__setattr__(self, "adapter", a)
        object.__setattr__(self, "head", h)
        object.__setattr__(self, "hparams", MappingProxyType(dict(self.hparams)))


@dataclass(frozen=True)
class ClientUpdate(_Picklable):
    client_id: int
    round: int
    adapter: Adapter
    head: Head
    n_examples: int
    local_loss: float
    train_time_s: float
    summary_c: tuple | torch.Tensor | None = None   # codebook-cell histogram h_k^c (Q)
    summary_y: tuple | torch.Tensor | None = None   # label histogram h_k^y (C)

    def __post_init__(self):
        for nm in ("client_id", "round", "n_examples"):
            if not isinstance(getattr(self, nm), int) or isinstance(getattr(self, nm), bool):
                raise BoundaryViolation(f"ClientUpdate.{nm} must be int")
        for nm in ("local_loss", "train_time_s"):
            if not _is_number(getattr(self, nm)):
                raise BoundaryViolation(f"ClientUpdate.{nm} must be a number")
        _check_adapter("ClientUpdate.adapter", self.adapter)
        _check_head("ClientUpdate.head", self.head)
        _check_numeric_tuple("ClientUpdate.summary_c", self.summary_c)
        _check_numeric_tuple("ClientUpdate.summary_y", self.summary_y)
        a, h = _freeze(self.adapter, self.head)
        object.__setattr__(self, "adapter", a)
        object.__setattr__(self, "head", h)

    def is_finite(self) -> bool:
        ts = [t for ab in self.adapter.values() for t in ab.values()] + list(self.head.values())
        return all(bool(torch.isfinite(t).all()) for t in ts) and torch.isfinite(torch.tensor(self.local_loss)).item()

    @property
    def has_summary(self) -> bool:
        return self.summary_c is not None and self.summary_y is not None


# ---------------------------------------------------------------------------
# Byte accounting on the serialised form (FR 24, NFR 3)
# ---------------------------------------------------------------------------
def _tensor_dict(adapter: Adapter, head: Head) -> dict[str, torch.Tensor]:
    d = {f"{m}.{k}": t.detach().contiguous().cpu() for m, ab in adapter.items() for k, t in ab.items()}
    d.update({f"head.{k}": t.detach().contiguous().cpu() for k, t in head.items()})
    return d


def adapter_bytes(adapter: Adapter, head: Head) -> int:
    return len(st_save(_tensor_dict(adapter, head)))


def summary_bytes(u: ClientUpdate) -> int:
    if not u.has_summary:
        return 0
    return 4 * (len(u.summary_c) + len(u.summary_y))   # (Q + C) x 4 bytes, float32


def update_bytes(u: ClientUpdate) -> int:
    meta = json.dumps({"client_id": u.client_id, "round": u.round, "n": u.n_examples,
                       "loss": u.local_loss, "t": u.train_time_s}).encode()
    return adapter_bytes(u.adapter, u.head) + summary_bytes(u) + len(meta)


def broadcast_bytes(b: GlobalBroadcast) -> int:
    meta = json.dumps({"round": b.round, "seed": b.client_seed, **dict(b.hparams)}).encode()
    cb = len(st_save({"codebook": b.codebook.contiguous()})) if b.codebook is not None else 0
    return adapter_bytes(b.adapter, b.head) + len(meta) + cb
