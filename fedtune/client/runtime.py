"""Client side (simulated): LocalTrainer, SummaryBuilder, ClientRuntime (FR 9, 11-12, Alg. 3).

A client owns its shard and local validation set. It sees the coordinator only
through ``GlobalBroadcast`` in and ``ClientUpdate`` out; raw text never leaves.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from ..data.loader import TextDataset
from ..infra.model_host import ModelHost
from ..protocol import ClientUpdate, GlobalBroadcast


class SummaryBuilder:
    """Bounded client summary h_k = (h_k^c, h_k^y): (Q + C) x 4 bytes.

    h^c[q] = fraction of examples whose frozen-base embedding is nearest codebook centre q
    h^y[c] = fraction of examples with label c; both rounded to 3 decimals.
    Embeddings use the frozen base, so the summary never changes and is cached.
    """

    def __init__(self, projection: torch.Tensor):
        self.projection = projection

    def build(self, host: ModelHost, shard: TextDataset, codebook: torch.Tensor, num_classes: int):
        emb = host.embed(shard.texts, self.projection)
        cell = torch.cdist(emb, codebook).argmin(1).numpy()
        hc = np.bincount(cell, minlength=len(codebook)) / max(len(shard), 1)
        hy = np.bincount(shard.labels, minlength=num_classes) / max(len(shard), 1)
        return tuple(round(float(x), 3) for x in hc), tuple(round(float(x), 3) for x in hy)


class ClientRuntime:
    def __init__(self, k: int, shard: TextDataset, val: TextDataset, host: ModelHost,
                 summary_builder: SummaryBuilder, num_classes: int):
        self.k, self.shard, self.val, self.host = k, shard, val, host
        self.summary_builder, self.num_classes = summary_builder, num_classes
        self._summary_cache = None
        self.codebook: torch.Tensor | None = None
        self.fault_rounds: set[int] = set()     # test hook: return a NaN update in these rounds

    # Algorithm 3 ---------------------------------------------------------------
    def train(self, b: GlobalBroadcast) -> ClientUpdate:
        t0 = time.perf_counter()
        hp = b.hparams
        g = np.random.default_rng(b.client_seed)         # reproducible shuffling per (k, r)
        torch.manual_seed(b.client_seed)
        if b.codebook is not None:                       # codebook arrives once, in round 1
            self.codebook = b.codebook
        self.host.load_adapter(b.adapter, b.head)
        params = self.host.trainable_parameters()
        opt = torch.optim.AdamW(params, lr=hp["lr"], weight_decay=hp["weight_decay"])   # fresh state each round
        self.host.model.train()

        bs, steps, losses = int(hp["batch_size"]), 0, []
        max_steps = hp.get("max_steps") or float("inf")
        labels = torch.as_tensor(self.shard.labels)
        for _ in range(int(hp["local_epochs"])):
            order = g.permutation(len(self.shard))
            for i in range(0, len(order), bs):
                idx = order[i:i + bs]
                logits = self.host.logits([self.shard.texts[j] for j in idx])
                loss = F.cross_entropy(logits, labels[idx].to(logits.device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if hp["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(params, hp["grad_clip"])
                opt.step()
                losses.append(float(loss.detach()))
                steps += 1
                if steps >= max_steps:
                    break
            if steps >= max_steps:
                break

        adapter, head = self.host.export_adapter()        # copied to CPU; host reused by next client
        if b.round in self.fault_rounds:
            first = next(iter(adapter))
            adapter[first]["A"] = torch.full_like(adapter[first]["A"], float("nan"))

        hc = hy = None
        if b.needs_summary:
            if self._summary_cache is None:
                if self.codebook is None:
                    raise RuntimeError("diversity strategy needs a codebook before the first summary")
                self._summary_cache = self.summary_builder.build(self.host, self.shard, self.codebook, self.num_classes)
            hc, hy = self._summary_cache
        return ClientUpdate(client_id=self.k, round=b.round, adapter=adapter, head=head,
                            n_examples=len(self.shard), local_loss=float(np.mean(losses)) if losses else 0.0,
                            train_time_s=time.perf_counter() - t0, summary_c=hc, summary_y=hy)

    # local validation (FR 7, 24): run on the client, only the metric is reported
    def validate(self, adapter, head) -> dict:
        from ..evaluation import classification_metrics
        if len(self.val) == 0:
            return {"accuracy": float("nan"), "macro_f1": float("nan")}
        self.host.load_adapter(adapter, head)
        pred = self.host.predict(self.val.texts)
        return classification_metrics(self.val.labels, pred, self.num_classes)
