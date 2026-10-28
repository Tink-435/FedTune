"""ModelHost: one frozen base model, swapped LoRA adapters (R5, NFR 2, FR 9-10).

The base weights are loaded once; clients are light records whose adapter state is
loaded only while they train. ``base: tiny-random`` builds a small randomly
initialised DistilBERT and a hashing tokenizer so the smoke demo and tests run
fully offline; any Hugging Face id up to ~1B parameters works otherwise.
"""
from __future__ import annotations

import zlib
from typing import Sequence

import numpy as np
import torch

LORA_TAGS = (".lora_A.", ".lora_B.")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():          # Apple-Silicon GPU backend (reference hardware)
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class HashingTokenizer:
    """Offline whitespace tokenizer for ``tiny-random``: stable crc32 word hashing."""
    PAD, CLS, SEP = 0, 1, 2

    def __init__(self, vocab_size: int = 4096):
        self.vocab_size = vocab_size

    def _ids(self, text: str) -> list[int]:
        return [3 + zlib.crc32(w.lower().encode()) % (self.vocab_size - 3) for w in text.split()]

    def __call__(self, texts, max_length: int):
        rows = []
        for t in texts:
            if isinstance(t, (tuple, list)):
                ids = [self.CLS] + self._ids(t[0]) + [self.SEP] + self._ids(t[1]) + [self.SEP]
            else:
                ids = [self.CLS] + self._ids(t) + [self.SEP]
            rows.append(ids[:max_length])
        L = max(map(len, rows))
        input_ids = torch.zeros(len(rows), L, dtype=torch.long)
        mask = torch.zeros(len(rows), L, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, :len(r)] = torch.tensor(r)
            mask[i, :len(r)] = 1
        return {"input_ids": input_ids, "attention_mask": mask}


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

    def __call__(self, texts, max_length: int):
        if texts and isinstance(texts[0], (tuple, list)):
            a, b = zip(*texts)
            enc = self.tok(list(a), list(b), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        else:
            enc = self.tok(list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}


class ModelHost:
    def __init__(self, mcfg: dict, num_labels: int, init_seed: int, max_length: int, device: torch.device | None = None):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForSequenceClassification, DistilBertConfig, DistilBertForSequenceClassification

        self.device = device or (pick_device() if mcfg.get("device", "auto") == "auto" else torch.device(mcfg["device"]))
        self.max_length = max_length
        torch.manual_seed(init_seed)
        if mcfg["base"] == "tiny-random":
            cfg = DistilBertConfig(vocab_size=4096, dim=64, hidden_dim=128, n_layers=2, n_heads=4,
                                   max_position_embeddings=256, num_labels=num_labels)
            base = DistilBertForSequenceClassification(cfg)
            self.tokenizer = HashingTokenizer(4096)
            default_targets = ["q_lin", "v_lin"]
        else:
            base = AutoModelForSequenceClassification.from_pretrained(mcfg["base"], num_labels=num_labels)
            if getattr(base.config, "pad_token_id", None) is None:
                base.config.pad_token_id = base.config.eos_token_id
            self.tokenizer = HFTokenizer(mcfg["base"])
            default_targets = ["q_lin", "v_lin"] if base.config.model_type == "distilbert" else None
        lc = mcfg["lora"]
        peft_cfg = LoraConfig(task_type=TaskType.SEQ_CLS, r=lc["rank"], lora_alpha=lc["alpha"],
                              lora_dropout=lc["dropout"], target_modules=lc["target_modules"] or default_targets)
        self.model = get_peft_model(base, peft_cfg).to(self.device)
        self.full_model_bytes = sum(p.numel() * p.element_size() for p in base.parameters())
        self._trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

    # ---------------------------------------------------------------- adapter state <-> model
    @staticmethod
    def _module_of(name: str) -> tuple[str, str] | None:
        for tag, which in ((".lora_A.", "A"), (".lora_B.", "B")):
            if tag in name:
                return name.split(tag)[0], which
        return None

    def export_adapter(self):
        """Copy trainable state to CPU as ({module: {A, B}}, head)."""
        adapter, head = {}, {}
        for n, p in self._trainable.items():
            mw = self._module_of(n)
            t = p.detach().to("cpu", copy=True).float()
            if mw:
                adapter.setdefault(mw[0], {})[mw[1]] = t
            else:
                head[n] = t
        return adapter, head

    @torch.no_grad()
    def load_adapter(self, adapter, head) -> None:
        for n, p in self._trainable.items():
            mw = self._module_of(n)
            src = adapter[mw[0]][mw[1]] if mw else head[n]
            p.copy_(src.to(p.device, p.dtype))

    def trainable_parameters(self):
        return list(self._trainable.values())

    # ---------------------------------------------------------------- forward helpers
    def encode(self, texts: Sequence) -> dict:
        enc = self.tokenizer(list(texts), self.max_length)
        return {k: v.to(self.device) for k, v in enc.items()}

    def logits(self, texts: Sequence) -> torch.Tensor:
        return self.model(**self.encode(texts)).logits

    @torch.no_grad()
    def predict(self, texts: Sequence, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        out = [self.logits(texts[i:i + batch_size]).argmax(-1).cpu().numpy() for i in range(0, len(texts), batch_size)]
        return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)

    @torch.no_grad()
    def embed(self, texts: Sequence, projection: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Mean-pooled hidden states of the *frozen base* (adapter disabled), projected to D dims."""
        self.model.eval()
        outs = []
        with self.model.disable_adapter():
            for i in range(0, len(texts), batch_size):
                enc = self.encode(texts[i:i + batch_size])
                hs = self.model(**enc, output_hidden_states=True).hidden_states[-1]
                m = enc["attention_mask"].unsqueeze(-1).to(hs.dtype)
                outs.append(((hs * m).sum(1) / m.sum(1).clamp(min=1)).float().cpu())
        return torch.cat(outs) @ projection
