"""Factor-space linear algebra (R4): update similarity (Alg. 4) and LoRAMerger (Alg. 9).

Neither routine forms a d_out x d_in matrix, so 1B-parameter models and 50 clients
fit in laptop memory (NFR 2, 4).

Trace identity (Eq. 1):  <B1 A1, B2 A2>_F = tr[(B1^T B2)(A2 A1^T)]
Update as a product (Eq. 2): U_k = B_k A_k - B_g A_g = [B_k, -B_g] . [A_k; A_g]
"""
from __future__ import annotations

import numpy as np
import torch

DT = torch.float64


# --------------------------------------------------------------------------- Algorithm 4
def update_similarity(adapters: list, global_adapter) -> np.ndarray:
    """Pairwise cosine similarity S in [-1, 1]^{m x m} of client updates U_k (classification head excluded)."""
    m = len(adapters)
    G = torch.zeros(m, m, dtype=DT)
    for mod, gab in global_adapter.items():
        Ag, Bg = gab["A"].to(DT), gab["B"].to(DT)
        r2 = Ag.shape[0] + Ag.shape[0]
        Bs = torch.cat([torch.cat([a[mod]["B"].to(DT), -Bg], 1) for a in adapters], 1)   # d_out x 2rm
        As = torch.cat([torch.cat([a[mod]["A"].to(DT), Ag], 0) for a in adapters], 0)    # 2rm x d_in
        MB = (Bs.T @ Bs).reshape(m, r2, m, r2)       # Gram matrices, 2rm x 2rm
        MA = (As @ As.T).reshape(m, r2, m, r2)
        G += torch.einsum("iajb,jbia->ij", MB, MA)   # G_ij += tr(MB[i,j] . MA[j,i])
    G = 0.5 * (G + G.T)
    d = torch.sqrt(torch.clamp(torch.diag(G), min=0))
    denom = d[:, None] * d[None, :]
    S = torch.where(denom > 0, G / torch.where(denom > 0, denom, torch.ones_like(denom)), torch.zeros_like(G))
    return S.clamp(-1, 1).numpy()


def explicit_inner(a1, a2, global_adapter) -> float:
    """Reference implementation that forms Delta W (tests only)."""
    tot = 0.0
    for mod, g in global_adapter.items():
        gW = g["B"].to(DT) @ g["A"].to(DT)
        d1 = a1[mod]["B"].to(DT) @ a1[mod]["A"].to(DT) - gW
        d2 = a2[mod]["B"].to(DT) @ a2[mod]["A"].to(DT) - gW
        tot += float((d1 * d2).sum())
    return tot


# --------------------------------------------------------------------------- Algorithm 9
def merge(adapters: list, heads: list, w: np.ndarray, method: str = "adapter_average"):
    """Return (global adapter, global head, residual epsilon or None)."""
    w_t = torch.as_tensor(np.asarray(w), dtype=DT)
    head = {k: sum(float(wi) * h[k].to(DT) for wi, h in zip(w, heads)).float() for k in heads[0]}
    out, eps = {}, []
    for mod in adapters[0]:
        As = [a[mod]["A"].to(DT) for a in adapters]
        Bs = [a[mod]["B"].to(DT) for a in adapters]
        r = As[0].shape[0]
        if method == "adapter_average":
            out[mod] = {"A": sum(wi * A for wi, A in zip(w_t, As)).float(),
                        "B": sum(wi * B for wi, B in zip(w_t, Bs)).float()}
            continue
        # product aggregation: sum_k w_k B_k A_k = B_hat A_hat (Eq. 3), re-factorised to rank r
        B_hat = torch.cat([wi * B for wi, B in zip(w_t, Bs)], 1)        # d_out x rm
        A_hat = torch.cat(As, 0)                                         # rm x d_in
        QB, RB = torch.linalg.qr(B_hat, mode="reduced")
        QA, RA = torch.linalg.qr(A_hat.T, mode="reduced")
        U, S, Vh = torch.linalg.svd(RB @ RA.T)                           # small core
        k = min(r, S.numel())
        sq = torch.sqrt(S[:k])
        Bn = torch.zeros(B_hat.shape[0], r, dtype=DT)
        An = torch.zeros(r, A_hat.shape[1], dtype=DT)
        Bn[:, :k] = QB @ U[:, :k] * sq                                   # balanced split Sigma^{1/2}
        An[:k] = (sq[:, None] * Vh[:k]) @ QA.T
        tot = float((S ** 2).sum())
        eps.append(float(torch.sqrt((S[k:] ** 2).sum() / tot)) if tot > 0 else 0.0)
        out[mod] = {"A": An.float(), "B": Bn.float()}
    return out, head, (max(eps) if eps else None)
