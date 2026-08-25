
from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np


def fedavg(updates: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    tot = float(sum(weights))
    if tot <= 0:
        raise ValueError("total weight must be positive")
    U = np.stack([u.astype(np.float64) for u in updates], axis=0)
    w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    return ((w * U).sum(axis=0) / tot).astype(np.float32)


def fedmedian(updates, weights=None) -> np.ndarray:
    return np.median(np.stack([u.astype(np.float64) for u in updates], 0),
                     axis=0).astype(np.float32)


def trimmed_mean(updates, weights=None, trim_ratio: float = 0.2) -> np.ndarray:
    n = len(updates)
    beta = int(np.floor(float(trim_ratio) * n))
    if 2 * beta >= n:
        return fedmedian(updates)
    S = np.sort(np.stack([u.astype(np.float64) for u in updates], 0), axis=0)
    return S[beta:n - beta].mean(axis=0).astype(np.float32)


def _krum_scores(updates, f: int) -> np.ndarray:
    n = len(updates)
    k = n - int(f) - 2
    if k < 1:
        warnings.warn(f"Krum undefined for n={n}, f={f} (needs n > 2f+2)",
                      RuntimeWarning)
        k = 1
    U = np.stack([u.astype(np.float64).reshape(-1) for u in updates], 0)
    sq = np.sum(U * U, axis=1)
    D = sq[:, None] + sq[None, :] - 2.0 * (U @ U.T)
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, :k].sum(axis=1)


def multi_krum(updates, weights=None, num_byzantine: int = 1,
               m: Optional[int] = None) -> np.ndarray:
    n = len(updates)
    if n == 1:
        return updates[0].astype(np.float32)
    m = max(1, min(int(m if m is not None else n - int(num_byzantine)), n))
    top = np.argsort(_krum_scores(updates, num_byzantine))[:m]
    return fedavg([updates[i] for i in top], [1.0] * m)


def bulyan(updates, weights=None, num_byzantine: int = 1) -> np.ndarray:
    n = len(updates)
    if n == 1:
        return updates[0].astype(np.float32)


    f = min(max(0, int(num_byzantine)), max(0, (n - 3) // 4))
    m = max(1, n - 2 * f)
    sel = np.argsort(_krum_scores(updates, f))[:m]
    chosen = [updates[i] for i in sel]
    beta = min(f, max(0, (m - 1) // 2))
    return trimmed_mean(chosen, trim_ratio=beta / max(m, 1))


def foolsgold(updates, weights=None, kappa: float = 1.0,
              eps: float = 1e-5) -> np.ndarray:
    n = len(updates)
    if n == 1:
        return updates[0].astype(np.float32)
    U = np.stack([u.astype(np.float64).reshape(-1) for u in updates], 0)
    N = U / (np.linalg.norm(U, axis=1, keepdims=True) + eps)
    cs = N @ N.T
    np.fill_diagonal(cs, 0.0)
    v = cs.max(axis=1)
    for i in range(n):
        for j in range(n):
            if i != j and v[j] > eps and v[i] < v[j]:
                cs[i, j] *= v[i] / v[j]
    v = cs.max(axis=1)
    a = np.clip(1.0 - v, 0.0, 1.0)
    if a.max() > eps:
        a = a / a.max()
    ae = np.clip(a, eps, 1 - eps)
    a = np.clip(kappa * (np.log(ae / (1 - ae)) + 0.5), 0.0, 1.0)
    w = np.ones(n) / n if a.sum() <= eps else a / a.sum()
    return (w.reshape(-1, 1) * U).sum(axis=0).astype(np.float32)


_DISPATCH = {"fedavg": fedavg, "median": fedmedian, "fedmedian": fedmedian,
             "trimmed_mean": trimmed_mean, "multi_krum": multi_krum,
             "krum": multi_krum, "bulyan": bulyan, "foolsgold": foolsgold}


def aggregate(method: str, updates, weights=None, **kw) -> np.ndarray:
    key = (method or "fedavg").lower()
    if key not in _DISPATCH:
        raise ValueError(f"unknown aggregation method: {method}. "
                         f"Choose from {sorted(_DISPATCH)}")
    if weights is None:
        weights = [1.0 / max(len(updates), 1)] * len(updates)
    return _DISPATCH[key](updates, weights, **kw)
