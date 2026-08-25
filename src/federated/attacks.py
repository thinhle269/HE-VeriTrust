
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

COORDINATED = {"alie", "ipm", "min_max", "min_sum", "block_evasion",
               "unresolvable"}


def _stats(benign: np.ndarray):
    return benign.mean(axis=0), benign.std(axis=0)


def alie(benign: np.ndarray, n_total: int, n_malicious: int) -> np.ndarray:
    mu, sigma = _stats(benign)
    s = max(np.floor(n_total / 2 + 1) - n_malicious, 1)
    denom = max(n_total - n_malicious, 1)
    frac = float(np.clip((n_total - n_malicious - s) / denom, 1e-6, 1 - 1e-6))
    z = float(norm.ppf(frac))
    z = z if np.isfinite(z) else 0.0
    return (mu - z * sigma).astype(np.float32)


def ipm(benign: np.ndarray, epsilon: float = 0.5) -> np.ndarray:
    mu, _ = _stats(benign)
    return (-float(epsilon) * mu).astype(np.float32)


def _direction(mu, sigma, kind: str) -> np.ndarray:
    kind = (kind or "std").lower()
    if kind == "std":
        return -sigma
    if kind == "sign":
        return -np.sign(mu)
    if kind == "mean":
        return -mu
    raise ValueError(f"unknown perturbation direction {kind}")


def _search_gamma(objective, mu, p, threshold, hi=100.0, iters=25) -> float:
    lo, best = 0.0, 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if objective(mu + mid * p) <= threshold:
            best, lo = mid, mid
        else:
            hi = mid
    return best


def min_max(benign: np.ndarray, perturbation: str = "std") -> np.ndarray:
    mu, sigma = _stats(benign)
    p = _direction(mu, sigma, perturbation)
    sq = np.sum(benign * benign, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (benign @ benign.T), 0.0)
    thr = float(np.sqrt(d2.max())) if benign.shape[0] > 1 else 0.0
    g = _search_gamma(lambda c: max(np.linalg.norm(c - b) for b in benign),
                      mu, p, thr)
    return (mu + g * p).astype(np.float32)


def min_sum(benign: np.ndarray, perturbation: str = "std") -> np.ndarray:
    mu, sigma = _stats(benign)
    p = _direction(mu, sigma, perturbation)
    sums = [float(np.sum((benign - benign[i]) ** 2)) for i in range(benign.shape[0])]
    thr = max(sums) if sums else 0.0
    g = _search_gamma(lambda c: float(np.sum((benign - c) ** 2)), mu, p, thr)
    return (mu + g * p).astype(np.float32)


def craft(method: str, benign: np.ndarray, n_total: int, n_malicious: int,
          epsilon: float = 0.5, perturbation: str = "std") -> np.ndarray:
    key = (method or "").lower()
    if key == "alie":
        return alie(benign, n_total, n_malicious)
    if key == "ipm":
        return ipm(benign, epsilon)
    if key == "min_max":
        return min_max(benign, perturbation)
    if key == "min_sum":
        return min_sum(benign, perturbation)
    raise ValueError(f"unknown coordinated attack: {method}. "
                     f"Choose from {sorted(COORDINATED)}")


def apply_coordinated(updates: Sequence, method: str, epsilon: float = 0.5,
                      perturbation: str = "std", slots: int = 50,
                      reference=None) -> int:

    benign = [u.delta for u in updates if not u.is_malicious]
    malicious = [u for u in updates if u.is_malicious]
    if not benign or not malicious:
        return 0
    if (method or "").lower() == "unresolvable":


        ref = np.asarray(reference, dtype=np.float64).reshape(-1)             if reference is not None and np.linalg.norm(reference) > 1e-12             else np.mean(np.stack(benign, axis=0), axis=0)
        target = float(np.median([np.linalg.norm(b) for b in benign]))
        for i, u in enumerate(malicious):
            u.delta = unresolvable(ref, target, seed=i)
        return len(malicious)
    if (method or "").lower() == "block_evasion":


        for u in malicious:
            u.delta = block_evasion(u.delta, slots)
        return len(malicious)
    v = craft(method, np.stack(benign, axis=0), len(updates), len(malicious),
              epsilon, perturbation)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    for u in malicious:
        u.delta = v.copy()
    return len(malicious)


def block_evasion(honest: np.ndarray, slots: int) -> np.ndarray:

    h = np.asarray(honest, dtype=np.float64).reshape(-1)
    S = max(int(slots), 1)
    pad = (-h.size) % S
    hp = np.pad(h, (0, pad))
    p = -hp
    correction = (hp.reshape(-1, S).sum(1) - p.reshape(-1, S).sum(1)) / S
    return (p + np.repeat(correction, S))[:h.size].astype(np.float32)


def unresolvable(reference: np.ndarray, target_norm: float,
                 seed: int = 0) -> np.ndarray:

    rng = np.random.default_rng(int(seed))
    r = np.asarray(reference, dtype=np.float64).reshape(-1)
    rr = float(r @ r)


    for _ in range(8):
        v = rng.normal(size=r.size)
        if rr > 1e-12:
            v -= (v @ r) / rr * r
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            return (v / n * float(target_norm)).astype(np.float32)
    raise RuntimeError("could not draw a vector orthogonal to the reference")
