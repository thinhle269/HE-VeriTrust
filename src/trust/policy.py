
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class Decision:
    accepted: List[int]
    rejected: List[int]
    weights: Dict[int, float]
    smoothed: Dict[int, float]
    raw: Dict[int, float]
    forced: List[int] = field(default_factory=list)


class ZeroTrustPolicy:
    def __init__(self, threshold: float = 0.40, ema_beta: float = 0.6,
                 reject_decay: float = 0.7, max_reject_streak: int = 3,
                 min_accept_fraction: float = 0.5):
        if not 0.0 <= ema_beta <= 1.0:
            raise ValueError("ema_beta must lie in [0, 1]")
        if not 0.0 <= min_accept_fraction <= 1.0:
            raise ValueError("min_accept_fraction must lie in [0, 1]")
        self.threshold = float(threshold)
        self.ema_beta = float(ema_beta)
        self.reject_decay = float(reject_decay)
        self.max_reject_streak = max(0, int(max_reject_streak))
        self.min_accept_fraction = float(min_accept_fraction)
        self._ema: Dict[int, float] = {}
        self._streak: Dict[int, int] = {}


    def evaluate(self, client_ids: Sequence[int],
                 trust: Sequence[float]) -> Decision:
        if len(client_ids) != len(trust):
            raise ValueError("client_ids/trust length mismatch")
        smoothed: Dict[int, float] = {}
        accepted: List[int] = []
        rejected: List[int] = []

        for cid, t in zip(client_ids, trust):
            cid = int(cid)
            t = float(t)
            prev = self._ema.get(cid)
            ema = t if prev is None else self.ema_beta * t + (1 - self.ema_beta) * prev
            self._ema[cid] = ema
            streak = self._streak.get(cid, 0)
            eff = min(streak, self.max_reject_streak)
            score = ema * (self.reject_decay ** eff) if eff > 0 else ema
            smoothed[cid] = score
            if score < self.threshold:
                rejected.append(cid)
                self._streak[cid] = streak + 1
            else:
                accepted.append(cid)
                self._streak[cid] = 0

        forced: List[int] = []
        n = len(client_ids)
        min_k = math.ceil(self.min_accept_fraction * n) if self.min_accept_fraction > 0 else 0
        if len(accepted) < min_k:
            for c in sorted(rejected, key=lambda x: smoothed[x], reverse=True)[:min_k - len(accepted)]:
                accepted.append(c)
                rejected.remove(c)
                forced.append(c)

        weights: Dict[int, float] = {}
        if accepted:
            adj = {c: max(smoothed[c], 1e-3) for c in accepted}
            tot = sum(adj.values())
            weights = {c: v / tot for c, v in adj.items()}
        return Decision(accepted, rejected, weights, smoothed,
                        {int(c): float(t) for c, t in zip(client_ids, trust)},
                        forced)

    def state(self) -> Dict[str, object]:
        return {"threshold": self.threshold, "ema": dict(self._ema),
                "streak": dict(self._streak)}


def quantise_weights(weights: Dict[int, float], order: Sequence[int],
                     scale_bits: int = 10,
                     max_share: float = 0.5) -> List[int]:

    ids = list(order)
    if not ids:
        return []
    w = np.array([max(float(weights.get(int(c), 0.0)), 0.0) for c in ids],
                 dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(ids), dtype=np.float64)
    w = w / w.sum()

    cap = float(max_share)
    if len(ids) * cap < 1.0 - 1e-9:


        raise ValueError(
            f"max_share={cap} is unsatisfiable for {len(ids)} clients "
            f"(needs >= {1.0 / len(ids):.3f})")
    for _ in range(64):
        over = w > cap + 1e-12
        if not over.any():
            break
        surplus = float((w[over] - cap).sum())
        w[over] = cap
        room = ~over
        if not room.any():
            break
        w[room] += surplus * (w[room] / max(w[room].sum(), 1e-12))

    scale = (1 << int(scale_bits))
    q = np.maximum(np.rint(w * scale).astype(np.int64), 0)
    if q.sum() == 0:
        q = np.ones(len(ids), dtype=np.int64)

    while q.max() / q.sum() > cap + 1e-9:
        q[int(np.argmax(q))] -= 1
    return [int(v) for v in q]
