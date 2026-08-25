
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

from .features import TrustFeatures

_EPS = 1e-12


def ema_effective_rounds(alpha: float) -> float:

    a = float(np.clip(alpha, _EPS, 1.0))
    return (2.0 - a) / a


class EvidenceAccumulator:


    def __init__(self, alpha: float = 0.4, z0: float = 2.0,
                 sigma: float = 0.03) -> None:
        self.alpha = float(np.clip(alpha, 1e-3, 1.0))
        self.z0 = max(float(z0), _EPS)
        self.sigma = max(float(sigma), 1e-4)
        self._state: Dict[int, np.ndarray] = {}

    @property
    def sigma_eff(self) -> float:

        return self.sigma / np.sqrt(ema_effective_rounds(self.alpha))

    def reset(self) -> None:
        self._state.clear()

    def _squash(self, cos_value: float) -> float:
        return float(np.tanh(cos_value / self.sigma_eff / self.z0))

    def update(self, feats: Dict[int, TrustFeatures]
               ) -> Dict[int, TrustFeatures]:

        out: Dict[int, TrustFeatures] = {}
        for cid, f in feats.items():
            v = f.as_array()
            prev = self._state.get(cid)
            cur = v if prev is None else self.alpha * v + (1.0 - self.alpha) * prev
            self._state[cid] = cur
            out[cid] = TrustFeatures(
                proj_ref=self._squash(float(cur[0])),
                norm_ratio=float(np.clip(cur[1], -1.0, 1.0)),
                peer_agreement=self._squash(float(cur[2])),
            )
        return out
