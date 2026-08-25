
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

from .shifted_sketch import ShiftedProbeSet, project_plaintext_shifted
from .sketch import ProbeSet, project_plaintext

_CACHE: Dict[Tuple, float] = {}

AnyProbeSet = Union[ProbeSet, ShiftedProbeSet]


def _project(vec: np.ndarray, probes: AnyProbeSet) -> np.ndarray:
    if isinstance(probes, ShiftedProbeSet):
        return project_plaintext_shifted(vec, probes)
    return project_plaintext(vec, probes)


def cosine_noise_std(probes: AnyProbeSet, dim: int, trials: int = 24,
                     seed: int = 0, anchor_cos: float = 0.10) -> float:


    n_values = getattr(probes, "n_values", None) or probes.k
    key = (int(dim), int(n_values), float(anchor_cos), int(trials), int(seed))
    if key in _CACHE:
        return _CACHE[key]

    rng = np.random.default_rng(seed)
    est = []
    for _ in range(int(trials)):
        a = rng.normal(size=int(dim)); a /= np.linalg.norm(a)
        w = rng.normal(size=int(dim)); w -= (w @ a) * a; w /= np.linalg.norm(w)
        b = anchor_cos * a + np.sqrt(1.0 - anchor_cos ** 2) * w
        sa, sb = _project(a, probes), _project(b, probes)
        na, nb = np.linalg.norm(sa), np.linalg.norm(sb)
        if na < 1e-12 or nb < 1e-12:
            continue
        est.append(float(sa @ sb / (na * nb)))
    sigma = float(np.std(est)) if len(est) > 1 else 0.03


    sigma = max(sigma, 1e-4)
    _CACHE[key] = sigma
    return sigma
