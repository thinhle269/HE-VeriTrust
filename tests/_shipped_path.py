from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from src.crypto.sketch_noise import cosine_noise_std
from src.trust.anfis import MamdaniTrust
from src.trust.features import TrustFeatures

def gate_scores(feats: Dict[int, TrustFeatures], probes=None, sigma: Optional[float]=None, alpha: float=1.0) -> np.ndarray:
    from src.trust.normalise import EvidenceAccumulator
    if sigma is None:
        sigma = cosine_noise_std(probes, probes.dim, trials=12) if probes is not None else 0.03
    acc = EvidenceAccumulator(alpha=alpha, z0=2.0, sigma=sigma)
    ev = acc.update(feats)
    return MamdaniTrust().score_many([ev[c] for c in sorted(feats)])
