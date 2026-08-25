
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class TrustFeatures:
    proj_ref: float
    norm_ratio: float
    peer_agreement: float

    def as_array(self) -> np.ndarray:
        return np.array([self.proj_ref, self.norm_ratio, self.peer_agreement],
                        dtype=np.float64)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def build_features(sketches: Dict[int, np.ndarray],
                   reference_sketch: Optional[np.ndarray] = None
                   ) -> Dict[int, TrustFeatures]:

    ids = sorted(sketches)
    if not ids:
        return {}
    S = np.stack([np.asarray(sketches[c], dtype=np.float64) for c in ids], axis=0)
    norms = np.linalg.norm(S, axis=1)
    med_norm = float(np.median(norms)) if len(norms) else 0.0


    if reference_sketch is None or np.linalg.norm(reference_sketch) < _EPS:
        ref = np.median(S, axis=0)
    else:
        ref = np.asarray(reference_sketch, dtype=np.float64).reshape(-1)


    unit = S / np.maximum(np.linalg.norm(S, axis=1, keepdims=True), _EPS)
    C = np.clip(unit @ unit.T, -1.0, 1.0)
    np.fill_diagonal(C, np.nan)

    out: Dict[int, TrustFeatures] = {}
    for i, cid in enumerate(ids):
        proj = _cos(S[i], ref)
        r = norms[i] / max(med_norm, _EPS)
        norm_ratio = float(np.tanh(np.log(max(r, _EPS)))) if med_norm > _EPS else 0.0
        peers = C[i][~np.isnan(C[i])]
        peer = float(np.median(peers)) if peers.size else 0.0
        out[cid] = TrustFeatures(proj, norm_ratio, peer)
    return out


def features_matrix(feats: Sequence[TrustFeatures]) -> np.ndarray:
    if not feats:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([f.as_array() for f in feats]).astype(np.float32)
