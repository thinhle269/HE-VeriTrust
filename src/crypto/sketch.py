
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from .packing import (PackingScheme, assert_no_carry, block_widths, read_field,
                      unpack_slot_sum)
from .paillier_backend import PublicContext, SecretContext

_PROBE_DOMAIN = b"HE-VeriTrust/v2/probes"


@dataclass(frozen=True)
class ProbeSet:


    pos: Tuple[Tuple[int, ...], ...]
    neg: Tuple[Tuple[int, ...], ...]
    dim: int
    widths: Tuple[int, ...]

    @property
    def k(self) -> int:
        return len(self.pos)

    @property
    def n_blocks(self) -> int:
        return len(self.widths)

    def coeff_matrix(self) -> np.ndarray:

        M = np.zeros((self.k, self.n_blocks), dtype=np.float64)
        for m in range(self.k):
            if self.pos[m]:
                M[m, list(self.pos[m])] = 1.0
            if self.neg[m]:
                M[m, list(self.neg[m])] = -1.0
        return M

    def probe_widths(self) -> np.ndarray:

        w = np.asarray(self.widths, dtype=np.float64)
        return np.array([w[list(self.pos[m])].sum() + w[list(self.neg[m])].sum()
                         for m in range(self.k)], dtype=np.float64)

    def spectral_norm(self) -> float:

        if self.k == 0:
            return 0.0
        M = self.coeff_matrix()
        w = np.asarray(self.widths, dtype=np.float64)
        norms = np.sqrt(np.maximum(self.probe_widths(), 1.0))
        G = ((M * w) @ M.T) / np.outer(norms, norms)
        return float(np.sqrt(max(np.linalg.eigvalsh(G).max(), 0.0)))


def probe_seed(round_idx: int, transcript_root: bytes,
               authority_nonce: bytes) -> bytes:

    h = hashlib.sha256()
    h.update(_PROBE_DOMAIN)
    h.update(int(round_idx).to_bytes(8, "big"))
    h.update(transcript_root)
    h.update(authority_nonce)
    return h.digest()


def derive_probes(seed: bytes, dim: int, scheme: PackingScheme,
                  k: int, density: float = 0.5) -> ProbeSet:

    widths = tuple(int(x) for x in block_widths(dim, scheme))
    n_blocks = len(widths)
    if k < 1:
        return ProbeSet((), (), int(dim), widths)
    if not 0 < density <= 1:
        raise ValueError("probe density must lie in (0, 1]")
    rng = np.random.default_rng(
        np.frombuffer(hashlib.sha512(seed).digest(), dtype=np.uint32))
    pos: List[Tuple[int, ...]] = []
    neg: List[Tuple[int, ...]] = []
    for _ in range(int(k)):

        u = rng.random(n_blocks)
        c = np.zeros(n_blocks, dtype=np.int8)
        c[u < density / 2] = 1
        c[(u >= density / 2) & (u < density)] = -1
        p = tuple(np.flatnonzero(c > 0).tolist())
        n = tuple(np.flatnonzero(c < 0).tolist())

        if not p and not n:
            p = (int(rng.integers(n_blocks)),)
        pos.append(p)
        neg.append(n)
    return ProbeSet(tuple(pos), tuple(neg), int(dim), widths)


def _probe_halves(cts: Sequence[int], pos: Sequence[int], neg: Sequence[int],
                  nsquare: int, zero: int) -> Tuple[int, int]:

    cp = zero
    for b in pos:
        cp = (cp * int(cts[b])) % nsquare
    cn = zero
    for b in neg:
        cn = (cn * int(cts[b])) % nsquare
    return cp, cn


def sketch_ciphertexts_parallel(pub: PublicContext, ciphertexts: Sequence[int],
                                probes: ProbeSet, scheme: PackingScheme,
                                n_jobs: int = -1) -> List[int]:

    if len(ciphertexts) != probes.n_blocks:
        raise ValueError(
            f"expected {probes.n_blocks} packed ciphertexts, "
            f"got {len(ciphertexts)}")
    if probes.k == 0:
        return []
    for m in range(probes.k):
        assert_no_carry(scheme, max(len(probes.pos[m]), len(probes.neg[m])) or 1)
    if n_jobs == 1 or probes.k < 4:
        return sketch_ciphertexts(pub, ciphertexts, probes, scheme)
    cts = [int(c) for c in ciphertexts]
    nsq = pub.nsquare
    pairs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_probe_halves)(cts, probes.pos[m], probes.neg[m], nsq,
                               pub.encrypt_zero())
        for m in range(probes.k))
    out: List[int] = []
    for cp, cn in pairs:
        out.append(cp)
        out.append(cn)
    return out


def sketch_ciphertexts_batch(pub: PublicContext,
                             per_client: Sequence[Sequence[int]],
                             probes: ProbeSet, scheme: PackingScheme,
                             n_jobs: int = -1) -> List[List[int]]:

    for cts in per_client:
        if len(cts) != probes.n_blocks:
            raise ValueError(
                f"expected {probes.n_blocks} packed ciphertexts, got {len(cts)}")
    if probes.k == 0:
        return [[] for _ in per_client]
    for m in range(probes.k):
        assert_no_carry(scheme, max(len(probes.pos[m]), len(probes.neg[m])) or 1)
    if n_jobs == 1 or probes.k * len(per_client) < 4:
        return [sketch_ciphertexts(pub, c, probes, scheme) for c in per_client]

    nsq = pub.nsquare
    tasks = [(i, m) for i in range(len(per_client)) for m in range(probes.k)]
    cts_int = [[int(c) for c in g] for g in per_client]


    zero = pub.encrypt_zero()
    pairs = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_probe_halves)(cts_int[i], probes.pos[m], probes.neg[m], nsq, zero)
        for i, m in tasks)
    out: List[List[int]] = [[0] * (2 * probes.k) for _ in per_client]
    for (i, m), (cp, cn) in zip(tasks, pairs):
        out[i][2 * m] = cp
        out[i][2 * m + 1] = cn
    return out


def sketch_ciphertexts(pub: PublicContext, ciphertexts: Sequence[int],
                       probes: ProbeSet, scheme: PackingScheme) -> List[int]:

    if len(ciphertexts) != probes.n_blocks:
        raise ValueError(
            f"expected {probes.n_blocks} packed ciphertexts, got {len(ciphertexts)}")
    out: List[int] = []
    for m in range(probes.k):


        assert_no_carry(scheme, max(len(probes.pos[m]), len(probes.neg[m])) or 1)
        cp = pub.add_many([ciphertexts[b] for b in probes.pos[m]]) \
            if probes.pos[m] else pub.encrypt_zero()
        cn = pub.add_many([ciphertexts[b] for b in probes.neg[m]]) \
            if probes.neg[m] else pub.encrypt_zero()
        out.append(cp)
        out.append(cn)
    return out


def decode_sketch(plaintexts: Sequence[int], probes: ProbeSet,
                  scheme: PackingScheme, normalise: bool = True) -> np.ndarray:

    w = np.asarray(probes.widths, dtype=np.int64)
    vals = np.zeros(probes.k, dtype=np.float64)
    for m in range(probes.k):
        wp = int(w[list(probes.pos[m])].sum()) if probes.pos[m] else 0
        wn = int(w[list(probes.neg[m])].sum()) if probes.neg[m] else 0
        sp = unpack_slot_sum(plaintexts[2 * m], scheme, wp) if probes.pos[m] else 0.0
        sn = unpack_slot_sum(plaintexts[2 * m + 1], scheme, wn) if probes.neg[m] else 0.0
        v = (sp - sn) / scheme.scale
        if normalise:
            v /= math.sqrt(max(wp + wn, 1))
        vals[m] = v
    return vals


def dp_sigma(probes: ProbeSet, clip_norm: float, relative_sigma: float) -> float:

    return float(relative_sigma) * probes.spectral_norm() * float(clip_norm)


def dp_epsilon(relative_sigma: float, delta: float = 1e-5) -> float:

    if relative_sigma <= 0:
        return float("inf")
    return float(math.sqrt(2.0 * math.log(1.25 / float(delta))) / float(relative_sigma))


def dp_epsilon_composed(relative_sigma: float, rounds: int,
                        delta: float = 1e-5) -> float:

    if rounds <= 0:
        return 0.0
    eps = dp_epsilon(relative_sigma, delta / (2.0 * rounds))
    d_prime = delta / 2.0
    return float(math.sqrt(2.0 * rounds * math.log(1.0 / d_prime)) * eps
                 + rounds * eps * (math.exp(eps) - 1.0))


def add_dp_noise(sketch: np.ndarray, sigma_abs: float,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:

    if sigma_abs <= 0:
        return sketch
    rng = rng or np.random.default_rng()
    return sketch + rng.normal(0.0, sigma_abs, size=sketch.shape)


def project_plaintext(vec: np.ndarray, probes: ProbeSet) -> np.ndarray:

    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size < probes.dim:
        v = np.pad(v, (0, probes.dim - v.size))
    w = np.asarray(probes.widths, dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(w)[:-1]])
    block_sums = np.array([v[starts[b]:starts[b] + w[b]].sum()
                           for b in range(probes.n_blocks)], dtype=np.float64)
    out = np.zeros(probes.k, dtype=np.float64)
    pw = probes.probe_widths()
    for m in range(probes.k):
        s = block_sums[list(probes.pos[m])].sum() if probes.pos[m] else 0.0
        s -= block_sums[list(probes.neg[m])].sum() if probes.neg[m] else 0.0
        out[m] = s / math.sqrt(max(pw[m], 1.0))
    return out
