
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed

from .packing import PackingScheme, assert_no_carry, block_widths, read_field
from .paillier_backend import PublicContext

_SHIFT_DOMAIN = b"HE-VeriTrust/v2/shifted-probes"


@dataclass(frozen=True)
class ShiftedProbeSet:


    blocks: Tuple[Tuple[int, ...], ...]
    signs: Tuple[Tuple[int, ...], ...]
    fields: Tuple[int, ...]
    shift: Tuple[int, ...]
    dim: int
    slots: int
    widths: Tuple[int, ...]

    @property
    def k(self) -> int:

        return len(self.blocks)

    @property
    def n_values(self) -> int:

        return self.k * len(self.fields)

    @property
    def n_blocks(self) -> int:
        return len(self.shift)

    def spectral_norm(self) -> float:

        return spectral_norm(self)

    def contributors(self, m: int, t: int) -> Tuple[List[int], List[int]]:

        pos: List[int] = []
        neg: List[int] = []
        for b, sgn in zip(self.blocks[m], self.signs[m]):
            s = t - self.shift[b]
            if 0 <= s < self.slots and s < self.widths[b]:
                (pos if sgn > 0 else neg).append(b)
        return pos, neg


def derive_shifted_probes(seed: bytes, dim: int, scheme: PackingScheme,
                          k: int, density: float = 0.5) -> ShiftedProbeSet:

    if scheme.shift_slots < 1:
        raise ValueError("shifted probes require scheme.shift_slots >= 1")
    if k < 1:
        raise ValueError("k must be >= 1")
    widths = tuple(int(x) for x in block_widths(dim, scheme))
    nb = len(widths)
    rng = np.random.default_rng(
        np.frombuffer(hashlib.sha512(_SHIFT_DOMAIN + seed).digest(),
                      dtype=np.uint32))
    shift = tuple(int(x) for x in rng.integers(0, scheme.shift_slots + 1, size=nb))
    blocks, signs = [], []
    for _ in range(int(k)):
        sel = np.flatnonzero(rng.random(nb) < density)
        if sel.size == 0:
            sel = np.array([int(rng.integers(nb))])
        sg = rng.choice(np.array([-1, 1]), size=sel.size)
        blocks.append(tuple(int(b) for b in sel))
        signs.append(tuple(int(x) for x in sg))


    fields = tuple(range(scheme.shift_slots, scheme.slots))
    return ShiftedProbeSet(tuple(blocks), tuple(signs), fields, shift,
                           int(dim), scheme.slots, widths)


def _shift_block(ct: int, delta: int, slot_bits: int, nsquare: int) -> int:

    if delta == 0:
        return int(ct)
    return pow(int(ct), 1 << (int(delta) * int(slot_bits)), nsquare)


def _shift_chunk(cts, deltas, slot_bits, nsquare):
    return [_shift_block(c, d, slot_bits, nsquare) for c, d in zip(cts, deltas)]


def shifted_sketch_ciphertexts_batch(pub: PublicContext,
                                     per_client: Sequence[Sequence[int]],
                                     probes: ShiftedProbeSet,
                                     scheme: PackingScheme,
                                     n_jobs: int = -1) -> List[List[int]]:

    nsq = pub.nsquare
    sb = scheme.slot_bits
    deltas = list(probes.shift)

    if n_jobs == 1 or len(per_client) * probes.n_blocks < 256:
        shifted_all = [[_shift_block(c, d, sb, nsq) for c, d in zip(cts, deltas)]
                       for cts in per_client]
    else:
        flat = [(i, b) for i in range(len(per_client))
                for b in range(probes.n_blocks)]
        n_chunks = min(128, max(1, len(flat)))
        size = (len(flat) + n_chunks - 1) // n_chunks
        chunks = [flat[i:i + size] for i in range(0, len(flat), size)]
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_shift_chunk)([int(per_client[i][b]) for i, b in ch],
                                  [deltas[b] for _, b in ch], sb, nsq)
            for ch in chunks)
        shifted_all = [[0] * probes.n_blocks for _ in per_client]
        for ch, vals in zip(chunks, res):
            for (i, b), v in zip(ch, vals):
                shifted_all[i][b] = v

    out: List[List[int]] = []
    for shifted in shifted_all:
        row: List[int] = []
        for m in range(probes.k):


            worst = max((max(len(a), len(b))
                         for a, b in (probes.contributors(m, t)
                                      for t in probes.fields)), default=1)
            assert_no_carry(scheme, worst or 1)


            cp = pub.encrypt_zero()
            cn = pub.encrypt_zero()
            for b, sgn in zip(probes.blocks[m], probes.signs[m]):
                if sgn > 0:
                    cp = (cp * shifted[b]) % nsq
                else:
                    cn = (cn * shifted[b]) % nsq
            row.append(cp)
            row.append(cn)
        out.append(row)
    return out


def decode_shifted_sketch(plaintexts: Sequence[int], probes: ShiftedProbeSet,
                          scheme: PackingScheme) -> np.ndarray:

    vals = np.zeros(probes.n_values, dtype=np.float64)
    i = 0
    for m in range(probes.k):
        for t in probes.fields:
            pos, neg = probes.contributors(m, t)
            fp = read_field(plaintexts[2 * m], scheme, t) - len(pos) * scheme.offset
            fn = read_field(plaintexts[2 * m + 1], scheme, t) - len(neg) * scheme.offset
            n = max(len(pos) + len(neg), 1)
            vals[i] = (fp - fn) / (scheme.scale * math.sqrt(n))
            i += 1
    return vals


def project_plaintext_shifted(vec: np.ndarray,
                              probes: ShiftedProbeSet) -> np.ndarray:

    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    if v.size < probes.dim:
        v = np.pad(v, (0, probes.dim - v.size))
    S = probes.slots
    out = np.zeros(probes.n_values, dtype=np.float64)
    i = 0
    for m in range(probes.k):
        for t in probes.fields:
            pos, neg = probes.contributors(m, t)
            acc = 0.0
            for b in pos:
                acc += v[b * S + (t - probes.shift[b])]
            for b in neg:
                acc -= v[b * S + (t - probes.shift[b])]
            out[i] = acc / math.sqrt(max(len(pos) + len(neg), 1))
            i += 1
    return out


def spectral_norm(probes: ShiftedProbeSet) -> float:

    if probes.k == 0:
        return 0.0
    coords = []
    for m in range(probes.k):
        for t in probes.fields:
            pos, neg = probes.contributors(m, t)
            d = {}
            for b in pos:
                d[b * probes.slots + (t - probes.shift[b])] = 1.0
            for b in neg:
                d[b * probes.slots + (t - probes.shift[b])] = -1.0
            n = math.sqrt(max(len(pos) + len(neg), 1))
            coords.append({j: s / n for j, s in d.items()})
    k = len(coords)
    G = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            a, b = coords[i], coords[j]
            small, large = (a, b) if len(a) < len(b) else (b, a)
            G[i, j] = G[j, i] = sum(v * large.get(key, 0.0)
                                    for key, v in small.items())
    return float(np.sqrt(max(np.linalg.eigvalsh(G).max(), 0.0)))
