
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed

from .identity import Registry, Submission, fresh_nonce, transcript_root
from .packing import PackingScheme, assert_no_carry, unpack
from .paillier_backend import PublicContext, SecretContext
from .sketch import (ProbeSet, add_dp_noise, decode_sketch, derive_probes,
                     dp_sigma, probe_seed, sketch_ciphertexts_batch)
from .shifted_sketch import (ShiftedProbeSet, decode_shifted_sketch,
                             derive_shifted_probes,
                             shifted_sketch_ciphertexts_batch)


class PolicyViolation(Exception):
    pass


@dataclass
class AuthorityPolicy:
    min_accept_fraction: float = 0.5
    max_client_weight: float = 0.5
    require_signatures: bool = True
    single_decrypt_per_round: bool = True
    audit_log: bool = True

    @classmethod
    def from_config(cls, cfg) -> "AuthorityPolicy":
        g = cfg.get if hasattr(cfg, "get") else (lambda k, d: d)
        return cls(
            min_accept_fraction=float(g("min_accept_fraction", 0.5)),
            max_client_weight=float(g("max_client_weight", 0.5)),
            require_signatures=bool(g("require_signatures", True)),
            single_decrypt_per_round=bool(g("single_decrypt_per_round", True)),
            audit_log=bool(g("audit_log", True)),
        )


@dataclass
class Measurement:

    round_idx: int
    client_ids: List[int]
    sketches: Dict[int, np.ndarray]
    probes: ProbeSet
    nonce: bytes
    transcript: bytes
    dp_sigma_abs: float


@dataclass
class AuditRecord:
    round_idx: int
    phase: str
    client_ids: List[int]
    weights: Optional[List[int]]
    transcript_hex: str
    timestamp: float
    note: str = ""


    nonce_hex: str = ""


class DecryptionAuthority:


    def __init__(self, secret: SecretContext, scheme: PackingScheme,
                 policy: AuthorityPolicy, dim: int,
                 sketch_k: int = 32, sketch_density: float = 0.5,
                 dp_enabled: bool = True, dp_relative_sigma: float = 0.05,
                 clip_norm: float = 1.0, n_jobs: int = -1,
                 sketch_mode: str = "shifted",
                 rng: Optional[np.random.Generator] = None):
        self._secret = secret
        self.pub: PublicContext = secret.public
        self.scheme = scheme
        self.policy = policy
        self.dim = int(dim)
        self.sketch_k = int(sketch_k)
        self.sketch_density = float(sketch_density)
        self.dp_enabled = bool(dp_enabled)
        self.dp_relative_sigma = float(dp_relative_sigma)
        self.clip_norm = float(clip_norm)
        self.n_jobs = int(n_jobs)


        self.sketch_mode = str(sketch_mode)
        if self.sketch_mode == "shifted" and scheme.shift_slots < 1:
            raise ValueError(
                "sketch_mode='shifted' needs crypto.sketch.shift_slots >= 1; "
                "without reserved headroom a shifted field would run past the "
                "modulus and corrupt the packing")
        self.registry = Registry()
        self.audit: List[AuditRecord] = []

        self._rng = rng or np.random.default_rng()

        self._last_completed_round = -1
        self._cur_round: Optional[int] = None
        self._cur_subs: Dict[int, Submission] = {}
        self._cur_probes: Optional[ProbeSet] = None
        self._cur_transcript: Optional[bytes] = None
        self._opened_rounds: set = set()


    def enrol(self, client_id: int, public_bytes: bytes) -> None:
        self.registry.enrol(client_id, public_bytes)

    def _record(self, rec: AuditRecord) -> None:
        if self.policy.audit_log:
            self.audit.append(rec)


    def measure(self, round_idx: int,
                submissions: Sequence[Submission]) -> Measurement:

        round_idx = int(round_idx)
        if round_idx <= self._last_completed_round:
            raise PolicyViolation(
                f"round {round_idx} already completed "
                f"(last={self._last_completed_round}); rounds must strictly increase")
        if not submissions:
            raise PolicyViolation("no submissions")

        seen: Dict[int, Submission] = {}
        n_blocks_expected = self.scheme.n_blocks(self.dim)
        nsq = self.pub.nsquare
        for sub in submissions:
            cid = int(sub.client_id)
            if cid in seen:
                raise PolicyViolation(f"duplicate submission for client {cid}")
            if int(sub.round_idx) != round_idx:
                raise PolicyViolation(
                    f"client {cid} submitted for round {sub.round_idx}, "
                    f"expected {round_idx} (replay?)")
            if cid not in self.registry:
                raise PolicyViolation(f"client {cid} is not enrolled")
            if self.policy.require_signatures and not self.registry.verify(sub):
                raise PolicyViolation(f"invalid signature for client {cid}")
            if len(sub.ciphertexts) != n_blocks_expected:
                raise PolicyViolation(
                    f"client {cid} sent {len(sub.ciphertexts)} blocks, "
                    f"expected {n_blocks_expected}")


            for c in sub.ciphertexts:
                if not (0 < int(c) < nsq):
                    raise PolicyViolation(f"client {cid} sent an out-of-range ciphertext")
            seen[cid] = sub

        subs = [seen[c] for c in sorted(seen)]
        root = transcript_root(subs)
        nonce = fresh_nonce()
        seed = probe_seed(round_idx, root, nonce)
        if self.sketch_mode == "shifted":
            probes = derive_shifted_probes(seed, self.dim, self.scheme,
                                           k=self.sketch_k,
                                           density=self.sketch_density)
        else:
            probes = derive_probes(seed, self.dim, self.scheme,
                                   k=self.sketch_k, density=self.sketch_density)

        sigma_abs = (dp_sigma(probes, self.clip_norm, self.dp_relative_sigma)
                     if self.dp_enabled else 0.0)


        builder = (shifted_sketch_ciphertexts_batch
                   if self.sketch_mode == "shifted" else sketch_ciphertexts_batch)
        sketch_cts = builder(self.pub, [s.ciphertexts for s in subs], probes,
                             self.scheme, n_jobs=self.n_jobs)
        flat = [c for per_client in sketch_cts for c in per_client]
        flat_pts = self._secret.decrypt_many(flat, n_jobs=self.n_jobs)
        width = 2 * probes.k
        sketches: Dict[int, np.ndarray] = {}
        for i, sub in enumerate(subs):
            pts = flat_pts[i * width:(i + 1) * width]
            s = (decode_shifted_sketch(pts, probes, self.scheme)
                 if self.sketch_mode == "shifted"
                 else decode_sketch(pts, probes, self.scheme))
            sketches[int(sub.client_id)] = add_dp_noise(s, sigma_abs, self._rng)

        self._cur_round = round_idx
        self._cur_subs = seen
        self._cur_probes = probes
        self._cur_transcript = root
        self._cur_nonce = nonce
        self._record(AuditRecord(round_idx, "measure", sorted(seen),
                                 None, root.hex(), time.time(),
                                 f"k={probes.k} dp_sigma={sigma_abs:.4g}",
                                 nonce_hex=nonce.hex()))
        return Measurement(round_idx, sorted(seen), sketches, probes, nonce,
                           root, sigma_abs)


    def open_aggregate(self, round_idx: int, accepted: Sequence[int],
                       weights_q: Sequence[int]) -> np.ndarray:

        round_idx = int(round_idx)
        if self._cur_round != round_idx or self._cur_probes is None:
            raise PolicyViolation(
                f"no measurement phase recorded for round {round_idx}")
        if self.policy.single_decrypt_per_round and round_idx in self._opened_rounds:
            raise PolicyViolation(
                f"round {round_idx} has already been opened; a second opening "
                f"with a different accepted set would enable a difference attack")

        accepted = [int(c) for c in accepted]
        weights_q = [int(w) for w in weights_q]
        if len(accepted) != len(weights_q):
            raise PolicyViolation("accepted/weights length mismatch")
        if len(set(accepted)) != len(accepted):
            raise PolicyViolation("duplicate client in accepted set")
        unknown = set(accepted) - set(self._cur_subs)
        if unknown:
            raise PolicyViolation(f"accepted set contains non-submitters {sorted(unknown)}")
        if any(w < 0 for w in weights_q):
            raise PolicyViolation("negative aggregation weight")

        total_w = sum(weights_q)
        if total_w <= 0:
            raise PolicyViolation("total aggregation weight must be positive")

        n_sub = len(self._cur_subs)
        min_k = math.ceil(self.policy.min_accept_fraction * n_sub)


        effective = [c for c, w in zip(accepted, weights_q) if w > 0]
        if len(effective) < min_k:
            raise PolicyViolation(
                f"participation floor violated: {len(effective)} effective "
                f"contributors < {min_k} required (of {n_sub} submitters)")

        max_share = max(weights_q) / float(total_w)
        if max_share > self.policy.max_client_weight + 1e-12:
            raise PolicyViolation(
                f"weight concentration {max_share:.3f} exceeds cap "
                f"{self.policy.max_client_weight:.3f}: a near-singleton "
                f"aggregate would expose one client's update")

        assert_no_carry(self.scheme, total_w)

        n_blocks = self.scheme.n_blocks(self.dim)
        cts = [[self._cur_subs[c].ciphertexts[b] for c in accepted]
               for b in range(n_blocks)]
        agg_cts = _weighted_products(cts, weights_q, self.pub.nsquare, self.n_jobs)
        pts = self._secret.decrypt_many(agg_cts, n_jobs=self.n_jobs)
        vec = unpack(pts, self.scheme, self.dim, total_w)

        self._opened_rounds.add(round_idx)
        self._last_completed_round = round_idx
        self._record(AuditRecord(round_idx, "open_aggregate", accepted,
                                 weights_q, self._cur_transcript.hex(),
                                 time.time(),
                                 f"n_effective={len(effective)} max_share={max_share:.3f}",
                                 nonce_hex=self._cur_nonce.hex()
                                 if getattr(self, "_cur_nonce", None) else ""))
        return vec


    def audit_table(self) -> List[dict]:
        return [
            {"round": r.round_idx, "phase": r.phase, "clients": r.client_ids,
             "weights": r.weights, "transcript": r.transcript_hex[:16],
             "nonce": r.nonce_hex, "t": r.timestamp, "note": r.note}
            for r in self.audit
        ]


def _weighted_products(per_block: Sequence[Sequence[int]],
                       weights: Sequence[int], nsquare: int,
                       n_jobs: int = -1) -> List[int]:

    if len(per_block) < 32 or n_jobs == 1:
        return [_wp(cs, weights, nsquare) for cs in per_block]
    chunks = np.array_split(np.arange(len(per_block)), min(64, len(per_block)))
    res = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_wp_chunk)([per_block[i] for i in idx], list(weights), nsquare)
        for idx in chunks if len(idx))
    return [c for r in res for c in r]


def _wp(cs: Sequence[int], weights: Sequence[int], nsquare: int) -> int:
    acc = 1
    for c, w in zip(cs, weights):
        if w == 0:
            continue
        acc = (acc * (int(c) if w == 1 else pow(int(c), int(w), nsquare))) % nsquare
    return acc


def _wp_chunk(blocks, weights, nsquare):
    return [_wp(cs, weights, nsquare) for cs in blocks]
