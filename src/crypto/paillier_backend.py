
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import joblib
from joblib import Parallel, delayed
from phe import paillier
from phe.util import invert


_PARALLEL_MIN = 64


@dataclass(frozen=True)
class PublicContext:


    public_key: paillier.PaillierPublicKey
    n_jobs: int = -1


    @property
    def nsquare(self) -> int:
        return self.public_key.nsquare

    def encrypt(self, plaintext: int) -> int:
        return self.public_key.raw_encrypt(int(plaintext))

    def encrypt_many(self, plaintexts: Sequence[int]) -> List[int]:
        pts = [int(p) for p in plaintexts]
        if self.n_jobs == 1 or len(pts) < _PARALLEL_MIN:
            return [self.public_key.raw_encrypt(p) for p in pts]
        n_chunks = max(1, min(64, len(pts) // 8))
        size = (len(pts) + n_chunks - 1) // n_chunks
        chunks = [pts[i:i + size] for i in range(0, len(pts), size)]
        res = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(_encrypt_chunk)(c, self.public_key) for c in chunks)
        return [c for r in res for c in r]

    def encrypt_batch(self, groups: Sequence[Sequence[int]]) -> List[List[int]]:

        flat: List[int] = []
        bounds: List[int] = [0]
        for g in groups:
            flat.extend(int(x) for x in g)
            bounds.append(len(flat))
        if not flat:
            return [[] for _ in groups]
        if self.n_jobs == 1 or len(flat) < _PARALLEL_MIN:
            enc = [self.public_key.raw_encrypt(p) for p in flat]
        else:
            workers = max(1, joblib.cpu_count() if self.n_jobs < 0
                          else int(self.n_jobs))
            n_chunks = max(1, min(len(flat), workers * 4))
            size = (len(flat) + n_chunks - 1) // n_chunks
            chunks = [flat[i:i + size] for i in range(0, len(flat), size)]
            res = Parallel(n_jobs=self.n_jobs, backend="loky")(
                delayed(_encrypt_chunk)(c, self.public_key) for c in chunks)
            enc = [c for r in res for c in r]
        return [enc[bounds[i]:bounds[i + 1]] for i in range(len(groups))]

    def add(self, c1: int, c2: int) -> int:

        return (int(c1) * int(c2)) % self.nsquare

    def add_many(self, cts: Sequence[int]) -> int:
        acc = 1
        ns = self.nsquare
        for c in cts:
            acc = (acc * int(c)) % ns
        return acc

    def mul_scalar(self, c: int, k: int) -> int:

        k = int(k)
        if k < 0:
            raise ValueError("use negate() for negative scalars")
        if k == 0:


            return self.public_key.raw_encrypt(0)
        if k == 1:
            return int(c)
        return pow(int(c), k, self.nsquare)

    def negate(self, c: int) -> int:

        return invert(int(c), self.nsquare)

    def encrypt_zero(self) -> int:
        return self.public_key.raw_encrypt(0)


@dataclass(frozen=True)
class SecretContext:


    private_key: paillier.PaillierPrivateKey
    public: PublicContext

    def decrypt(self, ciphertext: int) -> int:
        return self.private_key.raw_decrypt(int(ciphertext))

    def decrypt_many(self, cts: Sequence[int], n_jobs: int = -1) -> List[int]:
        cts = [int(c) for c in cts]
        if n_jobs == 1 or len(cts) < _PARALLEL_MIN:
            return [self.private_key.raw_decrypt(c) for c in cts]
        workers = max(1, joblib.cpu_count() if n_jobs < 0 else int(n_jobs))
        n_chunks = max(1, min(len(cts), workers * 4))
        size = (len(cts) + n_chunks - 1) // n_chunks
        chunks = [cts[i:i + size] for i in range(0, len(cts), size)]
        res = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_decrypt_chunk)(c, self.private_key) for c in chunks)
        return [p for r in res for p in r]


def _encrypt_chunk(pts, pk):
    return [pk.raw_encrypt(p) for p in pts]


def _decrypt_chunk(cts, sk):
    return [sk.raw_decrypt(c) for c in cts]


def generate_contexts(key_size: int = 2048, n_jobs: int = -1):

    pk, sk = paillier.generate_paillier_keypair(n_length=int(key_size))
    pub = PublicContext(pk, n_jobs=n_jobs)
    return pub, SecretContext(sk, pub)


def ciphertext_bytes(public_key: paillier.PaillierPublicKey) -> int:

    return (public_key.nsquare.bit_length() + 7) // 8
