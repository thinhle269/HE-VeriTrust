
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey, Ed25519PublicKey)

_DOMAIN = b"HE-VeriTrust/v2/submission"


def _canonical(client_id: int, round_idx: int,
               ciphertexts: Sequence[int]) -> bytes:

    h = hashlib.sha256()
    h.update(_DOMAIN)
    h.update(int(client_id).to_bytes(8, "big"))
    h.update(int(round_idx).to_bytes(8, "big"))
    h.update(len(ciphertexts).to_bytes(8, "big"))
    for c in ciphertexts:
        b = int(c).to_bytes((int(c).bit_length() + 7) // 8 or 1, "big")
        h.update(len(b).to_bytes(4, "big"))
        h.update(b)
    return h.digest()


@dataclass
class ClientIdentity:


    client_id: int
    _sk: Ed25519PrivateKey

    @classmethod
    def generate(cls, client_id: int) -> "ClientIdentity":
        return cls(int(client_id), Ed25519PrivateKey.generate())

    @property
    def public_bytes(self) -> bytes:
        return self._sk.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw)

    def sign_submission(self, round_idx: int,
                        ciphertexts: Sequence[int]) -> bytes:
        return self._sk.sign(_canonical(self.client_id, round_idx, ciphertexts))


@dataclass
class Submission:


    client_id: int
    round_idx: int
    ciphertexts: List[int]
    signature: bytes
    n_samples: int = 0

    def digest(self) -> bytes:
        return _canonical(self.client_id, self.round_idx, self.ciphertexts)


class Registry:


    def __init__(self) -> None:
        self._keys: Dict[int, Ed25519PublicKey] = {}

    def enrol(self, client_id: int, public_bytes: bytes) -> None:
        cid = int(client_id)
        if cid in self._keys:
            raise ValueError(f"client {cid} already enrolled")
        self._keys[cid] = Ed25519PublicKey.from_public_bytes(public_bytes)

    def __contains__(self, client_id: int) -> bool:
        return int(client_id) in self._keys

    def __len__(self) -> int:
        return len(self._keys)

    def verify(self, sub: Submission) -> bool:
        pk = self._keys.get(int(sub.client_id))
        if pk is None:
            return False
        try:
            pk.verify(sub.signature, sub.digest())
            return True
        except InvalidSignature:
            return False


def transcript_root(submissions: Sequence[Submission]) -> bytes:

    h = hashlib.sha256()
    h.update(b"HE-VeriTrust/v2/transcript")
    for sub in sorted(submissions, key=lambda s: int(s.client_id)):
        d = sub.digest()
        h.update(int(sub.client_id).to_bytes(8, "big"))
        h.update(d)
    return h.digest()


def fresh_nonce(n_bytes: int = 32) -> bytes:

    return os.urandom(n_bytes)
