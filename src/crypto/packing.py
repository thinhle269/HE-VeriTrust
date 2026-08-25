
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass(frozen=True)
class PackingScheme:


    modulus_bits: int
    value_bits: int
    headroom_bits: int
    scale: float


    shift_slots: int = 0

    @property
    def slot_bits(self) -> int:
        return self.value_bits + self.headroom_bits

    @property
    def slots(self) -> int:

        n = self.modulus_bits // self.slot_bits - 1 - int(self.shift_slots)
        if n < 1:
            raise ValueError(
                f"slot_bits={self.slot_bits} and shift_slots={self.shift_slots} "
                f"leave no room in a {self.modulus_bits}-bit modulus")
        return int(n)

    @property
    def max_field_index(self) -> int:

        return self.slots - 1 + int(self.shift_slots)

    @property
    def offset(self) -> int:
        return 1 << (self.value_bits - 1)

    @property
    def max_abs_q(self) -> int:
        return (1 << (self.value_bits - 1)) - 1

    @property
    def max_total_weight(self) -> int:

        return ((1 << self.slot_bits) - 1) // ((1 << self.value_bits) - 1)

    def n_blocks(self, dim: int) -> int:
        return (int(dim) + self.slots - 1) // self.slots

    def describe(self) -> dict:
        return {
            "modulus_bits": self.modulus_bits,
            "value_bits": self.value_bits,
            "headroom_bits": self.headroom_bits,
            "slot_bits": self.slot_bits,
            "slots_per_ciphertext": self.slots,
            "shift_slots": self.shift_slots,
            "max_total_weight": self.max_total_weight,
            "scale": self.scale,
        }


def assert_no_carry(scheme: PackingScheme, total_weight: int) -> None:

    bound = (1 << scheme.slot_bits)
    used = int(total_weight) * ((1 << scheme.value_bits) - 1)
    if used >= bound:
        raise OverflowError(
            f"packing would carry between slots: total_weight={total_weight} "
            f"needs {used.bit_length()} bits but slot_bits={scheme.slot_bits}. "
            f"Reduce weight_scale_bits or raise headroom_bits "
            f"(max_total_weight={scheme.max_total_weight}).")


def quantise(vec: np.ndarray, scheme: PackingScheme) -> np.ndarray:

    v = np.asarray(vec, dtype=np.float64).reshape(-1)
    v = np.where(np.isfinite(v), v, 0.0)
    q = np.rint(v * scheme.scale)
    return np.clip(q, -scheme.max_abs_q, scheme.max_abs_q).astype(object)


def pack(vec: np.ndarray, scheme: PackingScheme) -> List[int]:

    q = quantise(vec, scheme)
    slots, off = scheme.slots, scheme.offset
    out: List[int] = []
    for start in range(0, q.size, slots):
        block = q[start:start + slots]
        acc = 0
        for t, qi in enumerate(block):
            acc |= (int(qi) + off) << (t * scheme.slot_bits)
        out.append(acc)
    return out


def unpack(plaintexts: Sequence[int], scheme: PackingScheme, dim: int,
           total_weight: int) -> np.ndarray:

    assert_no_carry(scheme, total_weight)
    mask = (1 << scheme.slot_bits) - 1
    off_total = int(total_weight) * scheme.offset
    vals: List[float] = []
    for p in plaintexts:
        p = int(p)
        for _ in range(scheme.slots):
            vals.append(float((p & mask) - off_total))
            p >>= scheme.slot_bits
    arr = np.asarray(vals[:dim], dtype=np.float64)
    return arr / (float(total_weight) * scheme.scale)


def block_widths(dim: int, scheme: PackingScheme) -> np.ndarray:

    n = scheme.n_blocks(dim)
    w = np.full(n, scheme.slots, dtype=np.int64)
    rem = int(dim) % scheme.slots
    if rem:
        w[-1] = rem
    return w


def read_field(plaintext: int, scheme: PackingScheme, index: int) -> int:

    mask = (1 << scheme.slot_bits) - 1
    return (int(plaintext) >> (int(index) * scheme.slot_bits)) & mask


def unpack_slot_sum(plaintext: int, scheme: PackingScheme,
                    n_real_coords: int) -> float:

    mask = (1 << scheme.slot_bits) - 1
    p = int(plaintext)
    total = 0
    for _ in range(scheme.slots):
        total += (p & mask)
        p >>= scheme.slot_bits
    return float(total - int(n_real_coords) * scheme.offset)


def build_scheme(cfg_crypto) -> PackingScheme:

    get = cfg_crypto.get if hasattr(cfg_crypto, "get") else (lambda k, d: d)
    sk = get("sketch", {})
    sk_get = sk.get if hasattr(sk, "get") else (lambda k, d: d)
    return PackingScheme(
        modulus_bits=int(get("key_size", 2048)),
        value_bits=int(get("value_bits", 20)),
        headroom_bits=int(get("headroom_bits", 20)),
        scale=float(get("quantization_scale", 1e6)),
        shift_slots=int(sk_get("shift_slots", 0)),
    )
