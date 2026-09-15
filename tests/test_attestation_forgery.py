from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.identity import ClientIdentity, Submission
from src.crypto.packing import PackingScheme, pack
from src.crypto.paillier_backend import generate_contexts
from src.crypto.sketch import decode_sketch, derive_probes, project_plaintext, sketch_ciphertexts
from src.trust.features import TrustFeatures, build_features
from src.trust.policy import ZeroTrustPolicy
KEY_BITS = 1024
DIM = 600
N_HONEST, N_MAL = (7, 3)

def _population(rng):
    base = rng.normal(0, 0.02, size=DIM)
    honest = [base + rng.normal(0, 0.004, size=DIM) for _ in range(N_HONEST)]
    mal = [-(base + rng.normal(0, 0.004, size=DIM)) for _ in range(N_MAL)]
    return (honest, mal)

def _claimed_features(honest, mal):
    ref = np.mean(honest, axis=0)
    feats = {}
    for i, u in enumerate(honest):
        c = float(u @ ref / (np.linalg.norm(u) * np.linalg.norm(ref)))
        feats[i] = TrustFeatures(c, 0.0, c)
    for j in range(len(mal)):
        feats[N_HONEST + j] = TrustFeatures(0.95, 0.0, 0.9)
    return feats

def _sketch_features(honest, mal):
    scheme = PackingScheme(KEY_BITS, 20, 20, 1000000.0)
    pub, sec = generate_contexts(KEY_BITS, n_jobs=1)
    probes = derive_probes(b'round-seed', DIM, scheme, k=16, density=0.5)
    sketches = {}
    for cid, u in enumerate(honest + mal):
        cts = pub.encrypt_many(pack(u, scheme))
        pts = sec.decrypt_many(sketch_ciphertexts(pub, cts, probes, scheme), n_jobs=1)
        sketches[cid] = decode_sketch(pts, probes, scheme)
    ref = project_plaintext(np.mean(honest, axis=0), probes)
    return build_features(sketches, ref)

def _gate(feats):
    from _shipped_path import gate_scores
    ids = sorted(feats)
    scores = gate_scores({c: feats[c] for c in ids})
    dec = ZeroTrustPolicy(threshold=0.4, ema_beta=1.0, min_accept_fraction=0.0).evaluate(ids, scores)
    mal_ids = set(range(N_HONEST, N_HONEST + N_MAL))
    return (dec, mal_ids, dict(zip(ids, scores)))

def test_forgery_defeats_self_reported_attestation():
    rng = np.random.default_rng(0)
    honest, mal = _population(rng)
    dec, mal_ids, scores = _gate(_claimed_features(honest, mal))
    admitted = mal_ids - set(dec.rejected)
    assert admitted == mal_ids, f'self-reported attestation should admit every forging attacker; scores={scores}'

def test_forgery_is_inert_against_the_sketch():
    rng = np.random.default_rng(0)
    honest, mal = _population(rng)
    feats = _sketch_features(honest, mal)
    dec, mal_ids, scores = _gate(feats)
    assert mal_ids.issubset(set(dec.rejected)), f'sketch-derived attestation should reject the sign-flipping clients regardless of what they claim; scores={scores}'
    assert not set(range(N_HONEST)) & set(dec.rejected), f'no honest client should be rejected; scores={scores}'

def test_sketch_features_separate_the_populations():
    rng = np.random.default_rng(1)
    honest, mal = _population(rng)
    feats = _sketch_features(honest, mal)
    h = np.mean([feats[c].proj_ref for c in range(N_HONEST)])
    m = np.mean([feats[c].proj_ref for c in range(N_HONEST, N_HONEST + N_MAL)])
    assert h > 0.8, f'honest proj_ref should be near +1, got {h}'
    assert m < -0.8, f'sign-flipped proj_ref should be near -1, got {m}'

def test_client_cannot_choose_its_measured_subspace():
    scheme = PackingScheme(KEY_BITS, 20, 20, 1000000.0)
    from src.crypto.sketch import probe_seed
    root = b'\x11' * 32
    a = derive_probes(probe_seed(3, root, b'\xaa' * 32), DIM, scheme, k=16)
    b = derive_probes(probe_seed(3, root, b'\xbb' * 32), DIM, scheme, k=16)
    assert a.pos != b.pos, 'probe set must depend on the authority nonce'
    c = derive_probes(probe_seed(3, root, b'\xaa' * 32), DIM, scheme, k=16)
    assert a.pos == c.pos and a.neg == c.neg
