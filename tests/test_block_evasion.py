from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.packing import PackingScheme, pack
from src.crypto.paillier_backend import generate_contexts
from src.crypto.shifted_sketch import decode_shifted_sketch, derive_shifted_probes, project_plaintext_shifted, shifted_sketch_ciphertexts_batch
from src.crypto.sketch import derive_probes, project_plaintext
from src.federated.attacks import block_evasion
from src.crypto.sketch_noise import cosine_noise_std
from src.trust.anfis import MamdaniTrust
from src.trust.features import build_features
from src.trust.normalise import EvidenceAccumulator
DIM = 8400
N_HONEST, N_MAL = (7, 3)
HONEST_NORM = 4.7

def _population(seed=0, spread_deg=45.0):
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=DIM)
    truth /= np.linalg.norm(truth)
    het = np.deg2rad(spread_deg)
    H = []
    for _ in range(N_HONEST + N_MAL):
        w = rng.normal(size=DIM)
        w -= w @ truth * truth
        w /= np.linalg.norm(w)
        H.append((np.cos(het) * truth + np.sin(het) * w) * HONEST_NORM)
    return (H, truth)

def _separation(sketches, ref, probes=None):
    feats = build_features(sketches, ref)
    sigma = cosine_noise_std(probes, probes.dim, trials=12) if probes is not None else 0.03
    feats = EvidenceAccumulator(alpha=1.0, z0=2.0, sigma=sigma).update(feats)
    scores = MamdaniTrust().score_many([feats[c] for c in sorted(feats)])
    hon = scores[:N_HONEST]
    mal = scores[N_HONEST:]
    return (float(np.min(hon) - np.max(mal)), hon, mal)

def test_attack_preserves_every_block_sum_and_reverses_the_update():
    H, _ = _population()
    S = 42
    h = H[0]
    x = block_evasion(h, S)
    bs = lambda v: np.pad(v, (0, -len(v) % S)).reshape(-1, S).sum(1)
    assert np.abs(bs(x) - bs(h)).max() < 0.0001, 'block sums must be preserved'
    cos = float(x @ h / (np.linalg.norm(x) * np.linalg.norm(h)))
    assert cos < -0.9, f'the update must be reversed, got cos={cos}'
    ratio = np.linalg.norm(x) / np.linalg.norm(h)
    assert 0.9 < ratio < 1.1, f'norm should be preserved, got {ratio}'

def test_block_constant_probes_are_blind_to_it():
    scheme = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=0)
    probes = derive_probes(b'round', DIM, scheme, k=32, density=0.5)
    H, _ = _population()
    mal = [block_evasion(h, scheme.slots) for h in H[N_HONEST:]]
    sk = {i: project_plaintext(v, probes) for i, v in enumerate(H[:N_HONEST] + mal)}
    ref = project_plaintext(np.mean(H[:N_HONEST], axis=0), probes)
    for i in range(N_MAL):
        a = project_plaintext(H[N_HONEST + i], probes)
        b = sk[N_HONEST + i]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        assert cos > 0.99, f'block-constant sketch should not see it, cos={cos}'
    sep, _, _ = _separation(sk, ref, probes)
    assert sep < 0.2, f'gate should fail to separate, got separation={sep}'

def test_shifted_probes_see_it():
    scheme = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=8)
    probes = derive_shifted_probes(b'round', DIM, scheme, k=32, density=0.5)
    H, _ = _population()
    mal = [block_evasion(h, scheme.slots) for h in H[N_HONEST:]]
    sk = {i: project_plaintext_shifted(v, probes) for i, v in enumerate(H[:N_HONEST] + mal)}
    ref = project_plaintext_shifted(np.mean(H[:N_HONEST], axis=0), probes)
    sep, hon, malsc = _separation(sk, ref, probes)
    assert sep > 0.3, f'shifted sketch must separate the evasive clients; honest={np.round(hon, 3)} evasive={np.round(malsc, 3)}'

@pytest.mark.parametrize('seed', [0, 1, 2])
def test_shifted_probes_see_it_across_seeds(seed):
    scheme = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=8)
    probes = derive_shifted_probes(f'round{seed}'.encode(), DIM, scheme, k=32)
    H, _ = _population(seed=seed)
    mal = [block_evasion(h, scheme.slots) for h in H[N_HONEST:]]
    sk = {i: project_plaintext_shifted(v, probes) for i, v in enumerate(H[:N_HONEST] + mal)}
    ref = project_plaintext_shifted(np.mean(H[:N_HONEST], axis=0), probes)
    sep, _, _ = _separation(sk, ref, probes)
    assert sep > 0.3, f'seed {seed}: separation only {sep:.3f}'

def test_shifted_sketch_homomorphic_matches_plaintext():
    key, dim = (1024, 400)
    scheme = PackingScheme(key, 20, 20, 1000000.0, shift_slots=4)
    pub, sec = generate_contexts(key_size=key, n_jobs=1)
    probes = derive_shifted_probes(b'seed', dim, scheme, k=8, density=0.5)
    v = np.random.default_rng(0).normal(0, 0.01, dim)
    cts = pub.encrypt_many(pack(v, scheme))
    got = decode_shifted_sketch(sec.decrypt_many(shifted_sketch_ciphertexts_batch(pub, [cts], probes, scheme, n_jobs=1)[0], n_jobs=1), probes, scheme)
    want = project_plaintext_shifted(v, probes)
    assert np.abs(got - want).max() < 1e-05

def test_shift_headroom_is_enforced():
    scheme = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=0)
    with pytest.raises(ValueError, match='shift_slots'):
        derive_shifted_probes(b'seed', 1000, scheme, k=4)
    shifted = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=8)
    top_bit = (shifted.max_field_index + 1) * shifted.slot_bits
    assert top_bit <= shifted.modulus_bits, 'a maximal shift must still fit'
