from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.packing import PackingScheme
from src.crypto.shifted_sketch import derive_shifted_probes, project_plaintext_shifted
from src.crypto.sketch_noise import cosine_noise_std
from src.federated.attacks import unresolvable
from src.trust.anfis import MamdaniTrust
from src.trust.features import TrustFeatures, build_features
from src.trust.normalise import EvidenceAccumulator, ema_effective_rounds
TAU = 0.4
NH, NM = (7, 3)

def _decayed_round(rng, signal, sigma):
    true = np.array([signal] * NH + [-signal] * NM)
    return true + rng.normal(0, sigma, NH + NM)

def _score(feats, domain='normalised'):
    return MamdaniTrust(domain=domain).score_many(feats)

def test_ema_variance_reduction_is_what_we_claim():
    assert ema_effective_rounds(0.4) == pytest.approx(4.0)
    acc = EvidenceAccumulator(alpha=0.4, sigma=0.03)
    assert acc.sigma_eff == pytest.approx(0.015, rel=1e-06)

def test_raw_features_reject_honest_clients_once_the_model_converges():
    rng = np.random.default_rng(0)
    fp = 0
    for r in range(30):
        signal = 0.3 * np.exp(-r / 9.0) + 0.02
        obs = _decayed_round(rng, signal, 0.03)
        s = _score([TrustFeatures(o, 0.0, 0.0) for o in obs], domain='raw')
        fp += int((s[:NH] < TAU).sum())
    assert fp > 0.5 * NH * 30, f'the raw-feature gate is expected to reject most honest client-rounds; it rejected {fp}/{NH * 30}'

def test_normalised_evidence_holds_the_gate_as_the_signal_decays():
    rng = np.random.default_rng(0)
    acc = EvidenceAccumulator(alpha=0.4, z0=2.0, sigma=0.03)
    fp = miss = 0
    for r in range(30):
        signal = 0.3 * np.exp(-r / 9.0) + 0.02
        obs = _decayed_round(rng, signal, 0.03)
        feats = {i: TrustFeatures(o, 0.0, 0.0) for i, o in enumerate(obs)}
        ev = acc.update(feats)
        s = _score([ev[i] for i in range(NH + NM)])
        fp += int((s[:NH] < TAU).sum())
        miss += int((s[NH:] >= TAU).sum())
    assert fp == 0, f'{fp}/{NH * 30} honest client-rounds falsely rejected'
    assert miss <= 0.1 * NM * 30, f'{miss}/{NM * 30} attacker client-rounds missed'

def test_all_honest_population_is_never_rejected():
    rng = np.random.default_rng(1)
    acc = EvidenceAccumulator(alpha=0.4, z0=2.0, sigma=0.03)
    for r in range(30):
        signal = 0.3 * np.exp(-r / 9.0) + 0.02
        obs = signal + rng.normal(0, 0.03, NH)
        feats = {i: TrustFeatures(o, 0.0, 0.0) for i, o in enumerate(obs)}
        ev = acc.update(feats)
        s = _score([ev[i] for i in range(NH)])
        assert (s >= TAU).all(), f'round {r}: false rejection, scores={np.round(s, 3)}'

def test_dropping_out_does_not_launder_a_bad_history():
    acc = EvidenceAccumulator(alpha=0.4, z0=2.0, sigma=0.03)
    bad = {0: TrustFeatures(-0.5, 0.0, 0.0), 1: TrustFeatures(0.5, 0.0, 0.0)}
    for _ in range(5):
        acc.update(bad)
    after_absence = acc.update({0: TrustFeatures(0.0, 0.0, 0.0)})[0]
    assert after_absence.proj_ref < 0, f'history must persist across missed rounds, got {after_absence.proj_ref}'

def test_sigma_is_a_property_of_the_operator_and_is_stable():
    scheme = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=8)
    probes = derive_shifted_probes(b'r', 4000, scheme, k=16, density=0.5)
    a = cosine_noise_std(probes, 4000, trials=16, seed=0)
    b = cosine_noise_std(probes, 4000, trials=16, seed=0)
    assert a == b, 'must be cached / deterministic'
    assert 0.0 < a < 0.5, f'implausible noise estimate {a}'

def test_unresolvable_attack_evades_projection_but_not_the_norm_check():
    dim = 4000
    scheme = PackingScheme(2048, 20, 20, 1000000.0, shift_slots=8)
    probes = derive_shifted_probes(b'r', dim, scheme, k=16, density=0.5)
    rng = np.random.default_rng(0)
    truth = rng.normal(size=dim)
    truth /= np.linalg.norm(truth)
    H = []
    for _ in range(NH):
        w = rng.normal(size=dim)
        w -= w @ truth * truth
        w /= np.linalg.norm(w)
        H.append((0.7 * truth + np.sqrt(1 - 0.49) * w) * 4.7)
    ref = np.mean(H, axis=0)
    med = float(np.median([np.linalg.norm(h) for h in H]))
    quiet = [unresolvable(ref, med, seed=i) for i in range(NM)]
    loud = [unresolvable(ref, 3 * med, seed=i) for i in range(NM)]
    for label, M, expect_norm_flag in (('matched', quiet, False), ('3x', loud, True)):
        sk = {i: project_plaintext_shifted(v, probes) for i, v in enumerate(H + M)}
        f = build_features(sk, project_plaintext_shifted(ref, probes))
        proj = np.mean([f[i].proj_ref for i in range(NH, NH + NM)])
        nr = np.mean([f[i].norm_ratio for i in range(NH, NH + NM)])
        assert abs(proj) < 0.15, f'{label}: the attack is supposed to be unresolvable in proj_ref, got {proj}'
        if expect_norm_flag:
            assert nr > 0.5, f'{label}: norm_ratio must flag it, got {nr}'
        else:
            assert abs(nr) < 0.3, f'{label}: norm-matched, got {nr}'
