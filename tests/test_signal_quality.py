from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.packing import PackingScheme, build_scheme
from src.crypto.sketch import derive_probes, dp_sigma, project_plaintext
from _shipped_path import gate_scores
from src.trust.features import build_features
from src.utils.common import load_config
HONEST_UPDATE_NORM = 4.7
D = 52040
K = 32

def _perturb(rng, dim: int, target_norm: float) -> np.ndarray:
    v = rng.normal(size=dim)
    return v * (float(target_norm) / np.linalg.norm(v))

def _sketches(n_honest=7, n_mal=3, norm=HONEST_UPDATE_NORM, sigma_abs=0.0, seed=0):
    scheme = PackingScheme(2048, 20, 20, 1000000.0)
    probes = derive_probes(b'seed', D, scheme, k=K, density=0.5)
    rng = np.random.default_rng(seed)
    base = rng.normal(size=D)
    base *= norm / np.linalg.norm(base)
    out, truth = ({}, {})
    for i in range(n_honest + n_mal):
        u = base + _perturb(rng, D, norm * 0.25)
        if i >= n_honest:
            u = -u
        s = project_plaintext(u, probes)
        if sigma_abs:
            s = s + rng.normal(0, sigma_abs, size=s.shape)
        out[i] = s
        truth[i] = i >= n_honest
    ref = project_plaintext(base, probes)
    return (out, truth, ref, probes)

def test_sketch_snr_is_usable_without_dp():
    sk, _, _, _ = _sketches()
    norms = [np.linalg.norm(v) for v in sk.values()]
    assert min(norms) > 0.001, f'sketch collapsed to nothing: {norms}'

def test_gate_separates_populations_at_the_operating_point():
    sk, truth, ref, probes = _sketches()
    feats = build_features(sk, ref)
    scores = gate_scores(feats, probes)
    hon = [s for c, s in zip(sorted(feats), scores) if not truth[c]]
    mal = [s for c, s in zip(sorted(feats), scores) if truth[c]]
    assert min(hon) > 0.4, f'honest clients fell below the gate: {hon}'
    assert max(mal) < 0.4, f'malicious clients passed the gate: {mal}'
    assert min(hon) - max(mal) > 0.25, 'separation is too narrow to be robust'

def test_configured_dp_would_or_would_not_swamp_the_signal():
    cfg = load_config(ROOT / 'configs' / 'cic_iot.yaml')
    scheme = build_scheme(cfg.crypto)
    probes = derive_probes(b'seed', D, scheme, k=K, density=0.5)
    clip = float(cfg.federated.max_update_norm)
    sigma_rel = float(cfg.crypto.sketch.dp.sigma)
    enabled = bool(cfg.crypto.sketch.dp.enabled)
    sk, _, _, _ = _sketches()
    signal = float(np.median([np.linalg.norm(v) for v in sk.values()]))
    noise = dp_sigma(probes, clip, sigma_rel) * np.sqrt(probes.k)
    snr = signal / max(noise, 1e-12)
    if enabled:
        assert snr > 1.0, f'DP is enabled but drowns the trust signal: SNR={snr:.4f} (signal={signal:.4f}, noise={noise:.4f}). Lower crypto.sketch.dp.sigma or disable DP.'
    else:
        assert snr < 1.0, f'DP is disabled, but at sigma={sigma_rel} it would leave SNR={snr:.2f} > 1 - i.e. it is now affordable. Re-evaluate the default and the justification in configs/cic_iot.yaml.'

def test_gate_fails_under_dp_noise_as_documented():
    scheme = PackingScheme(2048, 20, 20, 1000000.0)
    probes = derive_probes(b'seed', D, scheme, k=K, density=0.5)
    sigma_abs = dp_sigma(probes, 10.0, 0.05)
    sk, truth, ref, probes = _sketches(sigma_abs=sigma_abs)
    feats = build_features(sk, ref)
    scores = gate_scores(feats, probes)
    hon = [s for c, s in zip(sorted(feats), scores) if not truth[c]]
    mal = [s for c, s in zip(sorted(feats), scores) if truth[c]]
    sep = min(hon) - max(mal)
    assert sep < 0.25, f'separation survived DP noise (gap={sep:.3f}); if this now holds, the DP-off default in configs/cic_iot.yaml is no longer justified'

@pytest.mark.parametrize('k', [8, 16, 32, 64])
def test_separation_holds_across_the_k_sweep(k):
    scheme = PackingScheme(2048, 20, 20, 1000000.0)
    probes = derive_probes(b's', D, scheme, k=k, density=0.5)
    rng = np.random.default_rng(3)
    base = rng.normal(size=D)
    base *= HONEST_UPDATE_NORM / np.linalg.norm(base)
    sk, truth = ({}, {})
    for i in range(10):
        u = base + _perturb(rng, D, HONEST_UPDATE_NORM * 0.25)
        if i >= 7:
            u = -u
        sk[i] = project_plaintext(u, probes)
        truth[i] = i >= 7
    feats = build_features(sk, project_plaintext(base, probes))
    scores = gate_scores(feats, probes)
    hon = [s for c, s in zip(sorted(feats), scores) if not truth[c]]
    mal = [s for c, s in zip(sorted(feats), scores) if truth[c]]
    assert min(hon) > max(mal), f'k={k} cannot separate honest from malicious'
