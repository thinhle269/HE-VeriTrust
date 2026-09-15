from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.authority import AuthorityPolicy, DecryptionAuthority, PolicyViolation
from src.crypto.identity import ClientIdentity, Submission
from src.crypto.packing import PackingScheme, assert_no_carry, pack, quantise, unpack
from src.crypto.paillier_backend import generate_contexts
from src.crypto.sketch import decode_sketch, derive_probes, dp_epsilon, sketch_ciphertexts
KEY_BITS = 1024
DIM = 240
N_CLIENTS = 6

@pytest.fixture(scope='module', params=['shifted', 'block'])
def env(request):
    mode = request.param
    scheme = PackingScheme(modulus_bits=KEY_BITS, value_bits=20, headroom_bits=20, scale=1000000.0, shift_slots=4 if mode == 'shifted' else 0)
    pub, sec = generate_contexts(key_size=KEY_BITS, n_jobs=1)
    policy = AuthorityPolicy(min_accept_fraction=0.5, max_client_weight=0.5)
    da = DecryptionAuthority(sec, scheme, policy, dim=DIM, sketch_k=8, dp_enabled=False, clip_norm=1.0, n_jobs=1, sketch_mode=mode)
    ids = {}
    for c in range(N_CLIENTS):
        ident = ClientIdentity.generate(c)
        da.enrol(c, ident.public_bytes)
        ids[c] = ident
    return (scheme, pub, sec, da, ids)

def _updates(rng, n=N_CLIENTS, dim=DIM):
    return {c: rng.normal(0, 0.01, size=dim) for c in range(n)}

def _submit(round_idx, updates, scheme, pub, ids):
    subs = []
    for cid, vec in updates.items():
        cts = pub.encrypt_many(pack(vec, scheme))
        subs.append(Submission(cid, round_idx, cts, ids[cid].sign_submission(round_idx, cts)))
    return subs

def test_singleton_aggregate_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(0)
    subs = _submit(0, _updates(rng), scheme, pub, ids)
    da.measure(0, subs)
    weights = [1] + [0] * (N_CLIENTS - 1)
    with pytest.raises(PolicyViolation, match='participation floor'):
        da.open_aggregate(0, list(range(N_CLIENTS)), weights)

def test_near_singleton_weight_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(1)
    subs = _submit(1, _updates(rng), scheme, pub, ids)
    da.measure(1, subs)
    weights = [100000] + [1] * (N_CLIENTS - 1)
    with pytest.raises(PolicyViolation, match='concentration'):
        da.open_aggregate(1, list(range(N_CLIENTS)), weights)

def test_difference_attack_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(2)
    subs = _submit(2, _updates(rng), scheme, pub, ids)
    da.measure(2, subs)
    every = list(range(N_CLIENTS))
    da.open_aggregate(2, every, [1] * N_CLIENTS)
    with pytest.raises(PolicyViolation, match='already been opened'):
        da.open_aggregate(2, every[1:], [1] * (N_CLIENTS - 1))

def test_round_replay_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(3)
    subs = _submit(2, _updates(rng), scheme, pub, ids)
    with pytest.raises(PolicyViolation, match='strictly increase'):
        da.measure(2, subs)

def test_forged_signature_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(4)
    subs = _submit(3, _updates(rng), scheme, pub, ids)
    tampered = pub.encrypt_many(pack(np.ones(DIM) * 0.5, scheme))
    subs[0] = Submission(subs[0].client_id, 3, tampered, subs[0].signature)
    with pytest.raises(PolicyViolation, match='invalid signature'):
        da.measure(3, subs)

def test_unenrolled_client_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(5)
    ghost = ClientIdentity.generate(999)
    cts = pub.encrypt_many(pack(np.zeros(DIM), scheme))
    sub = Submission(999, 3, cts, ghost.sign_submission(3, cts))
    with pytest.raises(PolicyViolation, match='not enrolled'):
        da.measure(3, [sub])

def test_cross_round_replay_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(6)
    old = _submit(1, _updates(rng), scheme, pub, ids)
    with pytest.raises(PolicyViolation, match='replay'):
        da.measure(4, old)

def test_out_of_range_ciphertext_is_refused(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(7)
    subs = _submit(5, _updates(rng), scheme, pub, ids)
    bad = list(subs[0].ciphertexts)
    bad[0] = pub.nsquare + 1
    subs[0] = Submission(subs[0].client_id, 5, bad, ids[subs[0].client_id].sign_submission(5, bad))
    with pytest.raises(PolicyViolation, match='out-of-range'):
        da.measure(5, subs)

def test_sketch_is_bound_to_the_ciphertext(env):
    scheme, pub, sec, da, ids = env
    honest = np.random.default_rng(8).normal(0, 0.01, size=DIM)
    poisoned = -honest

    def sketch_of(vec):
        subs = _submit(0, {0: vec}, scheme, pub, ids)
        probes = derive_probes(b'fixed-probe-seed', DIM, scheme, k=8)
        cts = sketch_ciphertexts(pub, subs[0].ciphertexts, probes, scheme)
        return decode_sketch(sec.decrypt_many(cts, n_jobs=1), probes, scheme)
    s_h, s_p = (sketch_of(honest), sketch_of(poisoned))
    assert np.allclose(s_h, -s_p, atol=1e-06)
    cos = float(np.dot(s_h, s_p) / (np.linalg.norm(s_h) * np.linalg.norm(s_p)))
    assert cos < -0.99, f'sign-flip must be visible in the sketch, got cos={cos}'

def test_packed_aggregate_matches_plaintext(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(9)
    ups = _updates(rng)
    subs = _submit(6, ups, scheme, pub, ids)
    da.measure(6, subs)
    w = [3, 2, 2, 1, 1, 1]
    got = da.open_aggregate(6, list(range(N_CLIENTS)), w)
    want = sum((wi * ups[c] for c, wi in zip(range(N_CLIENTS), w))) / sum(w)
    assert np.max(np.abs(got - want)) < 1e-05

def test_sketch_approximates_true_geometry():
    scheme = PackingScheme(2048, 20, 20, 1000000.0)
    d = 51144
    rng = np.random.default_rng(11)
    probes = derive_probes(b'seed', d, scheme, k=64, density=0.5)
    M = probes.coeff_matrix()
    w = np.asarray(probes.widths)
    blk = np.repeat(np.arange(probes.n_blocks), w)[:d]
    V = M[:, blk] / np.sqrt(probes.probe_widths())[:, None]
    errs = []
    for _ in range(12):
        a = rng.normal(size=d)
        b = rng.normal(size=d)
        ca = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        sa, sb = (V @ a, V @ b)
        cs = float(sa @ sb / (np.linalg.norm(sa) * np.linalg.norm(sb)))
        errs.append(abs(ca - cs))
    assert np.mean(errs) < 0.25, f'sketch cosine error too large: {np.mean(errs)}'

def test_no_carry_bound_is_enforced():
    scheme = PackingScheme(1024, 20, 20, 1000000.0)
    assert_no_carry(scheme, scheme.max_total_weight)
    with pytest.raises(OverflowError, match='carry'):
        assert_no_carry(scheme, scheme.max_total_weight + 1)

def test_quantisation_saturates_rather_than_wraps():
    scheme = PackingScheme(1024, 20, 20, 1000000.0)
    q = quantise(np.array([1e+30, -1e+30, 0.0, np.nan, np.inf]), scheme)
    assert int(q[0]) == scheme.max_abs_q
    assert int(q[1]) == -scheme.max_abs_q
    assert int(q[3]) == 0 and int(q[4]) == 0

def test_dp_epsilon_monotone():
    assert dp_epsilon(0.1) > dp_epsilon(1.0) > dp_epsilon(10.0)

def test_probe_derivation_is_reconstructible_from_the_audit_log(env):
    scheme, pub, sec, da, ids = env
    rng = np.random.default_rng(21)
    r = da._last_completed_round + 1
    subs = _submit(r, _updates(rng), scheme, pub, ids)
    meas = da.measure(r, subs)
    row = [a for a in da.audit_table() if a['phase'] == 'measure' and a['round'] == r][-1]
    assert row['nonce'], 'the audit log must carry the nonce'
    assert row['nonce'] == meas.nonce.hex()
    from src.crypto.identity import transcript_root
    from src.crypto.sketch import probe_seed
    root = transcript_root([subs[i] for i in np.argsort([s.client_id for s in subs])])
    assert root.hex().startswith(row['transcript']), 'logged root must match'
    seed = probe_seed(r, root, bytes.fromhex(row['nonce']))
    from src.crypto.shifted_sketch import derive_shifted_probes
    from src.crypto.sketch import derive_probes
    if da.sketch_mode == 'shifted':
        again = derive_shifted_probes(seed, DIM, scheme, k=da.sketch_k, density=da.sketch_density)
        assert again.blocks == meas.probes.blocks
        assert again.shift == meas.probes.shift
    else:
        again = derive_probes(seed, DIM, scheme, k=da.sketch_k, density=da.sketch_density)
        assert again.pos == meas.probes.pos
        assert again.neg == meas.probes.neg
