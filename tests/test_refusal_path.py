from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.authority import AuthorityPolicy, DecryptionAuthority, PolicyViolation
from src.crypto.identity import ClientIdentity, Submission
from src.crypto.packing import PackingScheme, pack
from src.crypto.paillier_backend import generate_contexts
from src.trust.policy import quantise_weights
KEY_BITS = 1024
DIM = 120
N = 6

@pytest.fixture(scope='module')
def env():
    scheme = PackingScheme(modulus_bits=KEY_BITS, value_bits=20, headroom_bits=20, scale=1000000.0, shift_slots=4)
    pub, sec = generate_contexts(key_size=KEY_BITS, n_jobs=1)
    policy = AuthorityPolicy(min_accept_fraction=0.5, max_client_weight=0.5)
    da = DecryptionAuthority(sec, scheme, policy, dim=DIM, sketch_k=4, dp_enabled=False, clip_norm=1.0, n_jobs=1, sketch_mode='shifted')
    ids = {}
    for c in range(N):
        ident = ClientIdentity.generate(c)
        da.enrol(c, ident.public_bytes)
        ids[c] = ident
    return (scheme, pub, da, ids)

def _subs(round_idx, scheme, pub, ids, rng):
    out = []
    for cid in range(N):
        cts = pub.encrypt_many(pack(rng.normal(0, 0.01, size=DIM), scheme))
        out.append(Submission(cid, round_idx, cts, ids[cid].sign_submission(round_idx, cts)))
    return out

def _phases(da):
    return [row['phase'] for row in da.audit_table()]

def test_refused_measurement_is_in_the_audit_log(env):
    scheme, pub, da, ids = env
    rng = np.random.default_rng(0)
    da.measure(0, _subs(0, scheme, pub, ids, rng))
    da.open_aggregate(0, list(range(N)), [1] * N)
    with pytest.raises(PolicyViolation):
        da.measure(0, _subs(0, scheme, pub, ids, rng))
    assert _phases(da)[-1] == 'refuse_measure'
    assert 'refused' in da.audit_table()[-1]['note']

def test_refused_opening_is_in_the_audit_log(env):
    scheme, pub, da, ids = env
    rng = np.random.default_rng(1)
    da.measure(1, _subs(1, scheme, pub, ids, rng))
    with pytest.raises(PolicyViolation):
        da.open_aggregate(1, list(range(N)), [100, 1, 1, 1, 1, 1])
    assert _phases(da)[-1] == 'refuse_open'

def test_carry_violation_is_a_policy_refusal_not_a_crash(env):
    scheme, pub, da, ids = env
    rng = np.random.default_rng(2)
    da.measure(2, _subs(2, scheme, pub, ids, rng))
    w = scheme.max_total_weight // 3 + 1
    weights = [w, w, w, 0, 0, 0]
    with pytest.raises(PolicyViolation):
        da.open_aggregate(2, list(range(N)), weights)
    assert _phases(da)[-1] == 'refuse_open'

def test_every_accepted_client_gets_at_least_one_unit():
    shares = {0: 0.5, 1: 0.4999, 2: 0.0001}
    q = quantise_weights(shares, [0, 1, 2], scale_bits=10, max_share=0.5)
    assert min(q) >= 1
    assert max(q) / sum(q) <= 0.5 + 1e-09
