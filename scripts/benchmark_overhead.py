from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.crypto.authority import AuthorityPolicy, DecryptionAuthority
from src.crypto.packing import PackingScheme, build_scheme, pack
from src.crypto.paillier_backend import ciphertext_bytes, generate_contexts
from src.data.cic_preprocessor import CicIotPreprocessor
from src.data.partition import dirichlet_partition
from src.federated.server import FederatedServer
from src.models.mlp import build_model
from src.trust.anfis import build_engine
from src.utils.common import get_logger, load_config, pick_device, set_global_seed
from run_experiment import build_clients, malicious_mask

def bench_scenario(cfg, scen, split, partition, device, dim, scheme, log, rounds: int, seed: int=42):
    cfg = load_config_like(cfg)
    cfg.federated['rounds'] = int(rounds)
    set_global_seed(seed)
    flags = malicious_mask(cfg, int(cfg.federated.num_clients), scen, np.random.default_rng(seed))
    model = build_model(cfg, split.num_features, split.num_classes).to(device)
    clients, idents = build_clients(cfg, split, partition, device, flags, seed)
    pub = authority = None
    if bool(scen.get('crypto', False)):
        pub, sec = generate_contexts(int(cfg.crypto.key_size), n_jobs=-1)
        sk = cfg.crypto.sketch
        authority = DecryptionAuthority(sec, scheme, AuthorityPolicy.from_config(cfg.crypto.authority), dim=dim, sketch_k=int(sk.get('k', 32)), sketch_density=float(sk.get('density', 0.5)), dp_enabled=bool(sk.dp.get('enabled', False)), dp_relative_sigma=float(sk.dp.get('sigma', 0.05)), clip_norm=float(cfg.federated.get('max_update_norm', 10.0)), n_jobs=-1)
        for cid, ident in idents.items():
            authority.enrol(cid, ident.public_bytes)
    Xv = torch.from_numpy(split.X_val[:8192]).to(device)
    yv = torch.from_numpy(split.y_val[:8192]).to(device)
    srv = FederatedServer(cfg, scen, model, clients, idents, (Xv, yv), (Xv, yv), split.num_classes, device, split.label_names, pub=pub, authority=authority, scheme=scheme, trust_engine=build_engine(cfg.trust, 'cpu', seed) if scen.get('trust') else None, logger=None)
    reps = srv.run()
    reps = reps[1:] if len(reps) > 1 else reps
    m = lambda f: float(np.mean([getattr(r, f) for r in reps]))
    return {'scenario': scen['name'], 'rounds_timed': len(reps), 'train_s': m('t_train'), 'encrypt_s': m('t_encrypt'), 'measure_s': m('t_measure'), 'trust_s': m('t_trust'), 'aggregate_s': m('t_aggregate'), 'total_s': m('t_total'), 'upload_MB': m('bytes_uploaded') / 1000000.0}

def load_config_like(cfg):
    import copy
    return copy.deepcopy(cfg)

def bench_packing(dim: int, key_bits: int, scheme: PackingScheme, log):
    pub, sec = generate_contexts(key_bits, n_jobs=-1)
    v = np.random.default_rng(0).normal(0, 0.01, size=dim)
    ct_bytes = ciphertext_bytes(pub.public_key)
    blocks = pack(v, scheme)
    t0 = time.time()
    pub.encrypt_batch([blocks])
    t_packed = time.time() - t0
    sample = 2000
    q = [int(round(x * scheme.scale)) for x in v[:sample]]
    t0 = time.time()
    pub.encrypt_batch([q])
    t_sample = time.time() - t0
    t_unpacked = t_sample * dim / sample
    return {'d': dim, 'key_bits': key_bits, 'slots_per_ct': scheme.slots, 'packed_ciphertexts': len(blocks), 'unpacked_ciphertexts': dim, 'packed_MB': len(blocks) * ct_bytes / 1000000.0, 'unpacked_MB': dim * ct_bytes / 1000000.0, 'float32_MB': dim * 4 / 1000000.0, 'packed_expansion': len(blocks) * ct_bytes / (dim * 4), 'unpacked_expansion': ct_bytes / 4, 'packed_encrypt_s': t_packed, 'unpacked_encrypt_s_estimated': t_unpacked, 'speedup': t_unpacked / max(t_packed, 1e-09)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'cic_iot.yaml'))
    ap.add_argument('--rounds', type=int, default=5)
    ap.add_argument('--scenarios', nargs='+', default=['fedavg', 'krum', 'he_only', 'veritrust', 'veritrust_mamdani'])
    ap.add_argument('--repeats', type=int, default=3, help='independent repetitions; the slowdown ratio needs them because its denominator is GPU-bound and noisy')
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = ROOT / cfg.paths.results_dir / 'overhead'
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger('overhead', ROOT / cfg.paths.logs_dir, cfg.logging.level)
    device = pick_device(cfg.device)
    split = CicIotPreprocessor(cfg, ROOT).run()
    partition = dirichlet_partition(split.y_train, int(cfg.federated.num_clients), float(cfg.federated.dirichlet_alpha), 42)
    scheme = build_scheme(cfg.crypto)
    probe = build_model(cfg, split.num_features, split.num_classes)
    dim = int(sum((v.numel() for v in probe.state_dict().values())))
    log.info('uncontended benchmark | d=%d | key=%d-bit | slots=%d', dim, cfg.crypto.key_size, scheme.slots)
    scen_by_name = {s['name']: s for s in cfg.experiments.scenarios}
    rows = []
    for rep in range(max(1, int(a.repeats))):
        for name in a.scenarios:
            if name not in scen_by_name:
                continue
            log.info('timing %s (repeat %d/%d) ...', name, rep + 1, a.repeats)
            r = bench_scenario(cfg, dict(scen_by_name[name]), split, partition, device, dim, scheme, log, a.rounds)
            r['repeat'] = rep
            rows.append(r)
            log.info('  %s', r)
    raw = pd.DataFrame(rows)
    raw.round(3).to_csv(out / 'round_breakdown_repeats.csv', index=False)
    num = [c for c in ('train_s', 'encrypt_s', 'measure_s', 'trust_s', 'aggregate_s', 'total_s', 'upload_MB') if c in raw]
    df = raw.groupby('scenario', as_index=False)[num].mean()
    sd = raw.groupby('scenario', as_index=False)[num].std().fillna(0.0)
    df['total_s_std'] = sd['total_s']
    df['crypto_added_s'] = df.get('encrypt_s', 0) + df.get('measure_s', 0) + df.get('aggregate_s', 0)
    df['crypto_added_s_std'] = sd.get('encrypt_s', 0) + sd.get('measure_s', 0) + sd.get('aggregate_s', 0)
    base = df.loc[df.scenario == 'fedavg', 'total_s']
    if len(base):
        b = float(base.iloc[0])
        df['slowdown'] = df['total_s'] / b
        per = raw.pivot_table(index='repeat', columns='scenario', values='total_s')
        if 'fedavg' in per:
            ratios = per.div(per['fedavg'], axis=0)
            df = df.merge(pd.DataFrame({'scenario': ratios.columns, 'slowdown_min': ratios.min().values, 'slowdown_max': ratios.max().values}), on='scenario', how='left')
    df['repeats'] = int(a.repeats)
    df.round(3).to_csv(out / 'round_breakdown.csv', index=False)
    print('\n=== per-round cost, one process, all cores ===')
    print(df.round(3).to_string(index=False))
    pk = bench_packing(dim, int(cfg.crypto.key_size), scheme, log)
    (out / 'packing.json').write_text(json.dumps(pk, indent=2), encoding='utf-8')
    print('\n=== packed vs unpacked Paillier (same vector, same key) ===')
    for k, v in pk.items():
        print(f'  {k:<30} {v}')
    print(f'\noutputs -> {out}')
if __name__ == '__main__':
    main()
