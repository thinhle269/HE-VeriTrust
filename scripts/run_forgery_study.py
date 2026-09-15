from __future__ import annotations
import argparse
import copy
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.crypto.authority import AuthorityPolicy, DecryptionAuthority
from src.crypto.packing import build_scheme
from src.crypto.paillier_backend import generate_contexts
from src.data.cic_preprocessor import CicIotPreprocessor
from src.data.edgeiiot_preprocessor import EdgeIIoTPreprocessor
from src.data.npz_preprocessor import NpzPreprocessor
from src.data.partition import dirichlet_partition
from src.federated.server import FederatedServer
from src.models.mlp import build_model
from src.trust.anfis import build_engine
from src.utils.common import get_logger, load_config, pick_device, set_global_seed
from run_experiment import build_clients, malicious_mask
CELLS = [('self_reported', False, 'v1 self-reported / honest attestation'), ('self_reported', True, 'v1 self-reported / FORGED attestation'), ('sketch', False, 'v2 sketch-derived / honest attestation'), ('sketch', True, 'v2 sketch-derived / FORGED attestation')]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'cic_iot.yaml'))
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 2024])
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=None)
    ap.add_argument('--engine', default='mamdani', choices=['mamdani', 'anfis'], help='Trust engine used for all four cells.  Default is the label-free Mamdani engine: the learned ANFIS is calibrated on ground-truth-labelled attestations, so it can exploit a residual bias between two otherwise overlapping distributions and makes the self-reported design look more robust than any deployment could reproduce.')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    base = load_config(args.config)
    base.federated['rounds'] = int(args.rounds)
    if args.jobs:
        base.crypto['n_jobs'] = int(args.jobs)
    base.trust['engine'] = str(args.engine)
    out = ROOT / base.paths.results_dir / f'forgery_{args.engine}{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger('forgery', ROOT / base.paths.logs_dir, base.logging.level)
    device = pick_device(base.device)
    source = str(base.data.get('source', 'cic_tree'))
    prep = {'npz': NpzPreprocessor, 'edgeiiot_csv': EdgeIIoTPreprocessor}.get(source, CicIotPreprocessor)
    split = prep(base, ROOT).run()
    Xv = torch.from_numpy(split.X_val).to(device)
    yv = torch.from_numpy(split.y_val).to(device)
    Xt = torch.from_numpy(split.X_test).to(device)
    yt = torch.from_numpy(split.y_test).to(device)
    scheme = build_scheme(base.crypto)
    probe = build_model(base, split.num_features, split.num_classes)
    dim = int(sum((v.numel() for v in probe.state_dict().values())))
    scen_by_name = {s['name']: s for s in base.experiments.scenarios}
    rows = []
    for seed in args.seeds:
        partition = dirichlet_partition(split.y_train, int(base.federated.num_clients), float(base.federated.dirichlet_alpha), seed)
        for name in ('fedavg_attack', 'fedavg'):
            cfg = copy.deepcopy(base)
            scen = dict(scen_by_name[name])
            set_global_seed(seed)
            flags = malicious_mask(cfg, int(cfg.federated.num_clients), scen, np.random.default_rng(seed))
            model = build_model(cfg, split.num_features, split.num_classes).to(device)
            clients, idents = build_clients(cfg, split, partition, device, flags, seed)
            srv = FederatedServer(cfg, scen, model, clients, idents, (Xv, yv), (Xt, yt), split.num_classes, device, split.label_names, logger=None)
            srv.run()
            t = srv.evaluate_final()
            rows.append({'cell': name, 'engine': args.engine, 'attestation': '-', 'forging': False, 'seed': seed, 'macro_f1': t['macro_f1'], 'accuracy': t['accuracy'], 'mal_rejected_rate': np.nan})
            log.info('[%s seed=%d] macro-F1=%.4f', name, seed, t['macro_f1'])
        for source, forging, label in CELLS:
            cfg = copy.deepcopy(base)
            cfg.experiments.attack['forge_attestation'] = bool(forging)
            scen = dict(scen_by_name['veritrust'])
            scen['attestation_source'] = source
            scen['trust_engine'] = str(args.engine)
            scen['crypto'] = source == 'sketch'
            set_global_seed(seed)
            flags = malicious_mask(cfg, int(cfg.federated.num_clients), scen, np.random.default_rng(seed))
            model = build_model(cfg, split.num_features, split.num_classes).to(device)
            clients, idents = build_clients(cfg, split, partition, device, flags, seed)
            pub = authority = None
            if scen['crypto']:
                pub, sec = generate_contexts(int(cfg.crypto.key_size), n_jobs=int(cfg.crypto.get('n_jobs', -1)))
                sk = cfg.crypto.sketch
                authority = DecryptionAuthority(sec, scheme, AuthorityPolicy.from_config(cfg.crypto.authority), dim=dim, sketch_k=int(sk.get('k', 32)), sketch_density=float(sk.get('density', 0.5)), dp_enabled=bool(sk.dp.get('enabled', True)), dp_relative_sigma=float(sk.dp.get('sigma', 0.05)), clip_norm=float(cfg.federated.get('max_update_norm', 10.0)), n_jobs=int(cfg.crypto.get('n_jobs', -1)))
                for cid, ident in idents.items():
                    authority.enrol(cid, ident.public_bytes)
            srv = FederatedServer(cfg, scen, model, clients, idents, (Xv, yv), (Xt, yt), split.num_classes, device, split.label_names, pub=pub, authority=authority, scheme=scheme, trust_engine=build_engine(cfg.trust, 'cpu', seed), logger=None)
            reps = srv.run()
            t = srv.evaluate_final()
            post = [r for r in reps if r.round_idx >= int(cfg.trust.get('calib_rounds', 5))]
            rate = float(np.mean([r.n_malicious_rejected / max(r.n_malicious, 1) for r in post])) if post else float('nan')
            hon = float(np.mean([r.n_honest_rejected for r in post])) if post else float('nan')
            rows.append({'cell': label, 'engine': args.engine, 'attestation': source, 'forging': forging, 'seed': seed, 'macro_f1': t['macro_f1'], 'accuracy': t['accuracy'], 'mal_rejected_rate': rate, 'honest_rejected_per_round': hon})
            log.info('[%s seed=%d] macro-F1=%.4f  malicious rejected=%.0f%%', label, seed, t['macro_f1'], 100 * rate)
            pd.DataFrame(rows).to_csv(out / 'forgery_raw.csv', index=False)
    raw = pd.DataFrame(rows)
    summ = raw.groupby(['cell', 'engine', 'attestation', 'forging'])[['macro_f1', 'accuracy', 'mal_rejected_rate']].agg(['mean', 'std']).reset_index()
    summ.columns = ['_'.join(c).strip('_') if isinstance(c, tuple) else c for c in summ.columns]
    summ.to_csv(out / 'forgery_summary.csv', index=False)
    log.info('\n%s', summ.round(4).to_string(index=False))
    log.info('outputs -> %s', out)
if __name__ == '__main__':
    main()
