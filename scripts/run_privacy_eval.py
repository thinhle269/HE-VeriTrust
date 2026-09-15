from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.crypto.packing import build_scheme
from src.crypto.sketch import ProbeSet, derive_probes, dp_epsilon, dp_epsilon_composed
from src.data.cic_preprocessor import CicIotPreprocessor
from src.models.mlp import build_model
from src.utils.common import get_logger, load_config, pick_device, set_global_seed

def dense_probe_matrix(probes: ProbeSet) -> np.ndarray:
    M = probes.coeff_matrix()
    blk = np.repeat(np.arange(probes.n_blocks), np.asarray(probes.widths))[:probes.dim]
    V = M[:, blk]
    return V / np.sqrt(np.maximum(probes.probe_widths(), 1.0))[:, None]

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = (np.linalg.norm(a), np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float('nan')
    return float(np.clip(a @ b / (na * nb), -1.0, 1.0))

def real_update(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device, lr: float, epochs: int, batch: int) -> np.ndarray:
    import copy
    local = copy.deepcopy(model).to(device)
    before = torch.cat([p.detach().reshape(-1) for p in local.state_dict().values()])
    opt = torch.optim.Adam(local.parameters(), lr=lr)
    lf = nn.CrossEntropyLoss()
    local.train()
    for _ in range(epochs):
        idx = torch.randperm(X.shape[0], device=device)
        for s in range(0, X.shape[0], batch):
            b = idx[s:s + batch]
            if b.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            lf(local(X[b]), y[b]).backward()
            opt.step()
    after = torch.cat([p.detach().reshape(-1) for p in local.state_dict().values()])
    return (after - before).cpu().numpy().astype(np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'cic_iot.yaml'))
    ap.add_argument('--trials', type=int, default=30)
    ap.add_argument('--k-values', nargs='+', type=int, default=[8, 16, 32, 64, 128, 512])
    ap.add_argument('--clients', type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ROOT / cfg.paths.results_dir / 'privacy'
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger('privacy', ROOT / cfg.paths.logs_dir, cfg.logging.level)
    device = pick_device(cfg.device)
    set_global_seed(cfg.seed)
    split = CicIotPreprocessor(cfg, ROOT).run()
    model = build_model(cfg, split.num_features, split.num_classes).to(device)
    dim = int(sum((v.numel() for v in model.state_dict().values())))
    scheme = build_scheme(cfg.crypto)
    clip = float(cfg.federated.get('max_update_norm', 10.0))
    sigma_rel = float(cfg.crypto.sketch.dp.get('sigma', 0.05))
    delta = float(cfg.crypto.sketch.dp.get('delta', 1e-05))
    rounds = int(cfg.federated.rounds)
    log.info('d=%d  clip=%.2f  dp_sigma_rel=%.3f', dim, clip, sigma_rel)
    Xtr = torch.from_numpy(split.X_train).to(device)
    ytr = torch.from_numpy(split.y_train).to(device)
    rng = np.random.default_rng(cfg.seed)
    shard = 2000
    rows = []
    for t in range(args.trials):
        sel = torch.from_numpy(rng.choice(len(ytr), shard, replace=False)).to(device)
        victim = real_update(model, Xtr[sel], ytr[sel], device, float(cfg.federated.client_lr), 1, int(cfg.federated.local_batch_size))
        n = np.linalg.norm(victim)
        if n > clip:
            victim = victim * (clip / n)
        cidx = torch.from_numpy(rng.choice(len(ytr), shard, replace=False)).to(device)
        control = real_update(model, Xtr[cidx], ytr[cidx], device, float(cfg.federated.client_lr), 1, int(cfg.federated.local_batch_size))
        cn = np.linalg.norm(control)
        if cn > clip:
            control = control * (clip / cn)

        def record(regime, k, rec, dp):
            cv, cc = (cosine(victim, rec), cosine(control, rec))
            rows.append({'regime': regime, 'k': int(k), 'trial': t, 'cosine': cv, 'cosine_control': cc, 'leakage': cv - cc, 'dp': dp, 'rel_l2': float(np.linalg.norm(rec - victim) / np.linalg.norm(victim))})
        rows.append({'regime': 'plaintext_full', 'k': dim, 'trial': t, 'cosine': 1.0, 'cosine_control': cosine(control, victim), 'leakage': 1.0 - cosine(control, victim), 'rel_l2': 0.0, 'dp': False})
        others = []
        for _ in range(args.clients - 1):
            s2 = torch.from_numpy(rng.choice(len(ytr), shard, replace=False)).to(device)
            u = real_update(model, Xtr[s2], ytr[s2], device, float(cfg.federated.client_lr), 1, int(cfg.federated.local_batch_size))
            nn_ = np.linalg.norm(u)
            others.append(u * (clip / nn_) if nn_ > clip else u)
        agg = (victim + sum(others)) / args.clients
        record('aggregate_only', 0, agg, False)
        for k in args.k_values:
            probes = derive_probes(f'trial{t}k{k}'.encode(), dim, scheme, k=min(k, scheme.n_blocks(dim)), density=0.5)
            V = dense_probe_matrix(probes)
            s_clean = V @ victim
            for dp_on in (False, True):
                s = s_clean.copy()
                if dp_on:
                    s = s + rng.normal(0.0, sigma_rel * probes.spectral_norm() * clip, size=s.shape)
                rec = np.linalg.lstsq(V, s, rcond=None)[0]
                record(f'sketch_k{k}', probes.k, rec, dp_on)
        if (t + 1) % 5 == 0:
            log.info('trial %d/%d', t + 1, args.trials)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / 'privacy_raw.csv', index=False)
    summ = raw.groupby(['regime', 'k', 'dp'])[['cosine', 'cosine_control', 'leakage']].agg(['mean', 'std'])
    summ.columns = [f'{a}_{b}' for a, b in summ.columns]
    summ = summ.reset_index()
    lo, hi = ([], [])
    for _, r in summ.iterrows():
        v = raw[(raw.regime == r['regime']) & (raw.dp == r['dp'])]['leakage'].dropna()
        b = np.array([rng.choice(v, len(v), replace=True).mean() for _ in range(2000)]) if len(v) else np.array([np.nan])
        lo.append(np.percentile(b, 2.5))
        hi.append(np.percentile(b, 97.5))
    summ['leakage_ci_lo'], summ['leakage_ci_hi'] = (lo, hi)
    summ['eps_round'] = [dp_epsilon(sigma_rel, delta) if d else np.inf for d in summ['dp']]
    summ['eps_total'] = [dp_epsilon_composed(sigma_rel, rounds, delta) if d else np.inf for d in summ['dp']]
    summ.to_csv(out / 'privacy_summary.csv', index=False)
    log.info('\n%s', summ.round(4).to_string(index=False))
    log.info('DP per-round eps=%.3f, %d-round composed eps=%.1f (delta=%g)', dp_epsilon(sigma_rel, delta), rounds, dp_epsilon_composed(sigma_rel, rounds, delta), delta)
    log.info('outputs -> %s', out)
if __name__ == '__main__':
    main()
