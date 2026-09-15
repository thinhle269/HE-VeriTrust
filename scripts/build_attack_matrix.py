from __future__ import annotations
import argparse
import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ATTACKS = ['ipm', 'sign_flip_scaled', 'block_evasion', 'alie', 'min_max', 'min_sum', 'unresolvable']
DEFENCES = ['fedavg_attack', 'fedmedian', 'trimmed_mean', 'krum', 'bulyan', 'foolsgold', 'veritrust', 'veritrust_mamdani']
PRIMARY = 'veritrust_mamdani'

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/cic_iot/tables')
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    f1: dict = {}
    gate: dict = {}
    seeds_seen = set()
    for atk in ATTACKS:
        ps = ROOT / 'results' / f'atk_{atk}_cic_iot' / 'csv' / 'per_seed.csv'
        if not ps.exists():
            print(f'  [skip] {atk}: not run')
            continue
        d = pd.read_csv(ps).drop_duplicates(['scenario', 'seed'])
        seeds_seen |= set(d.seed.unique())
        for scen, g in d.groupby('scenario'):
            f1.setdefault(scen, {})[atk] = (g.test_macro_f1.mean(), g.test_macro_f1.std())
        pc = ROOT / 'results' / f'atk_{atk}_cic_iot' / 'csv' / 'per_client.csv'
        if pc.exists():
            c = pd.read_csv(pc)
            for scen in ('veritrust', 'veritrust_mamdani'):
                x = c[c.scenario == scen]
                if x.empty:
                    continue
                m, h = (x[x.is_malicious], x[~x.is_malicious])
                gate.setdefault(scen, {})[atk] = (float((~m.accepted).mean()) if len(m) else np.nan, float((~h.accepted).mean()) if len(h) else np.nan)
    if not f1:
        print('no attack runs found')
        return 1
    cols = [a_ for a_ in ATTACKS if any((a_ in v for v in f1.values()))]
    rows = []
    for scen in DEFENCES:
        if scen not in f1:
            continue
        r = {'defence': scen}
        vals = []
        for atk in cols:
            mu, sd = f1[scen].get(atk, (np.nan, np.nan))
            r[atk] = round(mu, 4) if mu == mu else np.nan
            r[f'{atk}_std'] = round(sd, 4) if sd == sd else np.nan
            if mu == mu:
                vals.append((mu, atk))
        if vals:
            worst = min(vals)
            r['worst_macro_f1'] = round(worst[0], 4)
            r['worst_attack'] = worst[1]
        rows.append(r)
    mat = pd.DataFrame(rows)
    und = f1.get('fedavg_attack', {})
    ceiling = np.nan
    ceil_files = glob.glob(str(ROOT / 'results' / 's[0-9]*_cic_iot' / 'csv' / 'per_seed.csv'))
    if ceil_files:
        cf = pd.concat([pd.read_csv(f) for f in ceil_files]).drop_duplicates(['scenario', 'seed'])
        cl = cf[cf.scenario == 'fedavg'].test_macro_f1
        if len(cl):
            ceiling = float(cl.mean())
    dmg = pd.DataFrame([{'attack': atk, 'undefended_macro_f1': round(und.get(atk, (np.nan,))[0], 4), 'clean_ceiling': round(ceiling, 4) if ceiling == ceiling else np.nan, 'damage': round(ceiling - und[atk][0], 4) if atk in und and ceiling == ceiling else np.nan} for atk in cols])
    dmg['is_effective_at_this_scale'] = dmg['damage'] > 0.01
    mat.to_csv(out / 'attack_matrix.csv', index=False)
    dmg.to_csv(out / 'attack_damage.csv', index=False)
    grows = []
    for scen, per in gate.items():
        for atk, (det, fpr) in per.items():
            grows.append({'defence': scen, 'attack': atk, 'detection': round(det, 4), 'honest_rejected': round(fpr, 4)})
    if grows:
        pd.DataFrame(grows).sort_values(['defence', 'attack']).to_csv(out / 'attack_gate_quality.csv', index=False)
    print(f'seeds per cell: {sorted(seeds_seen)}\n')
    print('=== attack damage with no defence ===')
    print(dmg.to_string(index=False))
    print('\n=== macro-F1 matrix ===')
    show = ['defence'] + cols + ['worst_macro_f1', 'worst_attack']
    print(mat[show].to_string(index=False))
    eff = dmg[dmg.is_effective_at_this_scale].attack.tolist()
    print(f'\nOnly {len(eff)} of {len(cols)} attacks move the undefended baseline by more than 0.01 macro-F1: {eff}.')
    if PRIMARY in mat.defence.values:
        p = mat[mat.defence == PRIMARY].iloc[0]
        print(f'{PRIMARY}: worst column {p.worst_macro_f1} ({p.worst_attack}) against a clean ceiling of {ceiling:.4f}.')
    print(f'\ntables -> {out}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
