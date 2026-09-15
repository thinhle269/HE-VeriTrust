from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DIR_RE = re.compile('abl_(?P<param>.+)_(?P<value>[^_]+)_cic_iot$')

def collect() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(str(ROOT / 'results' / 'abl_*_cic_iot'))):
        m = DIR_RE.search(Path(d).name)
        if not m:
            continue
        param, raw = (m.group('param'), m.group('value'))
        try:
            value = float(raw)
        except ValueError:
            value = raw
        ps = Path(d) / 'csv' / 'per_seed.csv'
        pr = Path(d) / 'csv' / 'per_round.csv'
        if not ps.exists():
            continue
        seed_df = pd.read_csv(ps)
        round_df = pd.read_csv(pr) if pr.exists() else pd.DataFrame()
        for scen, g in seed_df.groupby('scenario'):
            rec = {'param': param, 'value': value, 'scenario': scen, 'macro_f1': g.test_macro_f1.mean(), 'macro_f1_std': g.test_macro_f1.std(), 'min_class_f1': g.test_min_class_f1.mean(), 'n_seeds': len(g)}
            if not round_df.empty:
                r = round_df[round_df.scenario == scen]
                if len(r) and 'n_malicious' in r.columns:
                    n_cli = int((r.n_accepted + r.n_rejected).max()) if {'n_accepted', 'n_rejected'}.issubset(r.columns) else 10
                    mal = r.n_malicious.sum()
                    hon = (n_cli - r.n_malicious).sum()
                    rec['malicious_rejected'] = r.n_malicious_rejected.sum() / mal if mal else np.nan
                    rec['honest_rejected'] = r.n_honest_rejected.sum() / hon if hon else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/cic_iot/tables')
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        raise SystemExit('no ablation directories found under results/abl_*_cic_iot')
    df = df.sort_values(['param', 'scenario', 'value'])
    df.round(4).to_csv(out / 'ablation_full.csv', index=False)
    cols = ['value', 'scenario', 'macro_f1', 'macro_f1_std', 'malicious_rejected', 'honest_rejected', 'min_class_f1']
    for param, g in df.groupby('param'):
        print(f'\n=== {param} ===')
        print(g[[c for c in cols if c in g.columns]].round(4).to_string(index=False))
        if 'malicious_rejected' in g.columns and g.scenario.nunique() == 1:
            f1 = g.macro_f1.to_numpy()
            det = g.malicious_rejected.to_numpy(dtype=float)
            if np.isfinite(det).sum() > 1:
                f1_span = float(np.nanmax(f1) - np.nanmin(f1))
                det_span = float(np.nanmax(det) - np.nanmin(det))
                if det_span > 0.2 and f1_span < 0.02:
                    print(f'  note: detection varies by {det_span:.2f} across this sweep while macro-F1 varies by only {f1_span:.3f} - the end-task metric does not resolve this parameter.')
    print(f"\ntables -> {out / 'ablation_full.csv'}")
if __name__ == '__main__':
    main()
