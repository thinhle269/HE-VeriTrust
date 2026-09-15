from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]

def load_many(pattern: str, name: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(ROOT / 'results' / pattern / 'csv' / name))):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def cohens_dz(diff: np.ndarray) -> float:
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else float(diff.mean() / sd)

def holm_bonferroni(pvals):
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='s[0-9]*_cic_iot', help='glob over results/ directories to pool')
    ap.add_argument('--ref', default='veritrust_mamdani')
    ap.add_argument('--metric', default='test_macro_f1')
    ap.add_argument('--out', default='results/cic_iot/tables')
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    per_seed = load_many(args.pattern, 'per_seed.csv')
    if per_seed.empty:
        raise SystemExit(f'no per_seed.csv found under results/{args.pattern}/csv/')
    per_seed = per_seed.drop_duplicates(subset=['scenario', 'seed'])
    cols = [c for c in per_seed.columns if c.startswith('test_')]
    head = per_seed.groupby('scenario')[cols].agg(['mean', 'std', 'count'])
    head.columns = [f'{a}_{b}' for a, b in head.columns]
    head = head.reset_index().sort_values('test_macro_f1_mean', ascending=False)
    head.to_csv(out / 'headline.csv', index=False)
    print('\n=== headline (mean +/- std over seeds) ===')
    show = head[['scenario', 'test_macro_f1_mean', 'test_macro_f1_std', 'test_accuracy_mean', 'test_weighted_f1_mean', 'test_min_class_f1_mean', 'test_macro_f1_count']]
    print(show.round(4).to_string(index=False))
    pc = load_many(args.pattern, 'per_class_f1.csv')
    if not pc.empty:
        piv = pc.groupby(['scenario', 'class'])['f1'].mean().unstack('class').round(4)
        piv.to_csv(out / 'per_class.csv')
        print('\n=== per-class F1 (mean over seeds) ===')
        print(piv.to_string())
    rows = []
    if args.ref in set(per_seed.scenario):
        ref = per_seed[per_seed.scenario == args.ref].set_index('seed')[args.metric]
        for scen in sorted(per_seed.scenario.unique()):
            if scen == args.ref:
                continue
            other = per_seed[per_seed.scenario == scen].set_index('seed')[args.metric]
            seeds = sorted(set(ref.index) & set(other.index))
            if len(seeds) < 2:
                continue
            a, b = (ref.loc[seeds].to_numpy(), other.loc[seeds].to_numpy())
            d = a - b
            try:
                _, wp = stats.wilcoxon(a, b)
            except ValueError:
                wp = np.nan
            _, tp = stats.ttest_rel(a, b)
            rows.append({'reference': args.ref, 'vs': scen, 'metric': args.metric, 'n_seeds': len(seeds), 'ref_mean': round(a.mean(), 4), 'other_mean': round(b.mean(), 4), 'mean_diff': round(d.mean(), 4), 'wins': int((d > 0).sum()), 'ttest_p': round(float(tp), 4), 'wilcoxon_p': round(float(wp), 4) if np.isfinite(wp) else np.nan, 'cohens_dz': round(cohens_dz(d), 3)})
        sig = pd.DataFrame(rows)
        fam = ~sig['vs'].isin(('fedavg', 'centralized', 'veritrust', 'veritrust_mamdani'))
        sig['holm_p'] = np.nan
        if fam.any():
            sig.loc[fam, 'holm_p'] = np.round(holm_bonferroni(sig.loc[fam, 'ttest_p'].to_numpy()), 4)
        sig['significant_holm_05'] = sig['holm_p'] < 0.05
        sig = sig.sort_values('mean_diff', ascending=False)
        sig.to_csv(out / 'significance.csv', index=False)
        n_fam = int(fam.sum())
        print(f'\n=== paired tests vs {args.ref} (Holm-Bonferroni over {n_fam} baseline comparisons) ===')
        print(sig.to_string(index=False))
        print("  cohens_dz is the PAIRED effect size mean(diff)/sd(diff), not Cohen's d.\n  wilcoxon_p cannot go below 0.0625 at n=5 - that is its floor, not a result.")
    rounds = load_many(args.pattern, 'per_round.csv')
    if not rounds.empty:
        r = rounds.copy()
        n_clients = int((r['n_accepted'] + r['n_rejected']).max()) if {'n_accepted', 'n_rejected'}.issubset(r.columns) else None
        if n_clients is None:
            pc = load_many(args.pattern, 'per_client.csv')
            n_clients = int(pc.client_id.nunique()) if not pc.empty else 10
        r['n_honest'] = n_clients - r['n_malicious']
        g = r.groupby('scenario').apply(lambda x: pd.Series({'malicious_rejected_rate': x.n_malicious_rejected.sum() / max(x.n_malicious.sum(), 1), 'honest_rejected_rate': x.n_honest_rejected.sum() / max(x.n_honest.sum(), 1), 'honest_rejected_per_round': x.n_honest_rejected.mean()}), include_groups=False).reset_index()
        g.to_csv(out / 'gate_quality.csv', index=False)
        print('\n=== trust-gate quality (all rounds, all seeds) ===')
        print(g.round(4).to_string(index=False))
        bench = out.parent / 'overhead' / 'round_breakdown.csv'
        if bench.exists():
            o = pd.read_csv(bench).rename(columns={'train_s': 't_train', 'encrypt_s': 't_encrypt', 'measure_s': 't_measure', 'trust_s': 't_trust', 'aggregate_s': 't_aggregate', 'total_s': 't_total', 'slowdown': 'slowdown_vs_fedavg'})
            o['source'] = 'dedicated benchmark (uncontended)'
        else:
            o = rounds.groupby('scenario').agg(t_train=('t_train', 'mean'), t_encrypt=('t_encrypt', 'mean'), t_measure=('t_measure', 'mean'), t_trust=('t_trust', 'mean'), t_aggregate=('t_aggregate', 'mean'), t_total=('t_total', 'mean'), upload_MB=('bytes_uploaded', lambda x: x.mean() / 1000000.0)).reset_index()
            base = o.loc[o.scenario == 'fedavg', 't_total']
            if len(base):
                o['slowdown_vs_fedavg'] = o['t_total'] / float(base.iloc[0])
            o['source'] = 'STUDY TIMINGS - CONTENDED, do not cite; run scripts/benchmark_overhead.py'
        o.round(3).to_csv(out / 'overhead.csv', index=False)
        print('\n=== per-round overhead ===')
        print(o.round(3).to_string(index=False))
    print(f'\ntables -> {out}')
if __name__ == '__main__':
    main()
