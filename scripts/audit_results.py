from __future__ import annotations
import glob
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OK, BAD, WARN = ('PASS', 'FAIL', 'WARN')
_results = []

def check(name: str, status: str, detail: str='') -> None:
    _results.append((status, name))
    mark = {OK: '  [PASS]', BAD: '  [FAIL]', WARN: '  [WARN]'}[status]
    print(f'{mark} {name}')
    for line in (detail or '').splitlines():
        if line.strip():
            print(f'         {line}')

def load(pattern: str, name: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / 'results' / pattern / 'csv' / name)))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()

def audit_data_hygiene():
    print('\n=== 1. DATA HYGIENE ===')
    metas = sorted(glob.glob(str(ROOT / 'data' / 'processed_cic' / '*.meta.json')))
    if not metas:
        return check('preprocessing metadata present', BAD, 'no cache metadata found')
    m = json.loads(Path(metas[0]).read_text(encoding='utf-8'))
    tr, va, te = (m['shapes'][k][0] for k in ('train', 'val', 'test'))
    tot = tr + va + te
    ratios = (tr / tot, va / tot, te / tot)
    check('split is 70/15/15', OK if abs(ratios[0] - 0.7) < 0.01 else BAD, f'train={tr} val={va} test={te}  ->  {ratios[0]:.3f}/{ratios[1]:.3f}/{ratios[2]:.3f}')
    src = (ROOT / 'src' / 'data' / 'cic_preprocessor.py').read_text(encoding='utf-8')
    i_split = src.index('train_test_split(idx')
    i_feat = src.index('self._select_features(X[tr]')
    i_scale = src.index('scaler.fit_transform')
    check('feature selection fitted on train only, after the split', OK if i_split < i_feat and 'X[tr]' in src[i_feat:i_feat + 40] else BAD, 'select_features receives X[tr]; both it and the scaler run after train_test_split')
    check('scaler fitted on train only', OK if i_split < i_scale else BAD)
    counts = m.get('train_class_counts', {})
    if counts:
        lo, hi = (min(counts.values()), max(counts.values()))
        check('class imbalance is moderate', OK if hi / lo < 5 else WARN, f'ratio {hi / lo:.1f}:1   min={lo} ({min(counts, key=counts.get)})  max={hi}')

def audit_protocol_fairness():
    print('\n=== 2. IS THE COMPARISON FAIR? ===')
    pc = load('s[0-9]*_cic_iot', 'per_client.csv')
    if pc.empty:
        return check('per-client records present', BAD)
    bad = []
    for seed, g in pc.groupby('seed'):
        sets = {s: frozenset(x[x.is_malicious].client_id.unique()) for s, x in g.groupby('scenario')}
        attacked = {s: v for s, v in sets.items() if v}
        if len(set(attacked.values())) > 1:
            bad.append((seed, attacked))
    check('identical Byzantine client set across scenarios within a seed', OK if not bad else BAD, '' if not bad else f'differs in seeds: {[b[0] for b in bad]}')
    per_seed_sets = {seed: frozenset(g[g.is_malicious].client_id.unique()) for seed, g in pc.groupby('seed')}
    n_distinct = len(set(per_seed_sets.values()))
    check('Byzantine set varies across seeds', OK if n_distinct > 1 else BAD, f'{n_distinct} distinct sets over {len(per_seed_sets)} seeds: ' + ', '.join((f'{k}:{sorted(v)}' for k, v in per_seed_sets.items())))
    sizes = pc.groupby(['seed', 'client_id']).n_samples.first().unstack()
    check('Dirichlet partition is reseeded per seed', OK if sizes.nunique().sum() > len(sizes.columns) else BAD, f'client-0 shard sizes by seed: {sizes.iloc[:, 0].to_dict()}')

def audit_crypto_consistency():
    print('\n=== 3. DOES THE CRYPTO PATH COMPUTE THE RIGHT THING? ===')
    ps = load('s[0-9]*_cic_iot', 'per_seed.csv').drop_duplicates(['scenario', 'seed'])
    if ps.empty:
        return check('per-seed records present', BAD)
    a = ps[ps.scenario == 'he_only'].set_index('seed').test_macro_f1
    b = ps[ps.scenario == 'fedavg_attack'].set_index('seed').test_macro_f1
    seeds = sorted(set(a.index) & set(b.index))
    if not seeds:
        return check('he_only vs fedavg_attack comparable', BAD)
    d = (a.loc[seeds] - b.loc[seeds]).abs()
    check('packed-HE aggregate matches plaintext FedAvg', OK if d.max() < 0.02 else BAD, f'|he_only - fedavg_attack| per seed: mean={d.mean():.4f} max={d.max():.4f}')
    _, p = stats.ttest_rel(a.loc[seeds], b.loc[seeds])
    check('...and the residual is not systematic', OK if p > 0.05 else WARN, f'paired t-test p={p:.3f} (a systematic offset would indicate bias, not rounding)')

def audit_statistics():
    print('\n=== 4. STATISTICS ===')
    ps = load('s[0-9]*_cic_iot', 'per_seed.csv').drop_duplicates(['scenario', 'seed'])
    ref = 'veritrust'
    r = ps[ps.scenario == ref].set_index('seed').test_macro_f1
    rows = []
    for scen in sorted(ps.scenario.unique()):
        if scen in (ref, 'centralized', 'fedavg', 'veritrust_mamdani'):
            continue
        o = ps[ps.scenario == scen].set_index('seed').test_macro_f1
        s = sorted(set(r.index) & set(o.index))
        if len(s) < 3:
            continue
        diff = r.loc[s].to_numpy() - o.loc[s].to_numpy()
        _, p = stats.ttest_rel(r.loc[s], o.loc[s])
        rows.append({'vs': scen, 'p': float(p), 'd': float(diff.mean() / diff.std(ddof=1)), 'wins': int((diff > 0).sum()), 'n': len(s)})
    df = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
    mtests = len(df)
    df['holm_threshold'] = [0.05 / (mtests - i) for i in range(mtests)]
    df['survives_holm'] = df.p < df.holm_threshold
    print(df.round(4).to_string(index=False))
    check('all comparisons survive Holm-Bonferroni correction', OK if df.survives_holm.all() else WARN, f'{int(df.survives_holm.sum())}/{mtests} survive; largest p = {df.p.max():.4f} against threshold {df.holm_threshold.iloc[-1]:.4f}')
    n_seeds = r.shape[0]
    check('seed count', WARN if n_seeds < 10 else OK, f'{n_seeds} seeds. The Wilcoxon floor at n=5 is p=0.0625, so the paired t-test and effect size carry the argument; state this.')
    sd = ps.groupby('scenario').test_macro_f1.std()
    tiny = sd[sd < 0.002]
    check('no scenario has a suspiciously tiny spread', OK if tiny.empty else WARN, '' if tiny.empty else f'std < 0.002 for: {tiny.round(5).to_dict()}')

def audit_gate_and_claims():
    print('\n=== 5. DO THE HEADLINE CLAIMS SURVIVE THEIR OWN DATA? ===')
    pr = load('s[0-9]*_cic_iot', 'per_round.csv')
    if pr.empty:
        return check('per-round records present', BAD)
    n = int((pr.n_accepted + pr.n_rejected).max())
    for scen in ('veritrust', 'veritrust_mamdani'):
        g = pr[pr.scenario == scen]
        if g.empty:
            continue
        det = g.n_malicious_rejected.sum() / max(g.n_malicious.sum(), 1)
        fpr = g.n_honest_rejected.sum() / max((n - g.n_malicious).sum(), 1)
        check(f'{scen}: gate detection >= 0.85 and false-reject <= 0.10', OK if det >= 0.85 and fpr <= 0.1 else BAD, f'detection={det:.4f}  false-reject={fpr:.4f}  ({g.n_honest_rejected.mean():.2f} honest clients per round)')
        late = g[g.round_idx >= g.round_idx.max() * 0.6] if 'round_idx' in g else g
        if not late.empty:
            l_det = late.n_malicious_rejected.sum() / max(late.n_malicious.sum(), 1)
            check(f'{scen}: detection still holds in the last 40% of rounds', OK if l_det >= 0.85 else BAD, f'late detection={l_det:.4f}')
    pcl = load('s[0-9]*_cic_iot', 'per_client.csv')
    if not pcl.empty and 'is_malicious' in pcl:
        for scen in ('veritrust', 'veritrust_mamdani'):
            hc = pcl[(pcl.scenario == scen) & ~pcl.is_malicious]
            if hc.empty:
                continue
            per_seed = [(g.client_id.nunique(), g[~g.accepted].client_id.nunique()) for _, g in hc.groupby('seed')]
            n_hon = max((n for n, _ in per_seed))
            spread = max((r for _, r in per_seed))
            check(f'{scen}: honest rejections stay concentrated, not population-wide', OK if spread <= max(1, n_hon // 3) else BAD, f'{spread} of {n_hon} honest clients rejected at least once (population-wide rejection means the gate lost its scale, not that everyone turned malicious)')
    ps = load('s[0-9]*_cic_iot', 'per_seed.csv')
    if not ps.empty and 'test_macro_f1' in ps:
        base = ps[ps.scenario == 'fedavg'].test_macro_f1.mean()
        for scen in ('veritrust', 'veritrust_mamdani'):
            v = ps[ps.scenario == scen].test_macro_f1
            if v.empty or not np.isfinite(base):
                continue
            check(f'{scen}: false rejections cost no accuracy vs honest-only FedAvg', OK if v.mean() >= base - 0.01 else BAD, f'{scen} macro-F1={v.mean():.4f} vs fedavg (no attack, no gate) {base:.4f}')
    tau = 0.4
    pcl0 = load('s[0-9]*_cic_iot', 'per_client.csv')
    for scen in ('veritrust', 'veritrust_mamdani'):
        x = pcl0[pcl0.scenario == scen] if not pcl0.empty else pd.DataFrame()
        if x.empty or 'trust_smoothed' not in x:
            continue
        forced = x[x.accepted & (x.trust_smoothed < tau)]
        n_mal = int(forced.is_malicious.sum())
        rounds = x.groupby(['seed', 'round_idx']).ngroups
        check(f'{scen}: participation floor never re-admitted an attacker', OK if n_mal == 0 else WARN, f'{len(forced)} acceptances below tau={tau} over {rounds} rounds, {n_mal} of them malicious. The floor is a liveness guarantee bought with safety; state the exchange rate rather than implying it never fired.')
    abl = ROOT / 'results' / 'cic_iot' / 'tables' / 'ablation_full.csv'
    if abl.exists():
        a = pd.read_csv(abl)
        vnum = pd.to_numeric(a['value'], errors='coerce')
        z = a[(a.param.astype(str) == 'malicious_fraction') & (vnum == 0.0)]
        if not z.empty:
            piv = z.set_index('scenario').macro_f1
            if {'veritrust', 'fedavg_attack'}.issubset(piv.index):
                delta = piv['veritrust'] - piv['fedavg_attack']
                check('no accuracy cost on a clean (unattacked) federation', OK if delta > -0.01 else WARN, f"at f=0.0: veritrust={piv['veritrust']:.4f} vs plain FedAvg={piv['fedavg_attack']:.4f} ({delta:+.4f})")

def audit_privacy_claims():
    print('\n=== 6. PRIVACY NUMBERS ===')
    p = ROOT / 'results' / 'cic_iot' / 'privacy' / 'privacy_summary.csv'
    if not p.exists():
        return check('privacy summary present', BAD)
    d = pd.read_csv(p)
    if 'leakage_mean' not in d.columns:
        return check('privacy reported against a control', BAD, 'raw cosine only - this measures task correlation, not leakage')
    agg = d[d.regime == 'aggregate_only'].leakage_mean.iloc[0]
    k32 = d[(d.regime == 'sketch_k32') & ~d.dp].leakage_mean.iloc[0]
    full = d[d.regime == 'plaintext_full'].leakage_mean.iloc[0]
    check('sketch leaks less than the aggregate the protocol must reveal', OK if k32 < agg else WARN, f'sketch(k=32)={k32:.4f}  aggregate={agg:.4f}  plaintext={full:.4f}')
    lo = d[(d.regime == 'sketch_k32') & ~d.dp].leakage_ci_lo.iloc[0]
    check('sketch leakage is nonetheless measurably above zero', OK if lo > 0 else WARN, f"95% CI lower bound = {lo:.4f}; the claim is 'small', not 'none'")

def audit_reproducibility():
    print('\n=== 7. REPRODUCIBILITY ===')
    cfgs = sorted(glob.glob(str(ROOT / 'results' / 's[0-9]*_cic_iot' / 'config_used.json')))
    check('the exact config is stored with the results', OK if cfgs else BAD, f'{len(cfgs)} config_used.json files')
    if cfgs:
        base = json.loads(Path(cfgs[0]).read_text(encoding='utf-8'))
        diffs = []
        for c in cfgs[1:]:
            o = json.loads(Path(c).read_text(encoding='utf-8'))
            for key in ('crypto', 'trust', 'zero_trust', 'model', 'training'):
                if o.get(key) != base.get(key):
                    diffs.append((Path(c).parts[-2], key))
        check('all seeds ran the same configuration', OK if not diffs else BAD, '' if not diffs else f'differences: {diffs}')
    audits = sorted(glob.glob(str(ROOT / 'results' / 's[0-9]*_cic_iot' / 'csv' / 'audit_*.csv')))
    check('decryption-authority audit logs retained', OK if audits else WARN, f'{len(audits)} audit logs')

def main():
    print('=' * 78)
    print('ADVERSARIAL AUDIT OF THE EXPERIMENTAL RECORD')
    print('=' * 78)
    for fn in (audit_data_hygiene, audit_protocol_fairness, audit_crypto_consistency, audit_statistics, audit_gate_and_claims, audit_privacy_claims, audit_reproducibility):
        try:
            fn()
        except Exception as exc:
            check(f'{fn.__name__} completed', BAD, f'{type(exc).__name__}: {exc}')
    n_fail = sum((1 for s, _ in _results if s == BAD))
    n_warn = sum((1 for s, _ in _results if s == WARN))
    print('\n' + '=' * 78)
    print(f'{len(_results)} checks: {len(_results) - n_fail - n_warn} pass, {n_warn} warn, {n_fail} FAIL')
    if n_fail:
        print('FAILURES:')
        for s, name in _results:
            if s == BAD:
                print(f'  - {name}')
    if n_warn:
        print('WARNINGS (must be stated in the paper, not hidden):')
        for s, name in _results:
            if s == WARN:
                print(f'  - {name}')
if __name__ == '__main__':
    main()
