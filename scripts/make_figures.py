from __future__ import annotations
import argparse
import glob
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
PRIMARY = 'veritrust_mamdani'
PROPOSED = {'veritrust_mamdani', 'veritrust', 'he_only'}
LABEL = {'centralized': 'Centralized', 'fedavg': 'FedAvg (clean)', 'fedavg_attack': 'FedAvg + attack', 'fedmedian': 'Median', 'trimmed_mean': 'Trimmed-Mean', 'krum': 'Multi-Krum', 'bulyan': 'Bulyan', 'foolsgold': 'FoolsGold', 'he_only': 'Packed HE only', 'veritrust_mamdani': 'HE-VeriTrust (label-free)', 'veritrust': 'HE-VeriTrust (learned engine)'}

def _load(pattern: str, name: str) -> pd.DataFrame:
    fs = sorted(glob.glob(str(ROOT / 'results' / pattern / 'csv' / name)))
    frames = [pd.read_csv(f) for f in fs]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {name}')

def fig_convergence(pattern, out):
    d = _load(pattern, 'per_round.csv')
    if d.empty:
        return print('  [skip] convergence: no per_round.csv')
    keep = [s for s in ['fedavg', 'fedavg_attack', 'krum', 'bulyan', 'he_only', 'veritrust', 'veritrust_mamdani'] if s in set(d.scenario)]
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    pal = plt.cm.tab10(np.linspace(0, 1, max(len(keep), 2)))
    for i, s in enumerate(keep):
        g = d[d.scenario == s].groupby('round_idx')['val_macro_f1'].agg(['mean', 'std']).reset_index()
        style = '-' if s in PROPOSED else '--'
        ax.plot(g.round_idx + 1, g['mean'], style, color=pal[i], lw=3.0 if s == PRIMARY else 2.0 if s in PROPOSED else 1.4, marker='o', ms=3.5 if s == PRIMARY else 3, zorder=5 if s == PRIMARY else 2, label=LABEL.get(s, s))
        ax.fill_between(g.round_idx + 1, g['mean'] - g['std'].fillna(0), g['mean'] + g['std'].fillna(0), color=pal[i], alpha=0.12)
    ax.set_xlabel('federated round')
    ax.set_ylabel('validation macro-F1')
    ax.set_title('Convergence under 30% Byzantine sign-flip (mean ± std)', fontsize=10, weight='bold')
    ax.legend(fontsize=8, ncol=2, loc='lower right')
    ax.grid(alpha=0.3)
    _save(fig, out, 'fig_convergence')

def fig_headline(pattern, out):
    d = _load(pattern, 'per_seed.csv').drop_duplicates(['scenario', 'seed'])
    if d.empty:
        return print('  [skip] headline: no per_seed.csv')
    g = d.groupby('scenario')[['test_macro_f1', 'test_weighted_f1']].agg(['mean', 'std'])
    g.columns = [f'{a}_{b}' for a, b in g.columns]
    g = g.reset_index().sort_values('test_macro_f1_mean', ascending=False)
    g = g[g.scenario != 'centralized']
    x = np.arange(len(g))
    w = 0.4
    fig, ax = plt.subplots(figsize=(9.5, 4.4))

    def _tier(s, primary, secondary, base):
        return primary if s == PRIMARY else secondary if s in PROPOSED else base
    c1 = [_tier(s, '#1b5e20', '#4c8c4a', '#78909c') for s in g.scenario]
    c2 = [_tier(s, '#43a047', '#8bc48a', '#b0bec5') for s in g.scenario]
    ax.bar(x - w / 2, g.test_macro_f1_mean, w, yerr=g.test_macro_f1_std.fillna(0), capsize=3, color=c1, edgecolor='black', lw=0.4, label='macro-F1')
    ax.bar(x + w / 2, g.test_weighted_f1_mean, w, yerr=g.test_weighted_f1_std.fillna(0), capsize=3, color=c2, edgecolor='black', lw=0.4, label='weighted-F1')
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(s, s) for s in g.scenario], rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('test F1')
    ax.set_title('CIC-IoT-2023 final test F1 (green = this work)', fontsize=10, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    _save(fig, out, 'fig_headline')

def fig_privacy(out):
    p = out.parent / 'privacy' / 'privacy_summary.csv'
    if not p.exists():
        return print(f'  [skip] privacy: no {p}')
    d = pd.read_csv(p)
    col_mean = 'leakage_mean' if 'leakage_mean' in d.columns else 'mean'
    lo_c = 'leakage_ci_lo' if 'leakage_ci_lo' in d.columns else 'ci_lo'
    hi_c = 'leakage_ci_hi' if 'leakage_ci_hi' in d.columns else 'ci_hi'
    sk = d[d.regime.str.startswith('sketch_k')].copy()
    sk['kk'] = sk['k'].astype(int)
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for dp, style, lab in [(False, '--', 'no DP noise'), (True, '-', 'with DP noise')]:
        s = sk[sk.dp == dp].sort_values('kk')
        if s.empty:
            continue
        ax.errorbar(s.kk, s[col_mean], yerr=[s[col_mean] - s[lo_c], s[hi_c] - s[col_mean]], marker='o', ls=style, lw=2, capsize=3, label=lab)
    for reg, col, lab in [('plaintext_full', '#d62728', 'plaintext update (what robust aggregators require)'), ('aggregate_only', '#2ca02c', 'decrypted aggregate (unavoidable floor)')]:
        r = d[d.regime == reg]
        if len(r):
            ax.axhline(float(r[col_mean].iloc[0]), color=col, ls=':', lw=1.6, label=lab)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.set_xlabel('sketch dimension  k   (of d = 52,040)')
    ax.set_ylabel('leakage above a non-participant control')
    ax.set_title("Leakage of the server's view (optimal linear reconstruction).\nBelow k=128 the attestation channel leaks less than the\naggregate the protocol must reveal anyway.", fontsize=9.5, weight='bold')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(alpha=0.3)
    _save(fig, out, 'fig_privacy_vs_k')

def fig_forgery(out):
    frames = {}
    for eng in ('mamdani', 'anfis'):
        f = out.parent / f'forgery_{eng}' / 'forgery_raw.csv'
        if f.exists():
            frames[eng] = pd.read_csv(f)
    if not frames:
        return print(f'  [skip] forgery: no forgery_* under {out.parent}')
    order = [('self_reported', False), ('self_reported', True), ('sketch', False), ('sketch', True)]
    names = ['self-reported\nhonest', 'self-reported\nFORGED', 'sketch\nhonest', 'sketch\nFORGED']
    cols = ['#90a4ae', '#c62828', '#66bb6a', '#1b5e20']
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6))
    width = 0.38
    for pi, (metric, ylab, title) in enumerate([('macro_f1', 'test macro-F1', '(a) end-task cost'), ('mal_rejected_rate', 'fraction of Byzantine updates rejected', '(b) what the gate actually does')]):
        ax = axes[pi]
        for ei, (eng, d) in enumerate(sorted(frames.items())):
            xs = np.arange(4) + (ei - 0.5) * width
            m = [d[(d.attestation == a) & (d.forging == f)][metric].mean() for a, f in order]
            e = [d[(d.attestation == a) & (d.forging == f)][metric].std() for a, f in order]
            ax.bar(xs, m, width, yerr=np.nan_to_num(e), capsize=3, color=cols, edgecolor='black', lw=0.5, hatch='' if eng == 'mamdani' else '//', label='fixed engine (no labels)' if eng == 'mamdani' else 'learned engine (label-calibrated)')
        ax.set_xticks(range(4))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10, weight='bold')
        ax.grid(axis='y', alpha=0.3)
        if pi == 0:
            d0 = list(frames.values())[0]
            for ref, style, lab in [('fedavg_attack', ':', 'undefended'), ('fedavg', '--', 'clean FedAvg')]:
                r = d0[d0.cell == ref]['macro_f1']
                if len(r):
                    ax.axhline(r.mean(), ls=style, color='black', lw=1.3, label=lab)
        ax.legend(fontsize=7.5, loc='lower left')
    fig.suptitle('Attestation forgery: a Byzantine client reports the statistics of the update it pretends to have sent.\nSelf-reported evidence loses half its detection; a sketch measured from the ciphertext loses none.', fontsize=10, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, out, 'fig_forgery')

def fig_ablation(out, param='trust_threshold', xlabel='Zero-Trust threshold $\\tau$'):
    ab = out.parent / 'ablation' / f'sweep_{param}_summary.csv'
    if not ab.exists():
        return print(f'  [skip] ablation {param}: not run for this dataset')
    d = pd.read_csv(ab)
    full = out.parent / 'tables' / 'ablation_full.csv'
    gate = pd.DataFrame()
    if full.exists():
        g = pd.read_csv(full)
        gate = g[g.param.astype(str) == param]
    categorical = not pd.api.types.is_numeric_dtype(d['value'])
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.3))

    def _x(sub):
        return np.arange(len(sub)) if categorical else sub['value'].astype(float).to_numpy()
    for scen, sub in d.groupby('scenario'):
        sub = sub.sort_values('value')
        axes[0].errorbar(_x(sub), sub.test_macro_f1_mean, yerr=sub.test_macro_f1_std.fillna(0), marker='o', lw=0 if categorical else 2, ms=8 if categorical else 6, capsize=3, label=LABEL.get(scen, scen))
    axes[0].set_ylabel('test macro-F1')
    axes[0].set_title('End-task metric', fontsize=10, weight='bold')
    if not gate.empty:
        for scen, sub in gate.groupby('scenario'):
            sub = sub.sort_values('value')
            if 'malicious_rejected' in sub:
                axes[1].plot(_x(sub), sub.malicious_rejected, marker='o', lw=0 if categorical else 2, ms=8 if categorical else 6, label=f'{LABEL.get(scen, scen)}: detection')
            if 'honest_rejected' in sub:
                axes[1].plot(_x(sub), sub.honest_rejected, marker='s', lw=0 if categorical else 2, ms=8 if categorical else 6, ls='none' if categorical else '--', label=f'{LABEL.get(scen, scen)}: honest rejected')
        axes[1].set_ylim(-0.02, 1.02)
    else:
        axes[1].text(0.5, 0.5, 'no gate metrics for this sweep', ha='center', va='center', transform=axes[1].transAxes)
    axes[1].set_ylabel('rate')
    axes[1].set_title('Gate behaviour (what macro-F1 cannot see)', fontsize=10, weight='bold')
    for ax, src in zip(axes, (d, gate if not gate.empty else d)):
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        if categorical:
            sub = src.drop_duplicates('value').sort_values('value')
            ax.set_xticks(np.arange(len(sub)))
            ax.set_xticklabels(sub['value'].astype(str), fontsize=9)
            ax.set_xlim(-0.5, len(sub) - 0.5)
    fig.suptitle(f'Sensitivity to {xlabel}', fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out, f'fig_ablation_{param}')

def fig_overhead(pattern, out):
    d = _load(pattern, 'per_round.csv')
    if d.empty:
        return print('  [skip] overhead: no per_round.csv')
    comps = [('t_train', 'Local training', '#8fb4d9'), ('t_encrypt', 'Encrypt (client)', '#e08214'), ('t_measure', 'Sketch measure (DA)', '#fdb863'), ('t_aggregate', 'Aggregate + decrypt', '#b2182b'), ('t_trust', 'Trust + gate', '#2ca02c')]
    keep = [s for s in ['fedavg', 'krum', 'he_only', 'veritrust'] if s in set(d.scenario)]
    g = d[d.scenario.isin(keep)].groupby('scenario')[[c for c, _, _ in comps]].mean()
    g = g.reindex(keep)
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    bottom = np.zeros(len(keep))
    for col, lab, c in comps:
        v = g[col].to_numpy()
        ax.bar(range(len(keep)), v, bottom=bottom, color=c, edgecolor='black', lw=0.4, label=lab)
        bottom += v
    for i, s in enumerate(keep):
        ax.text(i, bottom[i] + 0.3, f'{bottom[i]:.1f}s', ha='center', fontsize=9, weight='bold')
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels([LABEL.get(s, s) for s in keep], fontsize=9)
    ax.set_ylabel('wall-clock per federated round (s)')
    ax.set_ylim(0, bottom.max() * 1.22)
    ax.set_title('Per-round cost with slot packing (2048-bit Paillier)', fontsize=10, weight='bold')
    ax.legend(fontsize=8, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis='y', alpha=0.3)
    _save(fig, out, 'fig_overhead')

def fig_trust_trajectory(pattern, out):
    d = _load(pattern, 'per_client.csv')
    if d.empty:
        return print('  [skip] trajectory: no per_client.csv')
    d = d[(d.scenario == 'veritrust') & (d.seed == d.seed.min())]
    if d.empty:
        return print('  [skip] trajectory: no veritrust rows')
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for cid, s in d.groupby('client_id'):
        s = s.sort_values('round_idx')
        mal = bool(s.is_malicious.iloc[0])
        a1.plot(s.round_idx + 1, s.trust_smoothed, color='#d62728' if mal else '#2ca02c', lw=2.0 if mal else 1.2, alpha=1.0 if mal else 0.55, marker='o' if mal else None, ms=3)
    a1.axhline(0.4, color='#1f3864', ls='--', lw=1.5)
    a1.text(d.round_idx.max() * 0.98, 0.42, 'gate $\\tau=0.40$', ha='right', fontsize=9, color='#1f3864', weight='bold')
    a1.plot([], [], color='#2ca02c', lw=1.8, label='honest')
    a1.plot([], [], color='#d62728', lw=2.0, marker='o', ms=3, label='malicious')
    a1.set_xlabel('federated round')
    a1.set_ylabel('smoothed trust')
    a1.set_ylim(0, 1.02)
    a1.legend(fontsize=9)
    a1.grid(alpha=0.3)
    a1.set_title('(a) trust trajectories', fontsize=10, weight='bold')
    g = d.groupby(['round_idx', 'is_malicious'])['weight'].mean().reset_index()
    for mal, col, lab in [(False, '#2ca02c', 'honest'), (True, '#d62728', 'malicious')]:
        s = g[g.is_malicious == mal].sort_values('round_idx')
        a2.plot(s.round_idx + 1, s.weight, color=col, lw=2, marker='o', ms=3, label=lab)
    a2.axhline(0, color='gray', lw=0.8)
    a2.set_xlabel('federated round')
    a2.set_ylabel('mean aggregation weight')
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)
    a2.set_title('(b) aggregation weight', fontsize=10, weight='bold')
    fig.suptitle(f'Trust dynamics under 30% Byzantine sign-flip (seed {int(d.seed.iloc[0])})', fontsize=11, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, out, 'fig_trust_trajectory')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='s[0-9]*_cic_iot')
    ap.add_argument('--out', default='results/cic_iot/figures')
    a = ap.parse_args()
    out = ROOT / a.out
    print('generating figures ...')
    fig_convergence(a.pattern, out)
    fig_headline(a.pattern, out)
    fig_overhead(a.pattern, out)
    fig_trust_trajectory(a.pattern, out)
    fig_privacy(out)
    fig_forgery(out)
    for p, x in [('trust_threshold_30r', 'Zero-Trust threshold $\\tau$'), ('sketch_k_30r', 'sketch dimension $k$'), ('malicious_fraction', 'Byzantine fraction'), ('dp_sigma', 'DP noise multiplier $\\sigma$'), ('ema_alpha_30r', 'EMA weight $\\beta$ (smoothing the verdict)'), ('evidence_alpha_30r', 'evidence EMA $\\alpha$ (smoothing the feature)'), ('evidence_z0_30r', 'resolvability scale $z_0$'), ('sketch_mode_be', 'probe geometry, under the block-evasion attack'), ('dirichlet_alpha', 'Dirichlet $\\alpha$')]:
        fig_ablation(out, p, x)
    print(f'figures -> {out}')
if __name__ == '__main__':
    main()
