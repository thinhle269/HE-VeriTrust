from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
LABELS = {'centralized': 'Centralized reference', 'fedavg': 'FedAvg, no attack', 'veritrust_mamdani': 'HE-VeriTrust, label-free', 'veritrust': 'HE-VeriTrust, learned', 'krum': 'Multi-Krum', 'fedmedian': 'Median', 'trimmed_mean': 'Trimmed mean', 'foolsgold': 'FoolsGold-style', 'bulyan': 'Bulyan-like', 'he_only': 'Packed HE only', 'fedavg_attack': 'FedAvg, under attack'}
ORDER = list(LABELS)

def _read_headline(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path).set_index('scenario')
    missing = [scenario for scenario in ORDER if scenario not in table.index]
    if missing:
        raise ValueError(f'{path} is missing scenarios: {missing}')
    required = {'test_macro_f1_mean', 'test_macro_f1_std'}
    absent = sorted(required.difference(table.columns))
    if absent:
        raise ValueError(f'{path} is missing columns: {absent}')
    return table.loc[ORDER]

def generate(cic_csv: Path, edge_csv: Path, output_dir: Path) -> list[Path]:
    cic = _read_headline(cic_csv)
    edge = _read_headline(edge_csv)
    y = list(range(len(ORDER)))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.errorbar(cic['test_macro_f1_mean'], [value + 0.12 for value in y], xerr=cic['test_macro_f1_std'], fmt='o', markersize=5.2, capsize=2.5, linewidth=1.1, color='#2B6CB0', label='CIC-IoT-2023')
    ax.errorbar(edge['test_macro_f1_mean'], [value - 0.12 for value in y], xerr=edge['test_macro_f1_std'], fmt='s', markersize=4.8, capsize=2.5, linewidth=1.1, color='#D97706', label='Edge-IIoTset')
    ax.set_yticks(y, [LABELS[key] for key in ORDER])
    ax.set_xlim(0.6, 0.84)
    ax.set_xlabel('Test macro-$F_1$')
    ax.grid(axis='x', color='#D1D5DB', linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', frameon=False, ncol=2)
    fig.tight_layout(pad=1.2)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / 'fig_cross_dataset.png', output_dir / 'fig_cross_dataset.pdf', output_dir / 'fig_headline_two_datasets.png', output_dir / 'fig_headline_two_datasets.pdf', output_dir / 'fig_headline_two_datasets_v8.png', output_dir / 'fig_headline_two_datasets_v8.pdf']
    for output in outputs:
        kwargs = {'dpi': 300} if output.suffix == '.png' else {}
        fig.savefig(output, bbox_inches='tight', **kwargs)
    plt.close(fig)
    return outputs

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cic', type=Path, default=ROOT / 'results' / 'cic_iot' / 'tables' / 'headline.csv')
    parser.add_argument('--edge', type=Path, default=ROOT / 'results' / 'edgeiiot' / 'tables' / 'headline.csv')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'cic_iot' / 'figures')
    args = parser.parse_args()
    for output in generate(args.cic, args.edge, args.output_dir):
        print(f'wrote {output}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
