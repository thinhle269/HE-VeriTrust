from __future__ import annotations
import argparse
import glob
import hashlib
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TREE = 'D:/Chi Van/CIC_IoT_Attack_2023/CSV'

def dup_rate(df: pd.DataFrame) -> tuple:
    cols = [c for c in df.columns if c.strip().lower() not in ('label', 'class')]
    X = df[cols].to_numpy(dtype=np.float64, na_value=np.nan)
    X = np.ascontiguousarray(X)
    seen = {hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in X}
    return (len(X), len(seen))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', default=DEFAULT_TREE)
    ap.add_argument('--rows-per-dir', type=int, default=200000)
    ap.add_argument('--max-dirs', type=int, default=0, help='0 = all')
    a = ap.parse_args()
    dirs = sorted((d for d in glob.glob(os.path.join(a.tree, '*')) if os.path.isdir(d)))
    if a.max_dirs:
        dirs = dirs[:a.max_dirs]
    if not dirs:
        print(f'no attack directories under {a.tree}')
        return 1
    rows = []
    tot_n = tot_d = 0
    for d in dirs:
        files = sorted(glob.glob(os.path.join(d, '*.csv')))
        if not files:
            continue
        frames, got = ([], 0)
        for f in files:
            need = a.rows_per_dir - got
            if need <= 0:
                break
            try:
                chunk = pd.read_csv(f, nrows=need)
            except Exception as exc:
                print(f'  [skip] {os.path.basename(f)}: {exc}')
                continue
            frames.append(chunk)
            got += len(chunk)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        n, distinct = dup_rate(df)
        tot_n += n
        tot_d += distinct
        rows.append({'attack': os.path.basename(d), 'rows_read': n, 'distinct': distinct, 'duplicate_rate': round(1 - distinct / max(n, 1), 4)})
        print(f'  {os.path.basename(d):<32} {n:>8,} rows  {1 - distinct / max(n, 1):>7.2%} duplicates')
    out = ROOT / 'results' / 'cic_iot' / 'tables' / 'raw_duplicate_rate.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values('duplicate_rate', ascending=False)
    df.to_csv(out, index=False)
    overall = 1 - tot_d / max(tot_n, 1)
    print(f"\n{'=' * 66}")
    print(f'rows examined      : {tot_n:,} over {len(rows)} attack families')
    print(f'distinct rows      : {tot_d:,}')
    print(f'OVERALL DUPLICATE  : {overall:.2%}')
    print(f'\nWorst families:')
    for _, r in df.head(5).iterrows():
        print(f"  {r['attack']:<32} {r['duplicate_rate']:.2%}")
    print(f'\nA uniform random train/test split over this corpus places roughly')
    print(f'{overall:.0%} of the test rows verbatim in the training set. That is the')
    print(f'protocol difference to check before placing a published 0.95 macro-F1')
    print(f'beside a deduplicated one.')
    print(f'\n-> {out}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
