from __future__ import annotations
import glob
import hashlib
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]

def digest_rows(X: np.ndarray) -> list:
    X = np.ascontiguousarray(X)
    return [hashlib.blake2b(r.tobytes(), digest_size=16).digest() for r in X]

def main() -> int:
    caches = sorted(glob.glob(str(ROOT / 'data' / 'processed_cic' / '*.npz')))
    if not caches:
        print('no preprocessed cache found')
        return 1
    path = caches[0]
    print(f'cache: {Path(path).name}')
    d = np.load(path)
    keys = list(d.keys())
    print(f'arrays: {keys}\n')

    def pick(*names):
        for n in names:
            if n in d:
                return d[n]
        return None
    Xtr = pick('X_train', 'Xtr', 'train_X')
    Xva = pick('X_val', 'Xva', 'val_X')
    Xte = pick('X_test', 'Xte', 'test_X')
    if Xtr is None or Xte is None:
        print(f'could not find train/test arrays among {keys}')
        return 1
    dtr, dte = (digest_rows(Xtr), digest_rows(Xte))
    str_, ste = (set(dtr), set(dte))
    print(f'train {Xtr.shape}   test {Xte.shape}' + (f'   val {Xva.shape}' if Xva is not None else ''))
    print(f'\ninternal duplicates (identical feature vectors within a split)')
    print(f'  train: {len(Xtr) - len(str_):,} of {len(Xtr):,} rows are repeats ({100 * (1 - len(str_) / len(Xtr)):.1f}%)')
    print(f'  test : {len(Xte) - len(ste):,} of {len(Xte):,} rows are repeats ({100 * (1 - len(ste) / len(Xte)):.1f}%)')
    leaked = sum((1 for h in dte if h in str_))
    print(f'\nLEAKAGE - test rows whose exact feature vector also appears in train')
    print(f'  {leaked:,} of {len(Xte):,} test rows = {100 * leaked / len(Xte):.2f}%')
    if Xva is not None:
        dva = digest_rows(Xva)
        lv = sum((1 for h in dva if h in str_))
        print(f'  val: {lv:,} of {len(Xva):,} = {100 * lv / len(Xva):.2f}%')
    print()
    frac = leaked / max(len(Xte), 1)
    if frac > 0.05:
        print(f'VERDICT: {frac:.1%} of the test set is memorisable from training.')
        print('Our own numbers are inflated by this and the split must be made')
        print('duplicate-aware before anything is published.')
        return 2
    print(f'VERDICT: leakage is {frac:.2%}. Our ceiling of 0.7464 is a real')
    print('generalisation number. Published results in the 0.92-0.98 range on')
    print('this dataset should be checked for a row-level random split over')
    print('duplicated flow records before being placed in the same table.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
