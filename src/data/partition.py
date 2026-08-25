
from __future__ import annotations

from typing import Dict

import numpy as np


def dirichlet_partition(y: np.ndarray, num_clients: int, alpha: float,
                        seed: int, min_per_client: int = 32) -> Dict[int, np.ndarray]:

    rng = np.random.default_rng(int(seed))
    n_classes = int(y.max()) + 1
    parts = [[] for _ in range(num_clients)]
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        p = rng.dirichlet(np.repeat(float(alpha), num_clients))
        cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
        for cid, chunk in enumerate(np.split(idx, cuts)):
            parts[cid].extend(chunk.tolist())

    out = {i: np.array(sorted(p), dtype=np.int64) for i, p in enumerate(parts)}
    for cid in range(num_clients):
        while len(out[cid]) < min_per_client:
            donor = max(range(num_clients), key=lambda k: len(out[k]))
            if donor == cid or len(out[donor]) <= min_per_client:
                break
            take, out[donor] = out[donor][:min_per_client], out[donor][min_per_client:]
            out[cid] = np.concatenate([out[cid], take])
    return out
