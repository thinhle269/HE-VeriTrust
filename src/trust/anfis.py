
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from .features import TrustFeatures, features_matrix

_EPS = 1e-6


class ANFISHead(nn.Module):
    def __init__(self, n_inputs: int = 3, n_mf: int = 5, device: str = "cpu"):
        super().__init__()
        self.n_inputs, self.n_mf = int(n_inputs), int(n_mf)
        lo, hi = -1.0, 1.0
        means = torch.stack([torch.linspace(lo, hi, self.n_mf)
                             for _ in range(self.n_inputs)], dim=0)
        spread = (hi - lo) / max(self.n_mf - 1, 1)
        self.means = nn.Parameter(means)
        self.log_spreads = nn.Parameter(
            torch.full((self.n_inputs, self.n_mf), float(np.log(spread))))


        grid = torch.linspace(lo, hi, self.n_mf)
        logits = torch.zeros(self.n_mf ** self.n_inputs)
        idx = 0
        for i in range(self.n_mf):
            for j in range(self.n_mf):
                for k in range(self.n_mf):
                    score = 2.0 * grid[i] + 1.0 * grid[k] - 1.5 * grid[j].abs()
                    logits[idx] = score
                    idx += 1
        self.rule_logits = nn.Parameter(logits)

        self.base = nn.Linear(self.n_inputs, 1)
        with torch.no_grad():
            self.base.weight.copy_(torch.tensor([[2.0, -1.0, 1.0]]))
            self.base.bias.zero_()
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spreads = torch.exp(self.log_spreads)
        mus = []
        for i in range(self.n_inputs):
            xi = x[:, i:i + 1]
            mu = torch.exp(-0.5 * ((xi - self.means[i].unsqueeze(0))
                                   / (spreads[i].unsqueeze(0).abs() + _EPS)) ** 2)
            mus.append(mu)
        f = (mus[0].unsqueeze(2).unsqueeze(3)
             * mus[1].unsqueeze(1).unsqueeze(3)
             * mus[2].unsqueeze(1).unsqueeze(2)).reshape(x.shape[0], -1)
        w = torch.sigmoid(self.rule_logits).unsqueeze(0)
        t_fuzzy = (f * w).sum(1) / (f.sum(1) + _EPS)
        base = torch.sigmoid(self.base(x)).squeeze(1)
        g = torch.sigmoid(self.gate)
        return (g * base + (1.0 - g) * t_fuzzy).clamp(0.0, 1.0)


class NeuroFuzzyTrust:


    def __init__(self, n_mf: int = 5, lr: float = 0.02, device: str = "cpu",
                 seed: int = 0):
        self.device = torch.device(device)
        self.lr = float(lr)


        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2 ** 31 - 1, (1,), generator=gen)))
            self.model = ANFISHead(3, n_mf).to(self.device)
        self._fitted = False


    requires_calibration = True

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, feats: Sequence[TrustFeatures], labels: Sequence[float],
            epochs: int = 300) -> "NeuroFuzzyTrust":
        X = torch.from_numpy(features_matrix(feats)).to(self.device)
        if X.shape[0] == 0:
            return self
        y = torch.tensor(np.asarray(labels, dtype=np.float32).reshape(-1),
                         device=self.device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lf = nn.BCELoss()
        self.model.train()
        for _ in range(int(epochs)):
            opt.zero_grad(set_to_none=True)
            loss = lf(self.model(X).clamp(_EPS, 1 - _EPS), y)
            loss.backward()
            opt.step()
        self.model.eval()
        self._fitted = True
        return self

    def score_many(self, feats: Iterable[TrustFeatures]) -> np.ndarray:
        feats = list(feats)
        if not feats:
            return np.zeros(0)
        X = torch.from_numpy(features_matrix(feats)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X).cpu().numpy().astype(np.float64)


class MamdaniTrust:


    _SETS = {
        "normalised": (( -1.0, -1.0, -0.76, -0.38),
                       (-0.76, -0.38,  0.00,  0.46),
                       (  0.0,  0.46,  1.00,  1.00)),
        "raw":        (( -1.0, -1.0, -0.20,  0.20),
                       (  0.0,  0.30,  0.50,  0.80),
                       (  0.5,  0.80,  1.00,  1.00)),
    }


    requires_calibration = False

    def __init__(self, domain: str = "normalised", **_):
        if domain not in self._SETS:
            raise ValueError(f"domain must be one of {sorted(self._SETS)}")
        self.domain = domain
        self._fitted = True

    @property
    def is_fitted(self) -> bool:
        return True

    def fit(self, *_, **__):
        return self

    @staticmethod
    def _trap(x, a, b, c, d):


        if b <= x <= c:
            return 1.0
        if x <= a or x >= d:
            return 0.0
        if a < x < b:
            return (x - a) / max(b - a, _EPS)
        return (d - x) / max(d - c, _EPS)

    def score_many(self, feats: Iterable[TrustFeatures]) -> np.ndarray:
        out = []
        for f in feats:
            u_set, s_set, t_set = self._SETS[self.domain]
            lo = self._trap(f.proj_ref, *u_set)
            mid = self._trap(f.proj_ref, *s_set)
            hi = self._trap(f.proj_ref, *t_set)
            ok_norm = self._trap(f.norm_ratio, -0.8, -0.4, 0.4, 0.8)
            agree = self._trap(f.peer_agreement, -0.2, 0.2, 1.0, 1.0)
            untrusted = lo
            suspicious = max(mid, 1.0 - ok_norm)
            trusted = min(hi, max(ok_norm, agree))
            num = 0.10 * untrusted + 0.45 * suspicious + 0.90 * trusted
            den = untrusted + suspicious + trusted
            out.append(float(np.clip(num / den, 0.0, 1.0)) if den > _EPS else 0.5)
        return np.asarray(out, dtype=np.float64)


def build_engine(cfg_trust, device: str = "cpu", seed: int = 0):
    kind = str(cfg_trust.get("engine", "anfis")).lower()
    if kind == "mamdani":
        return MamdaniTrust()
    if kind in ("none", "off"):
        return None
    return NeuroFuzzyTrust(n_mf=int(cfg_trust.get("n_mf", 5)),
                           lr=float(cfg_trust.get("lr", 0.02)),
                           device=device, seed=seed)
