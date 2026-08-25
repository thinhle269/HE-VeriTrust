
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class ClientUpdate:
    client_id: int
    delta: np.ndarray
    n_samples: int
    loss_before: float
    loss_after: float
    train_time: float
    is_malicious: bool = False


    claimed: Dict[str, float] = field(default_factory=dict)


    honest_delta: Optional[np.ndarray] = None


def flatten_state(state: Dict[str, torch.Tensor]) -> np.ndarray:
    return np.concatenate([t.detach().cpu().numpy().astype(np.float32).reshape(-1)
                           for t in state.values()])


def state_schema(state: Dict[str, torch.Tensor]):
    return [(k, tuple(v.shape)) for k, v in state.items()]


def unflatten_state(vec: np.ndarray, schema) -> Dict[str, torch.Tensor]:
    out, off = {}, 0
    for name, shape in schema:
        n = int(np.prod(shape)) if shape else 1
        out[name] = torch.from_numpy(
            np.asarray(vec[off:off + n], dtype=np.float32).reshape(shape).copy())
        off += n
    if off != vec.size:
        raise ValueError(f"schema covers {off} elements, vector has {vec.size}")
    return out


class FederatedClient:
    def __init__(self, client_id: int, X: torch.Tensor, y: torch.Tensor,
                 device: torch.device, lr: float = 2e-3, local_epochs: int = 2,
                 batch_size: int = 256, optimizer: str = "adam",
                 grad_clip_norm: float = 1.0, max_update_norm: float = 10.0,
                 class_weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0,
                 balanced_sampler: bool = False,
                 is_malicious: bool = False, attack: str = "sign_flip",
                 noise_sigma: float = 1.0, num_classes: int = 8,
                 forge_attestation: bool = False, seed: int = 0):
        self.client_id = int(client_id)
        self.device = device
        self.X = X.to(device, non_blocking=True)
        self.y = y.to(device, non_blocking=True)
        self.n_samples = int(self.y.numel())
        self.lr = float(lr)
        self.local_epochs = int(local_epochs)
        self.batch_size = int(batch_size)
        self.optimizer_kind = str(optimizer).lower()
        self.grad_clip_norm = float(grad_clip_norm) if grad_clip_norm else None
        self.max_update_norm = float(max_update_norm) if max_update_norm else None
        self.label_smoothing = float(label_smoothing)
        self.class_weight = (class_weight.to(device)
                             if class_weight is not None else None)
        self.is_malicious = bool(is_malicious)
        self.attack = str(attack)
        self.noise_sigma = float(noise_sigma)
        self.num_classes = int(num_classes)
        self.forge_attestation = bool(forge_attestation)
        self._rng = np.random.default_rng(int(seed) + self.client_id)
        self._sample_p = self._build_sampling_weights() if balanced_sampler else None


    def _build_sampling_weights(self) -> torch.Tensor:
        counts = torch.bincount(self.y, minlength=self.num_classes).float()
        counts = torch.clamp(counts, min=1.0)
        w = 1.0 / counts
        p = w[self.y]
        return p / p.sum()

    def _loss_fn(self) -> nn.Module:
        return nn.CrossEntropyLoss(weight=self.class_weight,
                                   label_smoothing=self.label_smoothing)

    @torch.no_grad()
    def _epoch_loss(self, model: nn.Module, lf: nn.Module) -> float:
        model.eval()
        tot, n = 0.0, 0
        for i in range(0, self.n_samples, 8192):
            xb, yb = self.X[i:i + 8192], self.y[i:i + 8192]
            tot += float(lf(model(xb), yb).item()) * xb.shape[0]
            n += xb.shape[0]
        model.train()
        return tot / max(n, 1)


    def train_round(self, global_model: nn.Module,
                    global_flat: np.ndarray) -> ClientUpdate:
        local = copy.deepcopy(global_model).to(self.device)
        local.train()
        opt = (torch.optim.Adam(local.parameters(), lr=self.lr)
               if self.optimizer_kind == "adam"
               else torch.optim.SGD(local.parameters(), lr=self.lr, momentum=0.9))
        lf = self._loss_fn()
        loss_before = self._epoch_loss(local, lf)

        t0 = time.time()
        for _ in range(self.local_epochs):
            if self._sample_p is not None:
                idx = torch.multinomial(self._sample_p, self.n_samples,
                                        replacement=True)
            else:
                idx = torch.randperm(self.n_samples, device=self.device)
            for s in range(0, self.n_samples, self.batch_size):
                b = idx[s:s + self.batch_size]
                if b.numel() < 2:
                    continue
                xb, yb = self.X[b], self.y[b]
                if self.is_malicious and self.attack == "label_flip":
                    yb = (yb + 1) % self.num_classes
                opt.zero_grad(set_to_none=True)
                loss = lf(local(xb), yb)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if self.grad_clip_norm:
                    nn.utils.clip_grad_norm_(local.parameters(), self.grad_clip_norm)
                opt.step()
        train_time = time.time() - t0
        loss_after = self._epoch_loss(local, lf)

        delta = (flatten_state(local.state_dict()) - global_flat).astype(np.float32)
        honest_delta = delta.copy() if (self.is_malicious
                                        and self.forge_attestation) else None
        if self.is_malicious:
            if self.attack == "sign_flip":
                delta = -delta
            elif self.attack == "sign_flip_scaled":


                n = float(np.linalg.norm(delta))
                if n > 1e-12 and self.max_update_norm:
                    delta = (-delta / n * self.max_update_norm).astype(np.float32)
                else:
                    delta = -delta
            elif self.attack == "gaussian_noise":
                delta = delta + self._rng.normal(
                    0.0, self.noise_sigma, size=delta.shape).astype(np.float32)

        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        if self.max_update_norm:
            nrm = float(np.linalg.norm(delta))
            if np.isfinite(nrm) and nrm > self.max_update_norm:
                delta = delta * (self.max_update_norm / nrm)

        claimed = self._claimed_stats(delta, honest_delta, loss_before, loss_after)
        return ClientUpdate(self.client_id, delta, self.n_samples,
                            float(loss_before), float(loss_after), train_time,
                            self.is_malicious, claimed, honest_delta)


    def _claimed_stats(self, delta, honest_delta, loss_before,
                       loss_after) -> Dict[str, float]:

        denom = max(abs(loss_before), 1e-6)
        improvement = float(np.clip((loss_before - loss_after) / denom, -1.0, 1.0))
        if self.is_malicious and self.forge_attestation and honest_delta is not None:
            return {"cosine": float("nan"),
                    "loss_improvement": improvement,
                    "norm": float(np.linalg.norm(honest_delta)),
                    "n_samples": float(self.n_samples)}
        return {"cosine": float("nan"),
                "loss_improvement": improvement,
                "norm": float(np.linalg.norm(delta)),
                "n_samples": float(self.n_samples)}
