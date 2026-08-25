
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ..crypto.authority import DecryptionAuthority, PolicyViolation
from ..crypto.identity import ClientIdentity, Submission
from ..crypto.packing import PackingScheme, pack
from ..crypto.paillier_backend import PublicContext, ciphertext_bytes
from ..crypto.sketch import ProbeSet, project_plaintext
from ..crypto.shifted_sketch import (ShiftedProbeSet,
                                     project_plaintext_shifted)
from ..eval.metrics import compute_metrics
from ..crypto.sketch_noise import cosine_noise_std
from ..trust.features import TrustFeatures, build_features
from ..trust.normalise import EvidenceAccumulator
from ..trust.policy import Decision, ZeroTrustPolicy, quantise_weights
from .attacks import COORDINATED, apply_coordinated
from .client import ClientUpdate, FederatedClient, flatten_state, unflatten_state
from .robust import aggregate as robust_aggregate


def client_submit(pub: PublicContext, scheme: PackingScheme,
                  identity: ClientIdentity, round_idx: int,
                  delta: np.ndarray, n_samples: int = 0) -> Submission:

    cts = pub.encrypt_many(pack(delta, scheme))
    return Submission(identity.client_id, int(round_idx), cts,
                      identity.sign_submission(round_idx, cts), int(n_samples))


def client_submit_batch(pub: PublicContext, scheme: PackingScheme,
                        identities: Dict[int, ClientIdentity], round_idx: int,
                        updates: Sequence) -> List[Submission]:

    packed = [pack(u.delta, scheme) for u in updates]
    encrypted = pub.encrypt_batch(packed)
    out: List[Submission] = []
    for u, cts in zip(updates, encrypted):
        ident = identities[u.client_id]
        out.append(Submission(ident.client_id, int(round_idx), cts,
                              ident.sign_submission(round_idx, cts),
                              int(u.n_samples)))
    return out


@dataclass
class RoundReport:
    round_idx: int
    scenario: str
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    accepted: List[int]
    rejected: List[int]
    trust_raw: Dict[int, float]
    trust_smoothed: Dict[int, float]
    weights: Dict[int, float]
    n_malicious: int
    n_malicious_rejected: int
    n_honest_rejected: int


    n_accepted: int = 0
    n_rejected: int = 0
    t_train: float = 0.0
    t_encrypt: float = 0.0
    t_measure: float = 0.0
    t_trust: float = 0.0
    t_aggregate: float = 0.0
    t_total: float = 0.0
    bytes_uploaded: int = 0
    per_client: List[dict] = field(default_factory=list)


class FederatedServer:
    def __init__(self, cfg, scenario, model: nn.Module,
                 clients: Sequence[FederatedClient],
                 identities: Dict[int, ClientIdentity],
                 val: tuple, test: tuple, num_classes: int,
                 device: torch.device, label_names: Sequence[str],
                 pub: Optional[PublicContext] = None,
                 authority: Optional[DecryptionAuthority] = None,
                 scheme: Optional[PackingScheme] = None,
                 trust_engine=None, logger=None):
        self.cfg = cfg
        self.scenario = scenario
        self.name = str(scenario.get("name"))
        self.model = model.to(device)
        self.clients = list(clients)
        self.identities = identities
        self.Xv, self.yv = val
        self.Xt, self.yt = test
        self.num_classes = int(num_classes)
        self.device = device
        self.label_names = list(label_names)
        self.log = logger

        g = scenario.get
        self.aggregation = str(g("aggregation", "fedavg"))
        self.use_crypto = bool(g("crypto", False))
        self.use_trust = bool(g("trust", False))
        self.use_zt = bool(g("zero_trust", False))
        self.trim_ratio = float(g("trim_ratio", 0.2))
        self.attestation_source = str(g("attestation_source", "sketch"))

        self.pub = pub
        self.authority = authority
        self.scheme = scheme
        self.trust = trust_engine

        zt = cfg.zero_trust
        self.policy = ZeroTrustPolicy(
            threshold=float(zt.trust_threshold),
            ema_beta=float(zt.get("ema_alpha", 0.6)),
            reject_decay=float(zt.get("reject_decay", 0.7)),
            max_reject_streak=int(zt.get("max_reject_streak", 3)),
            min_accept_fraction=float(zt.get("min_accept_fraction", 0.5)),
        ) if self.use_zt else None

        att = cfg.experiments.attack
        self.attack_type = str(att.get("type", "sign_flip"))
        self.attack_eps = float(att.get("epsilon", 0.5))
        self.attack_pert = str(att.get("perturbation", "std"))
        self.num_byzantine = max(1, int(round(
            float(att.get("fraction", 0.3)) * len(self.clients))))

        self.weight_scale_bits = int(cfg.crypto.get("weight_scale_bits", 10))
        self.max_client_weight = float(
            cfg.crypto.authority.get("max_client_weight", 0.5))

        self._schema = [(k, tuple(v.shape)) for k, v in model.state_dict().items()]
        self._dim = int(sum(int(np.prod(s)) if s else 1 for _, s in self._schema))
        self._prev_aggregate: Optional[np.ndarray] = None
        self._calib_X: List[TrustFeatures] = []
        self._calib_y: List[float] = []


        self._calibrated = not (self.use_trust and self.trust is not None
                                and getattr(self.trust, "requires_calibration", False))


        if self.use_trust and self.attestation_source == "sketch"                 and not self.use_crypto:
            raise ValueError(
                f"scenario '{self.name}': attestation_source='sketch' requires "
                f"crypto=true (the sketch is derived from the submitted "
                f"ciphertexts). Set crypto: true, or use "
                f"attestation_source: 'self_reported'.")
        if self.use_crypto and (self.pub is None or self.authority is None
                                or self.scheme is None):
            raise ValueError(
                f"scenario '{self.name}': crypto=true but no public context, "
                f"decryption authority or packing scheme was supplied.")
        self.best_round = -1
        self.best_metric = float("nan")


    def run(self) -> List[RoundReport]:
        ms = self.cfg.get("model_selection")
        ms_on = bool(ms.get("enabled", True)) if ms else True
        best, best_state = -float("inf"), None
        reports: List[RoundReport] = []
        for r in range(int(self.cfg.federated.rounds)):
            rep = self._one_round(r)
            reports.append(rep)
            if self.log:
                self.log.info(
                    "[%s] r%02d/%d loss=%.4f acc=%.4f f1m=%.4f | acc/rej=%d/%d "
                    "mal_rej=%d/%d hon_rej=%d | %.1fs",
                    self.name, r + 1, self.cfg.federated.rounds, rep.val_loss,
                    rep.val_accuracy, rep.val_macro_f1, len(rep.accepted),
                    len(rep.rejected), rep.n_malicious_rejected, rep.n_malicious,
                    rep.n_honest_rejected, rep.t_total)
            if ms_on and np.isfinite(rep.val_macro_f1) and rep.val_macro_f1 > best:
                best = rep.val_macro_f1
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
                self.best_round = r
        if ms_on and best_state is not None:
            self.model.load_state_dict({k: v.to(self.device)
                                        for k, v in best_state.items()})
            self.best_metric = float(best)
            if self.log:
                self.log.info("[%s] restored best-validation checkpoint from "
                              "round %d (macro-F1=%.4f)", self.name,
                              self.best_round + 1, best)
        return reports


    def _one_round(self, r: int) -> RoundReport:
        t0 = time.time()
        global_flat = flatten_state(self.model.state_dict())


        ts = time.time()
        updates: List[ClientUpdate] = [c.train_round(self.model, global_flat)
                                       for c in self.clients]
        t_train = time.time() - ts


        if self.attack_type in COORDINATED:
            apply_coordinated(updates, self.attack_type, self.attack_eps,
                              self.attack_pert,
                              slots=(self.scheme.slots if self.scheme else 50),
                              reference=self._prev_aggregate)

        for u in updates:
            if not np.isfinite(u.delta).all():
                u.delta = np.nan_to_num(u.delta, nan=0.0, posinf=0.0, neginf=0.0)

        t_enc = t_meas = t_trust = 0.0
        n_bytes = 0
        sketches: Dict[int, np.ndarray] = {}
        probes: Optional[ProbeSet] = None
        subs: List[Submission] = []

        if self.use_crypto:

            ts = time.time()
            subs = client_submit_batch(self.pub, self.scheme, self.identities,
                                       r, updates)
            t_enc = time.time() - ts
            n_bytes = sum(len(s.ciphertexts) for s in subs) * \
                ciphertext_bytes(self.pub.public_key)


            ts = time.time()
            meas = self.authority.measure(r, subs)
            sketches, probes = meas.sketches, meas.probes
            t_meas = time.time() - ts
        else:
            n_bytes = len(updates) * self._dim * 4


        ids = [u.client_id for u in updates]
        ts = time.time()
        raw = self._score(r, updates, sketches, probes)
        t_trust = time.time() - ts


        if self.policy is not None:
            dec = self.policy.evaluate(ids, [raw[c] for c in ids])
        else:
            dec = Decision(list(ids), [], {c: 1.0 / len(ids) for c in ids},
                           dict(raw), dict(raw), [])


        ts = time.time()
        agg = self._aggregate(r, updates, dec)
        t_agg = time.time() - ts

        applied = self._apply(agg)
        if applied and dec.accepted:
            self._prev_aggregate = np.asarray(agg, dtype=np.float64).copy()


        vm = self._evaluate(self.Xv, self.yv)
        mal = {u.client_id for u in updates if u.is_malicious}
        rej = set(dec.rejected)
        per_client = [{
            "client_id": u.client_id, "is_malicious": u.is_malicious,
            "n_samples": u.n_samples, "loss_before": u.loss_before,
            "loss_after": u.loss_after,
            **{f"feat_{k}": float(v) for k, v in
               zip(("proj_ref", "norm_ratio", "peer_agreement"),
                   (getattr(self, "_last_features", {}).get(u.client_id).as_array()
                    if getattr(self, "_last_features", {}).get(u.client_id) is not None
                    else (float("nan"),) * 3))},
            "trust_raw": float(raw.get(u.client_id, 1.0)),
            "trust_smoothed": float(dec.smoothed.get(u.client_id, 1.0)),
            "weight": float(dec.weights.get(u.client_id, 0.0)),
            "accepted": u.client_id in dec.accepted,
        } for u in updates]

        return RoundReport(
            round_idx=r, scenario=self.name,
            train_loss=float(np.mean([u.loss_after for u in updates])),
            val_loss=vm["loss"], val_accuracy=vm["accuracy"],
            val_macro_f1=vm["macro_f1"], accepted=list(dec.accepted),
            rejected=list(dec.rejected), trust_raw=dict(raw),
            trust_smoothed=dict(dec.smoothed), weights=dict(dec.weights),
            n_malicious=len(mal), n_malicious_rejected=len(mal & rej),
            n_honest_rejected=len(rej - mal),
            n_accepted=len(dec.accepted), n_rejected=len(dec.rejected),
            t_train=t_train, t_encrypt=t_enc,
            t_measure=t_meas, t_trust=t_trust, t_aggregate=t_agg,
            t_total=time.time() - t0, bytes_uploaded=int(n_bytes),
            per_client=per_client)


    def _score(self, r: int, updates: Sequence[ClientUpdate],
               sketches: Dict[int, np.ndarray],
               probes: Optional[ProbeSet]) -> Dict[int, float]:
        ids = [u.client_id for u in updates]
        if not self.use_trust or self.trust is None:
            return {c: 1.0 for c in ids}

        if self.attestation_source == "self_reported":
            feats = self._legacy_features(updates)
        else:


            ref = None
            if self._prev_aggregate is not None and probes is not None:
                ref = (project_plaintext_shifted(self._prev_aggregate, probes)
                       if isinstance(probes, ShiftedProbeSet)
                       else project_plaintext(self._prev_aggregate, probes))
            feats = build_features(sketches, ref)


            feats = self._evidence(probes).update(feats)


        if not self._calibrated:
            for u in updates:
                self._calib_X.append(feats[u.client_id])
                self._calib_y.append(0.0 if u.is_malicious else 1.0)
            n_cal = int(self.cfg.trust.get("calib_rounds", 5))
            if r + 1 >= n_cal and len(set(self._calib_y)) > 1:
                self.trust.fit(self._calib_X, self._calib_y,
                               epochs=int(self.cfg.trust.get("calib_epochs", 300)))
                self._calibrated = True
                if self.log:
                    self.log.info("[%s] trust engine calibrated on %d labelled "
                                  "attestations", self.name, len(self._calib_y))

        scores = self.trust.score_many([feats[c] for c in ids])
        self._last_features = {c: feats[c] for c in ids}
        return {c: float(s) for c, s in zip(ids, scores)}

    def _evidence(self, probes) -> EvidenceAccumulator:

        acc = getattr(self, "_evidence_acc", None)
        if acc is None:
            tcfg = self.cfg.trust if hasattr(self.cfg, "trust") else {}
            sigma = 0.03
            if probes is not None:
                sigma = cosine_noise_std(
                    probes, int(getattr(probes, "dim", 0)) or 1,
                    trials=int(tcfg.get("noise_trials", 24)))
            acc = EvidenceAccumulator(
                alpha=float(tcfg.get("evidence_alpha", 0.4)),
                z0=float(tcfg.get("evidence_z0", 2.0)),
                sigma=sigma)
            self._evidence_acc = acc
            if self.log:
                self.log.info("[%s] sketch cosine noise sigma=%.4f, "
                              "effective sigma after smoothing=%.4f",
                              self.name, sigma, acc.sigma_eff)
        return acc

    def _legacy_features(self, updates) -> Dict[int, TrustFeatures]:

        ref = self._prev_aggregate
        out = {}
        norms = [float(u.claimed.get("norm", np.linalg.norm(u.delta)))
                 for u in updates]
        med = float(np.median(norms)) if norms else 1.0
        for u, nrm in zip(updates, norms):
            claimed_cos = u.claimed.get("cosine", float("nan"))
            if claimed_cos != claimed_cos:


                d = (u.honest_delta if u.honest_delta is not None
                     else u.delta).astype(np.float64)
                if ref is None or np.linalg.norm(ref) < 1e-12:
                    claimed_cos = 0.0
                else:
                    claimed_cos = float(np.clip(
                        d @ ref / (np.linalg.norm(d) * np.linalg.norm(ref) + 1e-12),
                        -1, 1))
            out[u.client_id] = TrustFeatures(
                float(claimed_cos),
                float(np.tanh(np.log(max(nrm / max(med, 1e-12), 1e-12)))),
                float(np.clip(u.claimed.get("loss_improvement", 0.0), -1, 1)))
        return out


    def _aggregate(self, r: int, updates: Sequence[ClientUpdate],
                   dec: Decision) -> np.ndarray:
        accepted = list(dec.accepted)
        if not accepted:
            if self.log:
                self.log.warning("[%s] round %d: no accepted clients; "
                                 "global model unchanged", self.name, r)
            return np.zeros(self._dim, dtype=np.float32)

        if self.use_crypto:
            wq = quantise_weights(dec.weights, accepted,
                                  scale_bits=self.weight_scale_bits,
                                  max_share=self.max_client_weight)
            try:
                return self.authority.open_aggregate(r, accepted, wq).astype(np.float32)
            except PolicyViolation as exc:


                if self.log:
                    self.log.error("[%s] round %d: authority refused the "
                                   "aggregate (%s); skipping round",
                                   self.name, r, exc)
                return np.zeros(self._dim, dtype=np.float32)

        by_id = {u.client_id: u for u in updates}
        deltas = [by_id[c].delta for c in accepted]
        weights = [dec.weights.get(c, 1.0 / len(accepted)) for c in accepted]
        kw = {}
        if self.aggregation == "trimmed_mean":
            kw["trim_ratio"] = self.trim_ratio
        if self.aggregation in ("krum", "multi_krum", "bulyan"):
            kw["num_byzantine"] = self.num_byzantine
        return robust_aggregate(self.aggregation, deltas, weights, **kw)


    def _apply(self, delta: np.ndarray) -> bool:
        if not np.isfinite(delta).all():
            if self.log:
                self.log.warning("[%s] non-finite aggregate; rolling back", self.name)
            return False
        flat = flatten_state(self.model.state_dict())
        new = flat + delta.astype(np.float32)
        if not np.isfinite(new).all():
            if self.log:
                self.log.warning("[%s] non-finite global weights; rolling back",
                                 self.name)
            return False
        state = unflatten_state(new, self._schema)
        self.model.load_state_dict({k: v.to(self.device) for k, v in state.items()})
        return True


    @torch.no_grad()
    def _evaluate(self, X: torch.Tensor, y: torch.Tensor,
                  with_confusion: bool = False) -> Dict:
        self.model.eval()
        lf = nn.CrossEntropyLoss(reduction="sum")
        tot, n, preds = 0.0, 0, []
        for i in range(0, X.shape[0], 16384):
            xb, yb = X[i:i + 16384], y[i:i + 16384]
            logits = self.model(xb)
            bl = float(lf(logits, yb).item())
            tot += bl if bl == bl else float("inf")
            preds.append(logits.argmax(1).cpu().numpy())
            n += xb.shape[0]
        mean = tot / max(n, 1)
        if not np.isfinite(mean):
            mean = 1e6
        return compute_metrics(y.cpu().numpy(), np.concatenate(preds),
                               self.num_classes, loss=mean,
                               label_names=self.label_names,
                               with_confusion=with_confusion)

    def evaluate_final(self) -> Dict:
        return self._evaluate(self.Xt, self.yt, with_confusion=True)
