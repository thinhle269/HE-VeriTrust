
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.crypto.authority import AuthorityPolicy, DecryptionAuthority
from src.crypto.identity import ClientIdentity
from src.crypto.packing import build_scheme
from src.crypto.paillier_backend import generate_contexts
from src.data.cic_preprocessor import CicIotPreprocessor
from src.data.npz_preprocessor import NpzPreprocessor
from src.data.partition import dirichlet_partition
from src.eval.metrics import class_weights, compute_metrics
from src.federated.client import FederatedClient
from src.federated.server import FederatedServer
from src.models.mlp import build_model, count_parameters
from src.trust.anfis import build_engine
from src.utils.common import get_logger, load_config, pick_device, set_global_seed
from src.utils.locking import ResultsLock


def parse_args():
    p = argparse.ArgumentParser(description="HE-VeriTrust experiments")
    p.add_argument("--config", default=str(ROOT / "configs" / "cic_iot.yaml"))
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--attack", default=None,
                   help="override experiments.attack.type")
    p.add_argument("--tag", default=None, help="suffix for the output directory")
    p.add_argument("--jobs", type=int, default=None,
                   help="worker processes for Paillier (default: crypto.n_jobs). "
                        "Set explicitly when running several seeds in parallel "
                        "so the pools do not oversubscribe the CPU.")
    p.add_argument("--force-lock", action="store_true",
                   help="take the results-directory lock even if another pid holds it")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def resolve_seeds(cfg, cli) -> List[int]:
    if cli:
        return [int(s) for s in cli]
    s = cfg.get("seeds")
    return [int(x) for x in s] if s else [int(cfg.seed)]


def apply_quick(cfg) -> None:
    cfg.federated["rounds"] = 2
    cfg.federated["num_clients"] = 4
    cfg.federated["local_epochs"] = 1
    cfg.crypto["key_size"] = 1024
    cfg.crypto.sketch["k"] = 8
    cfg.trust["calib_rounds"] = 1
    cfg.trust["calib_epochs"] = 50
    cfg.seeds = [42]


def run_centralized(cfg, split, device, logger, seed: int) -> Dict:

    set_global_seed(seed)
    model = build_model(cfg, split.num_features, split.num_classes).to(device)
    Xtr = torch.from_numpy(split.X_train).to(device)
    ytr = torch.from_numpy(split.y_train).to(device)
    Xv = torch.from_numpy(split.X_val).to(device)
    yv = torch.from_numpy(split.y_val).to(device)
    Xt = torch.from_numpy(split.X_test).to(device)
    yt = torch.from_numpy(split.y_test).to(device)

    counts = np.bincount(split.y_train, minlength=split.num_classes)
    cw = class_weights(counts, cfg.training.get("class_weighting", "none"),
                       float(cfg.training.get("cb_beta", 0.9999)))
    w = torch.tensor(cw, device=device) if cw is not None else None
    lf = nn.CrossEntropyLoss(weight=w,
                             label_smoothing=float(cfg.training.get("label_smoothing", 0.0)))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.federated.client_lr))
    bs = int(cfg.federated.local_batch_size)
    best, best_state = -float("inf"), None

    for _ in range(int(cfg.federated.rounds)):
        model.train()
        idx = torch.randperm(Xtr.shape[0], device=device)
        for s in range(0, Xtr.shape[0], bs):
            b = idx[s:s + bs]
            if b.numel() < 2:
                continue
            opt.zero_grad(set_to_none=True)
            lf(model(Xtr[b]), ytr[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([model(Xv[i:i + 16384]).argmax(1)
                            for i in range(0, Xv.shape[0], 16384)]).cpu().numpy()
        f = compute_metrics(split.y_val, pv, split.num_classes)["macro_f1"]
        if f > best:
            best = f
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        pt = torch.cat([model(Xt[i:i + 16384]).argmax(1)
                        for i in range(0, Xt.shape[0], 16384)]).cpu().numpy()
    test = compute_metrics(split.y_test, pt, split.num_classes,
                           label_names=split.label_names, with_confusion=True)
    logger.info("[centralized seed=%d] test macro-F1=%.4f acc=%.4f",
                seed, test["macro_f1"], test["accuracy"])
    return {"test": test, "state": model.state_dict()}


def build_clients(cfg, split, partition, device, malicious_flags,
                  seed: int) -> tuple:
    counts = np.bincount(split.y_train, minlength=split.num_classes)
    cw = class_weights(counts, cfg.training.get("class_weighting", "none"),
                       float(cfg.training.get("cb_beta", 0.9999)))
    cw_t = torch.tensor(cw) if cw is not None else None
    att = cfg.experiments.attack
    clients, identities = [], {}
    for cid, idx in partition.items():
        if len(idx) == 0:
            continue
        X = torch.from_numpy(split.X_train[idx])
        y = torch.from_numpy(split.y_train[idx])
        clients.append(FederatedClient(
            client_id=cid, X=X, y=y, device=device,
            lr=float(cfg.federated.client_lr),
            local_epochs=int(cfg.federated.local_epochs),
            batch_size=int(cfg.federated.local_batch_size),
            optimizer=str(cfg.federated.get("optimizer", "adam")),
            grad_clip_norm=float(cfg.federated.get("grad_clip_norm", 1.0)),
            max_update_norm=float(cfg.federated.get("max_update_norm", 10.0)),
            class_weight=cw_t,
            label_smoothing=float(cfg.training.get("label_smoothing", 0.0)),
            balanced_sampler=bool(cfg.training.get("balanced_sampler", False)),
            is_malicious=bool(malicious_flags[cid]),
            attack=str(att.get("type", "sign_flip")),
            noise_sigma=float(att.get("noise_sigma", 1.0)),
            num_classes=split.num_classes,
            forge_attestation=bool(att.get("forge_attestation", False)),
            seed=seed))
        identities[cid] = ClientIdentity.generate(cid)
    return clients, identities


def malicious_mask(cfg, n_clients: int, scen, rng) -> List[bool]:
    att = cfg.experiments.attack
    if not bool(att.get("enabled", True)) or not bool(scen.get("malicious", False)):
        return [False] * n_clients
    k = int(round(float(att.get("fraction", 0.3)) * n_clients))
    flags = [False] * n_clients
    for c in rng.choice(n_clients, size=max(k, 0), replace=False):
        flags[int(c)] = True
    return flags


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.quick:
        apply_quick(cfg)
    if args.rounds:
        cfg.federated["rounds"] = int(args.rounds)
    if args.jobs:
        cfg.crypto["n_jobs"] = int(args.jobs)
    if args.attack:
        cfg.experiments.attack["type"] = str(args.attack)
    if args.tag:
        for k in ("results_dir", "csv_dir", "figures_dir", "models_dir", "logs_dir"):
            cfg.paths[k] = str(cfg.paths[k]).replace("results/", f"results/{args.tag}_", 1)

    out = ROOT / cfg.paths.results_dir
    csv_dir = ROOT / cfg.paths.csv_dir
    for d in (out, csv_dir, ROOT / cfg.paths.models_dir, ROOT / cfg.paths.logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    lock = ResultsLock(out, label=f"run_experiment seeds={args.seeds}").acquire(
        force=args.force_lock)
    logger = get_logger("run", ROOT / cfg.paths.logs_dir, cfg.logging.level)

    device = pick_device(cfg.device)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    logger.info("device=%s | torch threads=%d", device, torch.get_num_threads())

    source = str(cfg.data.get('source', 'cic_tree'))
    prep = (NpzPreprocessor if source == 'npz' else CicIotPreprocessor)
    split = prep(cfg, ROOT).run()
    logger.info("data: train=%d val=%d test=%d features=%d classes=%d",
                len(split.y_train), len(split.y_val), len(split.y_test),
                split.num_features, split.num_classes)
    logger.info("train class counts: %s", split.class_counts())

    Xv = torch.from_numpy(split.X_val).to(device)
    yv = torch.from_numpy(split.y_val).to(device)
    Xt = torch.from_numpy(split.X_test).to(device)
    yt = torch.from_numpy(split.y_test).to(device)

    scen_by_name = {s["name"]: s for s in cfg.experiments.scenarios}
    wanted = args.scenarios or list(scen_by_name)
    seeds = resolve_seeds(cfg, args.seeds)
    logger.info("scenarios=%s seeds=%s rounds=%d", wanted, seeds,
                cfg.federated.rounds)

    probe = build_model(cfg, split.num_features, split.num_classes)


    dim = int(sum(v.numel() for v in probe.state_dict().values()))
    n_params = count_parameters(probe)
    scheme = build_scheme(cfg.crypto)
    logger.info("model params=%d | encrypted dim=%d | packing: %s",
                n_params, dim, scheme.describe())
    logger.info("ciphertexts/client/round = %d (vs %d unpacked)",
                scheme.n_blocks(dim), dim)

    rows_round, rows_client, rows_seed, rows_pc = [], [], [], []

    for seed in seeds:
        logger.info("=" * 70)
        logger.info("SEED %d", seed)
        partition = dirichlet_partition(
            split.y_train, int(cfg.federated.num_clients),
            float(cfg.federated.dirichlet_alpha), seed)

        for name in wanted:
            if name == "centralized":
                t0 = time.time()
                cr = run_centralized(cfg, split, device, logger, seed)
                rows_seed.append({"scenario": "centralized", "seed": seed,
                                  **_flat_test(cr["test"])})
                rows_pc += _pc_rows("centralized", seed, cr["test"])
                torch.save(cr["state"], ROOT / cfg.paths.models_dir /
                           f"centralized_seed{seed}.pt")
                logger.info("[centralized seed=%d] %.1fs", seed, time.time() - t0)
                continue
            if name not in scen_by_name:
                logger.warning("unknown scenario %s - skipped", name)
                continue

            scen = scen_by_name[name]
            set_global_seed(seed)
            rng = np.random.default_rng(seed)
            flags = malicious_mask(cfg, int(cfg.federated.num_clients), scen, rng)
            model = build_model(cfg, split.num_features, split.num_classes).to(device)
            clients, identities = build_clients(cfg, split, partition, device,
                                                flags, seed)

            pub = authority = None
            if bool(scen.get("crypto", False)) and bool(cfg.crypto.get("enabled", True)):
                t0 = time.time()
                pub, sec = generate_contexts(int(cfg.crypto.key_size),
                                             n_jobs=int(cfg.crypto.get("n_jobs", -1)))
                sk_cfg = cfg.crypto.sketch
                dp = sk_cfg.dp
                authority = DecryptionAuthority(
                    sec, scheme, AuthorityPolicy.from_config(cfg.crypto.authority),
                    dim=dim, sketch_k=int(sk_cfg.get("k", 32)),
                    sketch_density=float(sk_cfg.get("density", 0.5)),
                    dp_enabled=bool(dp.get("enabled", True)),
                    dp_relative_sigma=float(dp.get("sigma", 0.05)),
                    clip_norm=float(cfg.federated.get("max_update_norm", 10.0)),
                    sketch_mode=str(sk_cfg.get("mode", "shifted")),
                    n_jobs=int(cfg.crypto.get("n_jobs", -1)))
                for cid, ident in identities.items():
                    authority.enrol(cid, ident.public_bytes)
                logger.info("[%s] Paillier %d-bit keypair + authority ready (%.1fs)",
                            name, cfg.crypto.key_size, time.time() - t0)

            engine = None
            if bool(scen.get("trust", False)):

                tcfg = copy.deepcopy(cfg.trust)
                if scen.get("trust_engine"):
                    tcfg["engine"] = str(scen.get("trust_engine"))
                engine = build_engine(tcfg, device="cpu", seed=seed)

            server = FederatedServer(
                cfg, scen, model, clients, identities, (Xv, yv), (Xt, yt),
                split.num_classes, device, split.label_names, pub=pub,
                authority=authority, scheme=scheme, trust_engine=engine,
                logger=logger)

            t0 = time.time()
            reports = server.run()
            test = server.evaluate_final()
            logger.info("[%s seed=%d] %.1fs | test macro-F1=%.4f acc=%.4f",
                        name, seed, time.time() - t0, test["macro_f1"],
                        test["accuracy"])

            torch.save(server.model.state_dict(),
                       ROOT / cfg.paths.models_dir / f"{name}_seed{seed}.pt")
            for rep in reports:
                rows_round.append({"scenario": name, "seed": seed,
                                   **{k: v for k, v in rep.__dict__.items()
                                      if not isinstance(v, (dict, list))}})
                for pc in rep.per_client:
                    rows_client.append({"scenario": name, "seed": seed,
                                        "round_idx": rep.round_idx, **pc})
            rows_seed.append({"scenario": name, "seed": seed, **_flat_test(test)})
            rows_pc += _pc_rows(name, seed, test)
            if authority is not None:
                pd.DataFrame(authority.audit_table()).to_csv(
                    csv_dir / f"audit_{name}_seed{seed}.csv", index=False)

            _flush(csv_dir, rows_round, rows_client, rows_seed, rows_pc)

    summary = _summarise(csv_dir, rows_seed)
    logger.info("\n%s", summary.to_string(index=False))
    (out / "config_used.json").write_text(
        json.dumps(cfg.to_plain(), indent=2, default=str), encoding="utf-8")
    lock.release()
    logger.info("done -> %s", out)


def _flat_test(t: Dict) -> Dict:
    return {"test_accuracy": t["accuracy"], "test_macro_f1": t["macro_f1"],
            "test_weighted_f1": t["weighted_f1"],
            "test_min_class_f1": t.get("min_class_f1", float("nan")),
            "test_loss": t.get("loss", float("nan"))}


def _pc_rows(scen: str, seed: int, t: Dict) -> List[Dict]:
    return [{"scenario": scen, "seed": seed, "class": k, "f1": v}
            for k, v in (t.get("per_class_f1") or {}).items()]


def _flush(csv_dir: Path, rr, rc, rs, rp) -> None:
    if rr:
        pd.DataFrame(rr).to_csv(csv_dir / "per_round.csv", index=False)
    if rc:
        pd.DataFrame(rc).to_csv(csv_dir / "per_client.csv", index=False)
    if rs:
        pd.DataFrame(rs).to_csv(csv_dir / "per_seed.csv", index=False)
    if rp:
        pd.DataFrame(rp).to_csv(csv_dir / "per_class_f1.csv", index=False)


def _summarise(csv_dir: Path, rows_seed) -> pd.DataFrame:
    df = pd.DataFrame(rows_seed)
    if df.empty:
        return df
    cols = [c for c in df.columns if c.startswith("test_")]
    g = df.groupby("scenario")[cols].agg(["mean", "std", "count"]).reset_index()
    g.columns = ["scenario" if a == "scenario" else f"{a}_{b}"
                 for a, b in g.columns]
    g = g.sort_values("test_macro_f1_mean", ascending=False)
    g.to_csv(csv_dir / "summary.csv", index=False)
    return g


if __name__ == "__main__":
    main()
