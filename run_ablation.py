
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.common import get_logger, load_config


SWEEPS: Dict[str, Dict] = {
    "sketch_k": {
        "path": "crypto.sketch.k",
        "values": [8, 16, 32, 64, 128],
        "scenarios": ["veritrust"],
        "chosen": 32,
        "desc": "sketch probes k (privacy <-> robustness dial)",
    },
    "trust_threshold": {
        "path": "zero_trust.trust_threshold",
        "values": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "scenarios": ["veritrust"],
        "chosen": 0.40,
        "desc": "Zero-Trust acceptance threshold tau",
    },
    "evidence_alpha": {
        "path": "trust.evidence_alpha",


        "values": [0.20, 0.40, 0.60, 0.80, 1.00],
        "scenarios": ["veritrust"],
        "chosen": 0.40,
        "desc": "EMA weight on the newest FEATURE sample (evidence smoothing)",
    },
    "evidence_z0": {
        "path": "trust.evidence_z0",


        "values": [1.0, 1.5, 2.0, 3.0, 4.0],
        "scenarios": ["veritrust"],
        "chosen": 2.0,
        "desc": "resolvability scale z0 for the normalised trust features",
    },
    "ema_alpha": {
        "path": "zero_trust.ema_alpha",
        "values": [0.20, 0.40, 0.60, 0.80, 1.00],
        "scenarios": ["veritrust"],
        "chosen": 0.60,
        "desc": "EMA weight on the newest trust sample",
    },
    "dp_sigma": {
        "path": "crypto.sketch.dp.sigma",


        "values": [0.0, 0.0005, 0.001, 0.002, 0.005, 0.02],
        "scenarios": ["veritrust"],
        "chosen": 0.0,
        "desc": "Gaussian noise multiplier on the released sketch",


        "extra": {"crypto.sketch.dp.enabled": True},
    },
    "sketch_mode": {
        "path": "crypto.sketch.mode",


        "values": ["block", "shifted"],
        "scenarios": ["veritrust"],
        "chosen": "shifted",
        "desc": "probe geometry (block-constant vs post-commitment shifted)",
    },
    "malicious_fraction": {
        "path": "experiments.attack.fraction",
        "values": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
        "scenarios": ["fedavg_attack", "krum", "veritrust"],
        "chosen": 0.30,
        "desc": "Byzantine fraction",
    },
    "dirichlet_alpha": {
        "path": "federated.dirichlet_alpha",
        "values": [0.10, 0.30, 0.50, 1.00, 5.00],
        "scenarios": ["fedavg_attack", "veritrust"],
        "chosen": 0.50,
        "desc": "Dirichlet concentration (lower = more non-IID)",
    },
}


def set_nested(cfg, path: str, value) -> None:
    parts = path.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = obj[p]
    obj[parts[-1]] = value


def run_point(param: str, value, seeds: Sequence[int], rounds: int,
              jobs: int, base_config: Path, log, attack: str = None,
              suffix: str = "") -> pd.DataFrame:

    spec = SWEEPS[param]
    tag = (f"abl_{param}{suffix}_{value:g}" if isinstance(value, float)
           else f"abl_{param}{suffix}_{value}")
    cfg = load_config(base_config)
    set_nested(cfg, spec["path"], value)
    for path, val in (spec.get("extra") or {}).items():
        set_nested(cfg, path, val)
    cfg.federated["rounds"] = int(rounds)
    cfg["seeds"] = [int(s) for s in seeds]
    tmp = ROOT / "configs" / f"_{tag}.yaml"
    import yaml
    tmp.write_text(yaml.safe_dump(cfg.to_plain(), sort_keys=False), encoding="utf-8")

    cmd = [sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
           "--config", str(tmp), "--tag", tag, "--jobs", str(jobs),
           "--scenarios", *spec["scenarios"],
           "--seeds", *[str(s) for s in seeds]]
    if attack:
        cmd += ["--attack", str(attack)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    tmp.unlink(missing_ok=True)
    if proc.returncode != 0:
        log.error("%s=%s FAILED after %.0fs:\n%s", param, value,
                  time.time() - t0, proc.stderr[-2000:])
        return pd.DataFrame()
    log.info("%s=%-6s done in %.0fs", param, value, time.time() - t0)

    per_seed = ROOT / "results" / f"{tag}_cic_iot" / "csv" / "per_seed.csv"
    if not per_seed.exists():
        return pd.DataFrame()
    df = pd.read_csv(per_seed)
    df.insert(0, "param", param)
    df.insert(1, "value", value)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cic_iot.yaml"))
    ap.add_argument("--param", nargs="+", default=["all"])
    ap.add_argument("--values", nargs="+", type=float, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--attack", default=None,
                    help="override the configured attack for these sweeps; "
                         "sketch_mode is only informative under block_evasion")
    ap.add_argument("--suffix", default="",
                    help="appended to the output filenames, so a sweep re-run "
                         "under a different attack sits beside the default one "
                         "instead of overwriting it")
    args = ap.parse_args()

    params = list(SWEEPS) if args.param == ["all"] else args.param
    if args.values is not None and len(params) > 1:
        raise SystemExit("--values applies to a single --param")

    base = load_config(args.config)
    out = ROOT / base.paths.results_dir / "ablation"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("ablation", ROOT / base.paths.logs_dir, base.logging.level)
    log.info("sweeps=%s seeds=%s rounds=%d", params, args.seeds, args.rounds)

    index = []
    for param in params:
        spec = SWEEPS[param]
        values = args.values if args.values is not None else spec["values"]
        if param == "sketch_k":
            values = [int(v) for v in values]
        if param == "sketch_mode":
            values = [str(v) for v in values]
        frames = []
        for v in values:
            df = run_point(param, v, args.seeds, args.rounds, args.jobs,
                           Path(args.config), log, attack=args.attack,
                           suffix=args.suffix)
            if not df.empty:
                frames.append(df)
                pd.concat(frames, ignore_index=True).to_csv(
                    out / f"sweep_{param}{args.suffix}_raw.csv", index=False)
        if not frames:
            log.warning("no results for %s", param)
            continue
        raw = pd.concat(frames, ignore_index=True)
        summ = (raw.groupby(["param", "value", "scenario"])
                [["test_macro_f1", "test_accuracy", "test_min_class_f1"]]
                .agg(["mean", "std"]).reset_index())
        summ.columns = ["_".join(c).strip("_") if isinstance(c, tuple) else c
                        for c in summ.columns]
        summ.to_csv(out / f"sweep_{param}{args.suffix}_summary.csv", index=False)
        log.info("\n%s", summ.round(4).to_string(index=False))
        index.append({"param": param, "desc": spec["desc"],
                      "values": list(values), "chosen": spec.get("chosen"),
                      "seeds": args.seeds, "rounds": args.rounds,
                      "note": "trend sweep: fewer seeds and rounds than the "
                              "headline run"})
        (out / "index.json").write_text(json.dumps(index, indent=2, default=str),
                                        encoding="utf-8")
    log.info("ablations -> %s", out)


if __name__ == "__main__":
    main()
