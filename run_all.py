
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "runlogs"
PY = sys.executable

SEEDS = [42, 123, 2024, 7, 2025]


ABLATIONS = ["sketch_k", "trust_threshold", "dp_sigma", "ema_alpha",
             "evidence_alpha", "evidence_z0", "sketch_mode",
             "malicious_fraction", "dirichlet_alpha"]


def launch(name: str, cmd: Sequence[str]) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    fh = open(LOGS / f"{name}.log", "w", encoding="utf-8")
    print(f"  -> {name}: {' '.join(str(c) for c in cmd)}")
    return subprocess.Popen([str(c) for c in cmd], stdout=fh, stderr=subprocess.STDOUT)


def wait_all(procs: List[subprocess.Popen], label: str) -> None:
    t0 = time.time()
    codes = [p.wait() for p in procs]
    bad = [i for i, c in enumerate(codes) if c != 0]
    print(f"[{label}] finished in {(time.time() - t0) / 60:.1f} min"
          + (f"  ({len(bad)} of {len(codes)} failed - see runlogs/)" if bad else ""))


def stage_main(jobs: int) -> None:
    print("[main] 5 seeds in parallel")
    wait_all([launch(f"main_{s}",
                     [PY, ROOT / "scripts/run_experiment.py",
                      "--seeds", s, "--jobs", jobs, "--tag", f"s{s}"])
              for s in SEEDS], "main")


def stage_ablation(jobs: int, seeds: Sequence[int], rounds: int) -> None:


    print(f"[ablation] {len(ABLATIONS)} sweeps in parallel, "
          f"seeds={list(seeds)} rounds={rounds}")
    wait_all([launch(f"abl_{p}",
                     [PY, ROOT / "scripts/run_ablation.py", "--param", p,
                      "--seeds", *[str(s) for s in seeds],
                      "--rounds", rounds, "--jobs", jobs])
              for p in ABLATIONS], "ablation")


def stage_attacks(jobs: int, seeds: Sequence[int]) -> None:
    print("[attacks] coordinated-attack suite in parallel")
    procs = []
    for atk in ["sign_flip_scaled", "ipm", "alie", "min_max", "min_sum",
                "block_evasion", "unresolvable"]:
        procs.append(launch(f"atk_{atk}",
                            [PY, ROOT / "scripts/run_experiment.py",
                             "--attack", atk, "--tag", f"atk_{atk}",
                             "--jobs", jobs,
                             "--seeds", *[str(s) for s in seeds],
                             "--scenarios", "fedavg_attack", "fedmedian",
                             "krum", "bulyan", "foolsgold", "veritrust",
                             "veritrust_mamdani"]))
    wait_all(procs, "attacks")


def stage_forgery(jobs: int, seeds: Sequence[int]) -> None:
    print("[forgery] self-reported vs sketch, honest vs forging")
    wait_all([launch("forgery",
                     [PY, ROOT / "scripts/run_forgery_study.py",
                      "--seeds", *[str(s) for s in seeds], "--jobs", jobs])],
             "forgery")


def stage_privacy(trials: int) -> None:
    print("[privacy] leakage vs k, with DP epsilon")
    wait_all([launch("privacy",
                     [PY, ROOT / "scripts/run_privacy_eval.py",
                      "--trials", trials])], "privacy")


def stage_edgeiiot(jobs: int, seeds: Sequence[int]) -> None:

    print("[edgeiiot] second dataset, seeds in parallel")
    cfg = ROOT / "configs" / "edgeiiot.yaml"
    wait_all([launch(f"edge_{s}",
                     [PY, ROOT / "scripts/run_experiment.py",
                      "--config", cfg, "--seeds", s, "--jobs", jobs,
                      "--tag", f"e{s}"])
              for s in seeds], "edgeiiot")


def stage_tables() -> None:
    print("[tables] aggregating")
    subprocess.run([PY, str(ROOT / "scripts/analyze_results.py")], check=False)


    subprocess.run([PY, str(ROOT / "scripts/analyze_ablation.py")], check=False)
    subprocess.run([PY, str(ROOT / "scripts/make_figures.py")], check=False)

    if list((ROOT / "results").glob("e*_edgeiiot")):
        subprocess.run([PY, str(ROOT / "scripts/analyze_results.py"),
                        "--pattern", "e*_edgeiiot",
                        "--out", "results/edgeiiot/tables"], check=False)
        subprocess.run([PY, str(ROOT / "scripts/make_figures.py"),
                        "--pattern", "e*_edgeiiot",
                        "--out", "results/edgeiiot/figures"], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+",
                    default=["main", "privacy", "forgery", "attacks",
                             "ablation", "edgeiiot", "tables"])
    ap.add_argument("--skip-main", action="store_true")
    ap.add_argument("--jobs", type=int, default=4,
                    help="Paillier workers per child process")
    ap.add_argument("--abl-seeds", nargs="+", type=int, default=[42, 123])
    ap.add_argument("--abl-rounds", type=int, default=15)
    ap.add_argument("--attack-seeds", nargs="+", type=int, default=[42, 123, 2024])
    ap.add_argument("--privacy-trials", type=int, default=30)
    a = ap.parse_args()

    t0 = time.time()
    stages = [s for s in a.stages if not (s == "main" and a.skip_main)]
    print(f"stages: {stages}")
    if "main" in stages:
        stage_main(a.jobs)


    if "privacy" in stages:
        stage_privacy(a.privacy_trials)
    if "forgery" in stages:
        stage_forgery(a.jobs, a.attack_seeds)
    if "attacks" in stages:
        stage_attacks(a.jobs, a.attack_seeds)
    if "ablation" in stages:
        stage_ablation(a.jobs, a.abl_seeds, a.abl_rounds)
    if "edgeiiot" in stages:
        stage_edgeiiot(a.jobs, SEEDS)
    if "tables" in stages:
        stage_tables()
    print(f"\nall stages done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
