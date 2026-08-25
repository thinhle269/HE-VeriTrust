
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, RobustScaler,
                                   StandardScaler)

from ..utils.common import get_logger

CACHE_VERSION = "2.1.0"


SUPERCLASS_RULES: List[Tuple[str, str]] = [
    ("benign", "Benign"),
    ("ddos-", "DDoS"),
    ("dos-", "DoS"),
    ("mirai-", "Mirai"),
    ("recon-", "Recon"),
    ("vulnerabilityscan", "Recon"),
    ("dns_spoofing", "Spoofing"),
    ("mitm-", "Spoofing"),
    ("dictionarybruteforce", "BruteForce"),
    ("backdoor_malware", "Web"),
    ("browserhijacking", "Web"),
    ("commandinjection", "Web"),
    ("sqlinjection", "Web"),
    ("uploading_attack", "Web"),
    ("xss", "Web"),
]

CLASS_ORDER = ["Benign", "BruteForce", "DDoS", "DoS", "Mirai", "Recon",
               "Spoofing", "Web"]


def map_superclass(dirname: str) -> Optional[str]:
    key = dirname.strip().lower()
    for prefix, target in SUPERCLASS_RULES:
        if key.startswith(prefix):
            return target
    return None


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    label_names: List[str]
    scaler: object = None

    @property
    def num_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def class_counts(self) -> Dict[str, int]:
        c = np.bincount(self.y_train, minlength=self.num_classes)
        return {n: int(v) for n, v in zip(self.label_names, c)}


def _make_scaler(name: str):
    return {"standard": StandardScaler, "minmax": MinMaxScaler,
            "robust": RobustScaler}[str(name).lower()]()


class CicIotPreprocessor:
    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.root = Path(project_root)
        d = cfg.data
        self.csv_root = Path(d.csv_root)
        self.per_type_cap = int(d.get("per_type_cap", 60000))
        self.max_rows_per_class = int(d.get("max_rows_per_class", 40000))
        self.corr_threshold = float(d.get("drop_correlated_threshold", 0.98))
        self.scaler_name = str(d.get("scaler", "standard"))
        self.test_size = float(d.get("test_size", 0.15))
        self.val_size = float(d.get("val_size", 0.15))
        self.seed = int(cfg.seed)
        self.cache_dir = self.root / d.get("processed_dir", "data/processed_cic")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger("data.cic", self.root / cfg.paths.logs_dir,
                              level=cfg.logging.level)


    def _fingerprint(self) -> str:
        payload = {
            "version": CACHE_VERSION,
            "csv_root": str(self.csv_root),
            "per_type_cap": self.per_type_cap,
            "max_rows_per_class": self.max_rows_per_class,
            "corr_threshold": self.corr_threshold,
            "scaler": self.scaler_name,
            "test_size": self.test_size,
            "val_size": self.val_size,
            "seed": self.seed,
        }
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def _paths(self):
        fp = self._fingerprint()
        return (self.cache_dir / f"cic_{fp}.npz",
                self.cache_dir / f"cic_{fp}.meta.json",
                self.cache_dir / f"cic_{fp}.scaler.joblib")


    def run(self, force: bool = False) -> Split:
        npz_p, meta_p, sc_p = self._paths()
        if not force and npz_p.exists() and meta_p.exists() and sc_p.exists():
            self.log.info("Loading cached split %s", npz_p.name)
            with np.load(npz_p) as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return Split(feature_names=meta["feature_names"],
                         label_names=meta["label_names"],
                         scaler=joblib.load(sc_p), **arrays)

        df, provenance = self._load_raw()
        df = self._clean(df)
        df = self._cap_per_class(df)

        feat_cols = [c for c in df.columns if c != "Label"]
        X = df[feat_cols].to_numpy(dtype=np.float32)
        enc = LabelEncoder().fit(CLASS_ORDER)
        y = enc.transform(df["Label"].to_numpy()).astype(np.int64)
        del df


        idx = np.arange(len(y))
        tr, te = train_test_split(idx, test_size=self.test_size,
                                  stratify=y, random_state=self.seed)
        tr, va = train_test_split(tr, test_size=self.val_size / (1 - self.test_size),
                                  stratify=y[tr], random_state=self.seed)

        keep, feat_cols = self._select_features(X[tr], feat_cols)
        X_tr, X_va, X_te = X[tr][:, keep], X[va][:, keep], X[te][:, keep]

        scaler = _make_scaler(self.scaler_name)
        X_tr = scaler.fit_transform(X_tr).astype(np.float32)
        X_va = scaler.transform(X_va).astype(np.float32)
        X_te = scaler.transform(X_te).astype(np.float32)

        split = Split(X_train=X_tr, y_train=y[tr], X_val=X_va, y_val=y[va],
                      X_test=X_te, y_test=y[te], feature_names=feat_cols,
                      label_names=list(enc.classes_), scaler=scaler)

        np.savez_compressed(npz_p, X_train=X_tr, y_train=y[tr], X_val=X_va,
                            y_val=y[va], X_test=X_te, y_test=y[te])
        joblib.dump(scaler, sc_p)
        meta_p.write_text(json.dumps({
            "version": CACHE_VERSION,
            "fingerprint": self._fingerprint(),
            "feature_names": feat_cols,
            "label_names": list(enc.classes_),
            "train_class_counts": split.class_counts(),
            "shapes": {"train": list(X_tr.shape), "val": list(X_va.shape),
                       "test": list(X_te.shape)},
            "provenance": provenance,
        }, indent=2), encoding="utf-8")
        self.log.info("Cached split -> %s", npz_p.name)
        self.log.info("Train class counts: %s", split.class_counts())
        return split


    def _load_raw(self):
        if not self.csv_root.is_dir():
            raise FileNotFoundError(
                f"CIC-IoT CSV tree not found at {self.csv_root}. "
                f"Set data.csv_root in the config.")
        dirs = sorted(p for p in self.csv_root.iterdir() if p.is_dir())
        if not dirs:
            raise FileNotFoundError(f"No attack-type directories under {self.csv_root}")
        frames, provenance = [], []
        rng = np.random.default_rng(self.seed)
        for d in dirs:
            sup = map_superclass(d.name)
            if sup is None:
                self.log.warning("Unmapped attack directory %s - skipped", d.name)
                continue
            files = sorted(d.glob("*.csv"))
            if not files:
                continue
            got, parts = 0, []
            for f in files:
                if got >= self.per_type_cap:
                    break
                chunk = pd.read_csv(f, nrows=self.per_type_cap - got,
                                    low_memory=False)
                parts.append(chunk)
                got += len(chunk)
            sub = pd.concat(parts, ignore_index=True) if parts else None
            if sub is None or sub.empty:
                continue
            if "Label" in sub.columns:
                sub = sub.drop(columns=["Label"])
            sub["Label"] = sup
            frames.append(sub)
            provenance.append({"dir": d.name, "superclass": sup, "rows": int(len(sub))})
            self.log.info("  %-26s -> %-10s %7d rows", d.name, sup, len(sub))
        df = pd.concat(frames, ignore_index=True)

        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
        self.log.info("Loaded %d rows / %d columns from %d attack types",
                      len(df), df.shape[1], len(provenance))
        return df, provenance

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        df = df.drop_duplicates()
        self.log.info("Cleaning removed %d rows (non-finite + duplicates) -> %d",
                      before - len(df), len(df))
        return df

    def _cap_per_class(self, df: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        parts = []
        for label, grp in df.groupby("Label", sort=False):
            if len(grp) > self.max_rows_per_class:
                sel = rng.choice(len(grp), size=self.max_rows_per_class, replace=False)
                grp = grp.iloc[sel]
            parts.append(grp)
        out = pd.concat(parts, ignore_index=True)
        out = out.iloc[rng.permutation(len(out))].reset_index(drop=True)
        dist = out["Label"].value_counts().to_dict()
        self.log.info("After per-class cap %d: %d rows | %s",
                      self.max_rows_per_class, len(out), dist)
        return out

    def _select_features(self, X_train: np.ndarray, cols: List[str]):

        keep = np.ones(X_train.shape[1], dtype=bool)
        std = X_train.std(axis=0)
        const = std < 1e-8
        keep &= ~const
        if const.any():
            self.log.info("Dropping %d constant features: %s", int(const.sum()),
                          [cols[i] for i in np.where(const)[0]])
        thr = self.corr_threshold
        if 0 < thr < 1:
            idx = np.where(keep)[0]
            if len(idx) > 1:
                corr = np.nan_to_num(np.corrcoef(X_train[:, idx], rowvar=False))
                drop = set()
                for i in range(corr.shape[0]):
                    if i in drop:
                        continue
                    for j in range(i + 1, corr.shape[1]):
                        if j not in drop and abs(corr[i, j]) >= thr:
                            drop.add(j)
                if drop:
                    keep[idx[list(drop)]] = False
                    self.log.info("Dropping %d correlated features (|r|>=%.2f): %s",
                                  len(drop), thr, [cols[idx[j]] for j in sorted(drop)])
        return keep, [cols[i] for i in np.where(keep)[0]]
