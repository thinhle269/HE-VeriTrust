
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, RobustScaler,
                                   StandardScaler)

from ..utils.common import get_logger
from .cic_preprocessor import Split

CACHE_VERSION = "2.1.0"


def _make_scaler(name: str):
    return {"standard": StandardScaler, "minmax": MinMaxScaler,
            "robust": RobustScaler}.get(str(name).lower(), StandardScaler)()


class NpzPreprocessor:
    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.root = Path(project_root)
        d = cfg.data
        self.npz_path = Path(d.npz_path)
        self.name = str(d.get("name", self.npz_path.stem))
        self.max_rows_per_class = int(d.get("max_rows_per_class", 0) or 0)
        self.scaler_name = str(d.get("scaler", "standard"))
        self.test_size = float(d.get("test_size", 0.15))
        self.val_size = float(d.get("val_size", 0.15))
        self.seed = int(cfg.seed)
        self.cache_dir = self.root / d.get("processed_dir", "data/processed_npz")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = get_logger(f"data.{self.name}",
                              self.root / cfg.paths.logs_dir, cfg.logging.level)

    def _fingerprint(self) -> str:
        payload = {"version": CACHE_VERSION, "path": str(self.npz_path),
                   "cap": self.max_rows_per_class, "scaler": self.scaler_name,
                   "test": self.test_size, "val": self.val_size,
                   "seed": self.seed}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()
                              ).hexdigest()[:16]

    def run(self, force: bool = False) -> Split:
        fp = self._fingerprint()
        npz_p = self.cache_dir / f"{self.name}_{fp}.npz"
        meta_p = self.cache_dir / f"{self.name}_{fp}.meta.json"
        sc_p = self.cache_dir / f"{self.name}_{fp}.scaler.joblib"
        if not force and npz_p.exists() and meta_p.exists() and sc_p.exists():
            self.log.info("Loading cached split %s", npz_p.name)
            with np.load(npz_p) as z:
                arrays = {k: z[k] for k in z.files}
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            return Split(feature_names=meta["feature_names"],
                         label_names=meta["label_names"],
                         scaler=joblib.load(sc_p), **arrays)

        meta_src = self.npz_path.with_suffix(".meta.json")
        src_meta = json.loads(meta_src.read_text(encoding="utf-8")) \
            if meta_src.exists() else {}
        data = np.load(self.npz_path, allow_pickle=False)
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"]).astype(np.int64)

        finite = np.isfinite(X).all(axis=1)
        if not finite.all():
            self.log.info("Dropping %d non-finite rows", int((~finite).sum()))
            X, y = X[finite], y[finite]

        if self.max_rows_per_class > 0:
            rng = np.random.default_rng(self.seed)
            keep = []
            for c in np.unique(y):
                ci = np.where(y == c)[0]
                if len(ci) > self.max_rows_per_class:
                    ci = rng.choice(ci, self.max_rows_per_class, replace=False)
                keep.append(ci)
            keep = np.concatenate(keep)
            rng.shuffle(keep)
            X, y = X[keep], y[keep]

        enc = LabelEncoder()
        y = enc.fit_transform(y).astype(np.int64)
        names = src_meta.get("class_names")
        label_names = ([str(c) for c in names]
                       if names and len(names) == len(enc.classes_)
                       else [str(c) for c in enc.classes_])
        feature_names = (src_meta.get("feature_names")
                         or [f"f{i}" for i in range(X.shape[1])])

        idx = np.arange(len(y))
        strat = y if np.min(np.bincount(y)) >= 2 else None
        tr, te = train_test_split(idx, test_size=self.test_size, stratify=strat,
                                  random_state=self.seed)
        tr, va = train_test_split(
            tr, test_size=self.val_size / (1 - self.test_size),
            stratify=y[tr] if strat is not None else None,
            random_state=self.seed)

        scaler = _make_scaler(self.scaler_name)
        X_tr = scaler.fit_transform(X[tr]).astype(np.float32)
        X_va = scaler.transform(X[va]).astype(np.float32)
        X_te = scaler.transform(X[te]).astype(np.float32)

        split = Split(X_train=X_tr, y_train=y[tr], X_val=X_va, y_val=y[va],
                      X_test=X_te, y_test=y[te],
                      feature_names=list(feature_names),
                      label_names=list(label_names), scaler=scaler)
        np.savez_compressed(npz_p, X_train=X_tr, y_train=y[tr], X_val=X_va,
                            y_val=y[va], X_test=X_te, y_test=y[te])
        joblib.dump(scaler, sc_p)
        meta_p.write_text(json.dumps({
            "version": CACHE_VERSION, "feature_names": list(feature_names),
            "label_names": list(label_names),
            "train_class_counts": split.class_counts(),
            "shapes": {"train": list(X_tr.shape), "val": list(X_va.shape),
                       "test": list(X_te.shape)}}, indent=2), encoding="utf-8")
        self.log.info("%s ready: train=%d val=%d test=%d features=%d classes=%d",
                      self.name, len(split.y_train), len(split.y_val),
                      len(split.y_test), split.num_features, split.num_classes)
        self.log.info("train class counts: %s", split.class_counts())
        return split
