from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import List
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler
from ..utils.common import get_logger
from .cic_preprocessor import Split
CACHE_VERSION = 'edge-1.0.0'
LABEL_COL = 'Attack_type'
IDENTIFYING = ('frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4', 'http.request.full_uri', 'http.request.uri.query', 'http.file_data', 'http.referer', 'tcp.payload', 'tcp.options', 'tcp.srcport', 'dns.qry.name', 'mqtt.msg', 'mqtt.topic', 'mqtt.protoname', 'mqtt.msg_decoded_as', 'Attack_label')
PACKET_IDS = ('tcp.ack_raw', 'tcp.ack', 'tcp.checksum', 'tcp.seq')

def _make_scaler(name: str):
    return {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}.get(str(name).lower(), StandardScaler)()

class EdgeIIoTPreprocessor:

    def __init__(self, cfg, project_root: Path):
        self.cfg = cfg
        self.root = Path(project_root)
        d = cfg.data
        self.csv_path = Path(str(d.csv_path))
        self.cache_dir = self.root / d.get('processed_dir', 'data/processed_edge')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.test_size = float(d.get('test_size', 0.15))
        self.val_size = float(d.get('val_size', 0.15))
        self.scaler_kind = str(d.get('scaler', 'standard'))
        self.corr_threshold = float(d.get('drop_correlated_threshold', 0.98))
        self.deduplicate = bool(d.get('deduplicate', True))
        self.drop_packet_ids = bool(d.get('drop_packet_ids', False))
        self.max_rows_per_class = int(d.get('max_rows_per_class', 0))
        self.seed = int(cfg.get('seed', 42))
        self.log = get_logger('data.edge', self.root / cfg.paths.logs_dir, cfg.logging.level)

    def _fingerprint(self) -> str:
        payload = {'version': CACHE_VERSION, 'path': str(self.csv_path), 'test': self.test_size, 'val': self.val_size, 'scaler': self.scaler_kind, 'corr': self.corr_threshold, 'dedup': self.deduplicate, 'cap': self.max_rows_per_class, 'no_pkt_ids': self.drop_packet_ids, 'seed': self.seed}
        return hashlib.blake2b(json.dumps(payload, sort_keys=True).encode(), digest_size=8).hexdigest()

    def _paths(self):
        fp = self._fingerprint()
        return (self.cache_dir / f'edgeiiot_{fp}.npz', self.cache_dir / f'edgeiiot_{fp}.meta.json', self.cache_dir / f'edgeiiot_{fp}.scaler.joblib')

    def _select_features(self, X_train: np.ndarray, cols: List[str]):
        keep = np.ones(X_train.shape[1], dtype=bool)
        std = X_train.std(axis=0)
        const = std < 1e-08
        keep &= ~const
        if const.any():
            self.log.info('Dropping %d constant features', int(const.sum()))
        thr = self.corr_threshold
        if 0 < thr < 1:
            idx = np.where(keep)[0]
            if len(idx) > 1:
                C = np.corrcoef(X_train[:, idx], rowvar=False)
                C = np.nan_to_num(C)
                for a in range(len(idx)):
                    if not keep[idx[a]]:
                        continue
                    for b in range(a + 1, len(idx)):
                        if keep[idx[b]] and abs(C[a, b]) > thr:
                            keep[idx[b]] = False
        return (keep, [c for c, k in zip(cols, keep) if k])

    def run(self, force: bool=False) -> Split:
        npz_p, meta_p, sc_p = self._paths()
        if not force and npz_p.exists() and meta_p.exists() and sc_p.exists():
            self.log.info('Loading cached split %s', npz_p.name)
            with np.load(npz_p, allow_pickle=False) as z:
                meta = json.loads(meta_p.read_text(encoding='utf-8'))
                return Split(z['X_train'], z['y_train'], z['X_val'], z['y_val'], z['X_test'], z['y_test'], meta['feature_names'], meta['label_names'], joblib.load(sc_p))
        self.log.info('Reading %s', self.csv_path)
        df = pd.read_csv(self.csv_path, low_memory=False)
        n_raw = len(df)
        ident = tuple(IDENTIFYING) + (PACKET_IDS if self.drop_packet_ids else ())
        drop = [c for c in ident if c in df.columns]
        df = df.drop(columns=drop)
        self.log.info('Dropped %d identifying columns: %s', len(drop), drop)
        feat_cols = [c for c in df.columns if c != LABEL_COL]
        for c in list(feat_cols):
            if df[c].dtype == object:
                if df[c].nunique(dropna=False) <= 32:
                    df[c] = LabelEncoder().fit_transform(df[c].astype(str))
                else:
                    self.log.info('Dropping high-cardinality text column %s (%d levels)', c, df[c].nunique())
                    df = df.drop(columns=[c])
                    feat_cols.remove(c)
        df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors='coerce')
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if self.deduplicate:
            before = len(df)
            df = df.drop_duplicates(subset=feat_cols, keep='first')
            self.log.info('De-duplicated on the feature columns: %d -> %d rows (%.2f%% removed)', before, len(df), 100 * (1 - len(df) / before))
        if self.max_rows_per_class > 0:
            df = df.groupby(LABEL_COL, group_keys=False).apply(lambda g: g.sample(min(len(g), self.max_rows_per_class), random_state=self.seed))
        le = LabelEncoder()
        y = le.fit_transform(df[LABEL_COL].astype(str))
        X = df[feat_cols].to_numpy(dtype=np.float64)
        label_names = [str(c) for c in le.classes_]
        idx = np.arange(len(y))
        tr, tmp = train_test_split(idx, test_size=self.test_size + self.val_size, random_state=self.seed, stratify=y)
        rel = self.test_size / (self.test_size + self.val_size)
        va, te = train_test_split(tmp, test_size=rel, random_state=self.seed, stratify=y[tmp])
        keep, kept_names = self._select_features(X[tr], feat_cols)
        X = X[:, keep]
        scaler = _make_scaler(self.scaler_kind)
        X_tr = scaler.fit_transform(X[tr]).astype(np.float32)
        X_va = scaler.transform(X[va]).astype(np.float32)
        X_te = scaler.transform(X[te]).astype(np.float32)
        np.savez_compressed(npz_p, X_train=X_tr, y_train=y[tr], X_val=X_va, y_val=y[va], X_test=X_te, y_test=y[te])
        counts = {label_names[i]: int(c) for i, c in enumerate(np.bincount(y[tr], minlength=len(label_names)))}
        meta_p.write_text(json.dumps({'version': CACHE_VERSION, 'fingerprint': self._fingerprint(), 'source_csv': str(self.csv_path), 'rows_raw': n_raw, 'rows_after_dedup': int(len(df)), 'dropped_identifying': drop, 'feature_names': kept_names, 'label_names': label_names, 'train_class_counts': counts, 'shapes': {'train': list(X_tr.shape), 'val': list(X_va.shape), 'test': list(X_te.shape)}, 'provenance': 'preprocessed from the raw CSV by this project; de-duplicated before splitting; feature selection and scaler fitted on the training split only'}, indent=2), encoding='utf-8')
        joblib.dump(scaler, sc_p)
        self.log.info('Cached split -> %s  (%d features, %d classes)', npz_p.name, X_tr.shape[1], len(label_names))
        return Split(X_tr, y[tr], X_va, y[va], X_te, y[te], kept_names, label_names, scaler)
