
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml

_FMT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
_DATEFMT = "%H:%M:%S"


class AttrDict(dict):


    def __init__(self, data: Dict[str, Any]):
        super().__init__()
        for k, v in data.items():
            self[k] = self._convert(v)

    @classmethod
    def _convert(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._convert(x) for x in v]
        return v

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._convert(value)

    def to_plain(self) -> Dict[str, Any]:
        def back(v):
            if isinstance(v, AttrDict):
                return {k: back(x) for k, x in v.items()}
            if isinstance(v, list):
                return [back(x) for x in v]
            return v
        return {k: back(v) for k, v in self.items()}


def load_config(path: str | Path) -> AttrDict:
    with Path(path).open("r", encoding="utf-8") as fp:
        return AttrDict(yaml.safe_load(fp))


def get_logger(name: str, log_dir: Optional[Path] = None,
               level: str = "INFO") -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(level)
    lg.propagate = False
    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    lg.addHandler(sh)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)


        fh = logging.FileHandler(log_dir / "run.log", mode="w", encoding="utf-8")
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    return lg


def set_global_seed(seed: int) -> None:

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pick_device(spec: str = "auto") -> torch.device:
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
