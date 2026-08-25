
import argparse, sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.common import load_config
from src.data.cic_preprocessor import CicIotPreprocessor

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "cic_iot.yaml"))
ap.add_argument("--force", action="store_true")
a = ap.parse_args()
cfg = load_config(a.config)
sp = CicIotPreprocessor(cfg, ROOT).run(force=a.force)
print(json.dumps({"train": list(sp.X_train.shape), "val": list(sp.X_val.shape),
                  "test": list(sp.X_test.shape), "features": sp.feature_names,
                  "counts": sp.class_counts()}, indent=2))
