
from __future__ import annotations

import json
import os
import time
from pathlib import Path


class ResultsLocked(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import subprocess
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=10)
            return str(pid) in out.stdout
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ResultsLock:
    def __init__(self, results_dir: Path, label: str = ""):
        self.path = Path(results_dir) / ".run.lock"
        self.label = label

    def acquire(self, force: bool = False) -> "ResultsLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not force:
            try:
                info = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                info = {}
            pid = int(info.get("pid", -1))
            if _alive(pid):
                raise ResultsLocked(
                    f"{self.path.parent} is already being written by pid {pid} "
                    f"({info.get('label', '?')}, started "
                    f"{time.strftime('%H:%M:%S', time.localtime(info.get('t', 0)))}). "
                    f"Stop it first, or pass --force-lock.")
        self.path.write_text(json.dumps(
            {"pid": os.getpid(), "label": self.label, "t": time.time()}),
            encoding="utf-8")
        return self

    def release(self) -> None:
        try:
            self.path.unlink(missing_ok=True)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False
