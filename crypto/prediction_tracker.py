import json
import os
import logging
import tempfile
import time
from datetime import datetime, timedelta
from contextlib import contextmanager
import numpy as np

from utils.state_store import load_json, save_json, safe_float

logger = logging.getLogger(__name__)

TRACKER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prediction_accuracy.json")
_LOCK_FILE = TRACKER_FILE + ".lock"
_LOCK_TIMEOUT = 10  # seconds
_LOCK_POLL = 0.05


def _acquire_lock(timeout=_LOCK_TIMEOUT):
    """Cross-process lock via an exclusively-created lock file (Windows-safe)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            fd = os.open(_LOCK_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            # stale lock guard: if the owning PID is gone, steal it
            try:
                with open(_LOCK_FILE, "r") as f:
                    pid = int(f.read().strip())
                if pid and not _pid_alive(pid):
                    os.remove(_LOCK_FILE)
                    continue
            except Exception:
                pass
            time.sleep(_LOCK_POLL)
    return False


def _pid_alive(pid):
    try:
        import ctypes
        import ctypes.wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        return True  # assume alive if we can't check


def _release_lock():
    try:
        os.remove(_LOCK_FILE)
    except OSError:
        pass


@contextmanager
def _locked():
    if not _acquire_lock():
        logger.warning("prediction_tracker: lock timeout — proceeding unsafely")
    try:
        yield
    finally:
        _release_lock()


def normalize_symbol(symbol):
    """Canonical symbol key: uppercased, USDT-quote preserved.

    Prediction recording and verification must use the SAME key. Previously
    records used the full symbol (``NEARUSDT``) while the only verify caller
    passed a stripped one (``NEAR``), so nothing ever verified.
    """
    if not symbol:
        return symbol
    s = str(symbol).upper()
    if not s.endswith("USDT"):
        # tolerate a bare base asset but flag it: keep as-is (uppercased)
        return s
    return s


class PredictionTracker:

    def __init__(self):
        self.data = self._load()

    @staticmethod
    def _load():
        if os.path.exists(TRACKER_FILE):
            try:
                with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {"predictions": [], "stats": {}}
        return {"predictions": [], "stats": {}}

    @staticmethod
    def _save(data):
        try:
            fd, tmp = tempfile.mkstemp(prefix=".tracker_", dir=os.path.dirname(TRACKER_FILE))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                os.replace(tmp, TRACKER_FILE)
            except Exception:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning(f"Failed to save prediction tracker: {e}")

    def record_prediction(self, symbol, tf_hours, predicted_change_pct, confidence, grade, current_price, features=None):
        with _locked():
            data = self._load()
            data["predictions"].append({
                "symbol": normalize_symbol(symbol),
                "timeframe_hours": int(tf_hours) if tf_hours else 0,
                "predicted_change_pct": safe_float(predicted_change_pct),
                "confidence": int(safe_float(confidence)),
                "grade": str(grade),
                "current_price": safe_float(current_price),
                "timestamp": datetime.now().isoformat(),
                "actual_change_pct": None,
                "verified": False,
                "features": [safe_float(x) for x in (features or [])] if features else None,
            })
            if len(data["predictions"]) > 10000:
                data["predictions"] = data["predictions"][-5000:]
            self._save(data)
            self.data = data

    def verify_prediction(self, symbol, tf_hours, actual_change_pct, max_age_hours=720):
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        symbol = normalize_symbol(symbol)
        with _locked():
            data = self._load()
            verified_count = 0
            for pred in data["predictions"]:
                if pred["verified"]:
                    continue
                try:
                    pred_time = datetime.fromisoformat(pred["timestamp"])
                except Exception:
                    continue
                if pred_time < cutoff:
                    continue
                age_hours = (datetime.now() - pred_time).total_seconds() / 3600
                if (pred["symbol"] == symbol
                        and pred["timeframe_hours"] == tf_hours
                        and age_hours >= tf_hours * 0.9):
                    pred["actual_change_pct"] = safe_float(actual_change_pct) if actual_change_pct is not None else None
                    pred["verified"] = True
                    verified_count += 1
            if verified_count:
                self._update_stats(data)
                self._save(data)
            self.data = data
            return verified_count

    def verify_all(self, current_prices, max_age_hours=720):
        if not current_prices:
            return 0
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        now = datetime.now()
        with _locked():
            data = self._load()
            verified_count = 0
            for pred in data["predictions"]:
                if pred["verified"]:
                    continue
                sym = pred["symbol"]
                price = current_prices.get(sym)
                entry = safe_float(pred.get("current_price"))
                if price is None or price <= 0 or entry <= 0:
                    continue
                try:
                    pred_time = datetime.fromisoformat(pred["timestamp"])
                except Exception:
                    continue
                if pred_time < cutoff:
                    continue
                age_hours = (now - pred_time).total_seconds() / 3600
                if age_hours < pred["timeframe_hours"] * 0.9:
                    continue
                actual = (safe_float(price) / entry - 1.0) * 100.0
                pred["actual_change_pct"] = actual
                pred["verified"] = True
                verified_count += 1
            if verified_count:
                self._update_stats(data)
                self._save(data)
            self.data = data
            return verified_count

    @staticmethod
    def _update_stats(data):
        verified = [p for p in data["predictions"] if p["verified"]]
        if not verified:
            data["stats"] = {}
            return

        correct_direction = sum(
            1 for p in verified
            if (p["predicted_change_pct"] > 0 and p["actual_change_pct"] > 0)
            or (p["predicted_change_pct"] < 0 and p["actual_change_pct"] < 0)
        )
        total = len(verified)
        accuracy = correct_direction / total if total > 0 else 0

        abs_errors = [abs(p["predicted_change_pct"] - p["actual_change_pct"]) for p in verified]
        mae = np.mean(abs_errors) if abs_errors else 0

        calibration = {}
        bins = [(50, 60), (60, 70), (70, 80), (80, 90), (90, 101)]
        for lo, hi in bins:
            sel = [p for p in verified if lo <= p["confidence"] < hi]
            if sel:
                ok = sum(
                    1 for p in sel
                    if (p["predicted_change_pct"] > 0 and p["actual_change_pct"] > 0)
                    or (p["predicted_change_pct"] < 0 and p["actual_change_pct"] < 0)
                )
                key = f"{lo}-{hi if hi <= 100 else 100}"
                calibration[key] = {"n": len(sel), "realized_acc": round(ok / len(sel), 3)}

        data["stats"] = {
            "total_verified": int(total),
            "correct_direction": int(correct_direction),
            "direction_accuracy_pct": round(float(accuracy) * 100, 1),
            "mae_pct": round(float(mae), 2),
            "calibration": calibration,
            "last_updated": datetime.now().isoformat()
        }

    def get_stats(self):
        return self.data.get("stats", {})

    def get_recent_predictions(self, limit=20):
        return self.data["predictions"][-limit:]

    def get_accuracy_by_symbol(self, min_samples=5):
        verified = [p for p in self.data["predictions"] if p["verified"]]
        by_symbol = {}
        for p in verified:
            sym = p["symbol"]
            if sym not in by_symbol:
                by_symbol[sym] = {"total": 0, "correct": 0, "errors": []}
            by_symbol[sym]["total"] += 1
            if (p["predicted_change_pct"] > 0 and p["actual_change_pct"] > 0) or \
               (p["predicted_change_pct"] < 0 and p["actual_change_pct"] < 0):
                by_symbol[sym]["correct"] += 1
            by_symbol[sym]["errors"].append(abs(p["predicted_change_pct"] - p["actual_change_pct"]))

        result = {}
        for sym, d in by_symbol.items():
            if d["total"] >= min_samples:
                result[sym] = {
                    "samples": int(d["total"]),
                    "accuracy_pct": round(float(d["correct"] / d["total"] * 100), 1),
                    "mae_pct": round(float(np.mean(d["errors"])), 2)
                }
        return result


prediction_tracker = PredictionTracker()