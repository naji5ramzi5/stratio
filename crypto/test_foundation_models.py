"""
Tests for foundation_models.py (Chronos-2 + TimesFM 2.5 integration).
Run: python test_foundation_models.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ["FOUNDATION_MODELS"] = "1"


def _series(seed=0, n=400, start=30000.0, vol=0.02):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0005, vol, n)) + start


def test_imports():
    import foundation_models as fm
    assert fm.ENABLE_FOUNDATION is True
    assert hasattr(fm, "chronos_forecast")
    assert hasattr(fm, "timesfm_forecast")
    assert hasattr(fm, "foundation_forecast")
    print("OK test_imports")


def test_disabled_env():
    os.environ["FOUNDATION_MODELS"] = "0"
    import importlib
    import foundation_models as fm
    importlib.reload(fm)
    assert fm.foundation_forecast(_series(), 24) is None
    os.environ["FOUNDATION_MODELS"] = "1"
    importlib.reload(fm)
    print("OK test_disabled_env")


def test_chronos():
    import foundation_models as fm
    t0 = time.time()
    res = fm.chronos_forecast(_series(seed=10), 24)
    dt = time.time() - t0
    assert res is not None, "chronos returned None"
    for k in ("change_pct", "predicted_price", "q10", "q90", "spread_pct", "confidence"):
        assert k in res, f"missing {k}"
    assert 15 <= res["confidence"] <= 95
    assert res["q10"] <= res["q90"]
    print(f"OK test_chronos ({dt:.1f}s) change_pct={res['change_pct']}")


def test_timesfm():
    import foundation_models as fm
    t0 = time.time()
    res = fm.timesfm_forecast(_series(seed=11), 24)
    dt = time.time() - t0
    assert res is not None, "timesfm returned None"
    for k in ("change_pct", "predicted_price", "q10", "q90", "spread_pct", "confidence"):
        assert k in res, f"missing {k}"
    assert 15 <= res["confidence"] <= 95
    assert res["q10"] <= res["q90"]
    print(f"OK test_timesfm ({dt:.1f}s) change_pct={res['change_pct']}")


def test_combined():
    import foundation_models as fm
    res = fm.foundation_forecast(_series(seed=12), 16, use_cache=False)
    assert res is not None
    assert res["num_models"] == 2
    assert res["chronos"] is not None
    assert res["timesfm"] is not None
    assert "change_pct" in res and "confidence" in res
    print(f"OK test_combined num_models={res['num_models']}")


def test_short_context():
    import foundation_models as fm
    short = _series(seed=13, n=30)
    c = fm.chronos_forecast(short, 8)
    t = fm.timesfm_forecast(short, 8)
    assert c is not None, "chronos should handle 30 points"
    assert t is not None, "timesfm should handle 30 points"
    print("OK test_short_context")


def test_predictor_integration():
    from predictor import predict_price_movement
    r = predict_price_movement("BTCUSDT", 24)
    assert r is not None and "error" not in r, r
    assert "foundation_prediction" in r
    assert "ml_prediction" in r
    print(f"OK test_predictor_integration fnd={r['foundation_prediction']}")


if __name__ == "__main__":
    test_imports()
    test_disabled_env()
    test_chronos()
    test_timesfm()
    test_combined()
    test_short_context()
    test_predictor_integration()
    print("\nAll foundation_models tests passed.")
