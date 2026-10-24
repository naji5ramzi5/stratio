#!/usr/bin/env python3
"""
StratoCrypto live/server diagnostic.

Read-only: sends nothing to Telegram and writes no data files.
Run this ON THE SERVER from the crypto/ folder:

    python diagnose_live.py

It answers the only question that matters when Bot 3/Bot 4 go silent:
is it the tokens, the network, a missing package, or the predictor?
"""
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results = []


def line(status, name, detail=""):
    results.append((status, name, detail))
    print(f"  [{status:4s}] {name}" + (f" — {detail}" if detail else ""))


def section(title):
    print("\n" + "=" * 66)
    print(title)
    print("=" * 66)


# ── 1. Python ────────────────────────────────────────────────────────────────
section("1. PYTHON / WORKING DIRECTORY")
print("  python :", sys.version.replace("\n", " "))
print("  exe    :", sys.executable)
print("  cwd    :", os.getcwd())
line(PASS, "python runs", sys.version.split()[0])

# ── 2. .env files ────────────────────────────────────────────────────────────
section("2. .ENV FILES")
env_candidates = [
    os.path.join(HERE, ".env"),
    os.path.join(HERE, "..", ".env"),
    os.path.join(os.getcwd(), ".env"),
]
found_env = False
for p in env_candidates:
    ok = os.path.exists(p)
    found_env = found_env or ok
    print(f"  {'OK  ' if ok else 'MISS'} {p}")
line(PASS if found_env else FAIL, ".env found", "" if found_env else "no .env in crypto/ or project root")

try:
    from dotenv import load_dotenv
    for p in env_candidates:
        load_dotenv(p)
except Exception as e:
    line(FAIL, "python-dotenv", repr(e))

# ── 3. Tokens ────────────────────────────────────────────────────────────────
section("3. TELEGRAM TOKENS")
token_keys = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_PREDICTION_TOKEN",   # Bot 1
    "TELEGRAM_LIQUIDITY_TOKEN",    # Bot 2
    "TELEGRAM_PREDICTOR_TOKEN",    # Bot 3
    "BOT4_TOKEN",                  # Bot 4 (Boss)
    "ADMIN_CHAT_ID",
]
missing = []
for k in token_keys:
    v = os.getenv(k, "")
    if v:
        line(PASS, k, v[:4] + "..." + v[-4:])
    else:
        line(FAIL, k, "MISSING")
        missing.append(k)
if missing:
    print("\n  >> Fix: add these keys to crypto/.env on the server.")

# ── 4. Packages ──────────────────────────────────────────────────────────────
section("4. PYTHON PACKAGES")
for mod, label in [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
    ("requests", "requests"),
    ("dotenv", "python-dotenv"),
    ("pytz", "pytz"),
    ("lightgbm", "lightgbm (optional)"),
]:
    try:
        m = __import__(mod)
        line(PASS, label, getattr(m, "__version__", ""))
    except Exception as e:
        optional = "optional" in label
        line(WARN if optional else FAIL, label, f"missing ({e})")

# ── 5. Binance reachability ──────────────────────────────────────────────────
section("5. BINANCE MARKET DATA")
try:
    import requests
    url = "https://data-api.binance.vision/api/v3/ticker/24hr"
    t0 = time.time()
    r = requests.get(url, timeout=12)
    dt = round(time.time() - t0, 2)
    print(f"  GET {url} -> {r.status_code} in {dt}s ({len(r.content)} bytes)")
    if r.status_code == 200:
        data = r.json()
        if isinstance(data, list):
            usdt = [x for x in data if str(x.get("symbol", "")).endswith("USDT")]
            big = [x for x in usdt if float(x.get("quoteVolume", 0) or 0) >= 5_000_000]
            line(PASS, "spot ticker", f"{len(usdt)} USDT pairs, {len(big)} >= $5M")
        else:
            line(FAIL, "spot ticker payload", str(data)[:160])
    else:
        line(FAIL, "spot ticker HTTP", f"{r.status_code} {r.text[:120]}")
except Exception as e:
    line(FAIL, "spot ticker", repr(e))

try:
    import requests
    r = requests.get("https://fapi.binance.com/fapi/v1/ping", timeout=10)
    line(PASS if r.status_code == 200 else WARN, "futures ping", f"HTTP {r.status_code}")
except Exception as e:
    line(WARN, "futures ping", repr(e))

# ── 6. Predictor import ──────────────────────────────────────────────────────
section("6. PREDICTOR IMPORT")
predictor = None
try:
    import predictor as predictor_mod
    predictor = predictor_mod
    line(PASS, "import predictor", f"v{getattr(predictor, 'PREDICTOR_VERSION', '?')}")
except Exception as e:
    line(FAIL, "import predictor", repr(e))
    traceback.print_exc()

# ── 7. Live prediction sample ────────────────────────────────────────────────
section("7. LIVE PREDICTION SAMPLE (this is what Bot 3 does)")
if predictor:
    for sym in ("BTCUSDT", "ETHUSDT", "SOLUSDT"):
        try:
            t0 = time.time()
            res = predictor.predict_price_movement(sym, 24)
            dt = round(time.time() - t0, 1)
            if res.get("error"):
                line(FAIL, f"{sym} 24h", f"error: {res['error']} ({dt}s)")
            else:
                line(PASS, f"{sym} 24h",
                     f"{res.get('predicted_change_pct')}% conf={res.get('confidence')} ({dt}s)")
        except Exception as e:
            line(FAIL, f"{sym} 24h", repr(e))
else:
    line(FAIL, "prediction sample", "predictor did not import")

# ── 8. Prediction history freshness ─────────────────────────────────────────
section("8. PREDICTION HISTORY (Bot 4 only reads the last 2 hours)")
hist_path = os.path.join(HERE, "prediction_history.json")
if os.path.exists(hist_path):
    try:
        hist = json.load(open(hist_path, encoding="utf-8"))
        size_mb = round(os.path.getsize(hist_path) / 1e6, 2)
        print(f"  records={len(hist)}  size={size_mb} MB")
        fresh, last = 0, ""
        cutoff = time.time() - 7200
        from datetime import datetime as _dt
        for p in hist:
            if p.get("_verified") or p.get("error"):
                continue
            created = p.get("_created_at") or p.get("created_at") or ""
            last = max(last, created)
            try:
                ct = _dt.fromisoformat(created.replace("Z", "+00:00")).timestamp()
                if ct >= cutoff:
                    fresh += 1
            except Exception:
                continue
        print(f"  last _created_at = {last}")
        print(f"  usable records in last 2h = {fresh}")
        if fresh:
            line(PASS, "Bot4 cache", f"{fresh} fresh predictions")
        else:
            line(FAIL, "Bot4 cache",
                 "0 fresh predictions -> Bot 4 has nothing to scan (this is the Boss bug)")
    except Exception as e:
        line(FAIL, "read prediction_history.json", repr(e))
else:
    line(FAIL, "prediction_history.json", "MISSING")

# ── 9. Foundation models ─────────────────────────────────────────────────────
section("9. FOUNDATION MODELS (heavy; keep OFF on small VPS)")
print("  FOUNDATION_MODELS =", os.getenv("FOUNDATION_MODELS", "(unset -> ON by default)"))
try:
    import foundation_models as fm
    print("  availability:", fm.availability())
    line(PASS, "foundation module", "importable")
except Exception as e:
    line(WARN, "foundation module", repr(e))

# ── 10. Trained model artifacts (needed for calibrated confidence) ──────────
section("10. MODEL ARTIFACTS (Boss needs calibrated confidence for TRADE_READY)")
models_dir = os.path.join(HERE, "models")
needed = {
    "ensemble_v2.pkl": "ML ensemble (21 MB)",
    "calibration.json": "probability calibration",
    "meta_labeler.pkl": "meta-labelling gate",
}
if os.path.isdir(models_dir):
    present = set(os.listdir(models_dir))
    for name, why in needed.items():
        ok = name in present
        size = ""
        if ok:
            n = os.path.getsize(os.path.join(models_dir, name))
            size = f"{n/1e6:.2f} MB" if n >= 1e6 else f"{n/1e3:.1f} KB"
        line(PASS if ok else WARN, name, size if ok else f"MISSING — {why}")
else:
    line(WARN, "models/ folder", "MISSING — predictions will run uncalibrated")

# ── Summary ──────────────────────────────────────────────────────────────────
section("SUMMARY")
fails = [r for r in results if r[0] == FAIL]
warns = [r for r in results if r[0] == WARN]
print(f"  PASS={len(results) - len(fails) - len(warns)}  WARN={len(warns)}  FAIL={len(fails)}")
for status, name, detail in fails:
    print(f"  [FAIL] {name} — {detail}")
if not fails:
    print("  Everything critical passed. If Boss is still silent, check the bot log:")
    print("    tail -n 100 advanced_bot.log")
print("\nDONE")
