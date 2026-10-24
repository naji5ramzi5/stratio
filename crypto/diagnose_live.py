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
# The four sending bots are the hard requirement.  TELEGRAM_BOT_TOKEN is only a
# fallback for them, and the channel id can come from either variable.
required = [
    "TELEGRAM_PREDICTION_TOKEN",   # Bot 1
    "TELEGRAM_LIQUIDITY_TOKEN",    # Bot 2
    "TELEGRAM_PREDICTOR_TOKEN",    # Bot 3
    "BOT4_TOKEN",                  # Bot 4 (Boss)
]
missing = []
for k in required:
    v = os.getenv(k, "")
    if v:
        line(PASS, k, v[:4] + "..." + v[-4:])
    else:
        line(FAIL, k, "MISSING")
        missing.append(k)

master = os.getenv("TELEGRAM_BOT_TOKEN", "")
line(PASS if master else WARN, "TELEGRAM_BOT_TOKEN",
     (master[:4] + "..." + master[-4:]) if master else
     "not set (only a fallback; per-bot tokens above are what matter)")

chat = os.getenv("ADMIN_CHAT_ID", "") or os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")
line(PASS if chat else WARN, "ADMIN_CHAT_ID",
     chat if chat else "not set (code falls back to -1004373560971)")

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
    r, last_err = None, None
    for attempt in range(3):
        t0 = time.time()
        try:
            r = requests.get(url, timeout=12)
            break
        except Exception as e:
            last_err = e
            print(f"  attempt {attempt + 1}/3 failed: {e!r}")
            time.sleep(2)
    if r is None:
        raise last_err or RuntimeError("no response")
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
            line(WARN, "Bot4 cache",
                 "0 fresh predictions -> Bot 4 will use its live fallback")
    except Exception as e:
        line(FAIL, "read prediction_history.json", repr(e))
else:
    line(WARN, "prediction_history.json",
         "MISSING (normal on a fresh clone; Bot 4 will compute live fallback)")

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

# ── 11. Confidence gate ──────────────────────────────────────────────────────
section("11. CONFIDENCE GATE (why Boss may only ever say 'no signals')")
# predictor.py only sets confidence_available=True after 30 verified outcomes
# exist in prediction_accuracy.json.  Without it decision_engine can never
# return TRADE_READY — the bot is silent by design, not by accident.
acc_path = os.path.join(HERE, "prediction_accuracy.json")
n_ver = 0
if os.path.exists(acc_path):
    try:
        acc = json.load(open(acc_path, encoding="utf-8"))
        preds = acc.get("predictions", []) if isinstance(acc, dict) else acc
        n_ver = sum(1 for p in preds if p.get("verified"))
        print(f"  prediction_accuracy.json: {len(preds):,} records, {n_ver:,} verified")
    except Exception as e:
        line(FAIL, "prediction_accuracy.json", repr(e))
else:
    line(WARN, "prediction_accuracy.json", "MISSING -> 0 verified samples")
if n_ver >= 30:
    line(PASS, "confidence gate", f"{n_ver} verified -> calibrated confidence available")
else:
    line(WARN, "confidence gate",
         f"{n_ver} verified (need 30) -> Boss can only send 'no signals', "
         f"never TRADE_READY")

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
