"""
Standalone CryptoPredictions Dashboard Server
يشغل Flask مباشرة بدون Hydra أو orbit-ml
"""
import os
import sys
import json
import logging
import threading
import requests
from apscheduler.schedulers.background import BackgroundScheduler
import csv
import pytz
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv

BAGHDAD_TZ = pytz.timezone("Asia/Baghdad")

def now_baghdad():
    return datetime.now(BAGHDAD_TZ)

def fmt_baghdad(dt=None):
    dt = dt or now_baghdad()
    return dt.strftime("%Y-%m-%d %H:%M AST")

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
# Bot 1: توقعات العملات الصاعدة > 3%
TELEGRAM_PREDICTION_TOKEN = os.getenv("TELEGRAM_PREDICTION_TOKEN", "")
# Bot 2: مراقبة السيولة وقراءة السوق
TELEGRAM_LIQUIDITY_TOKEN = os.getenv("TELEGRAM_LIQUIDITY_TOKEN", "")
AUTHORIZED_USERS = [895650332, 991558864, 715531930, 117245128, 1796556765, 31128146]

LIQUIDITY_IMBALANCE_THRESHOLD = float(os.getenv("LIQUIDITY_IMBALANCE_THRESHOLD", "5"))
SPREAD_CHANGE_THRESHOLD = float(os.getenv("SPREAD_CHANGE_THRESHOLD", "0.2"))
PREDICTION_INCREASE_THRESHOLD = float(os.getenv("PREDICTION_INCREASE_THRESHOLD", "3"))
monitoring_active = True
training_lock = threading.Lock()
training_status = {
    "status": "idle",
    "current_symbol": "",
    "progress": "",
    "error": ""
}
# Load previous liquidity state
STATE_FILE = os.path.join(BASE_DIR, "liquidity_state.json")
if os.path.exists(STATE_FILE):
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            liquidity_state = json.load(f)
    except Exception:
        liquidity_state = {}
else:
    liquidity_state = {}

# ─────────────────────────────────────────────────────────────
# Helper: Binance Data Fetching
# ─────────────────────────────────────────────────────────────
def get_usd_and_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        trading_pairs = sorted([
            sym["symbol"] for sym in data["symbols"]
            if sym["status"] == "TRADING" and sym["symbol"].endswith("USDT")
        ])
        return trading_pairs
    except Exception as e:
        logger.error(f"Error fetching trading pairs: {e}")
        return []


def fetch_ohlcv(symbol, interval="1d", limit=365):
    """Fetch OHLCV data from Binance"""
    url = "https://api.binance.com/api/v3/klines"
    start_date = now_baghdad() - timedelta(days=limit)
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": int(start_date.timestamp() * 1000),
        "limit": 1000
    }
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY} if BINANCE_API_KEY else {}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        data = response.json()
        if isinstance(data, dict) and "code" in data:
            logger.error(f"Binance error for {symbol}: {data}")
            return []
        return data
    except Exception as e:
        logger.error(f"Error fetching OHLCV for {symbol}: {e}")
        return []


def get_current_price(symbol):
    """Get current price from Binance ticker"""
    url = "https://api.binance.com/api/v3/ticker/price"
    try:
        response = requests.get(url, params={"symbol": symbol}, timeout=10)
        data = response.json()
        return float(data.get("price", 0))
    except Exception as e:
        logger.error(f"Error fetching price for {symbol}: {e}")
        return 0.0


# ─────────────────────────────────────────────────────────────
# ML Prediction Engine (without TF/orbit-ml)
# ─────────────────────────────────────────────────────────────
def run_prediction_for_symbol(symbol):
    """
    Run XGBoost + RandomForest ensemble prediction for a single symbol.
    Returns dict with predicted_high, predicted_low, predicted_mean, etc.
    """
    import numpy as np

    klines = fetch_ohlcv(symbol, limit=500)
    if len(klines) < 30:
        raise ValueError(f"Not enough data for {symbol}: got {len(klines)} candles")

    closes = np.array([float(k[4]) for k in klines])
    highs  = np.array([float(k[2]) for k in klines])
    lows   = np.array([float(k[3]) for k in klines])
    volumes = np.array([float(k[5]) for k in klines])

    # Feature engineering
    def make_features(arr, n=20):
        features = []
        for i in range(n, len(arr)):
            window = arr[i-n:i]
            features.append([
                np.mean(window),
                np.std(window),
                np.min(window),
                np.max(window),
                window[-1] / window[0] - 1,  # momentum
                window[-1] / np.mean(window) - 1,  # deviation from mean
                arr[i-1],  # last close
                arr[i-2] if i >= 2 else arr[i-1],  # prev close
                np.mean(arr[max(0,i-5):i]),  # 5-day MA
                np.mean(arr[max(0,i-10):i]),  # 10-day MA
            ])
        return np.array(features)

    n_lag = 20
    X = make_features(closes, n_lag)
    y_close = closes[n_lag:]
    y_high  = highs[n_lag:]
    y_low   = lows[n_lag:]

    if len(X) < 10:
        raise ValueError(f"Insufficient features for {symbol}")

    split = int(len(X) * 0.85)
    X_train, X_test = X[:split], X[split:]
    yc_train = y_close[:split]
    yh_train = y_high[:split]
    yl_train = y_low[:split]

    # Try XGBoost + RandomForest ensemble
    try:
        from xgboost import XGBRegressor
        from sklearn.ensemble import RandomForestRegressor

        xgb_c = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                              verbosity=0, random_state=42)
        xgb_h = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                              verbosity=0, random_state=42)
        xgb_l = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                              verbosity=0, random_state=42)

        rf_c = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_h = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_l = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

        xgb_c.fit(X_train, yc_train)
        xgb_h.fit(X_train, yh_train)
        xgb_l.fit(X_train, yl_train)
        rf_c.fit(X_train, yc_train)
        rf_h.fit(X_train, yh_train)
        rf_l.fit(X_train, yl_train)

        # Predict next day features
        last_features = X[-1].reshape(1, -1)
        pred_close = (xgb_c.predict(last_features)[0] + rf_c.predict(last_features)[0]) / 2
        pred_high  = (xgb_h.predict(last_features)[0] + rf_h.predict(last_features)[0]) / 2
        pred_low   = (xgb_l.predict(last_features)[0] + rf_l.predict(last_features)[0]) / 2
        model_name = "XGBoost+RF Ensemble"

    except ImportError:
        # Fallback: simple linear regression
        from sklearn.linear_model import Ridge
        ridge_c = Ridge().fit(X_train, yc_train)
        ridge_h = Ridge().fit(X_train, yh_train)
        ridge_l = Ridge().fit(X_train, yl_train)
        last_features = X[-1].reshape(1, -1)
        pred_close = ridge_c.predict(last_features)[0]
        pred_high  = ridge_h.predict(last_features)[0]
        pred_low   = ridge_l.predict(last_features)[0]
        model_name = "Ridge Regression"

    yesterday_close = closes[-1]
    increase_pct = (pred_close - yesterday_close) / yesterday_close * 100

    return {
        "symbol": symbol.upper(),
        "yesterday_close": round(float(yesterday_close), 8),
        "predicted_high": round(float(max(pred_high, pred_close)), 8),
        "predicted_low": round(float(min(pred_low, pred_close)), 8),
        "predicted_mean": round(float(pred_close), 8),
        "increase_percentage": round(float(increase_pct), 2),
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment": "جاري التحليل..."
    }


def save_prediction(pred_dict):
    filename = os.path.join(BASE_DIR, "predictions.json")
    predictions = []
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        except Exception:
            predictions = []

    predictions = [p for p in predictions if p["symbol"].upper() != pred_dict["symbol"].upper()]
    predictions.append(pred_dict)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)


def run_prediction_worker(symbols_list):
    global training_status
    with training_lock:
        training_status["status"] = "running"
        training_status["error"] = ""
        training_status["current_symbol"] = ""

        for i, sym in enumerate(symbols_list):
            try:
                training_status["current_symbol"] = sym
                training_status["progress"] = f"جاري توقع العملة {sym} ({i+1}/{len(symbols_list)})..."
                logger.info(f"[PREDICT] {sym} ...")
                result = run_prediction_for_symbol(sym)
                save_prediction(result)
                # Conditional Telegram alert with liquidity info - BOT 1 (توقعات)
                if result.get("increase_percentage", 0) > PREDICTION_INCREASE_THRESHOLD:
                    liq = get_liquidity(sym)
                    token = TELEGRAM_PREDICTION_TOKEN
                    msg = f"\U0001f4c8 *\u062a\u0648\u0642\u0639 {sym}*\n"
                    msg += f"\U0001f539 \u0632\u064a\u0627\u062f\u0629 \u0645\u062a\u0648\u0642\u0639\u0629: {result.get('increase_percentage')}%\n"
                    msg += f"\U0001f539 \u0633\u0639\u0631 \u0627\u0644\u0625\u063a\u0644\u0627\u0642 \u0627\u0644\u0645\u062a\u0648\u0642\u0639: {result.get('predicted_mean')}\n"
                    msg += f"\U0001f539 \u0633\u064a\u0648\u0644\u0629: {liq.get('signal')} (\u0627\u062e\u062a\u0644\u0627\u0644 {liq.get('imbalance_pct')}%, \u0633\u0628\u0631\u064a\u062f {liq.get('spread_pct')}%)\n"
                    msg += f"\U0001f5e3\ufe0f \u0645\u0634\u0627\u0639\u0631: {result.get('sentiment')}\n"
                    msg += f"\u23f0 {result.get('timestamp')}\n"
                    for user_id in AUTHORIZED_USERS:
                        try:
                            url = f"https://api.telegram.org/bot{token}/sendMessage"
                            requests.post(url, json={"chat_id": user_id, "text": msg, "parse_mode": "Markdown"}, timeout=5)
                        except Exception:
                            pass
                logger.info(f"[PREDICT] {sym} → close={result['predicted_mean']:.6f} change={result['increase_percentage']:.2f}%")
            except Exception as e:
                logger.error(f"[PREDICT ERROR] {sym}: {e}")
                continue

        training_status["status"] = "completed"
        training_status["progress"] = f"اكتمل التوقع لـ {len(symbols_list)} عملة بنجاح!"
        training_status["current_symbol"] = ""


# ─────────────────────────────────────────────────────────────
# Liquidity Analysis
# ─────────────────────────────────────────────────────────────
def get_liquidity(symbol):
    """Get order book spread and depth imbalance"""
    url = f"https://api.binance.com/api/v3/depth"
    try:
        response = requests.get(url, params={"symbol": symbol, "limit": 20}, timeout=10)
        data = response.json()
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if not bids or not asks:
            return {"error": "No order book data"}

        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_ask) * 100

        bid_volume = sum(float(b[1]) for b in bids[:10])
        ask_volume = sum(float(a[1]) for a in asks[:10])
        total_volume = bid_volume + ask_volume
        imbalance = (bid_volume - ask_volume) / total_volume * 100 if total_volume > 0 else 0

        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "spread": round(spread, 8),
            "spread_pct": round(spread_pct, 4),
            "bid_depth": round(bid_volume, 4),
            "ask_depth": round(ask_volume, 4),
            "imbalance_pct": round(imbalance, 2),
            "signal": "شراء 🟢" if imbalance > LIQUIDITY_IMBALANCE_THRESHOLD else "بيع 🔴" if imbalance < -LIQUIDITY_IMBALANCE_THRESHOLD else "محايد ⚪"
        }
    except Exception as e:
        return {"error": str(e)}

def monitor_liquidity(symbols):
    """Check liquidity for each symbol and send Telegram alerts if thresholds are crossed."""
    global liquidity_state, monitoring_active
    if not monitoring_active:
        return
    alerts = []
    for sym in symbols:
        info = get_liquidity(sym)
        if "error" in info:
            continue
        prev = liquidity_state.get(sym, {})
        imbalance = info.get("imbalance_pct", 0)
        spread = info.get("spread_pct", 0)
        imbalance_cross = False
        spread_cross = False
        if prev:
            prev_imb = prev.get("imbalance_pct", 0)
            prev_spread = prev.get("spread_pct", 0)
            if abs(imbalance) > LIQUIDITY_IMBALANCE_THRESHOLD and abs(prev_imb) <= LIQUIDITY_IMBALANCE_THRESHOLD:
                imbalance_cross = True
            if spread > SPREAD_CHANGE_THRESHOLD and prev_spread <= SPREAD_CHANGE_THRESHOLD:
                spread_cross = True
        else:
            if abs(imbalance) > LIQUIDITY_IMBALANCE_THRESHOLD or spread > SPREAD_CHANGE_THRESHOLD:
                imbalance_cross = spread_cross = True
        # Update state
        liquidity_state[sym] = {"imbalance_pct": imbalance, "spread_pct": spread}
        # Log changes
        if imbalance_cross or spread_cross:
            alerts.append((sym, info, imbalance_cross, spread_cross))
            log_entry = {
                "timestamp": now_baghdad().isoformat(),
                "symbol": sym,
                "imbalance_pct": imbalance,
                "spread_pct": spread,
                "type": "imbalance" if imbalance_cross else "spread"
            }
            try:
                with open(os.path.join(BASE_DIR, "liquidity_log.json"), "a", encoding="utf-8") as log_f:
                    log_f.write(json.dumps(log_entry) + "\n")
            except Exception:
                pass
    # Send alerts via Telegram - BOT 2 (سيولة)
    if alerts:
        token = TELEGRAM_LIQUIDITY_TOKEN
        for sym, info, imb_cross, spr_cross in alerts:
            msg = f"\U0001f4ca *\u0633\u064a\u0648\u0644\u0629 {sym}*\n"
            msg += f"\U0001f539 \u0627\u062e\u062a\u0644\u0627\u0644: {info.get('imbalance_pct', 0)}% ({info.get('signal')})\n"
            msg += f"\U0001f539 \u0633\u0628\u0631\u064a\u062f: {info.get('spread_pct', 0)}%\n"
            msg += f"\U0001f552 {fmt_baghdad()}\n"
            for user_id in AUTHORIZED_USERS:
                try:
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    requests.post(url, json={"chat_id": user_id, "text": msg, "parse_mode": "Markdown"}, timeout=5)
                except Exception:
                    pass
    # Persist state
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(liquidity_state, f, ensure_ascii=False, indent=4)
    except Exception:
        pass
# ─────────────────────────────────────────────────────────────
# Flask Routes
# ─────────────────────────────────────────────────────────────
@app.route('/')
def home():
    try:
        with open(os.path.join(BASE_DIR, "templates", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"<h1>Error loading dashboard: {e}</h1>", 500


@app.route('/api/symbols')
def api_symbols():
    try:
        symbols = get_usd_and_usdt_pairs()
        return jsonify(symbols)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predictions')
def api_predictions():
    filename = os.path.join(BASE_DIR, "predictions.json")
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        except Exception:
            pass
    return jsonify([])


@app.route('/api/status')
def api_status():
    return jsonify(training_status)


@app.route('/api/run_prediction', methods=['POST'])
def api_run_prediction():
    global training_status
    if training_status["status"] == "running":
        return jsonify({"success": False, "message": "يوجد عملية توقع قيد التشغيل بالفعل."}), 400

    data = request.json or {}
    symbols_to_train = data.get("symbols", [])
    if not symbols_to_train:
        return jsonify({"success": False, "message": "لم يتم تحديد أي عملات لتوقعها."}), 400

    t = threading.Thread(target=run_prediction_worker, args=(symbols_to_train,), daemon=True)
    t.start()
    return jsonify({"success": True, "message": f"بدأت عملية التوقع لـ {len(symbols_to_train)} عملة في الخلفية."})


@app.route('/api/liquidity')
def api_liquidity():
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    result = get_liquidity(symbol)
    return jsonify({"symbol": symbol, "order_book": result})


@app.route('/api/send_telegram', methods=['POST'])
def api_send_telegram():
    data = request.json or {}
    symbol = data.get("symbol", "").upper()

    filename = os.path.join(BASE_DIR, "predictions.json")
    if not os.path.exists(filename):
        return jsonify({"success": False, "message": "لا توجد توقعات متاحة."}), 400

    try:
        with open(filename, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "message": f"فشل قراءة التوقعات: {e}"}), 500

    selected = None
    if symbol:
        for p in predictions:
            if p["symbol"].upper() == symbol:
                selected = p
                break
    elif predictions:
        selected = predictions[-1]

    if not selected:
        return jsonify({"success": False, "message": "لم يتم العثور على التوقع المطلوب."}), 404

    token = os.getenv("TELEGRAM_PREDICTION_TOKEN", TELEGRAM_TOKEN)
    if not token:
        return jsonify({"success": False, "message": "لم يتم إعداد رمز Telegram."}), 400

    msg = (
        f"📊 *توقع العملة: {selected['symbol']}*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"💰 سعر الإغلاق الأمس: `{selected['yesterday_close']}`\n"
        f"📈 أعلى سعر متوقع: `{selected['predicted_high']}`\n"
        f"📉 أدنى سعر متوقع: `{selected['predicted_low']}`\n"
        f"🎯 سعر الإغلاق المتوقع: `{selected['predicted_mean']}`\n"
        f"📊 نسبة التغيير: `{selected['increase_percentage']}%`\n"
        f"🤖 النموذج: {selected.get('model', 'Ensemble')}\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"⏰ {selected.get('timestamp', '')}"
    )

    sent_count = 0
    errors = []
    for user_id in AUTHORIZED_USERS:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            resp = requests.post(url, json={"chat_id": user_id, "text": msg, "parse_mode": "Markdown"}, timeout=10)
            if resp.status_code == 200:
                sent_count += 1
        except Exception as e:
            errors.append(str(e))

    return jsonify({
        "success": sent_count > 0,
        "message": f"تم الإرسال إلى {sent_count} مستخدم." + (f" أخطاء: {errors}" if errors else "")
    })


@app.route('/api/settings', methods=['POST'])
def api_settings():
    data = request.json or {}
    env_keys = {
        "BINANCE_API_KEY": data.get("binance_api_key"),
        "BINANCE_API_SECRET": data.get("binance_api_secret"),
        "TELEGRAM_PREDICTION_TOKEN": data.get("telegram_prediction_token"),
        "TELEGRAM_COMPARATOR_TOKEN": data.get("telegram_comparator_token"),
        "NEWS_API_KEY": data.get("news_api_key")
    }

    current_env = {}
    env_file = os.path.join(BASE_DIR, ".env")
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    parts = line.strip().split("=", 1)
                    if len(parts) == 2:
                        current_env[parts[0]] = parts[1]

    for k, v in env_keys.items():
        if v:
            current_env[k] = v

    try:
        with open(env_file, "w", encoding="utf-8") as f:
            for k, v in current_env.items():
                f.write(f"{k}={v}\n")
        load_dotenv(override=True)
        return jsonify({"success": True, "message": "تم تحديث الإعدادات بنجاح."})
    except Exception as e:
        return jsonify({"success": False, "message": f"فشل حفظ الإعدادات: {e}"}), 500

@app.route('/api/pause', methods=['POST'])
def api_pause():
    global monitoring_active
    monitoring_active = False
    return jsonify({"success": True, "message": "Monitoring paused."})

@app.route('/api/resume', methods=['POST'])
def api_resume():
    global monitoring_active
    monitoring_active = True
    return jsonify({"success": True, "message": "Monitoring resumed."})

@app.route('/api/price/<symbol>')
def api_price(symbol):
    try:
        price = get_current_price(symbol.upper())
        return jsonify({"symbol": symbol.upper(), "price": price})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("DASHBOARD_PORT", 8080))
    # APScheduler for background liquidity monitoring - started after all functions are defined
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        func=lambda: monitor_liquidity(get_usd_and_usdt_pairs()),
        trigger="interval",
        minutes=15,
        id="liquidity_monitor"
    )
    scheduler.start()
    logger.info("=" * 60)
    logger.info("  \U0001f680 CryptoPredictions Dashboard")
    logger.info(f"  \U0001f4e1 Running on: http://127.0.0.1:{port}")
    logger.info("  \U0001f916 Bot 1 (Predictions): TELEGRAM_PREDICTION_TOKEN")
    logger.info("  \U0001f4c8 Bot 2 (Liquidity):   TELEGRAM_LIQUIDITY_TOKEN")
    logger.info("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
