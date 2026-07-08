import logging
import os
import asyncio
import hydra
from omegaconf import DictConfig
from models import MODELS
from data_loader import get_dataset
from factory.trainer import Trainer
from factory.evaluator import Evaluator
from factory.profit_calculator import ProfitCalculator
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH

from utils.reporter import Reporter
from data_loader.creator import create_dataset, preprocess



import yaml
from datetime import datetime, timedelta

import requests
import csv
import os
import pytz
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)
import pandas as pd


from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"
symbols  = [];

from price_comparator import PriceComparator
price_comparator = PriceComparator()

def start_price_comparison(title):
    comparator = PriceComparator()
    comparator.start_comparator(title)
    
def check_and_delete_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"No data in file {filename}.")
                os.remove(filename)
                return None, False
            
            if lines[0].strip().startswith('timestamp'):
                print(f"Header detected in file {filename}, skipping the first line.")
                lines = lines[1:]

            if not lines:
                print(f"No valid data in file {filename} after removing header.")
                os.remove(filename)
                return None, False

            first_line = lines[0].strip()
            last_line = lines[-1].strip()

            try:
                first_date = datetime.strptime(first_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
                last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
            except ValueError as e:
                print(f"Error parsing dates in file {filename}: {e}")
                os.remove(filename)
                return None, False

            if first_date.date() < datetime(2020, 1, 1).date():
                os.remove(filename)
                print(f"File {filename} deleted because it doesn't start from 01/01/2020.")
                return None, False

            if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
                os.remove(filename)
                print(f"File {filename} deleted because it doesn't cover up to yesterday.")
                return None, False

            try:
                yesterday_close = float(last_line.split(',')[5])
            except (IndexError, ValueError) as e:
                print(f"Error extracting close value from file {filename}: {e}")
                os.remove(filename)
                return None, False

            return yesterday_close, True

    except Exception as e:
        print(f"An error occurred while checking file {filename}: {e}")
        return None, False

def add_future_dates(filename, symbol):
    today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    future_dates = [today + timedelta(days=i) for i in range(3)]

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for future_date in future_dates:
            formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([formatted_date, symbol,'1','1','1','1','1'])

data_folder = os.path.join(os.getcwd(), 'data')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_filename = os.path.join(data_folder, 'data1.csv')

def fetch_and_save_data(symbol, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"
    
    API_KEY = os.getenv("BINANCE_API_KEY", "")
    API_SECRET = os.getenv("BINANCE_API_SECRET", "")

    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": int(start_date.timestamp() * 1000),
        "endTime": int(end_date.timestamp() * 1000),
        "limit": 1000
    }
    headers = {"X-MBX-APIKEY": API_KEY}
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
    except Exception as e:
        print(f"Error fetching data from Binance: {e}")
        return False

    if isinstance(data, dict) and 'code' in data and data['code'] == -1121: 
        return False
    
    if not data:
        print(f"No data available for {symbol} from {start_date.date()}")
        return False

    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([
                timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]
            ])
    
    return True
def get_usd_and_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()
        # جمع أسماء الأزواج التي تنتهي بـ USDT أو USD فقط
        trading_pairs = sorted([
            symbol["symbol"] for symbol in data["symbols"]
            if symbol["status"] == "TRADING" and (symbol["symbol"].endswith("USDT"))
        ])
        return trading_pairs
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

from crypto_sentiment_analyzer import CryptoSentimentAnalyzer
import json
import threading

global_cfg = None

def save_prediction_to_json(symbol, yesterday_close, predicted_high, predicted_low, predicted_mean, increase_percentage, sentiment):
    filename = "predictions.json"
    predictions = []
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                predictions = json.load(f)
        except Exception as e:
            print(f"Error loading predictions.json: {e}")
            predictions = []
            
    # Remove older prediction for this symbol if it exists
    predictions = [p for p in predictions if p["symbol"].upper() != symbol.upper()]
    
    predictions.append({
        "symbol": symbol.upper(),
        "yesterday_close": yesterday_close,
        "predicted_high": predicted_high,
        "predicted_low": predicted_low,
        "predicted_mean": predicted_mean,
        "increase_percentage": increase_percentage,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sentiment": sentiment
    })
    
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"Error writing to predictions.json: {e}")

def train(cfg: DictConfig, target_symbols=None): 
    analyzer = CryptoSentimentAnalyzer()

    # جلب الأزواج النشطة التي تحتوي على USDT أو USD أو استخدام العملات المحددة
    if target_symbols is not None:
        symbols = [s.strip().upper() for s in target_symbols]
    else:
        symbols = get_usd_and_usdt_pairs()

    # طباعة اللائحة
    print("List of Active Trading Pairs (USDT/USD) on Binance:")
    print(len(symbols))
    start_date = datetime(2022, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)

    title = ''
    increase_threshold = 0.03
    saved_percentage = 0
  
    for symbol in symbols:
        try:
            # Check if file exists and delete it if necessary
            if os.path.exists(data_filename):
                os.remove(data_filename)  # Delete the file if it exists

            # Write headers only when creating the file
            with open(data_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

            current_start = start_date
            data_available = False

            while current_start < end_date:
                current_end = min(current_start + period, end_date)
                print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
                
                # Fetch data for each period
                if fetch_and_save_data(symbol, current_start, current_end):
                    data_available = True  # Data fetched successfully for at least one period
                if not data_available:
                    break
                current_start = current_end + timedelta(days=1)
            
            if not data_available:
                print(f"Skipping {symbol} due to insufficient data.")
                continue  # Skip symbol if no data is available

            yesterday_close, data_complete = check_and_delete_file(data_filename)
            if not data_complete:
                continue
            # Add three future dates at the end of the file with the symbol
            add_future_dates(data_filename, symbol)

            print("Data download complete for all symbols with data from the beginning of 2020.")
            
            # Load dataset or model based on configuration
            if cfg.load_path is None and cfg.model is None:
                msg = 'either specify a load_path or config a model.'
                logger.error(msg)
                raise Exception(msg)

            elif cfg.load_path is not None:
                dataset_ = pd.read_csv(cfg.load_path)
                if 'Date' not in dataset_.keys():
                    dataset_.rename(columns={'timestamp': 'Date'}, inplace=True)
                if 'High' not in dataset_.keys():
                    dataset_.rename(columns={'high': 'High'}, inplace=True)
                if 'Low' not in dataset_.keys():
                    dataset_.rename(columns={'low': 'Low'}, inplace=True)

                dataset, profit_calculator = preprocess(dataset_, cfg, logger)

            elif cfg.model is not None:
                dataset, profit_calculator = get_dataset(cfg.dataset_loader.name, "2022-01-01 13:30:00",
                                                        (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"), cfg)

            cfg.save_dir = os.getcwd()
            reporter = Reporter(cfg)
            reporter.setup_saving_dirs(cfg.save_dir)
            model = MODELS[cfg.model.type](cfg.model)

            dataset_for_profit = dataset.copy()
            dataset_for_profit.drop(['prediction'], axis=1, inplace=True)
            dataset.drop(['predicted_high', 'predicted_low'], axis=1, inplace=True)
         
            if cfg.validation_method == 'simple':
                train_dataset = dataset[ (dataset['Date'] > "2022-01-01 13:30:00") & (dataset['Date'] < (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 09:30:00"))]
                valid_dataset = dataset[ (dataset['Date'] > (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 10:30:00")) & (dataset['Date'] < (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"))]
                Trainer(cfg, train_dataset, None, model).train()
                mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()
              
            elif cfg.validation_method == 'cross_validation':
                n_split = 3
                tscv = TimeSeriesSplit(n_splits=n_split)

                for train_index, test_index in tscv.split(dataset):
                    train_dataset, valid_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
                    Trainer(cfg, train_dataset, None, model).train()
                    mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

                reporter.add_average()
            
            # Calculate profit and prediction data
            x = ProfitCalculator(cfg, dataset_for_profit, profit_calculator, mean_prediction, reporter).profit_calculator()
            predicted_high = x[0]['predicted_high'].iloc[0]
            predicted_low = x[0]['predicted_low'].iloc[0]
            predicted_mean = x[0]['predicted_mean'].iloc[0]
            predicted_high_formated = "{:.18f}".format(predicted_high)
            predicted_low_formated = "{:.18f}".format(predicted_low)
            predicted_mean_formated = "{:.18f}".format(predicted_mean)
            increase = (predicted_mean - yesterday_close) / yesterday_close
            
            # Save it anyway for UI display
            saved_percentage = increase * 100
            
            predicted_low_finally = 0
            predicted_high_finally = 0
            if predicted_low_formated > predicted_high_formated:
                predicted_low_finally = predicted_high_formated
                predicted_high_finally = predicted_low_formated
            else:
                predicted_low_finally = predicted_low_formated
                predicted_high_finally = predicted_high_formated

            sentiment_analysis = analyzer.get_sentiment_summary(symbol)
            
            # Save prediction to JSON file
            save_prediction_to_json(
                symbol=symbol,
                yesterday_close=yesterday_close,
                predicted_high=predicted_high_finally,
                predicted_low=predicted_low_finally,
                predicted_mean=predicted_mean_formated,
                increase_percentage=round(saved_percentage, 1),
                sentiment=sentiment_analysis.replace("تحليل المشاعر:\n", "").strip()
            )

            # Build alert message if it exceeds threshold
            if increase > increase_threshold:
                title += f'رمز العملة: {symbol}\n'
                title += f'نسبة الزيادة المتوقعة: {round(saved_percentage, 1)}%\n'
                title += f'اعلى سعر متوقع لليوم⬆️:\n {predicted_high_finally}\n'
                title += f'اقل سعر متوقع لليوم⬇️:\n {predicted_low_finally}\n'
                title += f'سعر الإغلاق المتوقع لليوم:\n {predicted_mean_formated}\n'
                title += '....................\n'
                title += sentiment_analysis
                title += '---\n'

            print('..............................d')
            print(yesterday_close)
            reporter.print_pretty_metrics(logger)
            reporter.save_metrics()

        except Exception as e:
            print(f"Error occurred while processing symbol {symbol}: {str(e)}")
            continue

    if title:
        print(title)
        price_comparator.update_predictions(title)
        price_comparator.start_comparing()
    return title

from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import hydra
from omegaconf import DictConfig
from functools import partial
from flask import Flask, jsonify, request
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import concurrent.futures

app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

TOKEN = os.getenv("TELEGRAM_PREDICTION_TOKEN", '7272871832:AAGa5-_FdFfziJqDG9pp4N9ljnZ2uyxslJ0')
AUTHORIZED_USERS = [895650332, 991558864, 715531930, 117245128, 1796556765, 31128146]

predicted_result = ""
training_lock = threading.Lock()
training_status = {
    "status": "idle",
    "current_symbol": "",
    "progress": "",
    "error": ""
}

# Run prediction worker in background thread
def run_prediction_worker(cfg, symbols_to_train):
    global training_status
    with training_lock:
        training_status["status"] = "running"
        training_status["error"] = ""
        try:
            for i, sym in enumerate(symbols_to_train):
                training_status["current_symbol"] = sym
                training_status["progress"] = f"جاري توقع العملة {sym} ({i+1}/{len(symbols_to_train)})..."
                train(cfg, target_symbols=[sym])
            training_status["status"] = "completed"
            training_status["progress"] = "اكتمل التوقع بنجاح!"
            training_status["current_symbol"] = ""
        except Exception as e:
            training_status["status"] = "failed"
            training_status["error"] = str(e)
            training_status["progress"] = f"فشل التوقع: {str(e)}"
            training_status["current_symbol"] = ""

def calculate_prediction(cfg: DictConfig) -> str:
    result = train(cfg)
    return result

async def send_prediction_to_users(application: Application, result: str):
    if not result:
        return
    parts = result.split('---')
    for user_id in AUTHORIZED_USERS:
        try:
            for part in parts:
                if part.strip():
                    await application.bot.send_message(user_id, part.strip())
        except Exception as e:
            print(f"فشل في إرسال التوقع إلى {user_id}: {e}")

async def daily_prediction(cfg: DictConfig, application: Application) -> None:
    global predicted_result
    print("[🔄] Daily prediction triggered in background...")
    # Run in thread executor to prevent blocking
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        predicted_result = await loop.run_in_executor(pool, calculate_prediction, cfg)
    await send_prediction_to_users(application, predicted_result)

async def check_authorized_user(update: Update) -> bool:
    user_id = update.message.from_user.id
    return user_id in AUTHORIZED_USERS

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_authorized_user(update):
        await update.message.reply_text('ليس لديك صلاحية للوصول إلى هذا البوت.')
        return
    
    keyboard = [[KeyboardButton("توقع")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text('مرحبًا! اضغط على الزر لتوقع النتيجة.', reply_markup=reply_markup)

async def handle_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    if not await check_authorized_user(update):
        await update.message.reply_text('ليس لديك صلاحية للوصول إلى هذا البوت.')
        return
    
    if update.message.text == "توقع":
        if predicted_result:
            await send_prediction_to_users(context.application, predicted_result)
        else:
            await update.message.reply_text("لم يتم حساب التوقع بعد. الرجاء المحاولة لاحقًا.")

# Flask Endpoints for Dashboard API
@app.route('/')
def home():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {str(e)}", 500

@app.route('/api/symbols')
def api_symbols():
    try:
        symbols = get_usd_and_usdt_pairs()
        return jsonify(symbols)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predictions')
def api_predictions():
    filename = "predictions.json"
    if os.path.exists(filename):
        try:
            with open(filename, "r", encoding="utf-8") as f:
                data = json.load(f)
                return jsonify(data)
        except Exception as e:
            return jsonify([])
    return jsonify([])

@app.route('/api/status')
def api_status():
    global training_status
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
        
    t = Thread(target=run_prediction_worker, args=(global_cfg, symbols_to_train))
    t.start()
    return jsonify({"success": True, "message": "بدأت عملية التوقع في الخلفية."})

@app.route('/api/liquidity', methods=['GET'])
def api_liquidity():
    symbol = request.args.get("symbol", "").upper()
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400
    try:
        from binance_liquidity import check_liquidity_and_price, get_orderbook_liquidity
        liq_msg = check_liquidity_and_price(symbol)
        ob = get_orderbook_liquidity(symbol)
        return jsonify({
            "symbol": symbol,
            "message": liq_msg,
            "order_book": ob
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/send_telegram', methods=['POST'])
def api_send_telegram():
    data = request.json or {}
    symbol = data.get("symbol", "").upper()
    
    filename = "predictions.json"
    if not os.path.exists(filename):
        return jsonify({"success": False, "message": "لا توجد توقعات متاحة لإرسالها."}), 400
        
    try:
        with open(filename, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    except Exception as e:
        return jsonify({"success": False, "message": f"فشل قراءة التوقعات: {e}"}), 500
        
    selected_prediction = None
    if symbol:
        for p in predictions:
            if p["symbol"].upper() == symbol:
                selected_prediction = p
                break
    else:
        if predictions:
            selected_prediction = predictions[-1]
            
    if not selected_prediction:
        return jsonify({"success": False, "message": "لم يتم العثور على التوقع المطلوب."}), 404
        
    title = ""
    title += f"رمز العملة: {selected_prediction['symbol']}\n"
    title += f"نسبة الزيادة المتوقعة: {selected_prediction['increase_percentage']}%\n"
    title += f"اعلى سعر متوقع لليوم⬆️:\n {selected_prediction['predicted_high']}\n"
    title += f"اقل سعر متوقع لليوم⬇️:\n {selected_prediction['predicted_low']}\n"
    title += f"سعر الإغلاق المتوقع لليوم:\n {selected_prediction['predicted_mean']}\n"
    title += '....................\n'
    title += f"تحليل المشاعر:\n{selected_prediction['sentiment']}\n"
    
    try:
        token = os.getenv("TELEGRAM_PREDICTION_TOKEN", TOKEN)
        chat_ids = AUTHORIZED_USERS
        
        import requests
        parts = title.split('---')
        for user_id in chat_ids:
            for part in parts:
                if part.strip():
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    requests.post(url, json={"chat_id": user_id, "text": part.strip()})
                    
        return jsonify({"success": True, "message": f"تم إرسال التوقع للعملة {selected_prediction['symbol']} إلى تلغرام بنجاح."})
    except Exception as e:
        return jsonify({"success": False, "message": f"فشل إرسال التلغرام: {str(e)}"}), 500

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
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    parts = line.strip().split("=", 1)
                    current_env[parts[0]] = parts[1]
                    
    for k, v in env_keys.items():
        if v:
            current_env[k] = v
            
    try:
        with open(".env", "w", encoding="utf-8") as f:
            for k, v in current_env.items():
                f.write(f"{k}={v}\n")
        load_dotenv(override=True)
        return jsonify({"success": True, "message": "تم تحديث الإعدادات وحفظها في .env بنجاح."})
    except Exception as e:
        return jsonify({"success": False, "message": f"فشل حفظ الإعدادات: {e}"}), 500

@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    global global_cfg
    global_cfg = cfg
    
    application = Application.builder().token(TOKEN).build()
    
    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=5, minute=30, second=0, timezone="Asia/Baghdad")
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])
    scheduler.start()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))
    
    # Run Telegram bot in polling
    application.run_polling()

if __name__ == '__main__':
    # Start Flask server
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.daemon = True
    flask_thread.start()
    main()
