import logging
import os
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
import asyncio
from datetime import datetime, timedelta
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from functools import partial
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# إعداد الـ Logging
logger = logging.getLogger(__name__)

# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"
symbols = []

# إعداد Flask للتعامل مع الوظائف المؤقتة
app = Flask(__name__)

# إعداد الـ Telegram Bot Token
TOKEN = '7272871832:AAGa5-_FdFfziJqDG9pp4N9ljnZ2uyxslJ0'
AUTHORIZED_USERS = [895650332, 991558864, 715531930, 117245128, 1796556765]

# دالة جلب الأزواج النشطة
def get_usd_and_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()
        trading_pairs = sorted([
            symbol["symbol"] for symbol in data["symbols"]
            if symbol["status"] == "TRADING" and (symbol["symbol"].endswith("USDT"))
        ])
        return trading_pairs
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

# دالة جلب البيانات
def fetch_and_save_data(symbol, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"
    API_KEY = 'ddCXARf1hp1OjbaLJInHpYnEhMqKziYs9ae8dEH1NbLaonYpkgPu0tX75DqnjaDD'
    API_SECRET = 'oFHovFudTJcj9UteGQa3VxxIOp9OqvlPn7t9HWiHJ62afPvgvZVo7Id01VsVRHW2'
    
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": int(start_date.timestamp() * 1000),
        "endTime": int(end_date.timestamp() * 1000),
        "limit": 1000
    }
    headers = {"X-MBX-APIKEY": API_KEY}
    
    response = requests.get(url, params=params, headers=headers)
    data = response.json()

    if isinstance(data, dict) and 'code' in data and data['code'] == -1121: 
        return False

    if not data:
        print(f"No data available for {symbol} from {start_date.date()}")
        return False

    data_filename = '/opt/render/project/src/data/data1.csv'
    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]])
    
    return True

# دالة التحقق من البيانات
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

# دالة إضافة تواريخ المستقبل
def add_future_dates(filename, symbol):
    today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    future_dates = [today + timedelta(days=i) for i in range(3)]

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for future_date in future_dates:
            formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([formatted_date, symbol, '1', '1', '1', '1', '1'])

# دالة التدريب
def train(cfg: DictConfig): 
    symbols = get_usd_and_usdt_pairs()
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
            if os.path.exists(data_filename):
                os.remove(data_filename)

            with open(data_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

            current_start = start_date
            data_available = False

            while current_start < end_date:
                current_end = min(current_start + period, end_date)
                print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
                
                if fetch_and_save_data(symbol, current_start, current_end):
                    data_available = True
                if not data_available:
                    break
                current_start = current_end + timedelta(days=1)

            if not data_available:
                print(f"Skipping {symbol} due to insufficient data.")
                continue

            yesterday_close, data_complete = check_and_delete_file(data_filename)
            if not data_complete:
                continue

            add_future_dates(data_filename, symbol)

            print("Data download complete for all symbols with data from the beginning of 2020.")
            
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

            trainer = Trainer(cfg, dataset, logger)
            trainer.create_model()
            trainer.train()
            report = Evaluator(cfg, profit_calculator).evaluate(trainer.model, cfg.save_dir)

            message = f"Training done! Final report: {report}"
            title = f"Training done for {symbol}:"
            
        except Exception as e:
            print(f"Error occurred for symbol {symbol}: {e}")
            continue

    return title + ' ' + message + '\n' + f"Accuracy: {report['accuracy']}"

# دالة الإرسال
async def daily_prediction(cfg: DictConfig, application: Application) -> None:
    for user_id in AUTHORIZED_USERS:
        try:
            # الحصول على النتيجة من دالة train
            result = train(cfg)

            # تقسيم الرسالة بناءً على الفاصل ---
            parts = result.split('---')

            # إرسال كل جزء من الأجزاء بشكل منفصل
            for part in parts:
                # التأكد من أن الجزء ليس فارغًا قبل إرساله
                if part.strip():  # التأكد من عدم إرسال جزء فارغ
                    await application.bot.send_message(user_id, part.strip())

        except Exception as e:
            print(f"فشل في إرسال التوقع إلى {user_id}: {e}")

# دالة جدولة الوظيفة
async def start_scheduler(scheduler):
    scheduler.start()
    while True:
        await asyncio.sleep(1)  # الحفاظ على الحلقة نشطة

@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()

    # إعداد الجدولة لتشغيل الوظيفة يوميًا عند الساعة الخامسة صباحًا بتوقيت بغداد
    scheduler = AsyncIOScheduler(timezone="Asia/Baghdad")
    trigger = CronTrigger(hour=5, minute=0, second=0)  # تنفيذ عند الساعة 5:00 صباحًا
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # تشغيل الجدولة في حلقة asyncio
    loop.run_until_complete(start_scheduler(scheduler))

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))  # تمرير cfg هنا
    application.run_polling()

if __name__ == '__main__':
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.start()
    main()
