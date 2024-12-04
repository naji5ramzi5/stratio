import logging
import os
import yfinance as yf  
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
import time
from utils.reporter import Reporter
from data_loader.creator import create_dataset, preprocess
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from functools import partial
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import yaml
from datetime import datetime, timedelta
import requests
import csv

logger = logging.getLogger(__name__)

API_KEY = 'ddCXARf1hp1OjbaLJInHpYnEhMqKziYs9ae8dEH1NbLaonYpkgPu0tX75DqnjaDD'
API_SECRET = 'oFHovFudTJcj9UteGQa3VxxIOp9OqvlPn7t9HWiHJ62afPvgvZVo7Id01VsVRHW2'

BASE_URL = "https://api.binance.com"
url_klines = f"{BASE_URL}/api/v3/klines"

symbols = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT"  
]

data_folder = '/opt/render/project/src/data'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_filename = os.path.join(data_folder, 'data1.csv')

def fetch_and_save_data_binance(symbol, start_date, end_date):
    """Fetch historical data from Binance API and save it to a CSV file."""
    params = {
        "symbol": symbol,
        "interval": "1d",
        "startTime": int(start_date.timestamp() * 1000),
        "endTime": int(end_date.timestamp() * 1000),
        "limit": 1000
    }
    headers = {
        "X-MBX-APIKEY": API_KEY
    }
    
    response = requests.get(url_klines, params=params, headers=headers)

    if response.status_code != 200:
        print(f"Failed to fetch data for {symbol}. Status Code: {response.status_code}")
        return False

    data = response.json()
    if not data:
        print(f"No data returned for {symbol}")
        return False

    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            timestamp = datetime.utcfromtimestamp(row[0] / 1000).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([timestamp, symbol, row[1], row[2], row[3], row[4], row[5]])

    return True

def train(cfg: DictConfig):
    start_date = datetime(2020, 1, 1)
    end_date = datetime.now() - timedelta(days=1)
    title = ''
    increase_threshold = 0.03
    saved_percentage = 0

    for symbol in symbols:
        print(symbol)
        if os.path.exists(data_filename):
            os.remove(data_filename)  # Delete the file if it exists

        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        if not fetch_and_save_data_binance(symbol, start_date, end_date):
            print(f"Skipping {symbol} due to insufficient data.")
            continue

        yesterday_close, data_complete = check_and_delete_file(data_filename)
        if not data_complete:
            continue

        add_future_dates(data_filename, symbol)


    print(title)
    return title

# Telegram Bot setup
TOKEN = '7272871832:AAGa5-_FdFfziJqDG9pp4N9ljnZ2uyxslJ0'
AUTHORIZED_USERS = [715531930, 117245128, 1796556765]

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
        await data(update, context, cfg)

# Flask and Bot running
app = Flask(__name__)

@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()

    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=5, minute=0, second=0, timezone="Asia/Baghdad")
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])
    scheduler.start()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))
    application.run_polling()

if __name__ == '__main__':
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.start()
    main()
