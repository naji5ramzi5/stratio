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
import pytz
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# تعريف تطبيق Flask
app = Flask(__name__)

# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"
symbols = []

data_folder = '/opt/render/project/src/data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
data_filename = os.path.join(data_folder, 'data1.csv')


def check_and_delete_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"No data in file {filename}.")
                os.remove(filename)
                return None, False
            
            # التحقق من وجود رأس (Header) وتجاهله
            if lines[0].strip().startswith('timestamp'):
                print(f"Header detected in file {filename}, skipping the first line.")
                lines = lines[1:]

            # التحقق من وجود بيانات بعد إزالة الرأس
            if not lines:
                print(f"No valid data in file {filename} after removing header.")
                os.remove(filename)
                return None, False

            # قراءة أول وآخر سطر
            first_line = lines[0].strip()  # أول سطر بيانات
            last_line = lines[-1].strip()  # آخر سطر بيانات

            # تحويل التواريخ
            try:
                first_date = datetime.strptime(first_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
                last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
            except ValueError as e:
                print(f"Error parsing dates in file {filename}: {e}")
                os.remove(filename)
                return None, False

            # التحقق من أن البيانات تبدأ من 1/1/2020
            if first_date.date() < datetime(2020, 1, 1).date():
                os.remove(filename)
                print(f"File {filename} deleted because it doesn't start from 01/01/2020.")
                return None, False

            # التحقق من أن البيانات تغطي حتى تاريخ الأمس
            if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
                os.remove(filename)
                print(f"File {filename} deleted because it doesn't cover up to yesterday.")
                return None, False

            # استخراج قيمة الإغلاق (close) من آخر سطر
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
    future_dates = [today + timedelta(days=i) for i in range(3)]  # today and the next two days

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for future_date in future_dates:
            formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([formatted_date, symbol, '1', '1', '1', '1', '1'])


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

    # Check if data is available
    if not data:
        print(f"No data available for {symbol} from {start_date.date()}")
        return False  # No data for this symbol

    # Write data to the file
    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]])

    return True  # Data fetched successfully


def get_usd_and_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()
        # جمع أسماء الأزواج التي تنتهي بـ USDT أو USD فقط
        trading_pairs = sorted([symbol["symbol"] for symbol in data["symbols"]
                                if symbol["status"] == "TRADING" and (symbol["symbol"].endswith("USDT"))])
        return trading_pairs
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


# وظائف التوقعات والردود للبوت
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    result = train(cfg)
    parts = result.split('---')
    for part in parts:
        if part.strip():  # تأكد من أن الجزء ليس فقط مسافات فارغة
            await update.message.reply_text(part.strip())  # إرفاق أي مسافات غير ضرورية


async def check_authorized_user(update: Update) -> bool:
    user_id = update.message.from_user.id
    if user_id in AUTHORIZED_USERS:
        return True
    return False


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


async def daily_prediction(cfg: DictConfig, application: Application) -> None:
    for user_id in AUTHORIZED_USERS:
        try:
            result = train(cfg)
            parts = result.split('---')

            for part in parts:
                if part.strip():
                    await application.bot.send_message(user_id, part.strip())
        except Exception as e:
            print(f"فشل في إرسال التوقع إلى {user_id}: {e}")


# تشغيل البوت وتحديد الجدولة
@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()

    # تعيين جدولة لتشغيل التوقعات اليومية في الساعة 6 صباحاً بتوقيت العراق
    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=6, minute=0, second=0, timezone="Asia/Baghdad")
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])
    scheduler.start()

    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))
    application.run_polling()


if __name__ == '__main__':
    # تشغيل Flask في Thread منفصل
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.start()

    # تشغيل التطبيق الرئيسي
    main(cfg)
