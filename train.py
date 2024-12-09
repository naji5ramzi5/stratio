from binance.client import Client
import pandas as pd
import pytz
from datetime import datetime, timedelta
import csv
import os
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
import hydra
from omegaconf import DictConfig
from functools import partial
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# إعدادات Binance API
API_KEY = os.getenv('BINANCE_API_KEY', 'ddCXARf1hp1OjbaLJInHpYnEhMqKziYs9ae8dEH1NbLaonYpkgPu0tX75DqnjaDD')  # تأكد من إضافة المفتاح عبر البيئة
API_SECRET = os.getenv('BINANCE_API_SECRET', 'oFHovFudTJcj9UteGQa3VxxIOp9OqvlPn7t9HWiHJ62afPvgvZVo7Id01VsVRHW2')  # تأكد من إضافة السر عبر البيئة
client = Client(API_KEY, API_SECRET)

# إعدادات Telegram
TOKEN = os.getenv('TELEGRAM_TOKEN', '7626181745:AAFmV0ctiYsj2SiecetN_GeLezMtBVDLx8E')  # تأكد من إضافة توكن التليجرام عبر البيئة
AUTHORIZED_USERS = [895650332, 991558864]  # قم بإضافة ID المستخدمين المصرح لهم

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# دالة لجلب البيانات من Binance API
def fetch_and_save_data(symbol: str, start_date: datetime, end_date: datetime) -> bool:
    start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
    end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
    try:
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_str, end_str)
    except Exception as e:
        print(f"خطأ في جلب البيانات من Binance: {e}")
        return False

    if klines:
        with open(f"{symbol}_data.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            for kline in klines:
                timestamp = datetime.utcfromtimestamp(kline[0] / 1000).replace(tzinfo=pytz.utc)
                symbol = symbol
                open_price = kline[1]
                high = kline[2]
                low = kline[3]
                close = kline[4]
                volume = kline[5]
                writer.writerow([timestamp, symbol, open_price, high, low, close, volume])
        return True
    return False

# دالة للتحقق من البيانات في الملف وحذفه إذا لم يكن كاملاً
def check_and_delete_file(filename: str):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        if df.empty:
            os.remove(filename)
            return None, False
        else:
            yesterday_close = df['close'].iloc[-1]
            return yesterday_close, True
    return None, False

# دالة لإضافة تواريخ مستقبلية إلى الملف
def add_future_dates(filename: str, symbol: str):
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        future_dates = ['2024-12-08', '2024-12-09', '2024-12-10']
        for date in future_dates:
            writer.writerow([date, symbol, '', '', '', '', ''])

# تدريب النموذج واستخراج التوقعات
def train(cfg: DictConfig):
    start_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)

    title = ''
    increase_threshold = 0.03
    saved_percentage = 0

    for symbol in symbols:
        data_filename = f"{symbol}_data.csv"
        if os.path.exists(data_filename):
            os.remove(data_filename)  # حذف الملف إذا كان موجودًا

        # كتابة رؤوس الأعمدة فقط عند إنشاء الملف
        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        current_start = start_date
        data_available = False

        while current_start < end_date:
            current_end = min(current_start + period, end_date)
            print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
        
            # جلب البيانات لكل فترة
            if fetch_and_save_data(symbol, current_start, current_end):
                data_available = True  # تم جلب البيانات بنجاح

            current_start = current_end + timedelta(days=1)
    
        if not data_available:
            print(f"Skipping {symbol} due to insufficient data.")
            continue  # تخطي الرمز إذا لم تكن هناك بيانات

        yesterday_close, data_complete = check_and_delete_file(data_filename)
        if not data_complete:
            continue

        # إضافة تواريخ مستقبلية في نهاية الملف
        add_future_dates(data_filename, symbol)

        print("Data download complete for all symbols with data from the beginning of 2020.")
        
        # المزيد من الكود الخاص بالتدريب والتوقعات ...

    print(title)

    return title

# التفاعل مع Telegram Bot
async def data(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    result = train(cfg)

    # تقسيم النص عند الفاصل ---
    parts = result.split('---')

    # إرسال كل جزء من الأجزاء بشكل منفصل
    for part in parts:
        if part.strip():
            await update.message.reply_text(part.strip())

# التحقق من المستخدمين المصرح لهم
async def check_authorized_user(update: Update) -> bool:
    user_id = update.message.from_user.id
    if user_id in AUTHORIZED_USERS:
        return True
    return False

# بدء التفاعل مع البوت
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await check_authorized_user(update):
        await update.message.reply_text('ليس لديك صلاحية للوصول إلى هذا البوت.')
        return
    
    keyboard = [[KeyboardButton("توقع")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text('مرحبًا! اضغط على الزر لتوقع النتيجة.', reply_markup=reply_markup)

# التعامل مع الرسائل والنصوص
async def handle_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    if not await check_authorized_user(update):
        await update.message.reply_text('ليس لديك صلاحية للوصول إلى هذا البوت.')
        return
    
    if update.message.text == "توقع":
        await data(update, context, cfg)

# التنبؤ اليومي
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

# تهيئة البوت مع الوظائف المقررة يوميًا
@hydra.main(config_path="configs/hydra", config_name="train")  # تصحيح المسار هنا
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()

    # استخدام asyncio لتشغيل الوظائف في حلقة حدث
    loop = asyncio.get_event_loop()

    scheduler = AsyncIOScheduler()
    trigger = CronTrigger(hour=13, minute=41, second=30, timezone="Asia/Baghdad")
    scheduler.add_job(daily_prediction, trigger, args=[cfg, application])

    # استخدام async loop لتشغيل scheduler
    loop.create_task(scheduler.start())
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))
    
    # تشغيل البوت بشكل غير متزامن
    loop.run_until_complete(application.run_polling())

if __name__ == '__main__':
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.start()
    main()
