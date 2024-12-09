import asyncio
from binance.client import Client
import pandas as pd
import pytz
from datetime import datetime, timedelta
import csv
import os
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# إعدادات Binance API
API_KEY = 'ddCXARf1hp1OjbaLJInHpYnEhMqKziYs9ae8dEH1NbLaonYpkgPu0tX75DqnjaDD'
API_SECRET = 'oFHovFudTJcj9UteGQa3VxxIOp9OqvlPn7t9HWiHJ62afPvgvZVo7Id01VsVRHW2'
client = Client(API_KEY, API_SECRET)

# إعدادات Telegram
TOKEN = "7626181745:AAFmV0ctiYsj2SiecetN_GeLezMtBVDLx8E"
AUTHORIZED_USERS = [123456789, 987654321]  # قم بإضافة ID المستخدمين المصرح لهم

# إعداد سجل الأخطاء
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# دالة لجلب البيانات من Binance API
def fetch_and_save_data(symbol: str, start_date: datetime, end_date: datetime) -> bool:
    try:
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_str, end_str)
    except Exception as e:
        logger.error(f"خطأ في جلب البيانات من Binance: {e}")
        return False

    if klines:
        with open(f"{symbol}_data.csv", 'a', newline='') as file:
            writer = csv.writer(file)
            for kline in klines:
                timestamp = datetime.utcfromtimestamp(kline[0] / 1000).replace(tzinfo=pytz.utc)
                open_price, high, low, close, volume = kline[1], kline[2], kline[3], kline[4], kline[5]
                writer.writerow([timestamp, symbol, open_price, high, low, close, volume])
        return True
    return False

# التحقق من الملف وحذفه إذا لم يكن كاملاً
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

# إضافة تواريخ مستقبلية للبيانات
def add_future_dates(filename: str, symbol: str):
    future_dates = ['2024-12-08', '2024-12-09', '2024-12-10']
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for date in future_dates:
            writer.writerow([date, symbol, '', '', '', '', ''])

# تدريب النموذج واستخراج التوقعات
def train():
    start_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)
    symbols = ["BTCUSDT", "ETHUSDT"]
    title = ''

    for symbol in symbols:
        data_filename = f"{symbol}_data.csv"
        if os.path.exists(data_filename):
            os.remove(data_filename)

        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        current_start = start_date
        data_available = False

        while current_start < end_date:
            current_end = min(current_start + period, end_date)
            if fetch_and_save_data(symbol, current_start, current_end):
                data_available = True
            current_start = current_end + timedelta(days=1)

        if not data_available:
            logger.info(f"Skipping {symbol} due to insufficient data.")
            continue

        yesterday_close, data_complete = check_and_delete_file(data_filename)
        if not data_complete:
            continue

        add_future_dates(data_filename, symbol)

    return title

# أوامر Telegram Bot
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.from_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("ليس لديك صلاحية للوصول إلى هذا البوت.")
        return
    keyboard = [[KeyboardButton("توقع")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    await update.message.reply_text("مرحبًا! اضغط على الزر لتوقع النتيجة.", reply_markup=reply_markup)

async def handle_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message.from_user.id not in AUTHORIZED_USERS:
        await update.message.reply_text("ليس لديك صلاحية للوصول إلى هذا البوت.")
        return
    if update.message.text == "توقع":
        result = train()
        await update.message.reply_text(result if result else "لا توجد بيانات كافية للتوقع.")

# التنبؤ اليومي
async def daily_prediction(application: Application) -> None:
    for user_id in AUTHORIZED_USERS:
        try:
            result = train()
            await application.bot.send_message(user_id, result if result else "لا توجد بيانات كافية للتوقع.")
        except Exception as e:
            logger.error(f"فشل في إرسال التوقع إلى {user_id}: {e}")

# الوظيفة الرئيسية
async def main():
    application = Application.builder().token(TOKEN).build()
    scheduler = AsyncIOScheduler()

    scheduler.add_job(daily_prediction, CronTrigger(hour=8, minute=0, timezone="Asia/Baghdad"), args=[application])
    scheduler.start()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prediction))

    await application.run_polling()  # لا حاجة لاستخدام asyncio.run()

if __name__ == '__main__':
    # لا تستخدم asyncio.run() هنا
    main()  # شغل main() مباشرة
