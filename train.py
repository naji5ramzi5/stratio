import asyncio
import os
from binance.client import Client
import pandas as pd
import pytz
from datetime import datetime, timedelta
import csv
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# إعداد Binance API
API_KEY = 'ddCXARf1hp1OjbaLJInHpYnEhMqKziYs9ae8dEH1NbLaonYpkgPu0tX75DqnjaDD'
API_SECRET = 'oFHovFudTJcj9UteGQa3VxxIOp9OqvlPn7t9HWiHJ62afPvgvZVo7Id01VsVRHW2'
client = Client(API_KEY, API_SECRET)

# إعدادات Telegram
TOKEN = "7626181745:AAFmV0ctiYsj2SiecetN_GeLezMtBVDLx8E"
AUTHORIZED_USERS = [123456789, 987654321]  # قم بإضافة ID المستخدمين المصرح لهم

# إعداد سجل الأخطاء
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

data_lock = asyncio.Lock()  # قفل لمنع التعارض

# دالة لضمان إنشاء المجلد
def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

# دالة لجلب البيانات من Binance API
def fetch_and_save_data(symbol: str, start_date: datetime, end_date: datetime) -> bool:
    try:
        start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
        end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
        klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, start_str, end_str)
        if not klines:
            logger.warning(f"لا توجد بيانات للفترة: {start_str} - {end_str}")
            return False
    except Exception as e:
        logger.error(f"خطأ في جلب البيانات من Binance: {e}")
        return False

    ensure_directory_exists(f"{symbol}_data.csv")
    with open(f"{symbol}_data.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        for kline in klines:
            timestamp = datetime.utcfromtimestamp(kline[0] / 1000).replace(tzinfo=pytz.utc)
            open_price, high, low, close, volume = kline[1], kline[2], kline[3], kline[4], kline[5]
            writer.writerow([timestamp, symbol, open_price, high, low, close, volume])
    return True

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

# تحليل البيانات والتنبؤ
def analyze_and_predict(filename: str):
    if not os.path.exists(filename):
        logger.warning(f"الملف {filename} غير موجود لتحليل التوقع.")
        return "لم يتم العثور على بيانات لتحليلها."

    # قراءة البيانات
    try:
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    except Exception as e:
        logger.error(f"خطأ في تحليل الملف {filename}: {e}")
        return "خطأ أثناء معالجة البيانات."

    # تحقق من البيانات الكافية
    if len(df) < 10:
        return "لا توجد بيانات كافية للتوقع."

    # تحليل البيانات (حساب متوسط الميل أو المتوسط العام)
    try:
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df = df.dropna(subset=['close'])
        prediction = df['close'].rolling(window=5).mean().iloc[-1]
        return f"التوقع التالي للسعر: {prediction:.2f} (بناءً على متوسط آخر 5 أيام)"
    except Exception as e:
        logger.error(f"خطأ أثناء تحليل البيانات في {filename}: {e}")
        return "خطأ أثناء التنبؤ بالنتائج."

# تدريب النموذج واستخراج التوقعات
async def train():
    async with data_lock:
        start_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
        end_date = datetime.now(pytz.utc) - timedelta(days=1)
        symbols = ["BTCUSDT", "ETHUSDT"]
        results = []

        for symbol in symbols:
            data_filename = f"{symbol}_data.csv"

            # حذف الملفات القديمة
            if os.path.exists(data_filename):
                os.remove(data_filename)

            # جمع البيانات
            current_start = start_date
            data_available = False
            while current_start < end_date:
                current_end = min(current_start + timedelta(days=90), end_date)
                if fetch_and_save_data(symbol, current_start, current_end):
                    data_available = True
                current_start = current_end + timedelta(days=1)

            if not data_available:
                logger.info(f"Skipping {symbol} due to insufficient data.")
                continue

            # التحقق وإضافة البيانات المستقبلية
            _, data_complete = check_and_delete_file(data_filename)
            if not data_complete:
                continue

            add_future_dates(data_filename, symbol)
            result = analyze_and_predict(data_filename)
            results.append(f"{symbol}: {result}")

        return "\n".join(results)

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
        result = await train()
        await update.message.reply_text(result if result else "لا توجد بيانات كافية للتوقع.")

# التنبؤ اليومي
async def daily_prediction(application: Application) -> None:
    async with data_lock:
        for user_id in AUTHORIZED_USERS:
            try:
                result = await train()
                await application.bot.send_message(
                    chat_id=user_id,
                    text=result if result else "لا توجد بيانات كافية للتوقع."
                )
            except Exception as e:
                logger.error(f"فشل في إرسال التوقع إلى المستخدم {user_id}: {e}")

# الوظيفة الرئيسية
async def main():
    application = Application.builder().token(TOKEN).build()

    # إعداد الجدولة
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        daily_prediction,
        CronTrigger(hour=8, minute=0, timezone="Asia/Baghdad"),
        args=[application]
    )
    scheduler.start()

    # إضافة معالجات الأوامر
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prediction))

    logger.info("البوت قيد التشغيل...")
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main())
