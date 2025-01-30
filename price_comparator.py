from apscheduler.schedulers.background import BackgroundScheduler
import requests
import json
from telegram import Bot
from telegram.ext import Updater

# إضافة مكتبة التحليل المشاعر
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer

# متغيرات البوت والتحديث
TOKEN = "YOUR_BOT_TOKEN"
AUTHORIZED_USERS = [YOUR_USER_IDS]  # قائمة مع معرفات المستخدمين
downSymbols = []  # مصفوفة لتخزين الرموز التي تم إخبارها بمستوى أدنى السعر

scheduler = BackgroundScheduler()

def get_current_price(symbol: str) -> float:
    """استرجاع السعر الحالي من باينانس"""
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
    response = requests.get(url)
    data = response.json()
    return float(data['price'])

def send_to_users(message: str):
    """إرسال الرسالة لجميع المستخدمين"""
    bot = Bot(token=TOKEN)
    for user_id in AUTHORIZED_USERS:
        bot.send_message(chat_id=user_id, text=message)

def get_sentiment_analysis(symbol: str) -> str:
    """جلب المشاعر من ملف التحليل"""
    analyzer = CryptoSentimentAnalyzer()
    return analyzer.get_sentiment_summary(symbol)

def compare_prices_and_send_notifications():
    """مقارنة الأسعار مع التوقعات وإرسال رسائل"""
    global downSymbols
    # هنا نقرأ المتغير title الذي يحتوي على التوقعات
    title = '... your title data from training process ...'  # تأكد من تمرير البيانات من مكان ما

    # تقسيم التوقعات حسب العملة
    predictions = title.split('---')
    
    for prediction in predictions:
        if "رمز العملة" not in prediction:
            continue

        # استخراج بيانات التوقع
        symbol = ...  # استخراج رمز العملة من التوقع
        predicted_low = ...  # استخراج أقل سعر متوقع
        predicted_high = ...  # استخراج أعلى سعر متوقع

        current_price = get_current_price(symbol)

        # مقارنة مع أقل سعر
        if current_price <= float(predicted_low):
            sentiment = get_sentiment_analysis(symbol)
            message = f"⚠️ تم الوصول إلى أقل سعر متوقع ل {symbol}!\n"
            message += f"السعر الحالي: {current_price}\n"
            message += f"أقل سعر متوقع: {predicted_low}\n"
            message += f"المشاعر: {sentiment}"

            send_to_users(message)
            downSymbols.append({'symbol': symbol, 'predicted_high': predicted_high})

    # مقارنة العملات في downSymbols مع أعلى سعر متوقع
    for item in downSymbols:
        symbol = item['symbol']
        predicted_high = item['predicted_high']
        current_price = get_current_price(symbol)
        if current_price >= float(predicted_high):
            sentiment = get_sentiment_analysis(symbol)
            message = f"🎯 تم الوصول إلى أعلى سعر متوقع ل {symbol}!\n"
            message += f"السعر الحالي: {current_price}\n"
            message += f"أعلى سعر متوقع: {predicted_high}\n"
            message += f"المشاعر: {sentiment}"

            send_to_users(message)
            downSymbols = [x for x in downSymbols if x['symbol'] != symbol]  # إزالة العملة من القائمة بعد الوصول إلى أعلى سعر

# جدولة المهمة لتعمل كل 15 دقيقة
scheduler.add_job(compare_prices_and_send_notifications, 'interval', minutes=15)

# بدء الجدولة
scheduler.start()

# التحقق من الجدولة في حلقة مستمرة
while True:
    pass
