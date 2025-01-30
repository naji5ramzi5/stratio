from apscheduler.schedulers.background import BackgroundScheduler
import requests
import json
from telegram import Bot
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer
import re

# 📌 متغيرات التوقعات والجدولة
predicted_title = None  # سيتم تحديثه بعد تدريب النموذج
TOKEN = "7693311875:AAExOMjL65mRn76jlV_P2XKTchpb6BQMXs8"
AUTHORIZED_USERS = [991558864]  
downSymbols = []  # قائمة العملات التي تم إرسال إشعار بشأنها
scheduler = BackgroundScheduler()

# ✅ دالة لجلب السعر الحالي من باينانس
def get_current_price(symbol: str) -> float:
    url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
    response = requests.get(url)
    data = response.json()
    return float(data['price'])

# ✅ دالة لإرسال رسالة للمستخدمين المصرح لهم
def send_to_users(message: str):
    bot = Bot(token=TOKEN)
    for user_id in AUTHORIZED_USERS:
        bot.send_message(chat_id=user_id, text=message)

# ✅ دالة تحليل المشاعر للعملات الرقمية
def get_sentiment_analysis(symbol: str) -> str:
    analyzer = CryptoSentimentAnalyzer()
    return analyzer.get_sentiment_summary(symbol)

# ✅ دالة لمقارنة الأسعار مع التوقعات وإرسال الإشعارات
def compare_prices_and_send_notifications():
    global downSymbols, predicted_title

    if not predicted_title:
        print("⚠️ لا توجد بيانات توقع بعد!")
        return
    
    predictions = predicted_title.split('---')  # تقسيم العملات المتوقعة

    for prediction in predictions:
        if "رمز العملة" not in prediction:
            continue

        # 🛠️ استخراج رمز العملة، أقل سعر متوقع، وأعلى سعر متوقع باستخدام Regex
        match_symbol = re.search(r"رمز العملة:\s*([\w]+)", prediction)
        match_low = re.search(r"اقل سعر متوقع لليوم⬇️:\s*([\d.]+)", prediction)
        match_high = re.search(r"اعلى سعر متوقع لليوم⬆️:\s*([\d.]+)", prediction)

        if not match_symbol or not match_low or not match_high:
            print(f"⚠️ فشل استخراج البيانات لعملة: {prediction}")
            continue

        symbol = match_symbol.group(1)
        predicted_low = float(match_low.group(1))
        predicted_high = float(match_high.group(1))

        # 🔍 جلب السعر الحالي
        current_price = get_current_price(symbol)

        # 🟢 مقارنة مع أقل سعر متوقع
        if current_price <= predicted_low:
            sentiment = get_sentiment_analysis(symbol)
            message = (f"⚠️ تم الوصول إلى أقل سعر متوقع ل {symbol}!\n"
                       f"📉 السعر الحالي: {current_price}\n"
                       f"📉 أقل سعر متوقع: {predicted_low}\n"
                       f"🧐 المشاعر: {sentiment}")
            send_to_users(message)
            downSymbols.append({'symbol': symbol, 'predicted_high': predicted_high})

    # 🔴 مقارنة العملات في downSymbols مع أعلى سعر متوقع
    for item in downSymbols[:]:  # نسخة من القائمة لمنع التعديل أثناء التكرار
        symbol = item['symbol']
        predicted_high = item['predicted_high']
        current_price = get_current_price(symbol)

        if current_price >= predicted_high:
            sentiment = get_sentiment_analysis(symbol)
            message = (f"🎯 تم الوصول إلى أعلى سعر متوقع ل {symbol}!\n"
                       f"📈 السعر الحالي: {current_price}\n"
                       f"📈 أعلى سعر متوقع: {predicted_high}\n"
                       f"🧐 المشاعر: {sentiment}")
            send_to_users(message)
            downSymbols.remove(item)  # حذف العملة بعد تحقيق الهدف

# 🕒 جدولة الفحص كل 15 دقيقة
scheduler.add_job(compare_prices_and_send_notifications, 'interval', minutes=15)
scheduler.start()
