from apscheduler.schedulers.background import BackgroundScheduler
import requests
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer
import re
from binance_liquidity import check_liquidity_and_price
import asyncio
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer

class PriceComparator:
    # توكن البوت الثاني
    TOKEN = '7693311875:AAExOMjL65mRn76jlV_P2XKTchpb6BQMXs8'
    AUTHORIZED_USERS = [991558864,895650332]  

    def __init__(self):
        print("[🔄] جاري تهيئة PriceComparator...")
        self.downSymbols = []  
        self.predicted_title = None  
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.compare_prices_and_send_notifications, 'interval', minutes=15)
        print("[✅] تم تهيئة PriceComparator بنجاح!")

    def get_current_price(self, symbol: str) -> float:
        print(f"[🔍] جلب السعر الحالي لـ {symbol}...")
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
        try:
            response = requests.get(url)
            data = response.json()
            price = float(data['price'])
            print(f"[✅] السعر الحالي لـ {symbol}: {price}")
            return price
        except Exception as e:
            print(f"[❌] خطأ أثناء جلب السعر لـ {symbol}: {e}")
            return None

    async def send_to_users(self, message: str):
        print("[📩] إرسال إشعار للمستخدمين...")
        application = Application.builder().token(self.TOKEN).build()
        
        async with application:
            for user_id in self.AUTHORIZED_USERS:
                try:
                    response = await application.bot.send_message(chat_id=user_id, text=message)
                    print(f"[✅] تم إرسال الرسالة بنجاح إلى المستخدم {user_id}")
                    print(f"[🔍] استجابة API: {response}")
                except Exception as e:
                    print(f"[❌] فشل إرسال الرسالة إلى {user_id}: {e}")

    def compare_prices_and_send_notifications(self):
        analyzer = CryptoSentimentAnalyzer()
        print("[🔄] بدء مقارنة الأسعار...")
        if not self.predicted_title:
            print("⚠️ لا توجد بيانات توقع بعد!")
            return

        predictions = self.predicted_title.split('---')
        for prediction in predictions:
            if "رمز العملة" not in prediction:
                continue

            match_symbol = re.search(r"رمز العملة:\s*([\w]+)", prediction)
            match_low = re.search(r"اقل سعر متوقع لليوم⬇️:\s*([\d.]+)", prediction)
            match_high = re.search(r"اعلى سعر متوقع لليوم⬆️:\s*([\d.]+)", prediction)

            if not match_symbol or not match_low or not match_high:
                print(f"[⚠️] فشل استخراج البيانات لعملة: {prediction}")
                continue

            symbol = match_symbol.group(1)
            predicted_low = float(match_low.group(1))
            predicted_high = float(match_high.group(1))
            
            current_price = self.get_current_price(symbol)
            if current_price is None:
                continue

            if current_price <= predicted_low:
                print(f"[⚠️] {symbol} وصل إلى أقل سعر متوقع! جاري إرسال إشعار...")
                result = check_liquidity_and_price(symbol)
                sentiment = analyzer.get_sentiment_summary(symbol)
                message = f"⚠️ تم الوصول إلى أقل سعر متوقع ل {symbol}!"
                asyncio.create_task(self.send_to_users(message))  # تشغيل الإرسال في الخلفية
                self.downSymbols.append({'symbol': symbol, 'predicted_high': predicted_high})

        for item in self.downSymbols[:]:
            symbol = item['symbol']
            predicted_high = item['predicted_high']
            current_price = self.get_current_price(symbol)

            if current_price and current_price >= predicted_high:
                print(f"[🎯] {symbol} وصل إلى أعلى سعر متوقع! جاري إرسال إشعار...")
                sentiment = analyzer.get_sentiment_summary(symbol)
                result = check_liquidity_and_price(symbol)
                message = f"🎯 تم الوصول إلى أعلى سعر متوقع ل {symbol}!"
                asyncio.create_task(self.send_to_users(message))  # تشغيل الإرسال في الخلفية
                self.downSymbols.remove(item)

    def update_predictions(self, new_predictions: str):
        print("[🔄] تحديث بيانات التوقعات...")
        self.predicted_title = new_predictions
        print("[✅] تم تحديث بيانات التوقعات بنجاح!")

    def start_comparing(self):
        print("[🚀] بدء الجدولة لمقارنة الأسعار...")
        self.scheduler.start()
        print("[✅] الجدولة قيد التشغيل بنجاح!")

# بوت التليجرام الثاني
async def start(update: Update, context):
    await update.message.reply_text(f"مرحبًا، {update.effective_user.first_name}!")

async def handle_message(update: Update, context):
    await update.message.reply_text(f"لقد استلمت رسالتك: {update.message.text}")

async def run_telegram_bot():
    application = Application.builder().token(PriceComparator.TOKEN).build()

    # إضافة معالج الأوامر والرسائل
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # بدء البوت
    await application.run_polling()

# تشغيل البوت الثاني فقط
if __name__ == "__main__":
    # بدء الجدولة لمقارنة الأسعار
    price_comparator = PriceComparator()
    price_comparator.start_comparing()

    # تشغيل البوت الثاني
    asyncio.run(run_telegram_bot())
