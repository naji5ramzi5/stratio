from apscheduler.schedulers.background import BackgroundScheduler
import requests
import json
from telegram import Bot
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer
import re
from binance_liquidity import check_liquidity_and_price


class PriceComparator:
    # ✅ تعريف المتغيرات كمتغيرات ثابتة داخل الكلاس
    TOKEN = "7693311875:AAExOMjL65mRn76jlV_P2XKTchpb6BQMXs8"
    AUTHORIZED_USERS = [991558864]  

    def __init__(self):
        """تهيئة الكلاس"""
        self.downSymbols = []  # قائمة العملات التي تم إرسال إشعار بشأنها
        self.predicted_title = None  # سيتم تحديثه بعد تدريب النموذج
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.compare_prices_and_send_notifications, 'interval', minutes=15)

    def get_current_price(self, symbol: str) -> float:
        """استرجاع السعر الحالي من باينانس"""
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
        response = requests.get(url)
        data = response.json()
        return float(data['price'])

    def send_to_users(self, message: str):
        """إرسال الرسالة لجميع المستخدمين المصرح لهم"""
        bot = Bot(token=self.TOKEN)
        for user_id in self.AUTHORIZED_USERS:
            bot.send_message(chat_id=user_id, text=message)

    def get_sentiment_analysis(self, symbol: str) -> str:
        """جلب تحليل المشاعر للعملات الرقمية"""
        analyzer = CryptoSentimentAnalyzer()
        return analyzer.get_sentiment_summary(symbol)

    def compare_prices_and_send_notifications(self):
        """مقارنة الأسعار مع التوقعات وإرسال الإشعارات"""
        if not self.predicted_title:
            print("⚠️ لا توجد بيانات توقع بعد!")
            return

        predictions = self.predicted_title.split('---')  # تقسيم العملات المتوقعة

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
            current_price = self.get_current_price(symbol)

            # 🟢 مقارنة مع أقل سعر متوقع
            if current_price <= predicted_low:
                result = check_liquidity_and_price(symbol)
                sentiment = self.get_sentiment_analysis(symbol)
                message = (f"⚠️ تم الوصول إلى أقل سعر متوقع ل {symbol}!\n"
                           f"📉 السعر الحالي: {current_price}\n"
                           f"📉 أقل سعر متوقع: {predicted_low}\n"
                           f"------------------------------\n"
                           f" {sentiment}\n"
                           f"------------------------------\n"
                           f"السيولة\n"
                           f" {result}")
                self.send_to_users(message)
                self.downSymbols.append({'symbol': symbol, 'predicted_high': predicted_high})

        # 🔴 مقارنة العملات في downSymbols مع أعلى سعر متوقع
        for item in self.downSymbols[:]:  # نسخة من القائمة لمنع التعديل أثناء التكرار
            symbol = item['symbol']
            predicted_high = item['predicted_high']
            current_price = self.get_current_price(symbol)

            if current_price >= predicted_high:
                sentiment = self.get_sentiment_analysis(symbol)
                result = check_liquidity_and_price(symbol)

                message = (f"🎯 تم الوصول إلى أعلى سعر متوقع ل {symbol}!\n"
                           f"📈 السعر الحالي: {current_price}\n"
                           f"📈 أعلى سعر متوقع: {predicted_high}\n"
                           f"---------------------------\n"
                           f": {sentiment}\n"
                           f"------------------------------\n"
                           f"السيولة\n"
                           f" {result}")
                self.send_to_users(message)
                self.downSymbols.remove(item)  # حذف العملة بعد تحقيق الهدف

    def update_predictions(self, new_predictions: str):
        """تحديث بيانات التوقعات بعد التدريب"""
        self.predicted_title = new_predictions
        print("✅ تم تحديث بيانات التوقعات بنجاح!")

    def start_comparing(self):
        """تشغيل الجدولة لمقارنة الأسعار"""
        self.scheduler.start()
        print("✅ تم بدء الجدولة بنجاح!")


# ✅ **تشغيل الكود إذا كان هذا هو الملف الرئيسي**
if __name__ == "__main__":
    price_comparator = PriceComparator()
    price_comparator.start_comparing()
