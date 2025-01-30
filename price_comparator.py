import re
import requests
import time
import asyncio
from telegram import Bot
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# بيانات تيليجرام
TOKEN = "7272871832:AAG...slJ0"  # ضع توكن البوت هنا
CHAT_ID = 895650332  # ضع معرف المستخدم أو المجموعة هنا

# رابط API Binance
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price"

# قائمة العملات التي انخفضت عن أقل سعر متوقع
downSymbol = {}

async def send_telegram_message(message):
    """
    إرسال رسالة عبر بوت تيليجرام.
    """
    bot = Bot(token=TOKEN)
    await bot.send_message(chat_id=CHAT_ID, text=message)

def get_binance_price(symbol: str):
    """
    جلب السعر الحالي للعملة من Binance API.
    """
    try:
        response = requests.get(BINANCE_API_URL, params={"symbol": symbol})
        data = response.json()
        return float(data["price"])
    except Exception as e:
        print(f"❌ خطأ في جلب سعر {symbol} من Binance: {e}")
        return None

def extract_currency_data(title: str):
    """
    استخراج رموز العملات وأقل وأعلى الأسعار المتوقعة من النص.
    """
    currency_data = []

    matches = re.findall(
        r'رمز العملة: (\w+).*?اعلى سعر متوقع لليوم⬆️:\n (\d+\.\d+).*?اقل سعر متوقع لليوم⬇️:\n (\d+\.\d+)',
        title, re.DOTALL
    )

    for match in matches:
        symbol = match[0] + "USDT"  # تحويل الرمز إلى تنسيق Binance
        predicted_high = float(match[1])
        predicted_low = float(match[2])
        currency_data.append({'symbol': symbol, 'predicted_high': predicted_high, 'predicted_low': predicted_low})

    return currency_data

async def check_prices(title: str):
    """
    مقارنة الأسعار الحالية مع التوقعات وإرسال تنبيهات عبر تيليجرام.
    """
    global downSymbol
    report = "📊 **تحديث أسعار العملات** 📊\n\n"
    currency_data = extract_currency_data(title)

    for data in currency_data:
        symbol = data['symbol']
        predicted_high = data['predicted_high']
        predicted_low = data['predicted_low']
        current_price = get_binance_price(symbol)

        if current_price is None:
            report += f"⚠️ لم يتم العثور على السعر الحالي لـ {symbol}.\n"
            continue

        # إذا كان السعر الحالي أقل من التوقع الأدنى
        if current_price <= predicted_low and symbol not in downSymbol:
            downSymbol[symbol] = predicted_high  # تخزين العملة مع أعلى سعر متوقع
            message = f"🔴 {symbol} وصل إلى أقل سعر متوقع: {current_price} (📉 {predicted_low})"
            await send_telegram_message(message)

        # إذا كانت العملة موجودة في downSymbol وارتفعت إلى أعلى سعر متوقع
        elif symbol in downSymbol and current_price >= downSymbol[symbol]:
            message = f"🟢 {symbol} ارتفع إلى أعلى سعر متوقع: {current_price} (📈 {downSymbol[symbol]})"
            await send_telegram_message(message)
            del downSymbol[symbol]  # إزالة العملة من القائمة

async def scheduled_task():
    """
    تشغيل المهمة كل 15 دقيقة.
    """
    title = """
    رمز العملة: BTC
    نسبة الزيادة المتوقعة: 2.5%
    اعلى سعر متوقع لليوم⬆️:
    48000.5
    اقل سعر متوقع لليوم⬇️:
    47000.2
    سعر الإغلاق المتوقع لليوم:
    47500.0
    ....................
    رمز العملة: ETH
    نسبة الزيادة المتوقعة: 1.8%
    اعلى سعر متوقع لليوم⬆️:
    3200.0
    اقل سعر متوقع لليوم⬇️:
    3100.5
    سعر الإغلاق المتوقع لليوم:
    3150.3
    ....................
    """

    await check_prices(title)

if __name__ == "__main__":
    scheduler = AsyncIOScheduler()
    scheduler.add_job(scheduled_task, "interval", minutes=15)
    scheduler.start()

    print("✅ بدأ تشغيل البرنامج...")
    loop = asyncio.get_event_loop()
    loop.run_forever()
