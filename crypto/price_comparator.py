from apscheduler.schedulers.background import BackgroundScheduler
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from binance_liquidity import check_liquidity_and_price
import re
import asyncio
from crypto_sentiment_analyzer import CryptoSentimentAnalyzer

import os
from dotenv import load_dotenv

load_dotenv()


def fmt_symbol(sym: str) -> str:
    """Format symbol with slash separator: WLDUSDT -> WLD/USDT"""
    if sym.endswith("USDT"):
        return sym[:-4] + "/USDT"
    if sym.endswith("USD"):
        return sym[:-3] + "/USD"
    if sym.endswith("BTC"):
        return sym[:-3] + "/BTC"
    return sym


class PriceComparator:
    TOKEN = os.getenv("TELEGRAM_COMPARATOR_TOKEN", "")
    AUTHORIZED_USERS = [895650332, 991558864, 715531930, 117245128, 1796556765, 31128146]

    def __init__(self):
        print("[*] Initializing PriceComparator...")
        self.predicted_title = None
        self.scheduler = BackgroundScheduler()
        self.scheduler.add_job(self.compare_prices_and_send_notifications, 'interval', minutes=15)
        self.application = Application.builder().token(self.TOKEN).build()
        print("[+] PriceComparator initialized successfully!")

    def start_comparator(self, title: str):
        print("[*] Starting comparator and setting up predictions...")
        self.update_predictions(title)
        try:
            self.start_comparing()
        except Exception as e:
            print(f"[!] Scheduler already running or failed to start: {e}")
        self.compare_prices_and_send_notifications()

    def get_current_price(self, symbol: str) -> float:
        print(f"[*] Fetching current price for {symbol}...")
        url = f'https://api.binance.com/api/v3/ticker/price?symbol={symbol}'
        try:
            response = requests.get(url)
            data = response.json()
            price = float(data['price'])
            print(f"[+] Current price for {symbol}: {price}")
            return price
        except Exception as e:
            print(f"[-] Error fetching price for {symbol}: {e}")
            return None

    async def send_to_users(self, message: str):
        print("[*] Sending notification to users...")
        for user_id in self.AUTHORIZED_USERS:
            try:
                await self.application.bot.send_message(chat_id=user_id, text=message)
                print(f"[+] Message sent successfully to user {user_id}")
            except Exception as e:
                print(f"[-] Failed to send message to {user_id}: {e}")

    def compare_prices_and_send_notifications(self):
        # Don't use asyncio.run() here since scheduler runs in existing loop
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._compare_prices_and_send_notifications())
        except RuntimeError:
            # No running loop, create one
            asyncio.run(self._compare_prices_and_send_notifications())

    async def _compare_prices_and_send_notifications(self):
        analyzer = CryptoSentimentAnalyzer()
        print("[*] Starting price comparison...")
        if not self.predicted_title:
            print("[!] No prediction data available yet!")
            return

        predictions = self.predicted_title.split('---')
        for prediction in predictions:
            if "رمز العملة" not in prediction:
                continue

            match_symbol = re.search(r"رمز العملة:\s*([\w]+)", prediction)
            match_low = re.search(r"اقل سعر متوقع لليوم⬇️:\s*([\d.]+)", prediction)
            match_high = re.search(r"اعلى سعر متوقع لليوم⬆️:\s*([\d.]+)", prediction)

            if not match_symbol or not match_low or not match_high:
                print(f"[!] Failed to extract data for coin: {prediction}")
                continue

            symbol = match_symbol.group(1)
            predicted_low = float(match_low.group(1))
            predicted_high = float(match_high.group(1))
            current_price = self.get_current_price(symbol)
            if current_price is None:
                continue

            result = check_liquidity_and_price(symbol)

            lower_threshold = predicted_low * 1.2
            upper_threshold = predicted_high * 0.8

            if current_price <= predicted_low:
                message = (f"[ALERT] Price reached predicted LOW for {fmt_symbol(symbol)}!\n"
                           f"Current price: {current_price}\n"
                           f"Predicted low: {predicted_low}\n"
                           f"Liquidity: {result}")
            elif current_price <= lower_threshold:
                message = (f"[WARN] Price approaching predicted LOW for {fmt_symbol(symbol)}!\n"
                           f"Current price: {current_price}\n"
                           f"Predicted low: {predicted_low}\n"
                           f"Liquidity: {result}")
            elif current_price >= predicted_high:
                message = (f"[ALERT] Price reached predicted HIGH for {fmt_symbol(symbol)}!\n"
                           f"Current price: {current_price}\n"
                           f"Predicted high: {predicted_high}\n"
                           f"Liquidity: {result}")
            elif current_price >= upper_threshold:
                message = (f"[WARN] Price approaching predicted HIGH for {fmt_symbol(symbol)}!\n"
                           f"Current price: {current_price}\n"
                           f"Predicted high: {predicted_high}\n"
                           f"Liquidity: {result}")
            else:
                continue

            await self.send_to_users(message)

    def update_predictions(self, new_predictions: str):
        print("[*] Updating prediction data...")
        self.predicted_title = new_predictions
        print("[+] Prediction data updated!")

    def start_comparing(self):
        print("[*] Starting price comparison scheduler...")
        self.scheduler.start()
        print("[+] Scheduler started successfully!")


async def start(update: Update, context):
    await update.message.reply_text(f"Welcome, {update.effective_user.first_name}!")


async def handle_message(update: Update, context):
    await update.message.reply_text(f"Received your message: {update.message.text}")


async def run_telegram_bot():
    application = Application.builder().token(PriceComparator.TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    await application.run_polling()


if __name__ == "__main__":
    price_comparator = PriceComparator()
    price_comparator.start_comparing()
    # Keep the main thread alive for the scheduler
    import time
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("[*] Shutting down...")
        price_comparator.scheduler.shutdown()