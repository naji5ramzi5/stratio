import os
import requests
import csv
import pandas as pd
from datetime import datetime, timedelta
import pytz
import logging
from omegaconf import DictConfig
import hydra
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from functools import partial
from flask import Flask
from threading import Thread
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from path_definition import HYDRA_PATH

# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"
symbols = []

# تعريف مسار البيانات
data_folder = '/opt/render/project/src/data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_filename = os.path.join(data_folder, 'data1.csv')

# دالة check_and_delete_file
def check_and_delete_file(filename):
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"No data in file {filename}.")
                os.remove(filename)
                return None, False

            if lines[0].strip().startswith('timestamp'):  # إذا كان السطر الأول رأس
                print(f"Header detected in file {filename}, skipping the first line.")
                lines = lines[1:]

            if not lines:
                print(f"No valid data in file {filename} after removing header.")
                os.remove(filename)
                return None, False

            first_line = lines[0].strip()
            last_line = lines[-1].strip()

            try:
                first_date = datetime.strptime(first_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
                last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
            except ValueError as e:
                print(f"Error parsing dates in file {filename}: {e}")
                os.remove(filename)
                return None, False

            if first_date.date() < datetime(2020, 1, 1).date():
                os.remove(filename)
                print(f"File {filename} deleted because it doesn't start from 01/01/2020.")
                return None, False

            if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
                os.remove(filename)
                print(f"File {filename} deleted because it doesn't cover up to yesterday.")
                return None, False

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


# دالة add_future_dates
def add_future_dates(filename, symbol):
    today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    future_dates = [today + timedelta(days=i) for i in range(3)]  # اليوم والأيام التالية

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for future_date in future_dates:
            formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([formatted_date, symbol, '1', '1', '1', '1', '1'])


# دالة get_usd_and_usdt_pairs
def get_usd_and_usdt_pairs():
    url = "https://api.binance.com/api/v3/exchangeInfo"
    try:
        response = requests.get(url)
        data = response.json()
        trading_pairs = sorted([
            symbol["symbol"] for symbol in data["symbols"]
            if symbol["status"] == "TRADING" and (symbol["symbol"].endswith("USDT"))
        ])
        return trading_pairs
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []


# دالة fetch_and_save_data
def fetch_and_save_data(symbol, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"
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

    if not data:
        print(f"No data available for {symbol} from {start_date.date()}")
        return False

    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([
                timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]
            ])
    
    return True


# دالة train
def train(cfg: DictConfig):
    symbols = get_usd_and_usdt_pairs()

    print("List of Active Trading Pairs (USDT/USD) on Binance:")
    print(len(symbols))
    start_date = datetime(2022, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)

    title = ''
    increase_threshold = 0.03
    saved_percentage = 0

    for symbol in symbols:
        try:
            if os.path.exists(data_filename):
                os.remove(data_filename)

            with open(data_filename, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

            current_start = start_date
            data_available = False

            while current_start < end_date:
                current_end = min(current_start + period, end_date)
                print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
                
                if fetch_and_save_data(symbol, current_start, current_end):
                    data_available = True

                if not data_available:
                    break
                current_start = current_end + timedelta(days=1)
            
            if not data_available:
                print(f"Skipping {symbol} due to insufficient data.")
                continue

            yesterday_close, data_complete = check_and_delete_file(data_filename)
            if not data_complete:
                continue
            add_future_dates(data_filename, symbol)

            print("Data download complete for all symbols with data from the beginning of 2020.")
            
            if cfg.load_path is None and cfg.model is None:
                msg = 'either specify a load_path or config a model.'
                logger.error(msg)
                raise Exception(msg)

            elif cfg.load_path is not None:
                dataset_ = pd.read_csv(cfg.load_path)
                if 'Date' not in dataset_.keys():
                    dataset_.rename(columns={'timestamp': 'Date'}, inplace=True)
                if 'High' not in dataset_.keys():
                    dataset_.rename(columns={'high': 'High'}, inplace=True)
                if 'Low' not in dataset_.keys():
                    dataset_.rename(columns={'low': 'Low'}, inplace=True)

                dataset, profit_calculator = preprocess(dataset_, cfg, logger)

            elif cfg.model is not None:
                dataset, profit_calculator = get_dataset(cfg.dataset_loader.name, "2022-01-01 13:30:00",
                                                        (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"), cfg)

            cfg.save_dir = os.getcwd()
            reporter = Reporter(cfg)
            reporter.setup_saving_dirs(cfg.save_dir)
            model = MODELS[cfg.model.type](cfg.model)

            dataset_for_profit = dataset.copy()
            dataset_for_profit.drop(['prediction'], axis=1, inplace=True)
            dataset.drop(['predicted_high', 'predicted_low'], axis=1, inplace=True)

            if cfg.validation_method == 'simple':
                train_dataset = dataset[ (dataset['Date'] > "2022-01-01 13:30:00") & (dataset['Date'] < (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 09:30:00"))]
                valid_dataset = dataset[ (dataset['Date'] > (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d 10:30:00")) & (dataset['Date'] < (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"))]
                Trainer(cfg, train_dataset, None, model).train()
                mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

            elif cfg.validation_method == 'cross_validation':
                n_split = 3
                tscv = TimeSeriesSplit(n_splits=n_split)

                for train_index, test_index in tscv.split(dataset):
                    train_dataset, valid_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
                    Trainer(cfg, train_dataset, None, model).train()
                    mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

                reporter.add_average()

            x = ProfitCalculator(cfg, dataset_for_profit, profit_calculator, mean_prediction, reporter).profit_calculator()
            predicted_high = x[0]['predicted_high'].iloc[0]
            predicted_low = x[0]['predicted_low'].iloc[0]
            predicted_mean = x[0]['predicted_mean'].iloc[0]
            predicted_high_formated = "{:.18f}".format(predicted_high)
            predicted_low_formated = "{:.18f}".format(predicted_low)
            predicted_mean_formated = "{:.18f}".format(predicted_mean)
            increase = (predicted_mean - yesterday_close) / yesterday_close
            if increase > increase_threshold:
                saved_percentage = increase * 100
            else:
                continue
            
            predicted_low_finally = 0
            predicted_high_finally = 0
            if predicted_low_formated > predicted_high_formated:
                predicted_low_finally = predicted_high_formated
                predicted_high_finally = predicted_low_formated
            else:
                predicted_low_finally = predicted_low_formated
                predicted_high_finally = predicted_high_formated

            title += f'رمز العملة: {symbol}\n'
            title += f'نسبة الزيادة المتوقعة: {round(saved_percentage, 1)}%\n'
            title += f'اعلى سعر متوقع لليوم⬆️:\n {predicted_high_finally}\n'
            title += f'اقل سعر متوقع لليوم⬇️:\n {predicted_low_finally}\n'
            title += f'سعر الإغلاق المتوقع لليوم:\n {predicted_mean_formated}\n'
            title += '---\n'
            print('..............................d')
            print(yesterday_close)
            reporter.print_pretty_metrics(logger)
            reporter.save_metrics()

        except Exception as e:
            print(f"Error occurred while processing symbol {symbol}: {str(e)}")
            continue

    print(title)
    return title


# التطبيق Flask لتشغيل البوت
app = Flask(__name__)

logging.basicConfig(level=logging.INFO)

TOKEN = '7272871832:AAGa5-_FdFfziJqDG9pp4N9ljnZ2uyxslJ0'
AUTHORIZED_USERS = [895650332, 991558864, 715531930, 117245128, 1796556765]

async def data(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    result = train(cfg)
    parts = result.split('---')
    for part in parts:
        if part.strip():
            await update.message.reply_text(part.strip())


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


@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))
    application.run_polling()


if __name__ == '__main__':
    flask_thread = Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 8080})
    flask_thread.start()
    main()
