# import logging
# import os

# import hydra
# from omegaconf import DictConfig
# from models import MODELS
# from data_loader import get_dataset
# from factory.trainer import Trainer
# from factory.evaluator import Evaluator
# from factory.profit_calculator import ProfitCalculator
# import pandas as pd

# from sklearn.model_selection import TimeSeriesSplit
# from path_definition import HYDRA_PATH

# from utils.reporter import Reporter
# from data_loader.creator import create_dataset, preprocess



# import yaml
# from datetime import datetime, timedelta

# import requests
# import csv
# import os
# import pytz
# from datetime import datetime, timedelta
# logger = logging.getLogger(__name__)
# import pandas as pd

# # إعداد تفاصيل API
# url = "https://api.binance.com/api/v3/klines"
# symbols = ['ETHBTC', 'LTCBTC', 'BNBBTC', 'NEOBTC', 'QTUMETH', 'EOSETH', 'SNTETH', 'BNTETH', 'BCCBTC', 'GASBTC', 'BNBETH', 'BTCUSDT']
# symbols = ['ETHBTC','BTCUSDT']

# # إعداد مجلد لحفظ البيانات


# # إعداد التاريخ الحالي وحساب التواريخ الديناميكية
# current_date = datetime.now()

# data = {
#     "window_size": 5,
#     "train_start_date": "2022-01-01 13:30:00",  # تاريخ ثابت
#     "train_end_date": (current_date - timedelta(days=5)).strftime("%Y-%m-%d 09:30:00"),
#     "valid_start_date": (current_date - timedelta(days=5)).strftime("%Y-%m-%d 10:30:00"),
#     "valid_end_date": (current_date + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"),
#     "features": "Date, open, High, Low, close, volume",
#     "indicators_names": "rsi macd"
# }

# # كتابة البيانات إلى ملف YAML بدون علامات التنصيص
# with open("configs/hydra/dataset_loader/common.yaml", "w") as file:
#     yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)



# def check_and_delete_file(filename):
#     try:
#         with open(filename, 'r') as file:
#             last_line = file.readlines()[-1]  # Read the last line
#             last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
#             # Check if the data covers up to yesterday
#             if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
#                 os.remove(filename)  # Delete file if data is outdated
#                 print(f"File {filename} deleted because it doesn't cover up to yesterday.")
#     except Exception as e:
#         print(f"An error occurred while checking file {filename}: {e}")

# def add_future_dates(filename, symbol):
#     today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
#     future_dates = [today + timedelta(days=i) for i in range(3)]  # today and the next two days

#     with open(filename, 'a', newline='') as file:
#         writer = csv.writer(file)
#         for future_date in future_dates:
#             formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
#             writer.writerow([formatted_date, symbol,'1','1','1','1','1'])


# data_folder = 'C:/Users/moham/Downloads/crypto/crypto/data'

# if not os.path.exists(data_folder):
#     os.makedirs(data_folder)

# data_filename = os.path.join(data_folder,'data1.csv')

# def fetch_and_save_data(symbol, start_date, end_date):
#     url = "https://api.binance.com/api/v3/klines"  # Add API URL here
#     params = {
#         'symbol': symbol,
#         'interval': '1d',
#         'startTime': int(start_date.timestamp() * 1000),
#         'endTime': int(end_date.timestamp() * 1000)
#     }
    
#     response = requests.get(url, params=params)
#     data = response.json()

#     # Check if data is available
#     if not data:
#         print(f"No data available for {symbol} from {start_date.date()}")
#         return False  # No data for this symbol

#     # Write data to the file
#     with open(data_filename, 'a', newline='') as file:
#         writer = csv.writer(file)
#         for entry in data:
#             timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
#             writer.writerow([
#                 timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]
#             ])
    
#     return True  # Data fetched successfully

# @hydra.main(config_path=HYDRA_PATH, config_name="train")
# def train(cfg: DictConfig):
#     start_date = datetime(2022, 1, 1, tzinfo=pytz.utc)
#     end_date = datetime.now(pytz.utc) - timedelta(days=1)
#     period = timedelta(days=90)

#     title=''
  
#     for symbol in symbols:
#         title+=f'رمز العملة: {symbol}🪙\n'
        
#         # //
#         if os.path.exists(data_filename):
#             os.remove(data_filename)  # Delete the file if it exists

#         # Write headers only when creating the file
#         with open(data_filename, 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

#         current_start = start_date
#         data_available = False

#         while current_start < end_date:
#             current_end = min(current_start + period, end_date)
#             print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
        
#         # Fetch data for each period
#             if fetch_and_save_data(symbol, current_start, current_end):
#                 data_available = True  # Data fetched successfully for at least one period

#             current_start = current_end + timedelta(days=1)
    
#         if not data_available:
#             print(f"Skipping {symbol} due to insufficient data.")
#             continue  # Skip symbol if no data is available

#         # Add three future dates at the end of the file with the symbol
#         add_future_dates(data_filename, symbol)

#         print("Data download complete for all symbols with data from the beginning of 2022.")
#         # //
#         if cfg.load_path is None and cfg.model is None:
#             msg = 'either specify a load_path or config a model.'
#             logger.error(msg)
#             raise Exception(msg)

#         elif cfg.load_path is not None:
#             dataset_ = pd.read_csv(cfg.load_path)
#             if 'Date' not in dataset_.keys():
#                 dataset_.rename(columns={'timestamp': 'Date'}, inplace=True)
#             if 'High' not in dataset_.keys():
#                 dataset_.rename(columns={'high': 'High'}, inplace=True)
#             if 'Low' not in dataset_.keys():
#                 dataset_.rename(columns={'low': 'Low'}, inplace=True)

#             dataset, profit_calculator = preprocess(dataset_, cfg, logger)

#         elif cfg.model is not None:
#             dataset, profit_calculator = get_dataset(cfg.dataset_loader.name, cfg.dataset_loader.train_start_date,
#                                 cfg.dataset_loader.valid_end_date, cfg)

#         cfg.save_dir = os.getcwd()
#         reporter = Reporter(cfg)
#         reporter.setup_saving_dirs(cfg.save_dir)
#         model = MODELS[cfg.model.type](cfg.model)

#         dataset_for_profit = dataset.copy()
#         dataset_for_profit.drop(['prediction'], axis=1, inplace=True)
#         dataset.drop(['predicted_high', 'predicted_low'], axis=1, inplace=True)
     
#         if cfg.validation_method == 'simple':
#             train_dataset = dataset[
#                 (dataset['Date'] > cfg.dataset_loader.train_start_date) & (
#                             dataset['Date'] < cfg.dataset_loader.train_end_date)]
#             valid_dataset = dataset[
#                 (dataset['Date'] > cfg.dataset_loader.valid_start_date) & (
#                             dataset['Date'] < cfg.dataset_loader.valid_end_date)]
#             Trainer(cfg, train_dataset, None, model).train()
#             mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()
#             print('..............................d')
#             print(mean_prediction)
#         elif cfg.validation_method == 'cross_validation':
#             n_split = 3
#             tscv = TimeSeriesSplit(n_splits=n_split)

#             for train_index, test_index in tscv.split(dataset):
#                 train_dataset, valid_dataset = dataset.iloc[train_index], dataset.iloc[test_index]
#                 Trainer(cfg, train_dataset, None, model).train()
#                 mean_prediction = Evaluator(cfg, test_dataset=valid_dataset, model=model, reporter=reporter).evaluate()

#             reporter.add_average()

#         x=ProfitCalculator(cfg, dataset_for_profit, profit_calculator, mean_prediction, reporter).profit_calculator()
        
#         predicted_high=x[0]['predicted_high']
#         predicted_low=x[0]['predicted_low']
#         predicted_mean=x[0]['predicted_mean']
#         title+=f'اعلى سعر متوقع لليوم: {predicted_high}⬆️🔮\n'
#         title+=f'اقل سعر متوقع لليوم: {predicted_low}⬇️🔮\n'
#         title+=f'السعر المتوقع لليوم: {predicted_mean}🔮\n'
#         title+='----------------------------------------'
#         print(title)

#         reporter.print_pretty_metrics(logger)
#         reporter.save_metrics()
#     title+='لا تجعل التنبؤات محور تداولك. ركز على التحليل العميق وإدارة المخاطر، واستند إلى البيانات والحقائق لاتخاذ قرارات مستنيرة.\n'
#     print(title)


# if __name__ == '__main__':
#     train()
import logging
import os

import hydra
from omegaconf import DictConfig
from models import MODELS
from data_loader import get_dataset
from factory.trainer import Trainer
from factory.evaluator import Evaluator
from factory.profit_calculator import ProfitCalculator
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from path_definition import HYDRA_PATH

from utils.reporter import Reporter
from data_loader.creator import create_dataset, preprocess



import yaml
from datetime import datetime, timedelta

import requests
import csv
import os
import pytz
from datetime import datetime, timedelta
logger = logging.getLogger(__name__)
import pandas as pd


# إعداد تفاصيل API
url = "https://api.binance.com/api/v3/klines"

symbols  = [
  "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", 
  "SOLUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT", "XLMUSDT", "BCHUSDT", 
  "TRXUSDT", "VETUSDT", "EOSUSDT", "XMRUSDT", "XTZUSDT", "THETAUSDT", 
  "ATOMUSDT", "ALGOUSDT", "FILUSDT", "ZECUSDT", "NEOUSDT",
  "DASHUSDT", "KSMUSDT", "XEMUSDT", "QTUMUSDT", "SNXUSDT", "MANAUSDT", 
  "AAVEUSDT", "MKRUSDT", "COMPUSDT", "CHZUSDT", "ENJUSDT", "SUSHIUSDT", 
  "YFIUSDT", "CRVUSDT", "ZRXUSDT", "FTMUSDT", "BNTUSDT", "RENUSDT", 
  "SRMUSDT", "BALUSDT", "BATUSDT", "CELOUSDT", "EGLDUSDT", "ONEUSDT", 
  "KAVAUSDT", "LUNAUSDT", "OCEANUSDT", "ICXUSDT", "RSRUSDT", "NEXOUSDT", 
  "POWRUSDT", "OGNUSDT", "SNTUSDT", "REEFUSDT", "ANKRUSDT", "NEARUSDT", 
  "PUNDIXUSDT", "KEEPUSDT", "QNTUSDT", "HBARUSDT", "AVAXUSDT", "ONTUSDT", 
  "HOTUSDT", "SCUSDT", "ANTUSDT", "EWTUSDT", "SXPUSDT", "LSKUSDT", 
  "OXTUSDT", "STORJUSDT", "USTUSDT", "RUNEUSDT", "AMPLUSDT", "CVCUSDT", 
  "FUNUSDT", "INJUSDT", "SKLUSDT", "CKBUSDT", "ARKUSDT", "FETUSDT", 
  "DGUSDT", "CELRUSDT", "AKROUSDT", "AKTUSDT", "AERGOUSDT", "BLZUSDT", 
  "CSPRUSDT", "CTKUSDT", "CTSIUSDT", "DUSKUSDT", "ALPHAUSDT", "AUDIOUSDT", 
  "BADGERUSDT", "BELUSDT", "BONDUSDT", "KP3RUSDT", "MATICUSDT", "IMXUSDT", 
  "SRMUSDT", "TVKUSDT", "DEGOUNDT", "PLUUSDT", "LTOUSDT", "TOMOUSDT", 
  "WANUSDT", "JSTUSDT", "LITUSDT", "ORNUSDT", "PHAUSDT", "RADUSDT", 
  "RGTUSDT", "RAMPUSDT", "REQUSDT", "RNDRUSDT", "RSVUSDT", "AGIXUSDT", 
  "SFIUSDT", "SANDUSDT", "SDTUSDT", "SLPUSDT", "STEEMUSDT", "TRBUSDT", 
  "TTUSDT", "TRYUSDT", "TRUUSDT", "TWTUSDT", "UOSUSDT", "UNFIUSDT", 
  "VELOUSDT", "VRAUSDT", "VGXUSDT", "WABIUSDT", "WOOUSDT", "WINGUSDT", 
  "WRXUSDT", "YGGUSDT", "YOYOWUSDT", "ZILUSDT", "MPHUSDT", "ALEPHUSDT", 
  "AMBUSDT", "FIDAUSDT", "CREAMUSDT", "DGBUSDT", "DENTUSDT", "DODOUSDT"
];




# إعداد مجلد لحفظ البيانات


# إعداد التاريخ الحالي وحساب التواريخ الديناميكية
current_date = datetime.now()

data = {
    "window_size": 5,
    "train_start_date": "2020-01-01 13:30:00",  # تاريخ ثابت
    "train_end_date": (current_date - timedelta(days=5)).strftime("%Y-%m-%d 09:30:00"),
    "valid_start_date": (current_date - timedelta(days=5)).strftime("%Y-%m-%d 10:30:00"),
    "valid_end_date": (current_date + timedelta(days=2)).strftime("%Y-%m-%d 10:30:00"),
    "features": "Date, open, High, Low, close, volume",
    "indicators_names": "rsi macd"
}

# كتابة البيانات إلى ملف YAML بدون علامات التنصيص
with open("configs/hydra/dataset_loader/common.yaml", "w") as file:
    yaml.dump(data, file, default_flow_style=False, allow_unicode=True, sort_keys=False)



def check_and_delete_file(filename):
    try:
        # فتح الملف والتحقق من آخر سطر
        with open(filename, 'r') as file:
            lines = file.readlines()
            if not lines:
                print(f"No data in file {filename}.")
                os.remove(filename)
                return None, False

            last_line = lines[-1]  # قراءة آخر سطر في الملف
            last_date = datetime.strptime(last_line.split(',')[0], '%Y-%m-%d %H:%M:%S%z')
            
            # إذا كانت البيانات غير كاملة حتى تاريخ الأمس
            if last_date.date() < (datetime.now(pytz.utc).date() - timedelta(days=1)):
                os.remove(filename)  # حذف الملف إذا كانت البيانات قديمة
                print(f"File {filename} deleted because it doesn't cover up to yesterday.")
                return None, False  # إرجاع None مع False للإشارة إلى أن البيانات ناقصة

            # إذا كانت البيانات كاملة حتى تاريخ الأمس، حفظ قيمة الإغلاق (close)
            yesterday_close = float(last_line.split(',')[5])  # قيمة الإغلاق
            return yesterday_close, True  # إرجاع قيمة الإغلاق وTrue للدلالة على أن البيانات كاملة

    except Exception as e:
        print(f"An error occurred while checking file {filename}: {e}")
        return None, False  # في حالة حدوث خطأ، إرجاع None وFalse

def add_future_dates(filename, symbol):
    today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    future_dates = [today + timedelta(days=i) for i in range(3)]  # today and the next two days

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for future_date in future_dates:
            formatted_date = future_date.strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([formatted_date, symbol,'1','1','1','1','1'])


data_folder = 'C:/Users/moham/Downloads/crypto/crypto/data'

if not os.path.exists(data_folder):
    os.makedirs(data_folder)

data_filename = os.path.join(data_folder,'data1.csv')

def fetch_and_save_data(symbol, start_date, end_date):
    url = "https://api.binance.com/api/v3/klines"  # Add API URL here
    params = {
        'symbol': symbol,
        'interval': '1d',
        'startTime': int(start_date.timestamp() * 1000),
        'endTime': int(end_date.timestamp() * 1000)
    }
    
    response = requests.get(url, params=params)
    data = response.json()

    if isinstance(data, dict) and 'code' in data and data['code'] == -1121: 
        return False
    
    # Check if data is available
    if not data:
        print(f"No data available for {symbol} from {start_date.date()}")
        return False  # No data for this symbol

    # Write data to the file
    with open(data_filename, 'a', newline='') as file:
        writer = csv.writer(file)
        for entry in data:
            timestamp = datetime.fromtimestamp(entry[0] / 1000, tz=pytz.utc).strftime('%Y-%m-%d 00:00:00+00:00')
            writer.writerow([
                timestamp, symbol, entry[1], entry[2], entry[3], entry[4], entry[5]
            ])
    
    return True  # Data fetched successfully

def train(cfg: DictConfig):
    start_date = datetime(2020, 1, 1, tzinfo=pytz.utc)
    end_date = datetime.now(pytz.utc) - timedelta(days=1)
    period = timedelta(days=90)

    title = ''
    increase_threshold = 0.03
    saved_percentage=0
  
    for symbol in symbols:
     
        
        # //
        if os.path.exists(data_filename):
            os.remove(data_filename)  # Delete the file if it exists

        # Write headers only when creating the file
        with open(data_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume'])

        current_start = start_date
        data_available = False

        while current_start < end_date:
            current_end = min(current_start + period, end_date)
            print(f"Fetching data for {symbol} from {current_start.date()} to {current_end.date()}")
        
            # Fetch data for each period
            if fetch_and_save_data(symbol, current_start, current_end):
                data_available = True  # Data fetched successfully for at least one period

            current_start = current_end + timedelta(days=1)
    
        if not data_available:
            print(f"Skipping {symbol} due to insufficient data.")
            continue  # Skip symbol if no data is available

   
        yesterday_close, data_complete = check_and_delete_file(data_filename)
        if not data_complete:
            continue
        # Add three future dates at the end of the file with the symbol

        add_future_dates(data_filename, symbol)

        print("Data download complete for all symbols with data from the beginning of 2020.")
        # //
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
            dataset, profit_calculator = get_dataset(cfg.dataset_loader.name, cfg.dataset_loader.train_start_date,
                                cfg.dataset_loader.valid_end_date, cfg)

        cfg.save_dir = os.getcwd()
        reporter = Reporter(cfg)
        reporter.setup_saving_dirs(cfg.save_dir)
        model = MODELS[cfg.model.type](cfg.model)

        dataset_for_profit = dataset.copy()
        dataset_for_profit.drop(['prediction'], axis=1, inplace=True)
        dataset.drop(['predicted_high', 'predicted_low'], axis=1, inplace=True)
     
        if cfg.validation_method == 'simple':
            train_dataset = dataset[
                (dataset['Date'] > cfg.dataset_loader.train_start_date) & (
                            dataset['Date'] < cfg.dataset_loader.train_end_date)]
            valid_dataset = dataset[
                (dataset['Date'] > cfg.dataset_loader.valid_start_date) & (
                            dataset['Date'] < cfg.dataset_loader.valid_end_date)]
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
        predicted_high_formated="{:.18f}".format(predicted_high)
        predicted_low_formated="{:.18f}".format(predicted_low)
        predicted_mean_formated="{:.18f}".format(predicted_mean)
        increase = (predicted_mean - yesterday_close) / yesterday_close
        if increase > increase_threshold:
           saved_percentage = increase * 100
        else:
            continue 
        title += f'رمز العملة: {symbol}\n'
        title += f'نسبة الزيادة المتوقعة: {round(saved_percentage, 1)}%\n'
        title += f'اعلى سعر متوقع لليوم⬆️:\n {predicted_high_formated}\n'
        title += f'اقل سعر متوقع لليوم⬇️:\n {predicted_low_formated}\n'
        title += f'سعر الإغلاق المتوقع لليوم:\n {predicted_mean_formated}\n'
        title += '-------------------------------------------------------\n'
        print('..............................d')
        print(yesterday_close)
        reporter.print_pretty_metrics(logger)
        reporter.save_metrics()

    title += 'لا تجعل التنبؤات محور تداولك. ركز على التحليل العميق وإدارة المخاطر، واستند إلى البيانات والحقائق لاتخاذ قرارات مستنيرة.\n'
    print(title)

    return title  # Return the title or any other relevant data


# @hydra.main(config_path=HYDRA_PATH, config_name="train")
# def main(cfg: DictConfig):
#     result = train(cfg)  # استدعاء الدالة وإرجاع النتيجة
#     print(result)

# ////
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

TOKEN = '7247002552:AAFfzqoRJ95XmwOLDB6Pn2etQTSCU3zT4Pc'

# دالة التوقع (data)
import logging
from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import hydra
from omegaconf import DictConfig
from functools import partial

# إعداد تسجيل الدخول
logging.basicConfig(level=logging.INFO)

TOKEN = "7247002552:AAFfzqoRJ95XmwOLDB6Pn2etQTSCU3zT4Pc"  # استبدل بـ توكن البوت الخاص بك
# HYDRA_PATH = "C:\Users\moham\Downloads\crypto\crypto\configs"  # مسار ملفات التكوين

async def data(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    # تنفيذ العمليات عند الضغط على "توقع"
    result = train(cfg)  # استدعاء دالة train باستخدام cfg
    await update.message.reply_text(result)

# دالة البدء، تقوم بعرض زر "توقع" كزر دائم
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # إعداد زر "توقع" كزر دائم
    keyboard = [[KeyboardButton("توقع")]]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)
    
    await update.message.reply_text('مرحبًا! اضغط على الزر لتوقع النتيجة.', reply_markup=reply_markup)

# دالة لمعالجة الرسائل التي تحتوي على نص "توقع"
async def handle_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE, cfg: DictConfig) -> None:
    if update.message.text == "توقع":
        await data(update, context, cfg)  # استدعاء دالة التوقع مع تمرير cfg

@hydra.main(config_path=HYDRA_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    # إنشاء التطبيق
    application = Application.builder().token(TOKEN).build()
    
    # إضافة الأوامر والمعالجات
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, partial(handle_prediction, cfg=cfg)))  # تمرير cfg هنا
    
    # بدء تشغيل البوت
    application.run_polling()

if __name__ == '__main__':
    main()


