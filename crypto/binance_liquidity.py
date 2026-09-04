import os
from binance.client import Client
from dotenv import load_dotenv

load_dotenv()

# إدخال API Key الخاص بك من Binance
API_KEY = os.getenv("BINANCE_API_KEY", "")
API_SECRET = os.getenv("BINANCE_API_SECRET", "")

# تهيئة الاتصال بواجهة Binance API
client = Client(API_KEY, API_SECRET)

def get_orderbook_liquidity(symbol: str) -> dict:
    try:
        depth = client.get_order_book(symbol=symbol, limit=20)
        bids = depth.get('bids', [])
        asks = depth.get('asks', [])
        
        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            spread = best_ask - best_bid
            spread_percentage = (spread / best_bid) * 100
        else:
            spread = 0.0
            spread_percentage = 0.0
            best_bid = 0.0
            best_ask = 0.0
            
        bid_depth = sum(float(bid[1]) for bid in bids)
        ask_depth = sum(float(ask[1]) for ask in asks)
        total_depth = bid_depth + ask_depth
        
        imbalance = 0.0
        if total_depth > 0:
            imbalance = (bid_depth - ask_depth) / total_depth
            
        if imbalance > 0.2:
            imbalance_desc = "سيولة شراء قوية (جدار شراء)"
        elif imbalance < -0.2:
            imbalance_desc = "سيولة بيع قوية (جدار بيع)"
        else:
            imbalance_desc = "سيولة متوازنة"
            
        return {
            "spread": f"{spread:.6f}",
            "spread_percentage": f"{spread_percentage:.4f}%",
            "bid_depth": f"{bid_depth:.2f}",
            "ask_depth": f"{ask_depth:.2f}",
            "imbalance": f"{imbalance * 100:.1f}%",
            "imbalance_desc": imbalance_desc
        }
    except Exception as e:
        print(f"Error fetching order book for {symbol}: {e}")
        return {
            "spread": "0.000000",
            "spread_percentage": "0.00%",
            "bid_depth": "0.00",
            "ask_depth": "0.00",
            "imbalance": "0.0%",
            "imbalance_desc": "غير متوفر"
        }

def check_liquidity_and_price(symbol: str) -> str:  
    liquidity=''
    message=''

    try:
        klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=8)
        old_volume = float(klines[0][7])
        new_volume = float(klines[6][7])
    except Exception as e:
        print(f"Error getting klines for liquidity: {e}")
        old_volume = 0.0
        new_volume = 0.0

    drop_percentage = 0.0
    if old_volume > 0 and new_volume > 0:
        drop_percentage = (old_volume - new_volume) / old_volume
    
    liquidity +=f'{drop_percentage * 100:.2f}'
    print(float(liquidity))
    
    if float(liquidity) <= 0:
        message='لا يوجد سحب للسيولة (تدفق سيولة إيجابي)'
    elif float(liquidity) > 0 and float(liquidity) <= 20 :
        message='سحب السيولة بسيط'
    else:
        message='سحب السيولة كبير ⚠️'

    ob = get_orderbook_liquidity(symbol)
    full_message = (
        f"{message}\n"
        f"📉 سحب الحجم: {liquidity}%\n"
        f"📊 عمق الطلبات: {ob['imbalance_desc']} (انحياز: {ob['imbalance']})\n"
        f"⚖️ الفارق (Spread): {ob['spread_percentage']}"
    )
    return full_message

