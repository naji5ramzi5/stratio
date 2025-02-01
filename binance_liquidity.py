from binance.client import Client

# إدخال API Key الخاص بك من Binance
API_KEY = "ddCXARf1hp1OjbaLJInHpYnEhMqKziYs9ae8dEH1NbLaonYpkgPu0tX75DqnjaDD"
API_SECRET = "oFHovFudTJcj9UteGQa3VxxIOp9OqvlPn7t9HWiHJ62afPvgvZVo7Id01VsVRHW2"

# تهيئة الاتصال بواجهة Binance API
client = Client(API_KEY, API_SECRET)

def check_liquidity_and_price(symbol: str) -> bool:  
    liquidity=''
    message=''

    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_15MINUTE, limit=8)
    
    old_volume = float(klines[0][7])
    new_volume = float(klines[6][7])

    if old_volume > 0 and new_volume > 0:
        drop_percentage = (old_volume - new_volume) / old_volume
    
    liquidity +=f'{drop_percentage * 100:.2f}'
    print(float(liquidity))
    if float(liquidity) <= 0:
        message='لايوجد سحب للسيولة'

    if float(liquidity) > 0 and float(liquidity) <= 20 :
        message='سحب السيولة بسيط'

    if float(liquidity) > 20 :
        message='سحب السيولة كبير'

    return message
