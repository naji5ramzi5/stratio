# إصلاح البوس (Bot 4) على سيرفر Hostinger

آخر تحديث: 2026-10-24

## 1) السبب الحقيقي لصمت البوس

البوس **لا يتوقع بنفسه**. هو يقرأ توقعات Bot 3 من ملف `prediction_history.json`
(آخر ساعتين فقط). إذا لم يجد شيئاً كان الكود يعمل `return` بهدوء — بدون أي رسالة
وبدون أي سطر واضح في اللوج. يعني: **Bot 3 إذا ما اشتغل على السيرفر، البوس يصمت تماماً**.

أسباب فرعية كانت موجودة في الكود:

| المشكلة | الأثر |
|---|---|
| `.env` كان يُقرأ من مجلد التشغيل (CWD) فقط | على السيرفر CWD غالباً `/root` أو جذر المشروع → كل التوكنات فاضية → لا رسائل |
| `FOUNDATION_MODELS` غير مضبوط | تحميل Chronos + TimesFM (مئات MB + torch) → كل توقع 80 ثانية → Bot 3 لا يكمل الدورة |
| `get_qualified_symbols()[:15]` | ترتيب أبجدي: AAVE…BTC فقط. ETH و SOL و XRP و DOGE ما تدخل أبداً |
| لا يوجد fallback في البوس | إذا الملف فاضي → صمت كامل بدون تنبيه |
| رسالة Markdown مكسورة | تلغرام يرفضها 400 → البوس "يرسل" لكن لا شيء يصل |

## 2) ما تم إصلاحه في الكود

- `crypto/advanced_bot.py`
  - تحميل `.env` من مجلد السكربت أولاً، ثم CWD، ثم جذر المشروع.
  - ترتيب العملات حسب حجم التداول 24h بدل الأبجدي (`get_symbols_by_volume`).
  - استثناء العملات المستقرة والذهب (USDC, USDT, FDUSD, XAUT, PAXG …) من المسح.
  - Bot 3: يسجّل فشل كل عملة + ملخص الفشل (قبل: `except: pass` صامت).
  - Bot 4: إذا ما لقى كاش → يحسب 5 توقعات مباشرة كخطة بديلة، وإذا فشل كلشي يرسل
    رسالة "مسح ماسح" بدل الصمت.
  - Bot 4: يفحص كل الرموز التي لها توقعات محفوظة، مو فقط أول 30 بالترتيب.
  - إذا تلغرام رفض Markdown → يعيد الإرسال نص عادي (الإشارة ما تضيع).
  - إذا Binance غير متاح → سطر واضح في اللوج بدل قائمة فاضية صامتة.

- `crypto/diagnose_live.py` — فحص كامل بدون إرسال أي رسالة (توكنات، باكجات، Binance، predictor، عمر التوقعات).
- `crypto/test_boss.py` — يرسل رسالة تجريبية واحدة من البوس ويطبع رد تلغرام الحقيقي.
- `crypto/server.env.example` — قالب `.env` للسيرفر.
- `crypto/run_server.sh` + `crypto/stratocrypto-bots.service` — تشغيل نظيف مع حدود threads.

## 3) الخطوات على السيرفر (بالترتيب)

### أ. ارفع الملفات المعدّلة
ارفع من جهازك إلى السيرفر داخل مجلد `crypto/`:

```
advanced_bot.py
prediction_tracker.py
diagnose_live.py
test_boss.py
accuracy_report.py
server.env.example
run_server.sh
stratocrypto-bots.service
```

> ملاحظة: آخر commit على GitHub حذف مجلد `crypto` بالكامل، لذلك **لا تعتمد على
> `git pull`** — ارفع الملفات مباشرة (File Manager أو scp/FTP).

> **مهم جداً:** مجلد `crypto/models/` مستثنى في `.gitignore`، يعني حتى لو عملت
> `git clone` لن يأتي معه. لازم ترفع هذه الملفات الثلاثة يدوياً:
> ```
> models/ensemble_v2.pkl      (~21 MB)
> models/calibration.json     (صغير، لكن بدونه لا يوجد TRADE_READY)
> models/meta_labeler.pkl
> ```
> بدونها البوت يشتغل لكن التوقعات تكون غير معايَرة، و Boss ما يرسل إشارات أبداً
> (فقط "لا توجد إشارات").

### ب. تأكد من `.env`
```bash
cd ~/stratocrypto/crypto        # عدّل المسار حسب مكان المشروع
ls -la .env
```
إذا غير موجود:
```bash
cp server.env.example .env
nano .env        # عبّي التوكنات
```
المفاتيح المطلوبة (كلها ضرورية):
```
TELEGRAM_PREDICTION_TOKEN=   # Bot 1
TELEGRAM_LIQUIDITY_TOKEN=    # Bot 2
TELEGRAM_PREDICTOR_TOKEN=    # Bot 3
BOT4_TOKEN=                  # Boss  ← أهم واحد
ADMIN_CHAT_ID=-100...        # قناة/مجموعة الإرسال
FOUNDATION_MODELS=0
```

### ج. شغّل الفحص (لا يرسل أي رسالة)
```bash
python3 diagnose_live.py
```
يجب أن تكون النتيجة النهائية `FAIL=0`. أرسل لي المخرجات كاملة إذا فيه أي FAIL.

### د. اختبر البوس وحده
```bash
python3 test_boss.py
```
- `200 ok: true` → التوكن والقناة سليمان، وصلت رسالة "اختبار البوس".
- `401` → التوكن غلط.
- `403` → البوت مو أدمن/مو عضو في القناة.
- `400 chat not found` → `ADMIN_CHAT_ID` غلط.

### هـ. شغّل البوتات
```bash
chmod +x run_server.sh
./run_server.sh
```
أو عبر systemd:
```bash
sudo cp stratocrypto-bots.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stratocrypto-bots
sudo journalctl -u stratocrypto-bots -f
```

### و. راقب اللوج
```bash
tail -n 100 logs/advanced_bot.log
grep -E "BOT3|BOT4|FILTER|Traceback" logs/advanced_bot.log | tail -n 40
```

الدورة السليمة تبدو هكذا:
```
[FILTER] top 15 by 24h volume: BTCUSDT, ETHUSDT, SOLUSDT, ...
[BOT3] Predicting 15 symbols x 1 TFs...
[BOT3] 15/15 symbols predicted
[BOT3] Complete | N coins > 3.0%
[BOT4] Loaded 15 cached predictions from history
[BOT4] Usable predictions for 15 symbols
[BOT4] ... -> TRADE_READY / NO_TRADE ...
```

## 4) ملاحظات مهمة (بصراحة)

### الأرقام الحقيقية للدقة (من 4,985 توقع مُتحقَّق منه)
شغّل على السيرفر:
```bash
python3 accuracy_report.py
```
النتيجة المقاسة اليوم:

| المقياس | القيمة |
|---|---|
| دقة الاتجاه (صعود/هبوط) | **52.0%** |
| من التوقعات التي قالت "صعود ≥3%"، كم فعلاً صعد ≥3% | **17.0%** (694 توقع) |
| نفس النسبة لو أخذنا كل التوقعات عشوائياً (المعيار) | 15.5% |
| الفرق (lift) | **+1.5 نقطة فقط** |

يعني فلتر ">3%" لا يضيف إلا 1.5 نقطة فوق الصدفة. **80% أو 91% غير موجودة
بهذا الموديل**، وأي رقم كهذا سيكون غير صحيح. الطريق الحقيقي لتحسينه هو إعادة
تدريب مع تحقق زمني (walk-forward) واختيار ميزات جديدة — شغل منفصل تماماً عن
مشكلة صمت البوس.

- `FOUNDATION_MODELS=0` ضروري على VPS صغير. تشغيلها يستهلك RAM ويبطئ الدورة
  لدرجة أن Bot 3 لا ينهي قبل الدورة التالية.
- السيرفر يحتاج اتصال بـ Binance. إذا `diagnose_live.py` طلع FAIL في قسم
  "BINANCE MARKET DATA" فالمشكلة حجب جغرافي من مزوّد السيرفر، وليست في الكود.
- قسم `MODEL ARTIFACTS` في الفحص مهم: إذا `calibration.json` أو `ensemble_v2.pkl`
  ناقصين، التوقعات تكون غير معايَرة (uncalibrated) و Boss ما يقدر يوصل
  `TRADE_READY` أبداً — يرسل فقط "لا توجد إشارات".

## 5) إذا بقى البوس صامت

أرسل لي مخرجات هذه الأوامر الثلاثة فقط:

```bash
python3 diagnose_live.py
python3 test_boss.py
grep -E "BOT3|BOT4|Telegram|Traceback" logs/advanced_bot.log | tail -n 50
```

منها أعرف بالضبط: توكن، أم قناة، أم Binance، أم predictor، أم توقعات قديمة.
