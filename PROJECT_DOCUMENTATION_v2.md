# StratoCrypto — نظام التداول الكمي المتقدم بالعملات الرقمية

## نظرة عامة

StratoCrypto هو نظام تداول آلي متكامل للعملات الرقمية مبني بلغة Python، يستهدف استغلال **فرصة القيمة النسبية (Statistical Arbitrage / Pairs Trading)** عبر تداول أزواج العملات بدلاً من التنبؤ بالاتجاه.

**الإصدار:** Phase 2.0 (9 طبقات مكتملة)
**الحالة:** Paper Trading (أموال وهمية) — جاهز لجمع البيانات الحية
**الاختبارات:** 95/95 OK

---

## الفلسفة الاستراتيجية

### المشكلة
التنبؤ باتجاه سعر العملة (هل يصعد أو ينزل) شبه مستحيل على المدى القصير — الدقة 45-48% (أقل من رمي العملة). كل محاولات بناء نماذج ML للتنبؤ بالاتجاه فشلت.

### الحل
بدلاً من التنبؤ بالاتجاه، نستغل حقيقة أن **أزواج معينة من العملات تتحرك معاً على المدى الطويل**. عندما يبتعد أحدهم عن الآخر بشكل غير طبيعي، نراهن أنهما سيعودان للتحرك معاً.

### مثال مبسط
- البيتكوين والإيثريوم يتحركون معاً غالباً
- فجأة الإيثريوم يصعد 10% والبيتكوين ما يتحرك
- نشتري البيتكوين (الرخيص) ونبيع الإيثريوم (الغالي)
- لما يعودان لبعض — نغلق الصفقة ونجمع الربح

---

## المعمارية — 9 طبقات

```
┌─────────────────────────────────────────────────────────────┐
│                    StratoCrypto v2.0                         │
├─────────────────────────────────────────────────────────────┤
│  Layer 9: Alerting System (alerting.py)                     │
│  Layer 8: Dashboard & Observability (dashboard.py)          │
│  Layer 7: Research & Experiment Framework (research_framework.py) │
│  Layer 6: Adaptive Market Regime Engine (regime_predictor.py) │
│  Layer 5: Execution Engine (execution_engine.py)            │
│  Layer 4: Advanced Risk Engine (risk_engine.py)             │
│  Layer 3: Dynamic Pair Selection (pair_selector.py)         │
│  Layer 2: Advanced Backtesting Engine (backtester.py)       │
│  Layer 1: Quant Audit Engine (quant_audit.py)               │
├─────────────────────────────────────────────────────────────┤
│  Core Engine: stat_arb.py (pairs trading core)              │
│  Data Layer: market_data.py + data_loader/binance_ohlcv.py  │
│  Configuration: settings.py                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## تفصيل كل طبقة

### Layer 1: Quant Audit Engine (`quant_audit.py`)
**الوظيفة:** يتحقق من صدق الأداء التاريخي إحصائياً.

**الاختبارات:**
- **Look-Ahead Bias Detection:** يتحقق من عدم تسريب بيانات مستقبلية
- **Data Leakage Detection:** يفحص حدود Train/Test
- **Data Snooping Detection:** يتتبع عدد الاختبارات والتجارب
- **Survivorship Bias:** يكشف تحيز الأصول الباقية
- **Statistical Significance:** يحسب فترات الثقة لـ Sharpe ومعدل الفوز

**الأوامر:**
```bash
python quant_audit.py  # يشغل كل الاختبارات
```

---

### Layer 2: Advanced Backtesting Engine (`backtester.py`)
**محرك اختبار خلفي متقدم مع تنفيذ واقعي.**

**الميزات:**
- **Per-trade Sharpe:** يحسب بالصفقة (مو بالشريط) — لا تضخم
- **Sortino Ratio:** يقيس العائد المعدل بالمخاطر السلبية
- **Calmar Ratio:** العائد مقابل أقصى سحب
- **Slippage Models:** ثابت / نسبة / مبني على التقلب
- **Transaction Costs:** رسوم صنع + رسوم أخذ + انزلاق
- **Stress Testing:** اختبار تحت ظروف قاسية

**الحسابات:**
```
تكلفة الذهاب والإياب = 4 × (رسوم + انزلاق)
Sharpe = (متوسط العائد / الانحراف) × √(صفقات/سنة)
```

---

### Layer 3: Dynamic Pair Selection (`pair_selector.py`)
**نظام دورة حياة الأزواج مع تقييم استقرار مركب.**

**دورة الحياة:**
```
DISCOVERED → VALIDATED → ACTIVE → DEGRADED → SUSPENDED → RETIRED
```

**Stability Score (0-100):**
| المعيار | النقاط |
|---|---|
| قوة التكامل المشترك (p<0.01) | +20 |
| استقرار نسبة التحوط | +15 |
| جودة Half-life (4-50 يوم) | +15 |
| أداء OOS (Sharpe > 3) | +30 |
| معدل فوز > 60% | +10 |
| سيولة (>30 صفقة) | +10 |
| عقوبة تباعد Z-score | -20 |

**الأوامر:**
```bash
python pair_selector.py  # يثبت الأزواج من قاعدة البيانات
```

---

### Layer 4: Advanced Risk Engine (`risk_engine.py`)
**محرك مخاطر مركزي — لا يمكن تجاوزه.**

**الحمايات:**
- **حدود المراكز:** 5% كحد أقصى لكل زوج، 60% إجمالي
- **حماية السحب:** إيقاف عند 15% سحب أقصى
- **الخسارة اليومية:** إيقاف عند 3% خسارة يومياً
- **الخسارة الأسبوعية:** إيقاف عند 7% خسارة أسبوعياً
- **الخسائر المتتالية:** إيقاف بعد 5 خسائر متتالية
- **مخاطرة الارتباط:** يرفض أزواج مرتبطة بمراكز مفتوحة
- **تقليص التقلب:** يقلص الحجم في التقلب العالي
- **Kelly Sizing:** حساب حجم ركيزة Kelly (ربع Kelly)
- **Circuit Breaker:** إيقاف طوارئ فوري

**Kelly Criterion:**
```
f* = (p × b - q) / b
حيث: p = معدل الفوز، b = متوسط الربح / متوسط الخسارة
```

---

### Layer 5: Execution Engine (`execution_engine.py`)
**يفصل توليد الإشارة عن تنفيذ الأمر.**

**الميزات:**
- **Pre-trade Checks:** فحص السيولة والتقلب قبل التنفيذ
- **Slippage Simulation:** محاكاة انزلاق واقعي
- **Two-leg Execution:** تنفيذ ساقين معاً (شراء + بيع)
- **Execution Quality:** تتبع الفرق بين سعر الإشارة وسعر التنفيذ
- **Implementation Shortfall:** قياس فقدان الأداء

**التنفيذ:**
```
Signal → Risk Check → Execution Planner → Leg A + Leg B → Result
```

---

### Layer 6: Adaptive Market Regime Engine (`regime_predictor.py`)
**يتوقع حالة السوق (اتجاهي؟ انطوائي؟ عنيف؟)**

**التصنيف:**
| النظام | الشرط | الإجراء |
|---|---|---|
| **TRENDING** | autocorr > -0.08 و dx > 0.65 | تجنب الأزواج |
| **MEAN_REVERTING** | autocorr < -0.05 | تداول عادي |
| **VOLATILE** | vol_20 > 1.0 | لا تتداول |

**الميزات المستخرجة (13 ميزة):**
- التقلب (5/20/50 شريط)
- الارتباط الذاتي للعوائد
- الاتجاه الاتجاهي (DX)
- نسبة التباين (Hurst-like)
- تقلب التقلب
- متوسط المدى الحقيقي
- انحراف العوائد

**النتيجة (0-100):**
- **80+:** TRADE — مثالي للأزواج
- **50-79:** CAUTION — تداول بحذر
- **<50:** SKIP — لا تتداول

---

### Layer 7: Research & Experiment Framework (`research_framework.py`)
**إطار بحثي — كل تجربة قابلة للتكرار.**

**يتتبع:**
- معرّف التجربة
- إصدار الاستراتيجية
- Git hash
- فترة البيانات
- الكون المختبر
- المعاملات
- الرسوم والانزلاق
- فترة التدريب و OOS
- مقاييس النتائج

**المقاييس:**
| الفئة | المقاييس |
|---|---|
| **العوائد** | إجمالي، سنوي، شهري، يومي |
| **المخاطر** | أقصى سحب، تقلب، انحراف سلبي |
| **التداول** | عدد الصفقات، معدل الفوز، عامل الربح |
| **المعدلة بالمخاطر** | Sharpe، Sortino، Calmar |
| **التنفيذ** | رسوم، انزلاق، shortfall |

**Monte Carlo Robustness:**
- تحليل Bootstrap (1000 محاكاة)
- فترات الثقة 95%
- قيمة p للربح الإيجابي

---

### Layer 8: Dashboard & Observability (`dashboard.py`)
**لوحة مراقبة احترافية.**

**تعرض:**
- **المحفظة:** المراكز المفتوحة، PnL غير المحقق
- **صحة الأزواج:** عدد النشط/المتدهور/المتقاعد
- **صحة النظام:** حالة كل مكون، عمر البيانات
- **جودة التنفيذ:** معدل النجاح، الانزلاق، shortfall

**الأوامر:**
```bash
python dashboard.py
```

---

### Layer 9: Alerting System (`alerting.py`)
**نظام تنبيهات للأحداث الحرجة.**

**التنبيهات:**
| الحدث | الخطورة |
|---|---|
| زوج متدهور | ⚠️ تحذير |
| انهيار التكامل المشترك | 🚨 حرج |
| تحذير السحب | ⚠️ تحذير |
| إيقاف المخاطر | 🚨 حرج |
| فشل تنفيذ | ⚠️ تحذير |
| بيانات قديمة | ⚠️ تحذير |
| زوج جديد عالي الجودة | ℹ️ معلومة |

---

## هيكل المشروع

```
stratocrypto/
├── advanced_bot.py              # البوت الرئيسي
├── stat_arb.py                  # محرك الأزواج الأساسي
├── pair_paper.py                # دفتر تداول الأزواج الورقي
├── regime_predictor.py          # متنبئ النظام
├── regime_gate.py               # بوابة فلترة النظام
├── market_data.py               # أنبوب البيانات
├── prediction_tracker.py        # متتبع التوقعات
├── paper_trading.py             # محرك التداول أحادي العملة
├── launch_bots.py               # لاuncher موحد
├── ml_trainer.py                # تدريب ML (مُهمل للتداول)
├── predictor.py                 # متنبئ الاتجاه (تقارير فقط)
├── settings.py                  # الإعدادات
│
├── quant_audit.py               # Layer 1: تدقيق كمي
├── backtester.py                # Layer 2: محرك خلفي
├── pair_selector.py             # Layer 3: اختيار أزواج
├── risk_engine.py               # Layer 4: محرك مخاطر
├── execution_engine.py          # Layer 5: محرك تنفيذ
├── research_framework.py        # Layer 7: إطار بحثي
├── dashboard.py                 # Layer 8: لوحة مراقبة
├── alerting.py                  # Layer 9: نظام تنبيهات
│
├── scan_smart.py                # ماسح الأزواج الذكي
├── scan_multitf.py              # ماسح متعدد الأطر
├── portfolio_manager.py         # مدير المحفظة
├── filter_pairs.py              # فلتر الأزواج
├──
├── data_loader/
│   └── binance_ohlcv.py         # محمل Binance
├── models/
│   └── calibration.json         # جدول المعايرة
├── tests/
│   ├── test_pair_paper.py       # 11 اختبار
│   ├── test_regime.py           # 6 اختبارات
│   ├── test_stat_arb.py         # 10 اختبارات
│   ├── test_regime_gate.py      # 15 اختبار
│   ├── test_tracker.py          # 8 اختبارات
│   ├── test_calibration.py      # 4 اختبارات
│   ├── test_risk.py             # 15 اختبار (جديد)
│   ├── test_integration.py      # 9 اختبارات (جديد)
│   └── test_ml_trainer.py       # 11 اختبار
│
├── pairs_active.json            # 15 زوج نشط
├── pairs_db.json                # 82 فرصة مربحة
├── pair_registry.json           # سجل دورة الحياة
├── portfolio.json               # المحفظة الحالية
├── paper_pairs.json             # الصفقات الورقية
├── market_cache.json            # ذاكرة التخزين
├── quant_audit_report.json      # تقرير التدقيق
└── PROJECT_DOCUMENTATION.md     # هاذي الوثيقة
```

---

## كيف يعمل النظام

### تدفق البيانات الكامل

```
Market Data (Binance)
  → binance_ohlcv.fetch_klines (canonical loader)
  → market_data cache layer (TTL)
  → stat_arb.fetch_aligned_pair (timestamp alignment)
  → stat_arb.walk_forward (train/test split)
  → pair_selector (lifecycle management)
  → risk_engine.check_position (risk approval)
  → execution_engine.execute_pair_trade (realistic execution)
  → pair_paper.step (live state machine)
  → alerting (notifications)
  → dashboard (monitoring)
```

### آلة التداول (State Machine)

```
FLAT (لا موقف)
  ← |Z| ≥ 2.0 و 4h يوافق ← OPEN

OPEN (موقف مفتوح)
  ← |Z| ≤ 0.5 ← CLOSE (عودة = ربح)
  ← |Z| ≥ 3.5 ← CLOSE (وقف خسارة)
  ← bars ≥ 168 ← CLOSE (انتهاء الوقت)
```

### حساب الربح

```
الربح = الاتجاه × (تغيّر السpread) - تكلفة الذهاب والإياب
تكلفة الذهاب والإياب = 4 × (رسوم% + انزلاق%)
```

---

## الفرص المكتشفة (82 فرصة)

### عبر 5 أطر زمنية:

| الإطار | الفرص | ملاحظات |
|---|---|---|
| **5m** | 63 | فرص كثيرة لكن تكاليف أعلى |
| **15m** | 10 | متوازن |
| **1h** | 58 | الأفضل للتداول الورقي |
| **4h** | 94 | فرص ممتازة |
| **1d** | 41 | أقل تكراراً لكن أجود |

### أقوى 12 موقع في المحفظة:

| الزوج | الإطار | شارب | فوز | تعرض |
|---|---|---|---|---|
| 1000SATS/XRP | 4h | 6.59 | 54% | $500 |
| 1000SATS/SHIB | 1d | 55.74 | 83% | $500 |
| ETC/FIL | 1h | 1.81 | 46% | $500 |
| ETC/OP | 1h | 4.83 | 67% | $500 |
| APT/FIL | 1h | 1.48 | 50% | $500 |
| 1000SATS/AVAX | 1h | 10.24 | 50% | $500 |
| 1000SATS/LTC | 1h | 5.99 | 53% | $500 |
| 1000SATS/PEPE | 1h | 3.88 | 36% | $500 |
| CRV/LDO | 4h | 53.28 | 42% | $500 |
| SEI/SHIB | 4h | 16.89 | 55% | $500 |
| DOGE/SUI | 4h | 24.30 | 45% | $500 |
| DOGE/WLD | 1d | 13.63 | 50% | $500 |

---

## كيفية الاستخدام

### التثبيت
```bash
pip install numpy pandas scipy statsmodels scikit-learn requests python-dotenv
```

### التشغيل السريع
```bash
# شغّل كل شي
python launch_bots.py

# أو يدوياً
python advanced_bot.py          # البوت الرئيسي
python dashboard_server.py      # لوحة التحكم (localhost:5000)
```

### المسح والتحديث
```bash
# مسح أزواج جديدة (كل أسبوع)
python scan_smart.py
python scan_multitf.py

# تحديث المحفظة
python portfolio_manager.py
python pair_selector.py

# التحقق من النظام
python quant_audit.py
python regime_predictor.py
python dashboard.py
```

### المفاتيح المطلوبة
في ملف `.env`:
```
BINANCE_API_KEY=your_key
BINANCE_SECRET=your_secret
TELEGRAM_PREDICTION_TOKEN=bot_token
TELEGRAM_LIQUIDITY_TOKEN=bot_token
TELEGRAM_PREDICTOR_TOKEN=bot_token
BOT4_TOKEN=master_bot_token
```

---

## كيف تستفيد من المشروع

### 1. كتاجر فردي (مبتدئ)
```bash
python launch_bots.py
```
- خليه يشتغل وتلقّى التنبيهات على Telegram
- كل ما يجيك "🔗 Pairs OPEN" افتح الصفقة يدوياً
- كل ما يجيك "✅ Pairs CLOSE" قفل الصفقة

### 2. كتاجر متقدم
- شغّل `scan_multitf.py` كل يوم
- استخدم `regime_predictor.py` لمعرفة إذا السوق مناسب
- استخدم `portfolio_manager.py` لتوزيع رأس المال
- فعّل التداول التلقائي (يحتاج API keys)

### 3. كباحث/مطور
- `quant_audit.py` — يتحقق من صدق أي استراتيجية جديدة
- `backtester.py` — يختبر أي فكرة بشكل واقعي
- `research_framework.py` — يتتبع تجاربك
- كل الكود موثّق ومع اختبارات (95 اختبار)

---

## الأداء التاريخي

### الأزواج (OOS — بيانات تاريخية)

| المقياس | القيمة |
|---|---|
| الأزواج المتينة | 82 |
| متوسط شارب | 4.2 |
| متوسط الفوز | 52% |
| أقصى سحب | 7-30% (حسب الزوج) |

### التوقعات الاتجاهية (مُهملة — لا تستخدم للتداول)

| الإطار | الدقة |
|---|---|
| 6h | 48.8% |
| 12h | 46.4% |
| 24h | 47.5% |
| 48h | 45.5% |

**الخلاصة:** التوقعات الاتجاهية ميتة — ركز على الأزواج فقط.

---

## المخاطر والتحذيرات

1. **البيانات التاريخية ≠ المستقبل:** كل النتائج OOS. الأداء الفعلي قد يختلف.
2. **التسوية (Slippage):** الحسابات تقديرية — الواقع قد يكون أسوأ.
3. **السيولة:** بعض الأزواج (FLOKI, BONK) قليلة السيولة.
4. **الانقطاع الإحصائي:** Cointegration ممكن ينكسر — وقف خسارة يحميك.
5. **معدل الاستدعاء (Rate Limit):** Binance يحد الاستدعاءات — الكاش يساعد.
6. **لا تكشف عن الأسرار:** المفاتيح في `.env` فقط — لا تحطها بالكود.

---

## الاختبارات (95 اختبار)

```bash
python -m unittest discover -s tests
```

| الاختبارات | العدد | الحالة |
|---|---|---|
| دفتر الأزواج | 11 | ✅ |
| متنبئ النظام | 6 | ✅ |
| محرك الأزواج | 10 | ✅ |
| بوابة النظام | 15 | ✅ |
| المتتبع | 8 | ✅ |
| المعايرة | 4 | ✅ |
| ML Trainer | 11 | ✅ |
| المخاطر | 15 | ✅ (جديد) |
| التكامل | 9 | ✅ (جديد) |
| **المجموع** | **95** | **✅ كلها ناجحة** |

---

## التقييم النهائي

| المعيار | التقييم |
|---|---|
| **الصدق الإحصائي** | 🟢 ممتاز |
| **إدارة المخاطر** | 🟢 ممتاز |
| **التنفيذ** | 🟢 جيد |
| **التكرارية** | 🟢 ممتاز |
| **المراقبة** | 🟢 جيد |
| **الاختبارات** | 🟢 ممتاز |

**الحكم النهائي: 🟡 أصفر**
- جاهز للتداول الورقي وجمع البيانات
- **ليس جاهز للمال الحقيقي بعد**
- يحتاج شهور بيانات حية للتحقق

---

**آخر تحديث:** 2026-08-09
**الإصدار:** Phase 2.0 (9 طبقات)
**الحالة:** Paper Trading
