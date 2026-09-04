"""
STRATOCRYPTO — News Intelligence : Telegram Bot (Bot #5 — News)
==============================================================

A Telegram bot that monitors crypto / macro / Fed / regulation / geopolitical
news via free RSS sources (Google News primary, crypto RSS secondary, official
RSS), turns it into structured Market Events, and pushes alerts to the owner.

Commands:
  /start /help /news /breaking /btc /eth /macro /fed /trump /regulation
  /market /impact /sources

Token: reads TELEGRAM_BOT_TOKEN from .env (same token as the rest of the
StratoCrypto fleet — one Telegram bot per token; this is the "news" agent).
Recipients: ADMIN_CHAT_ID in .env (falls back to hardcoded owner id).

No news is invented. Each alert carries VERIFIED / UNVERIFIED / CONFLICTING.
"""

import os
import sys
import json
import time
import logging
import threading
import urllib.parse
from datetime import datetime, timezone, timedelta

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from dotenv import load_dotenv
load_dotenv()
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

from news_agent import NewsAgent
from news_provider import NewsAggregator, reliability_for

log = logging.getLogger("news.bot")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [NEWS-BOT] %(message)s")

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Accept ALL users: anyone who messages the bot becomes a subscriber and
# receives alerts. Subscribers persist across restarts.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBS_FILE = os.path.join(BASE_DIR, "news_subscribers.json")


def load_subs():
    try:
        with open(SUBS_FILE, encoding="utf-8") as f:
            return set(int(x) for x in json.load(f))
    except Exception:
        return set()


def save_subs():
    try:
        with open(SUBS_FILE, "w", encoding="utf-8") as f:
            json.dump(sorted(SUBSCRIBERS), f)
    except Exception:
        pass


SUBSCRIBERS = load_subs()
_admin = os.getenv("ADMIN_CHAT_ID", "")
for _aid in _admin.split(","):
    _aid = _aid.strip()
    if _aid.isdigit():
        SUBSCRIBERS.add(int(_aid))
# Ensure friend is always subscribed
SUBSCRIBERS.add(715531930)
save_subs()


def subscribe(uid: int):
    if uid not in SUBSCRIBERS:
        SUBSCRIBERS.add(uid)
        save_subs()

try:
    from telebot import TeleBot
    from telebot import apihelper as _apihelper
except ImportError:
    log.error("pyTelegramBotAPI not installed. Run: py -m pip install pyTelegramBotAPI")
    sys.exit(1)

# Optional proxy (e.g. SOCKS5/HTTP) for networks that block Telegram.
# Set in .env:  TELEGRAM_PROXY=socks5://user:pass@host:port
PROXY = os.getenv("TELEGRAM_PROXY", "")
if PROXY:
    _apihelper.proxy = {"https": PROXY, "http": PROXY}
    log.info(f"Using Telegram proxy: {PROXY.split('@')[-1]}")

bot = TeleBot(TOKEN) if TOKEN else None
agent = NewsAgent()

# Sweep configuration
SWEEP_INTERVAL = int(os.getenv("NEWS_SWEEP_MINUTES", "10")) * 60
ALERT_MIN_LEVEL = os.getenv("NEWS_ALERT_MIN", "IMPORTANT")   # INFO/WATCH/IMPORTANT/BREAKING/CRITICAL
MAX_ALERTS_PER_SWEEP = int(os.getenv("NEWS_MAX_ALERTS", "8"))  # anti-spam cap per cycle

CAT_QUERIES = {
    "fed": "(Federal Reserve OR Fed OR FOMC OR Powell OR interest rate OR rate cut OR rate hike OR CPI OR PCE)",
    "us_gov": "(Trump OR White House OR Congress OR Treasury OR executive order)",
    "regulation": "(SEC OR CFTC OR crypto regulation OR stablecoin bill OR ETF regulation OR crypto legislation)",
    "crypto": "(Bitcoin OR Ethereum OR Solana OR XRP OR stablecoin OR DeFi OR crypto exchange OR ETF)",
    "market": "(ETF inflows OR liquidation OR whale OR exchange hack OR funding rate OR open interest)",
    "geopolitical": "(war OR sanctions OR tariffs OR trade restriction)",
    "macro": "(CPI OR PPI OR GDP OR jobs OR NFP OR PMI OR retail sales OR Treasury auction OR dollar index)",
}

LEVEL_ORDER = ["INFO", "WATCH", "IMPORTANT", "BREAKING", "CRITICAL"]


# ── formatting ──────────────────────────────────────────────────────
def baghdad_str(ts: float) -> str:
    if not ts:
        return "—"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=3)
    return dt.strftime("%Y-%m-%d %H:%M") + " AST"


def verification_label(v: str) -> str:
    return {"VERIFIED": "✅ VERIFIED (مؤكد من مصدر رسمي/مصدرين مستقلين)",
            "SINGLE_SOURCE": "ℹ️ SINGLE SOURCE (مصدر واحد موثوق)",
            "UNVERIFIED": "⚠️ UNVERIFIED REPORT",
            "CONFLICTING": "🔄 CONFLICTING REPORTS"}.get(v, v)


def impact_label(score: int) -> str:
    if score >= 90: return "EXTREME"
    if score >= 75: return "VERY HIGH"
    if score >= 60: return "HIGH"
    if score >= 40: return "MEDIUM"
    if score >= 20: return "LOW"
    return "VERY LOW"


CATEGORY_AR = {
    "FED / Monetary": "السياسة النقدية للاحتياطي الفيدرالي",
    "U.S. Government": "الحكومة الأمريكية",
    "Regulation": "التنظيم والرقابة",
    "Crypto": "سوق العملات الرقمية",
    "Market": "هيكل السوق",
    "Geopolitical": "الأحداث الجيوسياسية",
    "Macro": "الاقتصاد الكلي",
    "Other": "الأسواق العامة",
}
SENTIMENT_AR = {"BULLISH": "إيجابي", "BEARISH": "سلبي", "NEUTRAL": "محايد"}


def format_event(ev, compact=False) -> str:
    if ev.official_speech:
        header = "🔴 تنبيه رسمي — حكومة / بنك مركزي"
    elif ev.breaking:
        header = "🚨 خبر عاجل"
    else:
        header = "📡 وحدة تحليل أخبار ستراتو"
    L = []
    L.append(header)
    L.append("━━━━━━━━━━━━━━━━━━━━━━")
    if ev.official_speech:
        L.append("🏛 مصدر رسمي يتحدث عن العملات/الكريبتو — أولوية قصوى")
    L.append("")
    L.append("📰 الخبر:")
    L.append(f"{ev.headline}")
    L.append("")
    L.append(f"🏛 المصدر الرئيسي: {ev.primary_source or (ev.sources[0]['name'] if ev.sources else '—')}")
    independent = int(getattr(ev, "independent_sources_count", 0) or 0)
    if independent > 1:
        L.append(f"📚 تأكيدات مستقلة: {independent - 1}")
    else:
        L.append("📚 تأكيدات مستقلة: لا يوجد")
    L.append(f"⏱ الوقت: {baghdad_str(ev.first_ts)}")
    L.append(f"🎯 الأصول المتأثرة: {', '.join(ev.assets) if ev.assets else 'السوق بشكل عام'}")
    L.append(f"📊 التصنيف: {CATEGORY_AR.get(ev.category, ev.category)}")
    L.append(f"🧠 الميل: {SENTIMENT_AR.get(ev.sentiment, ev.sentiment)}")
    L.append(f"🔥 تأثير السوق: {impact_label(ev.impact_score)} — {ev.impact_score}/100")
    L.append(f"🎯 الثقة: {ev.confidence}/100")
    L.append(f"🔎 التحقق: {verification_label(ev.verification)}")
    if ev.price_before:
        pr = ", ".join(f"{a} ${v:,.2f}" for a, v in ev.price_before.items())
        L.append(f"💲 السعر عند الرصد: {pr}")
    if not compact:
        L.append("")
        L.append("🔎 التحليل:")
        L.append(ev.analysis or "—")
        L.append("")
        L.append("⚠️ ملاحظة: هذا تحليل للمعلومة وليس ضمانًا لاتجاه السوق.")
        L.append("")
        L.append(f"🔎 سبب التحقق: {getattr(ev, 'verification_reason', '') or '—'}")
        # Telegram text intentionally contains no URLs: Google RSS redirects are
        # unreadable and must not be mistaken for independent source evidence.
        displayed = []
        for source in ev.sources:
            if source.get("is_aggregated"):
                continue
            name = source.get("name", "?")
            if name not in displayed:
                displayed.append(name)
        if displayed:
            shown = ", ".join(displayed[:3])
            suffix = f" +{len(displayed) - 3}" if len(displayed) > 3 else ""
            L.append(f"📰 مصادر مباشرة: {shown}{suffix}")
        elif getattr(ev, "mention_count", len(ev.sources)):
            L.append("📰 مصدر الرصد: Google News (بانتظار رابط مباشر قابل للتحقق)")
    return "\n".join(L)


def send_to_users(text: str):
    if bot is None:
        log.warning("No token — cannot send Telegram message")
        return
    for uid in set(SUBSCRIBERS):
        try:
            bot.send_message(uid, text)
        except Exception as e:
            log.warning(f"send to {uid} failed: {e}")


# ── fetching helpers ───────────────────────────────────────────────
def fetch_events(query=None, category=None, asset=None, limit=25):
    agg = NewsAggregator()
    if category:
        items = agg.fetch_category(category, limit)
    elif query:
        items = agg.fetch_query(query, limit)
    else:
        items = agg.fetch_category("crypto", limit)
    evs = agent.process(items)
    if asset:
        evs = [e for e in evs if asset.upper() in e.assets]
    return evs


def sweep_once():
    """Full category sweep -> alert new/escalated events."""
    agg = NewsAggregator()
    items = []
    for cat in CAT_QUERIES:
        items.extend(agg.fetch_category(cat, 20))
    items.extend(agg.fetch_official(8))
    evs = agent.process(items)
    mi = LEVEL_ORDER.index(ALERT_MIN_LEVEL)
    cand = [e for e in evs if LEVEL_ORDER.index(e.alert_level) >= mi]
    cand = sorted(cand, key=lambda e: e.impact_score, reverse=True)[:MAX_ALERTS_PER_SWEEP]
    sent = 0
    for ev in cand:
        send_to_users(format_event(ev))
        sent += 1
        time.sleep(0.3)
    log.info(f"sweep done — {len(evs)} events, {sent} alerts sent (cap {MAX_ALERTS_PER_SWEEP})")
    return sent


def official_speech_sweep():
    """Priority sweep: alert whenever an official source speaks about currencies.

    Bypasses the normal level/cap filters — this is the White House / Fed / Treasury
    / SEC / CFTC linkage the owner requested. Dedup is handled by the agent so a
    story is not spammed.
    """
    agg = NewsAggregator()
    items = agg.fetch_official_speech(12)
    if not items:
        return 0
    evs = agent.process(items)
    sent = 0
    for ev in evs:
        if ev.official_speech:
            send_to_users(format_event(ev))
            sent += 1
            time.sleep(0.3)
    if sent:
        log.info(f"official speech sweep — {sent} priority alert(s)")
    return sent


# ── background loop ─────────────────────────────────────────────────
def _loop():
    time.sleep(20)   # let polling start first
    while True:
        try:
            sweep_once()
            official_speech_sweep()
        except Exception as e:
            log.warning(f"sweep error: {e}")
        time.sleep(SWEEP_INTERVAL)


# ── command handlers ───────────────────────────────────────────────
def _reply(message, text):
    try:
        bot.send_message(message.chat.id, text)
    except Exception as e:
        log.warning(f"reply failed: {e}")


@bot.message_handler(commands=["start", "help"])
def cmd_help(message):
    subscribe(message.chat.id)
    txt = (
        "🤖 *StratoCrypto News Intelligence*\n"
        "مراقبة الأخبار الاقتصادية والتنظيمية والسياسية المؤثرة على الكريبتو.\n\n"
        "تم اشتراكك ✅ وستصلك التنبيهات تلقائيًا.\n\n"
        "الأوامر:\n"
        "/news — أحدث الأخبار المهمة\n"
        "/breaking — الأخبار العاجلة فقط\n"
        "/btc — أخبار Bitcoin\n"
        "/eth — أخبار Ethereum\n"
        "/macro — الاقتصاد الكلي\n"
        "/fed — الفيدرالي الأمريكي\n"
        "/trump — أخبار Trump الاقتصادية\n"
        "/regulation — SEC / CFTC / التنظيم\n"
        "/gov — كلام المصادر الرسمية عن الكريبتو (البيت الأبيض/الفيدرالي/الخزينة)\n"
        "/market — الأكثر تأثيرًا على السوق\n"
        "/impact — الأعلى حسب Impact Score\n"
        "/sources — المصادر المستخدمة\n\n"
        "⚠️ الأخبار تُرسل تلقائيًا عند اكتشافها. هذا تحليل معلومات لا ضمان للسعر."
    )
    _reply(message, txt)


@bot.message_handler(func=lambda m: not (m.text or "").startswith("/"))
def cmd_any(message):
    # Accept ANY user: subscribe on first contact, then echo help.
    subscribe(message.chat.id)
    _reply(message,
           "👋 تم اشتراكك في تنبيهات StratoCrypto.\n"
           "اكتب /news أو /breaking أو /help لعرض الأخبار.\n"
           "الأخبار المهمة ستصلك تلقائيًا.")


@bot.message_handler(commands=["news"])
def cmd_news(message):
    evs = fetch_events(category="crypto", limit=30)
    evs = [e for e in evs if LEVEL_ORDER.index(e.alert_level) >= LEVEL_ORDER.index("WATCH")]
    evs = sorted(evs, key=lambda e: e.impact_score, reverse=True)[:5]
    if not evs:
        _reply(message, "لا توجد أخبار مهمة حاليًا.")
        return
    for ev in evs:
        _reply(message, format_event(ev, compact=True))


@bot.message_handler(commands=["breaking"])
def cmd_breaking(message):
    evs = fetch_events(category="fed", limit=20) + fetch_events(category="regulation", limit=20)
    evs = [e for e in evs if e.breaking]
    evs = sorted(evs, key=lambda e: e.impact_score, reverse=True)[:5]
    if not evs:
        _reply(message, "لا توجد أخبار عاجلة الآن.")
        return
    for ev in evs:
        _reply(message, format_event(ev))


@bot.message_handler(commands=["btc"])
def cmd_btc(message):
    evs = fetch_events(query="Bitcoin", limit=25)
    _show_asset(message, evs, "BTC")


@bot.message_handler(commands=["eth"])
def cmd_eth(message):
    evs = fetch_events(query="Ethereum", limit=25)
    _show_asset(message, evs, "ETH")


@bot.message_handler(commands=["macro"])
def cmd_macro(message):
    evs = fetch_events(category="macro", limit=25)
    _show(message, evs)


@bot.message_handler(commands=["fed"])
def cmd_fed(message):
    evs = fetch_events(category="fed", limit=25)
    _show(message, evs)


@bot.message_handler(commands=["trump"])
def cmd_trump(message):
    evs = fetch_events(query="(Trump AND (crypto OR economy OR tariff OR bitcoin))", limit=25)
    _show(message, evs)


@bot.message_handler(commands=["regulation"])
def cmd_regulation(message):
    evs = fetch_events(category="regulation", limit=25)
    _show(message, evs)


@bot.message_handler(commands=["market"])
def cmd_market(message):
    evs = fetch_events(category="crypto", limit=30) + fetch_events(category="market", limit=20)
    evs = [e for e in evs if LEVEL_ORDER.index(e.alert_level) >= LEVEL_ORDER.index("IMPORTANT")]
    evs = sorted(evs, key=lambda e: e.impact_score, reverse=True)[:5]
    _show(message, evs)


@bot.message_handler(commands=["impact"])
def cmd_impact(message):
    evs = agent.top_impact(10)
    _show(message, evs)


@bot.message_handler(commands=["sources"])
def cmd_sources(message):
    txt = (
        "🔗 *المصادر المستخدمة*\n"
        "Primary: Google News (مجاني، بدون مفتاح)\n"
        "Secondary: CoinDesk / Cointelegraph / Decrypt / The Block RSS\n"
        "Official: Federal Reserve / SEC / Treasury / White House / CFTC / ECB\n\n"
        "تقييم الموثوقية: المصادر الرسمية 100، المالية الكبرى 88-93، "
        "الإعلام الكريبتو 80-84، وسائل التواصل 10-40 (تُستخدم لاكتشاف فقط)."
    )
    _reply(message, txt)


def _show_asset(message, evs, asset):
    evs = [e for e in evs if asset in e.assets]
    _show(message, evs)


@bot.message_handler(commands=["gov"])
def cmd_gov(message):
    """Manual trigger: show latest official-source statements about currencies."""
    agg = NewsAggregator()
    items = agg.fetch_official_speech(15)
    evs = agent.process(items)
    off = [e for e in evs if e.official_speech]
    off = sorted(off, key=lambda e: e.impact_score, reverse=True)[:6]
    if not off:
        _reply(message, "🔴 لا توجد تصريحات رسمية حديثة عن العملات/الكريبتو حاليًا.")
        return
    for ev in off:
        _reply(message, format_event(ev))


def _show(message, evs):
    evs = sorted(evs, key=lambda e: e.impact_score, reverse=True)[:5]
    if not evs:
        _reply(message, "لا توجد نتائج حاليًا.")
        return
    for ev in evs:
        _reply(message, format_event(ev, compact=True))


# ── main ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not TOKEN:
        log.error("TELEGRAM_BOT_TOKEN missing in .env — bot cannot start.")
        sys.exit(1)
    log.info(f"News bot starting — token loaded, subscribers={SUBSCRIBERS}, "
             f"sweep every {SWEEP_INTERVAL//60} min, min alert={ALERT_MIN_LEVEL}")
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    # Auto-retry polling: Telegram API is intermittently blocked on this
    # network, so a timeout must not kill the bot — just back off and retry.
    while True:
        try:
            bot.infinity_polling(timeout=20, logger_level=logging.WARNING)
        except Exception as e:
            log.error(f"polling error: {e}; retrying in 30s")
            time.sleep(30)
