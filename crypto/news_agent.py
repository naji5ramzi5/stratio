"""
STRATOCRYPTO — News Intelligence : Agent Layer
==============================================

Turns raw NewsItems into structured Market Events:

  NewsItem -> classify -> detect assets -> sentiment -> impact score
            -> confidence -> verification -> event clustering (dedup)
            -> alert level -> price coupling -> dataset storage

Design rules (per spec):
  * Never invent news. UNVERIFIED / VERIFIED / CONFLICTING labels.
  * Confidence != probability. No "X% chance price will rise".
  * Dedup: many outlets => one Event, sources collected underneath.
  * No secrets logged.
"""

import os
import re
import json
import time
import logging
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta

import requests

from news_provider import NewsItem, reliability_for, NewsAggregator

log = logging.getLogger("news.agent")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(BASE_DIR, "news_events.jsonl")
PRICE_LOG = os.path.join(BASE_DIR, "news_price_reactions.jsonl")

BINANCE = "https://api.binance.com"

# ── Categories ─────────────────────────────────────────────────────
CATEGORY_KEYWORDS = {
    "FED / Monetary": ["federal reserve", "fed ", "fomc", "powell", "interest rate",
                        "rate cut", "rate hike", "inflation", "cpi", "pce", "treasury yield",
                        "dollar liquidity", "liquidity", "employment", "unemployment", "dxy"],
    "U.S. Government": ["trump", "white house", "congress", "treasury", "executive order",
                        "crypto legislation", "senate", "president"],
    "Regulation": ["sec", "cftc", "doj", "banking regulator", "stablecoin regulation",
                   "crypto regulation", "etf regulation", "lawsuit", "probe", "ban crypto",
                   "securities"],
    "Crypto": ["bitcoin", "ethereum", "solana", "xrp", "stablecoin", "defi", "exchange",
               "etf", "institutional adoption", "mining", "altcoin", "crypto"],
    "Market": ["etf inflow", "etf outflow", "liquidation", "whale", "exchange hack",
               "large transfer", "funding rate", "open interest", "volatility", "sell-off"],
    "Geopolitical": ["war", "sanction", "tariff", "trade restriction", "geopolitic", "invasion"],
    "Macro": ["cpi", "ppi", "gdp", "jobs", "nfp", "pmi", "retail sales", "treasury auction",
              "dollar index", "recession"],
}

# Map phrases -> canonical topic so outlets covering the SAME story cluster.
TOPIC_MAP = {
    "rate hike": "RATES", "rate increase": "RATES", "higher rates": "RATES",
    "rate cut": "RATES", "interest rate": "RATES", "rates": "RATES",
    "fomc": "FED_POLICY", "powell": "FED_POLICY", "federal reserve": "FED_POLICY",
    "fed minutes": "FED_POLICY", "cpi": "INFLATION", "inflation": "INFLATION",
    "pce": "INFLATION", "etf": "ETF", "stablecoin": "STABLECOIN",
    "sec": "SEC", "cftc": "CFTC", "hack": "HACK", "exchange hack": "HACK",
    "ban": "BAN", "tariff": "TARIFF", "sanction": "SANCTION", "war": "WAR",
    "treasury": "TREASURY", "white house": "WH_GOV", "trump": "WH_GOV",
    "bitcoin reserve": "STRAT_RESERVE", "defi": "DEFI", "mining": "MINING",
}

HIGH_IMPACT_KEYWORDS = {
    "rate cut": 10, "rate hike": 10, "interest rate": 8, "etf": 8, "etf inflow": 9,
    "etf outflow": 9, "hack": 12, "exchange hack": 13, "ban": 9, "ban crypto": 11,
    "sanction": 10, "tariff": 8, "reserve": 10, "bitcoin reserve": 12, "strategic reserve": 11,
    "lawsuit": 9, "sec ": 8, "cftc": 8, "approval": 8, "approved": 9, "reject": 8,
    "fraud": 9, "arrest": 7, "whale": 5, "liquidation": 6, "war": 10, "recession": 10,
    "fomc": 9, "powell": 7, "default": 9, "bailout": 8, "embargo": 8,
}

# ── Official sources & currency-speech detection ────────────────
# Entities whose statements about currencies/crypto are treated as PRIMARY
# authoritative sources => VERIFIED + highest-priority BREAKING alert.
OFFICIAL_ENTITIES = [
    "white house", "trump", "treasury", "u.s. treasury", "federal reserve",
    "fed ", "fomc", "powell", "sec", "cftc", "executive order", "congress",
    "senate", "president", "biden", "ecb", "bank of england", "imf", "bis",
]
# Terms indicating a statement is about currencies / crypto markets.
CURRENCY_TERMS = [
    "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
    "stablecoin", "dollar", "currency", "tariff", "etf", "xrp", "solana",
    "sol", "defi", "digital asset", "blockchain", "cbdc", "altcoin", "token",
    "fed rate", "interest rate", "rate cut", "rate hike",
]

# ── Asset detection ────────────────────────────────────────────────
ASSET_MAP = {
    "bitcoin": "BTC", "btc": "BTC", "btcusdt": "BTC",
    "ethereum": "ETH", "eth": "ETH", "ether": "ETH",
    "solana": "SOL", "sol": "SOL",
    "xrp": "XRP", "ripple": "XRP",
    "cardano": "ADA", "ada": "ADA",
    "dogecoin": "DOGE", "doge": "DOGE",
    "litecoin": "LTC", "ltc": "LTC",
    "binance": "BNB", "bnb": "BNB",
    "stablecoin": "USDT", "tether": "USDT", "usdc": "USDC", "usdt": "USDT",
    "defi": "DEFI", "defi ": "DEFI",
}
BROAD_TOKENS = {"BTC", "ETH", "SOL", "XRP", "USDT"}
EXCHANGE_WORDS = {"binance", "coinbase", "kraken", "bybit", "okx", "bitfinex", "upbit"}

# ── Sentiment lexicon (English) ───────────────────────────────────
POS = {"surge", "rally", "gain", "gains", "rise", "rises", "rising", "bullish", "soar",
       "soars", "jump", "jumps", "approve", "approved", "approval", "adoption", "inflow",
       "inflows", "record", "high", "support", "supports", "upgrade", "positive", "boost",
       "rebound", "recover", "recovery", "optimism", "win", "wins", "green", "outperform",
       "breakout", "accumulate", "accumulation", "expands", "partnership", "launch"}
NEG = {"plunge", "plunges", "crash", "crashes", "drop", "drops", "fall", "falls", "falling",
       "bearish", "slump", "decline", "declines", "sell", "selloff", "sell-off", "downgrade",
       "negative", "loss", "losses", "hack", "hacked", "ban", "banned", "probe", "lawsuit",
       "fraud", "arrest", "reject", "rejected", "rejection", "sanction", "tariff", "war",
       "recession", "fear", "panic", "outflow", "outflows", "liquidate", "liquidation",
       "default", "collapse", "warning", "risk", "volatile", "dump", "dumps", "red"}


# ── Helpers ────────────────────────────────────────────────────────
def _norm_title(t: str) -> str:
    t = t.lower()
    # strip trailing " - Source Name"
    t = re.sub(r"\s*-\s*[^-]+$", "", t)
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return t


def _tokens(t: str) -> set:
    return set(w for w in _norm_title(t).split() if len(w) > 2)


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _now_ts() -> float:
    return datetime.now(timezone.utc).timestamp()


# ── Event ─────────────────────────────────────────────────────────
@dataclass
class MarketEvent:
    event_id: str
    headline: str
    category: str
    topic: str = ""
    assets: list = field(default_factory=list)
    sentiment: str = "NEUTRAL"
    sentiment_score: float = 0.0
    impact_score: int = 0
    confidence: int = 0
    verification: str = "UNVERIFIED"
    alert_level: str = "INFO"
    breaking: bool = False
    official_speech: bool = False   # True when an official source speaks about currencies
    sources: list = field(default_factory=list)   # [{name,link,reliability}]
    mention_count: int = 0
    independent_sources_count: int = 0
    primary_source: str = ""
    verification_reason: str = ""
    confidence_components: dict = field(default_factory=dict)
    first_ts: float = 0.0
    last_ts: float = 0.0
    analysis: str = ""
    price_before: dict = field(default_factory=dict)
    created_at: float = 0.0

    def as_dict(self):
        return asdict(self)


class NewsAgent:
    def __init__(self, aggregator: NewsAggregator = None):
        self.agg = aggregator or NewsAggregator()
        self._events: dict[str, MarketEvent] = {}     # event_id -> event
        self._seen_links: set = set()

    @staticmethod
    def _source_record(item: NewsItem) -> dict:
        """Keep feed aggregation separate from independently verifiable reporting."""
        link = item.link or ""
        domain = urllib.parse.urlparse(link).netloc.lower().removeprefix("www.")
        is_aggregated = domain in {"news.google.com", "google.com"}
        name = item.source or domain or "Unknown"
        return {
            "name": name,
            "link": link,
            "reliability": reliability_for(name),
            "domain": domain,
            "is_aggregated": is_aggregated,
        }

    @staticmethod
    def _independent_sources(sources: list) -> list:
        """Only direct publishers and official feeds can confirm an event.

        Google News is valuable for discovery, but its RSS redirect does not prove
        that every listed publisher independently confirmed the same event.
        """
        unique = {}
        for source in sources:
            if source.get("is_aggregated") and source.get("reliability", 0) < 98:
                continue
            key = (source.get("domain") or source.get("name", "")).lower()
            if key and key not in unique:
                unique[key] = source
        return list(unique.values())

    def _refresh_evidence(self, ev: MarketEvent, sentiments: list[str]) -> None:
        independent = self._independent_sources(ev.sources)
        ev.mention_count = len(ev.sources)
        ev.independent_sources_count = len(independent)
        ev.primary_source = (independent[0].get("name") if independent else
                             (ev.sources[0].get("name") if ev.sources else "Unknown"))
        ev.verification = self.verify(ev.sources, sentiments)
        ev.verification_reason = self.verification_reason(ev.sources, ev.verification)
        ev.confidence, ev.confidence_components = self.confidence(ev.sources, ev.verification)

    # ── classification ──────────────────────────────────────────
    def classify(self, text: str) -> str:
        t = text.lower()
        best, best_score = "Other", 0
        for cat, kws in CATEGORY_KEYWORDS.items():
            s = sum(2 if kw in t else 0 for kw in kws)
            if s > best_score:
                best, best_score = cat, s
        return best

    def detect_assets(self, text: str) -> list:
        t = " " + text.lower() + " "
        found = set()
        for kw, tok in ASSET_MAP.items():
            if f" {kw}" in t or f"{kw} " in t or kw in t:
                found.add(tok)
        if any(w in t for w in EXCHANGE_WORDS):
            found.add("CRYPTO_MARKET")
        # broad macro / fed events hit the whole market
        if any(k in t for k in ("federal reserve", "fed ", "fomc", "interest rate",
                                 "rate cut", "rate hike", "inflation", "cpi", "dxy",
                                 "treasury yield", "tariff", "sanction", "war", "recession")):
            found.update(BROAD_TOKENS)
            found.add("CRYPTO_MARKET")
        if not found:
            # crypto-specific fallback if "crypto"/"coin"/"token" mentioned
            if any(k in t for k in ("crypto", "coin", "token", "blockchain")):
                found.update(BROAD_TOKENS)
        return sorted(found)

    def sentiment(self, text: str) -> tuple:
        toks = _tokens(text)
        pos = len(toks & POS)
        neg = len(toks & NEG)
        total = pos + neg
        if total == 0:
            return "NEUTRAL", 0.0
        score = (pos - neg) / total
        if score > 0.05:
            label = "BULLISH"
        elif score < -0.05:
            label = "BEARISH"
        else:
            label = "NEUTRAL"
        return label, round(score, 3)

    # ── impact & confidence ─────────────────────────────────────
    def impact_score(self, category, assets, text, sources, breaking, ts) -> int:
        t = text.lower()
        cat_base = {
            "FED / Monetary": 70, "Regulation": 68, "U.S. Government": 60,
            "Macro": 62, "Geopolitical": 65, "Crypto": 50, "Market": 55, "Other": 40,
        }.get(category, 42)
        score = cat_base
        # high-impact keyword bonus
        for kw, w in HIGH_IMPACT_KEYWORDS.items():
            if kw in t:
                score += w
        # asset breadth
        if "CRYPTO_MARKET" in assets:
            score += 8
        if any(a in BROAD_TOKENS for a in assets):
            score += 6
        if len(assets) >= 3:
            score += 4
        # source reliability (avg of top sources)
        rel = 0
        if sources:
            rel = sum(s.get("reliability", 55) for s in sources) / len(sources)
        score += (rel / 100.0) * 10
        # Only independently verifiable publishers count as confirmation.
        score += min(len(self._independent_sources(sources)), 5) / 5.0 * 8
        # recency
        age_min = (_now_ts() - ts) / 60.0 if ts else 999
        if age_min <= 15:
            score += 12
        elif age_min <= 60:
            score += 8
        elif age_min <= 360:
            score += 3
        score = max(0, min(100, int(score)))
        return score

    def confidence(self, sources, verification) -> tuple[int, dict]:
        independent = self._independent_sources(sources)
        n = len(independent)
        avg_rel = (sum(s.get("reliability", 55) for s in independent) / n) if n else 30
        if verification == "VERIFIED":
            base = 72 + min(n, 5) * 3
        elif verification == "CONFLICTING":
            base = 50
        elif verification == "SINGLE_SOURCE":
            base = 55
        else:  # UNVERIFIED
            base = 40
        reliability_bonus = (avg_rel / 100.0) * 10
        score = max(0, min(100, int(base + reliability_bonus)))
        return score, {
            "independent_sources": n,
            "aggregated_mentions": max(0, len(sources) - n),
            "average_reliability": round(avg_rel, 1),
            "reliability_bonus": round(reliability_bonus, 1),
        }

    def is_official_source(self, sources) -> bool:
        return any(s.get("reliability", 0) >= 98
                   for s in self._independent_sources(sources or []))

    def mentions_currency(self, text: str) -> bool:
        t = text.lower()
        return any(k in t for k in CURRENCY_TERMS)

    def is_official_speech_about_currency(self, sources, text) -> bool:
        """True when an official entity speaks ABOUT currencies/crypto.

        Either the reporting source IS an official primary source, OR the
        headline references an official entity speaking (e.g. 'White House
        says crypto...'). Always escalates to a priority BREAKING alert.
        """
        if not self.mentions_currency(text):
            return False
        if self.is_official_source(sources):
            return True
        t = text.lower()
        return any(ent in t for ent in OFFICIAL_ENTITIES)

    def detect_breaking(self, category, text, ts, impact, sources=None) -> bool:
        age_min = (_now_ts() - ts) / 60.0 if ts else 999
        if age_min > 180:
            return False
        t = text.lower()
        hot = any(k in t for k in ("breaking", "just in", "urgent", "alert",
                                    "rate cut", "rate hike", "fomc", "powell",
                                    "sec ", "cftc", "bitcoin etf", "hack",
                                    "exchange hack", "ban", "sanction", "war",
                                    "tariff", "reserve", "default", "collapse"))
        # Official source speaking about currencies => always breaking.
        official = any(s.get("reliability", 0) >= 95
                       for s in self._independent_sources(sources or []))
        if official and self.mentions_currency(text):
            return True
        if category in ("FED / Monetary", "Regulation", "U.S. Government", "Geopolitical"):
            return bool(hot) or impact >= 75
        return bool(hot) and impact >= 70

    def alert_level(self, breaking, impact) -> str:
        if breaking and impact >= 85:
            return "CRITICAL"
        if breaking or impact >= 75:
            return "BREAKING"
        if impact >= 60:
            return "IMPORTANT"
        if impact >= 40:
            return "WATCH"
        return "INFO"

    # ── verification ────────────────────────────────────────────
    def verify(self, sources, sentiments):
        independent = self._independent_sources(sources)
        n = len(independent)
        has_primary = any(s.get("reliability", 0) >= 98 for s in independent)
        conflict = ("BULLISH" in sentiments and "BEARISH" in sentiments
                    and len(set(sentiments)) == 2)
        # One authoritative primary source (official, reliability >= 98) suffices.
        if has_primary:
            return "CONFLICTING" if conflict else "VERIFIED"
        if n >= 2 and any(s.get("reliability", 0) >= 70 for s in sources):
            return "CONFLICTING" if conflict else "VERIFIED"
        # Single credible (non-primary) source => SINGLE SOURCE, not UNVERIFIED.
        if n == 1 and any(s.get("reliability", 0) >= 70 for s in sources):
            return "SINGLE_SOURCE"
        return "UNVERIFIED"

    def verification_reason(self, sources, verification: str) -> str:
        independent = self._independent_sources(sources)
        if verification == "VERIFIED" and self.is_official_source(independent):
            return "مصدر رسمي مباشر."
        if verification == "VERIFIED":
            return f"تم تأكيد الحدث من {len(independent)} مصادر مستقلة مباشرة."
        if verification == "SINGLE_SOURCE":
            name = independent[0].get("name", "موثوق") if independent else "غير قابل للتحقق مباشرة"
            return f"مصدر موثوق واحد فقط: {name}."
        if verification == "CONFLICTING":
            return "توجد مصادر مستقلة ذات روايات أو اتجاهات متعارضة."
        return "لا يوجد تأكيد مستقل كافٍ؛ التغطيات المجمعة لا تعد تأكيداً مستقلاً."

    # ── event clustering / dedup ─────────────────────────────────
    def _topic(self, text: str) -> str:
        t = text.lower()
        # first match wins (ordered by specificity above)
        for kw, topic in TOPIC_MAP.items():
            if kw in t:
                return topic
        return ""

    def _match_event(self, item: NewsItem, category: str, topic: str) -> MarketEvent | None:
        toks = _tokens(item.title)
        for ev in self._events.values():
            # strong match: same story text
            if _jaccard(toks, _tokens(ev.headline)) >= 0.5:
                return ev
            # same category + same canonical topic => same Market Event
            if ev.category == category and topic and ev.topic == topic:
                return ev
        return None

    def process(self, items: list) -> list:
        """Process raw items -> return list of *new or updated* events to alert."""
        to_alert = {}
        for it in items:
            if it.link in self._seen_links:
                continue
            self._seen_links.add(it.link)
            _txt0 = f"{it.title}. {it.description}"
            _cat0 = self.classify(_txt0)
            _topic0 = self._topic(_txt0)
            ev = self._match_event(it, _cat0, _topic0)
            src = self._source_record(it)
            if ev is None:
                # build new event
                text = f"{it.title}. {it.description}"
                cat = self.classify(text)
                topic = self._topic(text)
                assets = self.detect_assets(text)
                sent_label, sent_score = self.sentiment(text)
                breaking = self.detect_breaking(cat, text, it.pub_ts, 50, [src])
                # tentative impact (no sources yet)
                impact = self.impact_score(cat, assets, text, [src], breaking, it.pub_ts)
                eid = "EVT_" + datetime.fromtimestamp(it.pub_ts or _now_ts(), tz=timezone.utc).strftime("%Y%m%d_%H%M")
                ev = MarketEvent(
                    event_id=eid, headline=it.title, category=cat, topic=topic, assets=assets,
                    sentiment=sent_label, sentiment_score=sent_score,
                    impact_score=impact,
                    alert_level=self.alert_level(breaking, impact), breaking=breaking,
                    sources=[src], first_ts=it.pub_ts or _now_ts(),
                    last_ts=it.pub_ts or _now_ts(),
                    created_at=_now_ts(),
                )
                self._refresh_evidence(ev, [sent_label])
                ev.analysis = self.build_analysis(ev)
                # Official speech about currencies => priority escalation.
                if self.is_official_speech_about_currency(ev.sources, ev.headline):
                    ev.official_speech = True
                    ev.breaking = True
                    self._refresh_evidence(ev, [ev.sentiment])
                    ev.impact_score = max(ev.impact_score, 92)
                    ev.alert_level = "CRITICAL"
                self._events[eid] = ev
                to_alert[ev.event_id] = ev
            else:
                # merge into existing event
                if (src["name"], src["link"]) not in {(s["name"], s["link"]) for s in ev.sources}:
                    ev.sources.append(src)
                ev.last_ts = it.pub_ts or _now_ts()
                # re-evaluate verification / confidence / impact with more sources
                sentiments = [ev.sentiment]
                self._refresh_evidence(ev, sentiments)
                ev.impact_score = self.impact_score(ev.category, ev.assets,
                                                    ev.headline, ev.sources,
                                                    ev.breaking, ev.first_ts)
                ev.alert_level = self.alert_level(ev.breaking, ev.impact_score)
                # breaking can escalate as sources/priority grow
                if not ev.breaking and self.detect_breaking(ev.category, ev.headline,
                                                            ev.first_ts, ev.impact_score,
                                                            ev.sources):
                    ev.breaking = True
                    ev.alert_level = self.alert_level(True, ev.impact_score)
                # Official speech about currencies => priority escalation on merge too.
                if self.is_official_speech_about_currency(ev.sources, ev.headline):
                    ev.official_speech = True
                    ev.breaking = True
                    self._refresh_evidence(ev, [ev.sentiment])
                    ev.impact_score = max(ev.impact_score, 92)
                    ev.alert_level = "CRITICAL"
                # re-alert only if it became VERIFIED or escalated to BREAKING/CRITICAL
                if ev.verification == "VERIFIED" or ev.alert_level in ("BREAKING", "CRITICAL"):
                    to_alert[ev.event_id] = ev
        # store to dataset (new events)
        for ev in to_alert.values():
            self._store(ev)
        # single bounded parallel price snapshot for impactful events
        self._batch_snapshot(list(self._events.values()))
        return list(to_alert.values())

    def _batch_snapshot(self, events, budget=6, max_events=8):
        need = [e for e in events if (e.impact_score >= 55 or e.breaking)]
        need = sorted(need, key=lambda e: e.impact_score, reverse=True)[:max_events]
        assets = []
        for e in need:
            for a in e.assets:
                if a not in assets:
                    assets.append(a)
        assets = assets[:6]
        if not assets:
            return
        prices = self._snap_parallel(assets, budget)
        for e in need:
            e.price_before = {a: prices[a] for a in e.assets if a in prices}

    # ── price coupling ──────────────────────────────────────────
    def _snap_prices(self, assets) -> dict:
        return self._snap_parallel(list(assets), 4)

    def _snap_parallel(self, assets, budget=4) -> dict:
        out = {}
        syms = [(a, self._asset_to_symbol(a)) for a in assets if self._asset_to_symbol(a)]
        if not syms:
            return out
        import concurrent.futures
        deadline = time.time() + budget

        def fetch(pair):
            a, sym = pair
            try:
                timeout = max(1.0, deadline - time.time())
                r = requests.get(f"{BINANCE}/api/v3/ticker/price",
                                 params={"symbol": sym}, timeout=timeout)
                if r.status_code == 200:
                    return a, float(r.json()["price"])
            except Exception:
                pass
            return a, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(syms)) as ex:
            futs = [ex.submit(fetch, p) for p in syms]
            for f in concurrent.futures.as_completed(futs):
                try:
                    a, price = f.result()
                    if price is not None:
                        out[a] = price
                except Exception:
                    pass
                if time.time() > deadline:
                    break
        return out

    def _asset_to_symbol(self, a: str):
        m = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
             "ADA": "ADAUSDT", "DOGE": "DOGEUSDT", "LTC": "LTCUSDT", "BNB": "BNBUSDT",
             "USDT": None, "USDC": None, "DEFI": None, "CRYPTO_MARKET": None}
        return m.get(a)

    def record_reaction(self, ev: MarketEvent) -> dict:
        """Compare current price to price_before; store reaction row."""
        if not ev.price_before:
            return {}
        now = self._snap_prices(list(ev.price_before.keys()))
        reaction = {}
        for a, before in ev.price_before.items():
            after = now.get(a)
            if before and after:
                pct = (after - before) / before * 100.0
                reaction[a] = {"before": before, "after": after, "pct": round(pct, 3)}
        if reaction:
            row = {"event_id": ev.event_id, "ts": _now_ts(), "reaction": reaction}
            try:
                with open(PRICE_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            except Exception:
                pass
        return reaction

    def build_analysis(self, ev: MarketEvent) -> str:
        """Short Arabic rationale. Information only — not a price guarantee."""
        cat_ar = {
            "FED / Monetary": "السياسة النقدية للاحتياطي الفيدرالي",
            "U.S. Government": "الحكومة الأمريكية",
            "Regulation": "التنظيم والرقابة",
            "Crypto": "سوق العملات الرقمية",
            "Market": "هيكل السوق",
            "Geopolitical": "الأحداث الجيوسياسية",
            "Macro": "الاقتصاد الكلي",
            "Other": "الأسواق العامة",
        }.get(ev.category, "الأسواق")
        direction = {"BULLISH": "اتجاه إيجابي محتمل", "BEARISH": "اتجاه سلبي محتمل",
                     "NEUTRAL": "أثر محايد/غير واضح"}.get(ev.sentiment, "")
        assets = ", ".join(ev.assets[:5]) if ev.assets else "السوق بشكل عام"
        n_src = ev.independent_sources_count
        base = (f"يبدو أن الخبر ضمن تصنيف «{cat_ar}» وقد يؤثر على: {assets}. "
                f"الميل العام: {direction}. ")
        if ev.verification == "VERIFIED":
            base += f"تم التحقق من الخبر عبر {n_src} مصادر مستقلة. "
        elif ev.verification == "CONFLICTING":
            base += "توجد تقارير متعارضة بين المصادر — يُنصح بالحذر. "
        elif ev.verification == "SINGLE_SOURCE":
            base += f"الخبر حتى الآن من مصدر موثوق واحد ولم يُؤكد من مصدر ثانٍ. "
        else:
            base += "لا يوجد حتى الآن تأكيد مستقل كافٍ للخبر. "
        base += "هذا تحليل للمعلومة وليس ضمانًا لاتجاه السعر."
        return base

    def _store(self, ev: MarketEvent):
        try:
            with open(DATASET, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev.as_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            log.warning(f"dataset store failed: {e}")

    # ── queries for commands ─────────────────────────────────────
    def recent(self, n=10, min_level=None, category=None, asset=None) -> list:
        evs = sorted(self._events.values(), key=lambda e: e.last_ts, reverse=True)
        if min_level:
            order = ["INFO", "WATCH", "IMPORTANT", "BREAKING", "CRITICAL"]
            try:
                mi = order.index(min_level)
                evs = [e for e in evs if order.index(e.alert_level) >= mi]
            except Exception:
                pass
        if category:
            evs = [e for e in evs if e.category.lower() == category.lower()]
        if asset:
            evs = [e for e in evs if asset.upper() in e.assets]
        return evs[:n]

    def top_impact(self, n=10) -> list:
        return sorted(self._events.values(), key=lambda e: e.impact_score, reverse=True)[:n]


# ── standalone test ───────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    a = NewsAgent()
    items = a.agg.fetch_category("fed", 20)
    evs = a.process(items)
    for e in evs[:5]:
        print(f"[{e.alert_level}] {e.category} | {e.sentiment} | imp={e.impact_score} conf={e.confidence} {e.verification}")
        print("   ", e.headline[:80])
        print("    assets:", e.assets, "srcs:", [s['name'] for s in e.sources])
