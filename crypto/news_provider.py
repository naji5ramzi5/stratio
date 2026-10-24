"""
STRATOCRYPTO — News Intelligence : Provider Layer
=================================================

Abstraction over news sources so the analysis layer never depends on a
single provider. Swap providers via .env without touching the agent.

Primary    : Google News RSS  (free, no key, multi-source, near real-time)
Secondary  : CoinDesk + Cointelegraph RSS (crypto-specific)
Official   : Fed / SEC / Treasury / White House / ECB / CFTC RSS

Optional premium providers (CryptoPanic / NewsAPI) can be added later by
implementing NewsProvider and registering them in NewsAggregator.
"""

import os
import time
import logging
import urllib.parse
import xml.etree.ElementTree as ET
import requests
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta

log = logging.getLogger("news.provider")

UA = {"User-Agent": "Mozilla/5.0 (StratoCrypto NewsAgent)"}

# ── Source reliability (0-100) ──────────────────────────────────────
SOURCE_RELIABILITY = {
    # Official / government
    "federal reserve": 100, "federalreserve.gov": 100, "frb": 100,
    "sec": 100, "sec.gov": 100, "cfc": 100, "cftc": 100, "cftc.gov": 100,
    "u.s. treasury": 100, "treasury.gov": 100, "white house": 100, "whitehouse.gov": 100,
    "ecb": 98, "european central bank": 98, "bank of england": 98, "imf": 97, "bis": 97,
    "congress": 95, "senate": 95, "house of representatives": 95,
    # Major financial
    "reuters": 93, "bloomberg": 93, "associated press": 92, "ap news": 92,
    "wsj": 90, "wall street journal": 90, "financial times": 91, "ft.com": 91,
    "cnbc": 88, "the guardian": 82, "bbc": 88, "nytimes": 86, "the new york times": 86,
    "forbes": 80, "marketwatch": 84, "investing.com": 80, "yahoo finance": 82,
    # Established crypto media
    "coindesk": 84, "cointelegraph": 82, "decrypt": 82, "the block": 83,
    "bitcoin magazine": 80, "cryptoquant": 81, "glassnode": 81,
    # Lower trust
    "twitter": 10, "x.com": 10, "reddit": 25, "telegram": 20, "unknown": 30,
}


def reliability_for(source_name: str) -> int:
    if not source_name:
        return 30
    s = source_name.lower().strip()
    # exact / contains match
    if s in SOURCE_RELIABILITY:
        return SOURCE_RELIABILITY[s]
    for key, val in SOURCE_RELIABILITY.items():
        if key in s or s in key:
            return val
    return 55  # reputable-looking unknown outlet


@dataclass
class NewsItem:
    title: str
    link: str
    source: str
    pub_date: str = ""            # original RFC822 string
    pub_ts: float = 0.0           # epoch UTC
    description: str = ""
    query: str = ""               # which query/category pulled it
    provider: str = ""

    def as_dict(self):
        return asdict(self)


def _parse_date(s: str) -> float:
    if not s:
        return 0.0
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S%z"):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            continue
    return 0.0


def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\n", " ").replace("\r", " ")
    import re
    t = re.sub(r"<[^>]+>", " ", t)        # strip HTML in descriptions
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ── Google News redirect resolution ───────────────────────────────────
GOOGLE_NEWS_REDIRECT = "news.google.com/rss/articles/"

# Cache resolved URLs to avoid repeated HEAD requests
_RESOLVED_CACHE: dict[str, str] = {}


def _extract_real_url_from_desc(desc: str) -> str | None:
    """Try to extract the actual publisher URL from Google News RSS description HTML."""
    if not desc:
        return None
    # Google News RSS description often contains <a href="REAL_URL">Source</a>
    # Find all URLs in description, skip Google domains
    urls = re.findall(r'href=["\'](https?://[^"\']+)["\']', desc)
    for u in urls:
        host = urllib.parse.urlparse(u).netloc.lower()
        if "news.google.com" not in host and "google.com" not in host:
            return u
    return None


def _resolve_google_news_url(redirect_url: str, desc: str = "") -> str:
    """
    Resolve a Google News redirect URL to the actual publisher URL.
    Strategy:
      1. Try extracting from description (fast, no network)
      2. If that fails and not cached, do a quick HEAD request to follow redirect
      3. Cache result
    """
    if not redirect_url or GOOGLE_NEWS_REDIRECT not in redirect_url:
        return redirect_url
    if redirect_url in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[redirect_url]

    # 1) Fast path: extract from description
    real = _extract_real_url_from_desc(desc)
    if real:
        _RESOLVED_CACHE[redirect_url] = real
        return real

    # 2) Slow path: follow redirect (HEAD request, short timeout)
    try:
        r = requests.head(redirect_url, headers=UA, timeout=5, allow_redirects=True)
        final_url = r.url
        if final_url and GOOGLE_NEWS_REDIRECT not in final_url:
            _RESOLVED_CACHE[redirect_url] = final_url
            return final_url
    except Exception:
        pass

    # 3) Fallback: return original (will at least show Google domain)
    _RESOLVED_CACHE[redirect_url] = redirect_url
    return redirect_url


class NewsProvider:
    """Base class. Subclasses implement fetch()."""

    name = "base"

    def fetch(self, query: str, limit: int = 20) -> list:
        raise NotImplementedError

    def _http(self, url: str, timeout: int = 15) -> str:
        # Use requests (certifi) so official HTTPS feeds don't fail on
        # missing CA certs like urllib does.
        r = requests.get(url, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.text


class RssProvider(NewsProvider):
    """Generic RSS reader. Subclasses set `feeds` (list of URLs)."""

    feeds: list = []

    def fetch(self, query: str = "", limit: int = 20) -> list:
        items = []
        for feed_url in self.feeds:
            try:
                raw = self._http(feed_url)
                items.extend(self._parse(raw, feed_url))
            except Exception as e:
                log.warning(f"[{self.name}] feed {feed_url} failed: {e}")
        items.sort(key=lambda x: x.pub_ts, reverse=True)
        return items[:limit]

    def _parse(self, raw: str, feed_url: str) -> list:
        out = []
        try:
            root = ET.fromstring(raw)
        except Exception as e:
            log.warning(f"[{self.name}] parse error: {e}")
            return out
        for item in root.iter("item"):
            title = _clean_text(_text(item, "title"))
            link = _text(item, "link")
            src_el = item.find("source")
            source = _clean_text(_text(src_el)) if src_el is not None and _text(src_el) else self._feed_name(feed_url)
            pub = _text(item, "pubDate") or _text(item, "dc:date") or _text(item, "published")
            desc = _clean_text(_text(item, "description"))
            if not title:
                continue
            out.append(NewsItem(
                title=title, link=link, source=source,
                pub_date=pub, pub_ts=_parse_date(pub),
                description=desc[:600], query="", provider=self.name))
        return out

    def _feed_name(self, feed_url: str) -> str:
        return self.name


class GoogleNewsProvider(NewsProvider):
    """Primary provider — free, keyless, multi-source, query based."""

    name = "google_news"

    BASE = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    # category query builders
    CATEGORY_QUERIES = {
        "fed":        "(Federal Reserve OR Fed OR FOMC OR Powell OR interest rate OR rate cut OR rate hike OR CPI OR PCE)",
        "us_gov":     "(Trump OR White House OR Congress OR Treasury OR executive order)",
        "regulation": "(SEC OR CFTC OR crypto regulation OR stablecoin bill OR ETF regulation OR crypto legislation)",
        "crypto":     "(Bitcoin OR Ethereum OR Solana OR XRP OR stablecoin OR DeFi OR crypto exchange OR ETF)",
        "market":     "(ETF inflows OR liquidation OR whale OR exchange hack OR funding rate OR open interest)",
        "geopolitical": "(war OR sanctions OR tariffs OR trade restriction)",
        "macro":      "(CPI OR PPI OR GDP OR jobs OR NFP OR PMI OR retail sales OR Treasury auction OR dollar index)",
    }

    def fetch(self, query: str, limit: int = 25) -> list:
        q = urllib.parse.quote(query)
        url = self.BASE.format(q=q)
        try:
            raw = self._http(url)
            return self._parse(raw, query)[:limit]
        except Exception as e:
            log.warning(f"[google_news] query '{query}' failed: {e}")
            return []

    def fetch_category(self, category: str, limit: int = 25) -> list:
        q = self.CATEGORY_QUERIES.get(category, category)
        return self.fetch(q, limit)

    def _parse(self, raw: str, query: str) -> list:
        out = []
        try:
            root = ET.fromstring(raw)
        except Exception as e:
            log.warning(f"[google_news] parse error: {e}")
            return out
        for item in root.iter("item"):
            title = _clean_text(_text(item, "title"))
            link = _text(item, "link")
            src_el = item.find("source")
            source = _clean_text(_text(src_el)) if (src_el is not None and _text(src_el)) else "unknown"
            pub = _text(item, "pubDate")
            desc = _clean_text(_text(item, "description"))
            if not title:
                continue
            # Resolve Google News redirect to actual publisher URL
            resolved_link = _resolve_google_news_url(link, desc)
            out.append(NewsItem(
                title=title, link=resolved_link, source=source,
                pub_date=pub, pub_ts=_parse_date(pub),
                description=desc[:600], query=query, provider=self.name))
        return out


class CryptoRssProvider(RssProvider):
    """Secondary provider — crypto-native outlets."""

    name = "crypto_rss"
    feeds = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
        "https://decrypt.co/feed",
        "https://www.theblock.co/rss.xml",
    ]

    def _feed_name(self, feed_url: str) -> str:
        if "coindesk" in feed_url: return "CoinDesk"
        if "cointelegraph" in feed_url: return "Cointelegraph"
        if "decrypt" in feed_url: return "Decrypt"
        if "theblock" in feed_url: return "The Block"
        return "CryptoRSS"


class OfficialRssProvider(RssProvider):
    """Official / government sources — highest reliability."""

    name = "official_rss"
    feeds = [
        "https://www.federalreserve.gov/feeds.htm",   # placeholder, replaced below
    ]

    # Curated reliable official feeds
    OFFICIAL = [
        ("Federal Reserve", "https://www.federalreserve.gov/rss.htm"),
        ("SEC", "https://www.sec.gov/rss/litigation/litreleases.xml"),
        ("U.S. Treasury", "https://home.treasury.gov/feed/rss.xml"),
        ("White House", "https://www.whitehouse.gov/feed/"),
        ("CFTC", "https://www.cftc.gov/rss/index.htm"),
        ("ECB", "https://www.ecb.europa.eu/rss/rss.xml"),
    ]

    def fetch(self, query: str = "", limit: int = 15) -> list:
        items = []
        for name, url in self.OFFICIAL:
            try:
                raw = self._http(url)
                for it in self._parse(raw, url):
                    it.source = name
                    items.append(it)
            except Exception as e:
                log.warning(f"[official_rss] {name} failed: {e}")
        items.sort(key=lambda x: x.pub_ts, reverse=True)
        return items[:limit]


class NewsAggregator:
    """Tries providers in order with fallback. Merges + dedupes raw items."""

    def __init__(self):
        self.primary = GoogleNewsProvider()
        self.secondary = CryptoRssProvider()
        self.official = OfficialRssProvider()

    def fetch_query(self, query: str, limit: int = 25) -> list:
        items = self.primary.fetch(query, limit)
        if not items:
            log.warning("NEWS_FEED_DEGRADED: primary empty, trying secondary")
            items = self.secondary.fetch(query, limit)
        return items

    def fetch_category(self, category: str, limit: int = 25) -> list:
        items = self.primary.fetch_category(category, limit)
        if not items:
            log.warning("NEWS_FEED_DEGRADED: primary category empty")
        return items

    def fetch_official(self, limit: int = 15) -> list:
        return self.official.fetch(limit=limit)

    def fetch_official_speech(self, limit: int = 12) -> list:
        """Targeted sweep: when official entities speak ABOUT currencies/crypto.

        Official RSS feeds are often blocked/unreachable, so this uses a precise
        Google News query (official entity + currency term) which reliably catches
        statements such as 'White House says crypto...' or 'Treasury on stablecoins'.
        """
        q = ('("White House" OR Trump OR Treasury OR "Federal Reserve" OR Powell '
             'OR SEC OR CFTC OR Congress) (Bitcoin OR crypto OR cryptocurrency OR '
             'stablecoin OR dollar OR tariff OR currency OR ETF OR "executive order")')
        items = self.primary.fetch(q, limit)
        for it in items:
            it.query = "official_speech"
        return items

    def fetch_all_breaking_sweep(self, categories: dict, per: int = 20) -> list:
        """Sweep all categories from primary + official feeds."""
        items = []
        for cat in categories:
            items.extend(self.fetch_category(cat, per))
        items.extend(self.fetch_official(10))
        # dedupe by link
        seen, out = set(), []
        for it in items:
            key = it.link or it.title
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out


def _text(el, tag: str = None) -> str:
    if el is None:
        return ""
    if tag is not None:
        found = el.find(tag)
        if found is None:
            # namespace fallback
            for child in el.iter():
                if child.tag.endswith(tag) or child.tag == tag:
                    return child.text or ""
            return ""
        return found.text or ""
    return el.text or ""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    agg = NewsAggregator()
    res = agg.fetch_query("Bitcoin ETF", 5)
    for r in res:
        print(r.source, "|", r.title[:70], "|", datetime.fromtimestamp(r.pub_ts, tz=timezone.utc) if r.pub_ts else "")
