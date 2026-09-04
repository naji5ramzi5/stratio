import unittest

from news_agent import NewsAgent
from news_bot import format_event
from news_provider import NewsItem


class TestNewsEvidence(unittest.TestCase):
    def setUp(self):
        self.agent = NewsAgent()
        # Evidence tests must not request a live price snapshot.
        self.agent._batch_snapshot = lambda *_args, **_kwargs: None

    @staticmethod
    def _item(source, link):
        return NewsItem(
            title="Bitcoin approaches $80,000 as ETF demand weakens",
            link=link,
            source=source,
            pub_ts=1_700_000_000,
            description="Bitcoin ETF demand weakens over the weekend.",
        )

    def test_google_rss_mentions_do_not_create_independent_confirmation(self):
        items = [
            self._item("CryptoSlate", "https://news.google.com/rss/articles/one"),
            self._item("CoinDesk", "https://news.google.com/rss/articles/two"),
            self._item("The Block", "https://news.google.com/rss/articles/three"),
        ]
        event = self.agent.process(items)[0]

        self.assertEqual(event.mention_count, 3)
        self.assertEqual(event.independent_sources_count, 0)
        self.assertEqual(event.verification, "UNVERIFIED")
        self.assertLess(event.confidence, 60)

    def test_two_direct_publishers_are_verified(self):
        items = [
            self._item("CoinDesk", "https://www.coindesk.com/markets/event"),
            self._item("Cointelegraph", "https://cointelegraph.com/news/event"),
        ]
        event = self.agent.process(items)[0]

        self.assertEqual(event.independent_sources_count, 2)
        self.assertEqual(event.verification, "VERIFIED")

    def test_telegram_text_does_not_expose_rss_urls(self):
        event = self.agent.process([
            self._item("CryptoSlate", "https://news.google.com/rss/articles/very-long-tracking-url")
        ])[0]
        text = format_event(event)

        self.assertNotIn("https://", text)
        self.assertIn("مصدر الرصد: Google News", text)


if __name__ == "__main__":
    unittest.main()
