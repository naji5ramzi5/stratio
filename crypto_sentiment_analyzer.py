import requests
from textblob import TextBlob

class CryptoSentimentAnalyzer:
    NEWS_API_URL = "https://newsapi.org/v2/everything"
    API_KEY = "00b61f31bfce4f4282e44161fcadea48"  # استبدل بمفتاح NewsAPI الخاص بك

    def __init__(self):
        pass

    def fetch_news(self, crypto_pair):
        """
        يجلب الأخبار المتعلقة بالعملة.
        """
        crypto_name = crypto_pair.replace("USDT", "").upper()
        params = {
            "q": crypto_name,
            "apiKey": self.API_KEY,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10,
        }
        try:
            response = requests.get(self.NEWS_API_URL, params=params)
            response.raise_for_status()
            return response.json().get("articles", [])
        except requests.exceptions.RequestException as e:
            print(f"حدث خطأ أثناء جلب الأخبار: {e}")
            return []

    @staticmethod
    def analyze_sentiment(text):
        """
        يحلل المشاعر للنصوص ويعيد تصنيفها (إيجابي، محايد، سلبي).
        """
        if not text:  # إذا كان النص فارغًا
            return "محايد"
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return "إيجابي"
        elif analysis.sentiment.polarity == 0:
            return "محايد"
        else:
            return "سلبي"

    def get_sentiment_summary(self, crypto_pair):
        """
        جلب الأخبار وتحليل المشاعر وإرجاع النتائج كنص عربي.
        """
        articles = self.fetch_news(crypto_pair)
        if not articles:
            return "لم يتم العثور على أخبار لهذه العملة."

        positive_count = 0
        neutral_count = 0
        negative_count = 0
        total_articles = len(articles)

        for article in articles:
            description = article.get("description", "")
            sentiment = self.analyze_sentiment(description)
            if sentiment == "إيجابي":
                positive_count += 1
            elif sentiment == "محايد":
                neutral_count += 1
            elif sentiment == "سلبي":
                negative_count += 1

        # حساب النسب
        positive_percentage = (positive_count / total_articles) * 100
        neutral_percentage = (neutral_count / total_articles) * 100
        negative_percentage = (negative_count / total_articles) * 100

        # صياغة النتائج النصية
        result = (
            f"تحليل المشاعر:\n"
            f"النسبة الإيجابية: {positive_percentage:.2f}%\n"
            f"النسبة المحايدة: {neutral_percentage:.2f}%\n"
            f"النسبة السلبية: {negative_percentage:.2f}%"
        )
        return result
