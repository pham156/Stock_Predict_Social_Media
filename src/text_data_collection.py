import yfinance as yf
import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta


def _to_date(d):
    """Normalize input to a date object.

    Accepts a string in ISO format 'YYYY-MM-DD', a datetime, or a date
    and returns a datetime.date.
    """
    from datetime import date as _date

    if isinstance(d, str):
        # Try ISO format first (YYYY-MM-DD)
        try:
            return datetime.fromisoformat(d).date()
        except Exception:
            # Fallback to common format
            return datetime.strptime(d, "%Y-%m-%d").date()
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, _date):
        return d
    raise TypeError(f"Unsupported date type: {type(d)}")

def get_stock_news(ticker, start_date, end_date):
    """Get news headlines for a stock using yfinance"""
    stock = yf.Ticker(ticker)
    
    # Get news (limited historical data in yfinance)
    news = []
    try:
        # Normalize the start/end to date objects so comparisons are safe
        start_date = _to_date(start_date)
        end_date = _to_date(end_date)

        for item in stock.news:
            pub_date = datetime.fromtimestamp(item['providerPublishTime'])
            pub_date_only = pub_date.date()
            if start_date <= pub_date_only <= end_date:
                news.append({
                    'date': pub_date_only,
                    'title': item.get('title', ''),
                    'content': item.get('content', '')
                })
    except:
        pass
    
    return pd.DataFrame(news)

def get_twitter_posts(query, start_date, end_date, max_tweets=1000):
    """Get tweets mentioning a stock ticker"""
    tweets = []
    # Build search strings. Accepts either strings or date/datetime objects.
    if isinstance(start_date, (str,)):
        start_str = start_date
    else:
        start_str = _to_date(start_date).isoformat()

    if isinstance(end_date, (str,)):
        end_str = end_date
    else:
        end_str = _to_date(end_date).isoformat()

    search_query = f"${query} since:{start_str} until:{end_str}"
    
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(search_query).get_items()):
        if i >= max_tweets:
            break
        tweets.append({
            'date': tweet.date.date(),
            'content': getattr(tweet, 'rawContent', getattr(tweet, 'content', '')),
            'url': getattr(tweet, 'url', None)
        })
    
    return pd.DataFrame(tweets)

# Example usage for Apple
start_date = "2024-01-01"
end_date = "2024-03-01"
ticker = "AAPL"

news_df = get_stock_news(ticker, start_date, end_date)
tweets_df = get_twitter_posts(ticker, start_date, end_date)

print(f"Collected {len(news_df)} news articles and {len(tweets_df)} tweets")