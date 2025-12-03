from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import torch
import numpy as np

# Load FinBERT model
finbert = pipeline("sentiment-analysis", 
                  model="ProsusAI/finbert",
                  tokenizer="ProsusAI/finbert")

def analyze_sentiment_batch(texts):
    """Analyze sentiment for a batch of texts"""
    results = finbert(texts)
    
    # Convert to numerical scores: positive=1, negative=-1, neutral=0
    sentiment_scores = []
    for result in results:
        if result['label'] == 'positive':
            sentiment_scores.append(1.0)
        elif result['label'] == 'negative':
            sentiment_scores.append(-1.0)
        else:
            sentiment_scores.append(0.0)
    
    return sentiment_scores

# Analyze sentiment for collected data
news_df['sentiment'] = analyze_sentiment_batch(news_df['title'].tolist())
tweets_df['sentiment'] = analyze_sentiment_batch(tweets_df['content'].tolist())