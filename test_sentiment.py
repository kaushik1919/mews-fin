"""
Test GNews Sentiment Analysis Integration
"""

import os
import sys
sys.path.append('.')

import pandas as pd
import requests
from datetime import datetime, timedelta
from src.sentiment_analyzer import SentimentAnalyzer

def test_gnews_api():
    """Test the GNews API integration"""
    
    # Use the API key from env template
    api_key = "0903e69179300b9e3117cdc721c14366"  # From .env.template
    
    print("🔍 Testing GNews API Integration...")
    print("=" * 50)
    
    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Fetch news for the last 3 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3)
    
    all_news = []
    
    for symbol in symbols:
        print(f"Fetching news for {symbol}...")
        
        try:
            url = "https://gnews.io/api/v4/search"
            params = {
                'q': f"{symbol} stock market financial",
                'token': api_key,
                'lang': 'en',
                'country': 'us',
                'max': 5,
                'from': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'to': end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                print(f"  ✅ Found {len(articles)} articles")
                
                for article in articles:
                    all_news.append({
                        'symbol': symbol,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'published_date': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })
                    
            elif response.status_code == 429:
                print(f"  ⚠️  Rate limit exceeded for {symbol}")
            else:
                print(f"  ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Exception for {symbol}: {str(e)}")
    
    if not all_news:
        print("❌ No news articles retrieved")
        return
    
    # Convert to DataFrame
    news_df = pd.DataFrame(all_news)
    print(f"\n📰 Total articles retrieved: {len(news_df)}")
    
    # Analyze sentiment
    print("\n🎯 Analyzing Sentiment...")
    print("=" * 50)
    
    try:
        analyzer = SentimentAnalyzer()
        
        # Combine title and description
        news_df['full_text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
        
        sentiments = []
        for i, text in enumerate(news_df['full_text']):
            if text.strip():
                sentiment = analyzer.analyze_vader_sentiment(text)
                sentiments.append(sentiment)
                print(f"  Article {i+1}: {sentiment['compound']:.3f} - {text[:100]}...")
            else:
                sentiments.append({'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0})
        
        # Add sentiment scores
        for i, sent in enumerate(sentiments):
            news_df.loc[i, 'sentiment_compound'] = sent['compound']
            news_df.loc[i, 'sentiment_positive'] = sent['pos']
            news_df.loc[i, 'sentiment_negative'] = sent['neg']
            news_df.loc[i, 'sentiment_neutral'] = sent['neu']
        
        # Summary by symbol
        print("\n📊 Sentiment Summary by Symbol:")
        print("=" * 50)
        
        summary = news_df.groupby('symbol').agg({
            'sentiment_compound': ['mean', 'std', 'count'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean'
        }).round(3)
        
        summary.columns = ['Avg_Sentiment', 'Sentiment_Std', 'News_Count', 'Avg_Positive', 'Avg_Negative']
        print(summary)
        
        # Overall sentiment
        overall_sentiment = news_df['sentiment_compound'].mean()
        positive_pct = (news_df['sentiment_compound'] > 0.1).mean() * 100
        negative_pct = (news_df['sentiment_compound'] < -0.1).mean() * 100
        
        print(f"\n🌟 Overall Market Sentiment:")
        print(f"  Average Sentiment: {overall_sentiment:.3f}")
        print(f"  Positive News: {positive_pct:.1f}%")
        print(f"  Negative News: {negative_pct:.1f}%")
        
        # Save results
        news_df.to_csv('data/test_news_sentiment.csv', index=False)
        print(f"\n💾 Results saved to data/test_news_sentiment.csv")
        
        print("\n🎉 Sentiment analysis integration test completed successfully!")
        
    except Exception as e:
        print(f"❌ Sentiment analysis error: {str(e)}")

if __name__ == "__main__":
    test_gnews_api()
