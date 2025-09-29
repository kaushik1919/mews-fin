"""
Updated sentiment test with better error handling
"""

import os
import sys
sys.path.append('.')

import pandas as pd
import requests
from datetime import datetime, timedelta
from src.sentiment_analyzer import SentimentAnalyzer

def test_improved_sentiment():
    """Test the improved GNews API integration"""
    
    api_key = "0903e69179300b9e3117cdc721c14366"
    
    print("🔍 Testing Improved GNews Sentiment Analysis...")
    print("=" * 60)
    
    # Test with single symbol first
    symbol = 'AAPL'
    
    try:
        url = "https://gnews.io/api/v4/search"
        params = {
            'q': f"{symbol} stock",
            'token': api_key,
            'lang': 'en',
            'country': 'us',
            'max': 5,
            'sortby': 'publishedAt'
        }
        
        print(f"📡 Fetching news for {symbol}...")
        response = requests.get(url, params=params, timeout=15)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            
            # Show API info
            if 'information' in data:
                print("ℹ️  API Info:", data['information'])
            
            articles = data.get('articles', [])
            print(f"✅ Found {len(articles)} articles")
            
            if articles:
                # Create DataFrame
                news_data = []
                for article in articles:
                    news_data.append({
                        'symbol': symbol,
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', '')[:200] + '...',  # Truncate content
                        'published_date': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })
                
                news_df = pd.DataFrame(news_data)
                
                print("\n📰 Sample Headlines:")
                for i, row in news_df.head(3).iterrows():
                    print(f"  {i+1}. {row['title']}")
                    print(f"     Source: {row['source']}")
                    print()
                
                # Test sentiment analysis
                print("🎯 Testing Sentiment Analysis...")
                analyzer = SentimentAnalyzer()
                
                # Combine text
                news_df['full_text'] = (
                    news_df['title'].fillna('') + ' ' + 
                    news_df['description'].fillna('') + ' ' + 
                    news_df['content'].fillna('')
                )
                
                sentiments = []
                for text in news_df['full_text']:
                    if text.strip() and len(text) > 10:
                        sentiment = analyzer.analyze_vader_sentiment(text)
                        sentiments.append(sentiment)
                    else:
                        sentiments.append({'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0})
                
                # Add to dataframe
                for i, sent in enumerate(sentiments):
                    news_df.loc[i, 'sentiment_compound'] = sent['compound']
                    compound = sent['compound']
                    if compound >= 0.05:
                        news_df.loc[i, 'sentiment_label'] = 'Positive 📈'
                    elif compound <= -0.05:
                        news_df.loc[i, 'sentiment_label'] = 'Negative 📉'
                    else:
                        news_df.loc[i, 'sentiment_label'] = 'Neutral ➡️'
                
                print("\n📊 Sentiment Results:")
                for i, row in news_df.iterrows():
                    print(f"  {row['sentiment_label']} ({row['sentiment_compound']:.3f}): {row['title'][:80]}...")
                
                # Summary
                avg_sentiment = news_df['sentiment_compound'].mean()
                positive_count = (news_df['sentiment_compound'] > 0.05).sum()
                negative_count = (news_df['sentiment_compound'] < -0.05).sum()
                
                print(f"\n📈 Summary for {symbol}:")
                print(f"  Average Sentiment: {avg_sentiment:.3f}")
                print(f"  Positive Articles: {positive_count}")
                print(f"  Negative Articles: {negative_count}")
                print(f"  Neutral Articles: {len(news_df) - positive_count - negative_count}")
                
                # Save results
                news_df.to_csv('data/improved_sentiment_test.csv', index=False)
                print(f"\n💾 Results saved to data/improved_sentiment_test.csv")
                
                print("\n🎉 Sentiment analysis test completed successfully!")
                
            else:
                print("⚠️  No articles found in response")
                
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text[:300]}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_improved_sentiment()
