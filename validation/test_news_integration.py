"""
Complete News Integration Test

Tests the entire news pipeline with real data and validates integration
with the validated ML trading model.
"""

import sys
sys.path.insert(0, '/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime

print("Loading modules...")

from src.kyperian.news.client import StockNewsClient
from src.kyperian.news.sentiment import SentimentAnalyzer, quick_sentiment
from src.kyperian.news.pipeline import NewsPipeline
from src.kyperian.news.integrator import NewsSignalIntegrator

print("Modules loaded successfully!")


async def test_news_client():
    """Test the StockNews API client."""
    print("\n" + "=" * 60)
    print("TEST 1: StockNews API Client")
    print("=" * 60)
    
    client = StockNewsClient()
    
    try:
        # Test ticker news
        print("\n📰 Fetching AAPL news...")
        news = await client.get_ticker_news(['AAPL'], items=5)
        
        print(f"   Found {len(news)} articles:")
        for i, article in enumerate(news[:3], 1):
            title = article.get('title', 'No title')[:60]
            sentiment = article.get('sentiment', 'N/A')
            source = article.get('source_name', 'Unknown')
            print(f"   {i}. [{sentiment:>8}] {title}...")
            print(f"      Source: {source}")
        
        # Test trending headlines
        print("\n🔥 Fetching trending headlines...")
        trending = await client.get_trending_headlines()
        
        print(f"   Found {len(trending)} trending stories:")
        for i, headline in enumerate(trending[:3], 1):
            title = headline.get('title', 'No title')[:60]
            print(f"   {i}. {title}...")
        
        # Test multi-symbol
        print("\n📊 Fetching news for AAPL, MSFT, NVDA...")
        multi_news = await client.get_ticker_news(['AAPL', 'MSFT', 'NVDA'], items=10)
        
        # Count by ticker
        ticker_counts = {}
        for article in multi_news:
            for ticker in article.get('tickers', []):
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
        
        print(f"   Articles by ticker: {ticker_counts}")
        
        print("\n✅ StockNews API Client: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False
        
    finally:
        await client.close()


def test_sentiment_analyzer():
    """Test the FinBERT sentiment analyzer."""
    print("\n" + "=" * 60)
    print("TEST 2: Sentiment Analyzer (FinBERT)")
    print("=" * 60)
    
    try:
        # Test quick sentiment first (no model loading)
        print("\n⚡ Testing quick sentiment (keyword-based)...")
        
        test_texts = [
            "Apple stock surges 10% on record iPhone sales",
            "Tesla misses delivery targets, shares plunge",
            "Federal Reserve holds interest rates steady",
        ]
        
        for text in test_texts:
            score = quick_sentiment(text)
            emoji = "🟢" if score > 0.2 else ("🔴" if score < -0.2 else "⚪")
            print(f"   {emoji} ({score:+.2f}) {text[:50]}...")
        
        # Test FinBERT (will download model on first use)
        print("\n🤖 Testing FinBERT sentiment (ML-based)...")
        print("   (This may take a moment on first run as model downloads)")
        
        analyzer = SentimentAnalyzer()
        
        for text in test_texts:
            result = analyzer.analyze(text)
            emoji = "🟢" if result.normalized_score > 0.2 else ("🔴" if result.normalized_score < -0.2 else "⚪")
            print(f"   {emoji} [{result.label.value:>8}] ({result.normalized_score:+.2f}) {text[:40]}...")
        
        # Test batch analysis
        print("\n📦 Testing batch analysis...")
        batch_result = analyzer.analyze_batch(test_texts, return_average=True)
        print(f"   Average sentiment: {batch_result['average']:+.2f}")
        print(f"   Distribution: +{batch_result['positive_count']} / ={batch_result['neutral_count']} / -{batch_result['negative_count']}")
        
        print("\n✅ Sentiment Analyzer: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_news_pipeline():
    """Test the complete news pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: News Pipeline (Fetch + Analyze)")
    print("=" * 60)
    
    pipeline = NewsPipeline()
    
    try:
        # Test on-demand analysis for our validated symbols
        symbols = ['AAPL', 'MSFT', 'NVDA']
        
        for symbol in symbols:
            print(f"\n📊 Analyzing {symbol}...")
            signal = await pipeline.fetch_and_analyze(symbol, lookback_hours=24)
            
            emoji = "🟢" if signal.signal_type == 'BULLISH' else ("🔴" if signal.signal_type == 'BEARISH' else "⚪")
            
            print(f"   {emoji} Sentiment: {signal.sentiment_score:+.2f}")
            print(f"      Confidence: {signal.confidence:.1%}")
            print(f"      Signal: {signal.signal_type}")
            print(f"      Articles: {signal.article_count}")
            print(f"      Actionable: {signal.actionable}")
            print(f"      Top Headline: {signal.headline[:50]}...")
        
        print("\n✅ News Pipeline: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await pipeline.stop()


async def test_signal_integration():
    """Test ML + News signal integration."""
    print("\n" + "=" * 60)
    print("TEST 4: ML + News Signal Integration")
    print("=" * 60)
    
    pipeline = NewsPipeline()
    integrator = NewsSignalIntegrator(
        ml_model=None,  # We'll simulate ML predictions
        news_pipeline=pipeline
    )
    
    try:
        # Simulate ML predictions from our validated model
        # These would come from the actual model in production
        ml_predictions = {
            'AAPL': (0.35, 0.68),   # Bullish, 68% confident (from OOS test)
            'MSFT': (0.28, 0.62),   # Bullish, 62% confident
            'NVDA': (0.52, 0.75),   # Strong bullish, 75% confident
        }
        
        print("\n🔄 Getting combined signals...")
        print("-" * 50)
        
        for symbol, (ml_signal, ml_conf) in ml_predictions.items():
            signal = await integrator.get_signal(
                symbol,
                ml_prediction=(ml_signal, ml_conf)
            )
            
            print(f"\n{symbol}:")
            print(f"   ML Signal:      {signal.ml_signal:+.2f} ({signal.ml_direction}, {signal.ml_confidence:.0%})")
            print(f"   News Signal:    {signal.news_signal:+.2f} ({signal.news_direction}, {signal.news_confidence:.0%})")
            print(f"   Combined:       {signal.combined_signal:+.2f} ({signal.final_direction})")
            print(f"   Agreement:      {'✅ Yes' if signal.signal_agreement else '⚠️ No'}")
            print(f"   Actionable:     {'✅ Yes' if signal.actionable else '❌ No'}")
            print(f"   Reason:         {signal.reason[:60]}...")
        
        # Get portfolio allocation
        print("\n📈 Portfolio Allocation:")
        print("-" * 50)
        
        all_signals = await integrator.get_multi_symbol_signals(
            list(ml_predictions.keys()),
            ml_predictions=ml_predictions
        )
        
        allocation = integrator.get_portfolio_allocation(all_signals)
        
        if allocation:
            for symbol, pct in allocation.items():
                direction = "LONG" if pct > 0 else "SHORT"
                print(f"   {symbol}: {abs(pct):.1%} {direction}")
        else:
            print("   No actionable signals for allocation")
        
        print("\n✅ Signal Integration: WORKING")
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        await pipeline.stop()


async def main():
    """Run all tests."""
    print("=" * 60)
    print("KYPERIAN NEWS INTEGRATION TEST")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    results = []
    
    # Test 1: API Client
    results.append(("StockNews API", await test_news_client()))
    
    # Test 2: Sentiment Analyzer
    results.append(("Sentiment Analyzer", test_sentiment_analyzer()))
    
    # Test 3: News Pipeline
    results.append(("News Pipeline", await test_news_pipeline()))
    
    # Test 4: Signal Integration
    results.append(("Signal Integration", await test_signal_integration()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - News Integration Ready!")
    else:
        print("⚠️ Some tests failed - Review errors above")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    asyncio.run(main())
