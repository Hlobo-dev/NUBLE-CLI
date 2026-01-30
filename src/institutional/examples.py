"""
Example usage of the Institutional Research Platform.
Run this file to see the platform in action.
"""

import asyncio
import os
from datetime import date, timedelta

# Add src to path for direct execution
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from institutional import (
    Orchestrator,
    TechnicalAnalyzer,
    SentimentAnalyzer,
    PatternRecognizer,
    AnomalyDetector,
    load_config,
)


async def example_orchestrator():
    """Example: Using the Orchestrator for natural language queries"""
    print("\n" + "="*60)
    print("ORCHESTRATOR EXAMPLE")
    print("="*60)
    
    config = load_config()
    orchestrator = Orchestrator(config=config)
    
    print(f"\nAvailable providers: {orchestrator.get_available_providers()}")
    
    # Example query
    print("\nQuery: 'What is the current analysis for AAPL?'")
    result = await orchestrator.query("What is the current analysis for AAPL?")
    
    print(f"\nStatus: {result.status.value}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Providers used: {result.providers_used}")
    
    if result.data:
        for symbol, data in result.data.items():
            print(f"\nData for {symbol}:")
            if "quote" in data:
                print(f"  Quote: {data['quote']}")
            if "analytics" in data:
                analytics = data["analytics"]
                if "technical" in analytics:
                    tech = analytics["technical"]
                    print(f"  Technical: {tech.direction} ({tech.strength:.0%})")
    
    await orchestrator.close()


def example_technical_analysis():
    """Example: Direct technical analysis"""
    print("\n" + "="*60)
    print("TECHNICAL ANALYSIS EXAMPLE")
    print("="*60)
    
    # Sample price data (normally from a provider)
    closes = [
        150.0, 151.2, 149.8, 152.5, 153.1, 154.0, 152.8, 155.2, 156.7, 155.9,
        157.3, 158.1, 156.5, 159.2, 160.1, 158.8, 161.5, 162.3, 160.9, 163.7,
        165.2, 164.1, 166.8, 168.3, 167.5, 170.1, 171.8, 170.2, 172.5, 174.1,
        173.2, 175.8, 177.2, 176.1, 178.5, 180.2, 179.1, 181.8, 183.5, 182.3,
        185.1, 186.7, 184.9, 188.2, 190.1, 189.3, 192.5, 194.2, 193.1, 196.8,
    ]
    
    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    volumes = [1000000 + i * 10000 for i in range(len(closes))]
    
    analyzer = TechnicalAnalyzer()
    
    # RSI
    rsi = analyzer.rsi(closes, period=14)
    print(f"\nRSI (14): {rsi[-1]:.2f}")
    
    # MACD
    macd_line, signal_line, histogram = analyzer.macd(closes)
    if histogram:
        print(f"MACD Histogram: {histogram[-1]:.2f}")
    
    # Bollinger Bands
    upper, middle, lower = analyzer.bollinger_bands(closes, period=20)
    if upper:
        print(f"Bollinger Bands: Lower={lower[-1]:.2f}, Middle={middle[-1]:.2f}, Upper={upper[-1]:.2f}")
    
    # Full analysis
    trend = analyzer.analyze(highs, lows, closes, volumes, symbol="EXAMPLE")
    print(f"\nTrend Analysis:")
    print(f"  Direction: {trend.direction}")
    print(f"  Strength: {trend.strength:.0%}")
    print(f"  Support: {[f'${s:.2f}' for s in trend.support_levels[:3]]}")
    print(f"  Resistance: {[f'${r:.2f}' for r in trend.resistance_levels[:3]]}")
    
    # Trading signals
    signals = analyzer.get_signals(highs, lows, closes, volumes)
    print(f"\nTrading Signals:")
    for signal in signals:
        print(f"  {signal.name}: {signal.signal} (strength: {signal.strength:.0%})")


def example_sentiment_analysis():
    """Example: Sentiment analysis on financial text"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS EXAMPLE")
    print("="*60)
    
    analyzer = SentimentAnalyzer(use_ml=False)  # Rule-based
    
    # Sample headlines
    headlines = [
        "Apple beats earnings expectations, stock surges 5%",
        "Tesla faces production challenges amid supply chain issues",
        "Microsoft announces major AI partnership",
        "Fed signals potential rate cuts, markets rally",
        "Retail sales disappoint, raising recession concerns",
    ]
    
    print("\nAnalyzing headlines:")
    for headline in headlines:
        result = analyzer.analyze(headline)
        sentiment_emoji = "📈" if result.sentiment == "positive" else "📉" if result.sentiment == "negative" else "➖"
        print(f"\n{sentiment_emoji} {headline}")
        print(f"   Sentiment: {result.sentiment} (score: {result.score:.2f}, confidence: {result.confidence:.0%})")
    
    # Aggregate analysis
    articles = [{"title": h, "summary": "", "published_at": None} for h in headlines]
    aggregate = analyzer.analyze_news_feed(articles)
    
    print(f"\nAggregate Sentiment:")
    print(f"  Overall: {aggregate.overall_sentiment} ({aggregate.overall_score:.2f})")
    print(f"  Bullish: {aggregate.bullish_count}, Bearish: {aggregate.bearish_count}, Neutral: {aggregate.neutral_count}")


def example_pattern_recognition():
    """Example: Chart pattern detection"""
    print("\n" + "="*60)
    print("PATTERN RECOGNITION EXAMPLE")
    print("="*60)
    
    recognizer = PatternRecognizer(use_ml=False)
    
    # Sample OHLCV data with a double bottom pattern
    opens = [100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 92, 93, 94, 95, 94, 93, 92, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103]
    closes = [99, 98, 97, 96, 95, 94, 93, 92, 91, 92, 93, 94, 95, 94, 93, 92, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104]
    highs = [max(o, c) + 0.5 for o, c in zip(opens, closes)]
    lows = [min(o, c) - 0.5 for o, c in zip(opens, closes)]
    
    # Detect candlestick patterns
    candlestick_patterns = recognizer.scan_candlestick_patterns(opens, highs, lows, closes, lookback=10)
    print(f"\nCandlestick patterns found: {len(candlestick_patterns)}")
    for pattern in candlestick_patterns[:5]:
        print(f"  {pattern.pattern_type.value}: {pattern.direction} (confidence: {pattern.confidence:.0%})")
    
    # Full pattern analysis
    result = recognizer.analyze(opens, highs, lows, closes)
    print(f"\nFull Pattern Analysis:")
    print(f"  Patterns detected: {result['pattern_count']}")
    print(f"  Bullish patterns: {result['bullish_patterns']}")
    print(f"  Bearish patterns: {result['bearish_patterns']}")
    print(f"  Overall bias: {result['overall_bias']}")


def example_anomaly_detection():
    """Example: Anomaly detection"""
    print("\n" + "="*60)
    print("ANOMALY DETECTION EXAMPLE")
    print("="*60)
    
    detector = AnomalyDetector(z_threshold=2.0, lookback_period=10)
    
    # Normal data with an anomaly
    closes = [100, 101, 99, 102, 100, 101, 99, 100, 101, 100, 99, 101, 100, 99, 100, 101, 130, 102, 100, 101]  # 130 is anomaly
    volumes = [1000000] * 15 + [5000000, 5000000] + [1000000] * 3  # Volume spike
    opens = [c - 0.5 for c in closes]
    highs = [c + 1 for c in closes]
    lows = [c - 1 for c in closes]
    
    # Detect anomalies
    report = detector.analyze(opens, highs, lows, closes, volumes, symbol="EXAMPLE")
    
    print(f"\nAnomaly Report for EXAMPLE:")
    print(f"  Risk Score: {report.risk_score:.2f}")
    print(f"  Alert Level: {report.alert_level}")
    print(f"\n  Anomalies detected: {len(report.anomalies)}")
    for anomaly in report.anomalies:
        print(f"    - {anomaly.anomaly_type.value}: {anomaly.description}")
    
    print(f"\n  Recommendations:")
    for rec in report.recommendations:
        print(f"    • {rec}")
    
    print(f"\n  Summary: {report.summary}")


async def main():
    """Run all examples"""
    print("\n" + "🏛️ "*20)
    print("INSTITUTIONAL RESEARCH PLATFORM - EXAMPLES")
    print("🏛️ "*20)
    
    # Technical analysis (no API keys needed)
    example_technical_analysis()
    
    # Sentiment analysis (no API keys needed)
    example_sentiment_analysis()
    
    # Pattern recognition (no API keys needed)
    example_pattern_recognition()
    
    # Anomaly detection (no API keys needed)
    example_anomaly_detection()
    
    # Orchestrator (requires API keys)
    print("\n" + "="*60)
    print("ORCHESTRATOR EXAMPLE (requires API keys)")
    print("="*60)
    
    # Check for API keys
    if not any([
        os.getenv("POLYGON_API_KEY"),
        os.getenv("ALPHA_VANTAGE_API_KEY"),
        os.getenv("FINNHUB_API_KEY"),
    ]):
        print("\n⚠️  No API keys configured. Skipping orchestrator example.")
        print("Set environment variables to test:")
        print("  export POLYGON_API_KEY='your-key'")
        print("  export ALPHA_VANTAGE_API_KEY='your-key'")
        print("  export FINNHUB_API_KEY='your-key'")
        print("  export OPENAI_API_KEY='your-key'")
    else:
        await example_orchestrator()
    
    print("\n" + "="*60)
    print("EXAMPLES COMPLETE")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
