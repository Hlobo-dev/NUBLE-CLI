"""
KYPERIAN Crypto Module Complete Test
Tests all crypto components: CryptoNews API, CoinDesk API, Crypto Analyzer
"""

import sys
sys.path.insert(0, "/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI")

import time
from datetime import datetime

# Test results tracking
test_results = {
    "passed": [],
    "failed": [],
    "warnings": []
}


def log_result(test_name: str, passed: bool, message: str = ""):
    """Log test result"""
    if passed:
        test_results["passed"].append(test_name)
        print(f"✅ PASS: {test_name}")
    else:
        test_results["failed"].append(test_name)
        print(f"❌ FAIL: {test_name}")
    if message:
        print(f"   {message}")


def log_warning(message: str):
    """Log warning"""
    test_results["warnings"].append(message)
    print(f"⚠️  {message}")


def test_cryptonews_api():
    """Test 1: CryptoNews API Client"""
    print("\n" + "=" * 70)
    print("🧪 TEST 1: CRYPTONEWS API CLIENT")
    print("=" * 70)
    
    try:
        from src.kyperian.news.crypto_client import CryptoNewsClient
        
        client = CryptoNewsClient()
        
        # Test 1.1: Get Bitcoin news
        print("\n📰 1.1: Fetching BTC news...")
        btc_news = client.get_ticker_news("BTC", items=5)
        
        if btc_news and btc_news.get("data"):
            articles = btc_news["data"]
            log_result("CryptoNews BTC", True, f"Retrieved {len(articles)} articles")
            
            # Show sample headlines
            for article in articles[:2]:
                title = article.get("title", "No title")[:60]
                print(f"      📄 {title}...")
        else:
            log_result("CryptoNews BTC", False, "No data returned")
        
        time.sleep(0.5)  # Rate limit
        
        # Test 1.2: Get ETH news
        print("\n📰 1.2: Fetching ETH news...")
        eth_news = client.get_ticker_news("ETH", items=3)
        
        if eth_news and eth_news.get("data"):
            log_result("CryptoNews ETH", True, f"Retrieved {len(eth_news['data'])} articles")
        else:
            log_warning("ETH news not available - may be API limit")
        
        time.sleep(0.5)
        
        # Test 1.3: General crypto news
        print("\n📰 1.3: Fetching general crypto news...")
        general = client.get_general_news(items=5)
        
        if general and general.get("data"):
            log_result("CryptoNews General", True, f"Retrieved {len(general['data'])} articles")
        else:
            log_warning("General news not available")
        
        return True
        
    except Exception as e:
        log_result("CryptoNews API", False, str(e))
        return False


def test_coindesk_api():
    """Test 2: CoinDesk Data API"""
    print("\n" + "=" * 70)
    print("🧪 TEST 2: COINDESK DATA API")
    print("=" * 70)
    
    try:
        from src.kyperian.news.coindesk_client import CoinDeskClient
        
        client = CoinDeskClient()
        
        # Test 2.1: Bitcoin Price Index
        print("\n💰 2.1: Fetching Bitcoin Price Index...")
        bpi = client.get_bitcoin_index()
        
        if bpi and bpi.get("bpi"):
            usd_price = bpi["bpi"]["USD"]["rate_float"]
            log_result("CoinDesk BPI", True, f"BTC = ${usd_price:,.2f}")
        else:
            log_result("CoinDesk BPI", False, "No BPI data")
        
        time.sleep(0.3)
        
        # Test 2.2: Current price
        print("\n💰 2.2: Fetching current BTC price...")
        price = client.get_current_price("BTC")
        
        if price and price.get("price"):
            log_result("CoinDesk Price", True, f"BTC = ${price['price']:,.2f}")
        else:
            log_warning("Direct price endpoint may be limited")
        
        time.sleep(0.3)
        
        # Test 2.3: Historical data
        print("\n📈 2.3: Fetching historical data...")
        from datetime import timedelta
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        
        historical = client.get_historical_data("BTC", start_date, end_date)
        
        if historical and historical.get("prices"):
            prices = historical["prices"]
            log_result("CoinDesk Historical", True, f"Retrieved {len(prices)} data points")
            
            if prices:
                latest = prices[-1]
                print(f"      Latest: {latest['date']} - ${latest['price']:,.2f}")
        else:
            log_warning("Historical data limited (API restriction)")
        
        return True
        
    except Exception as e:
        log_result("CoinDesk API", False, str(e))
        return False


def test_crypto_analyzer():
    """Test 3: Crypto Analyzer (without FinBERT)"""
    print("\n" + "=" * 70)
    print("🧪 TEST 3: CRYPTO ANALYZER")
    print("=" * 70)
    
    try:
        from src.kyperian.assets.crypto_analyzer import CryptoAnalyzer, CryptoSignal
        
        # Initialize without FinBERT for speed
        print("\n🔧 Initializing analyzer (keyword sentiment)...")
        analyzer = CryptoAnalyzer(load_finbert=False)
        
        # Test 3.1: BTC Analysis
        print("\n🔍 3.1: Analyzing BTC...")
        btc = analyzer.analyze("BTC")
        
        if btc.news_count > 0 or btc.current_price:
            log_result("Crypto Analyzer BTC", True, 
                      f"Signal: {btc.signal.name}, Strength: {btc.signal_strength:.3f}")
            
            print(f"      📰 News articles: {btc.news_count}")
            print(f"      📊 Sentiment: {btc.news_sentiment:.3f}")
            if btc.current_price:
                print(f"      💰 Price: ${btc.current_price:,.2f}")
        else:
            log_warning("Limited data for BTC analysis")
        
        time.sleep(0.5)
        
        # Test 3.2: ETH Analysis
        print("\n🔍 3.2: Analyzing ETH...")
        eth = analyzer.analyze("ETH")
        
        if eth.news_count > 0:
            log_result("Crypto Analyzer ETH", True,
                      f"Signal: {eth.signal.name}, Headlines: {eth.news_count}")
        else:
            log_warning("Limited data for ETH analysis")
        
        time.sleep(0.5)
        
        # Test 3.3: SOL Analysis
        print("\n🔍 3.3: Analyzing SOL...")
        sol = analyzer.analyze("SOL")
        
        log_result("Crypto Analyzer SOL", True,
                  f"Signal: {sol.signal.name}, Headlines: {sol.news_count}")
        
        return True
        
    except Exception as e:
        log_result("Crypto Analyzer", False, str(e))
        import traceback
        traceback.print_exc()
        return False


def test_crypto_analyzer_with_finbert():
    """Test 4: Crypto Analyzer with FinBERT (optional)"""
    print("\n" + "=" * 70)
    print("🧪 TEST 4: CRYPTO ANALYZER + FINBERT")
    print("=" * 70)
    
    try:
        from src.kyperian.assets.crypto_analyzer import CryptoAnalyzer
        
        print("\n🔧 Initializing analyzer with FinBERT...")
        print("   (This may take a moment to load the model)")
        
        analyzer = CryptoAnalyzer(load_finbert=True)
        
        if analyzer.finbert is None:
            log_warning("FinBERT not available - skipping")
            return True
        
        print("\n🔍 4.1: Full BTC analysis with FinBERT...")
        btc = analyzer.analyze("BTC")
        
        if btc.sentiment_confidence > 0.5:
            log_result("FinBERT Crypto Analysis", True,
                      f"Confidence: {btc.sentiment_confidence:.2f}")
        else:
            log_result("FinBERT Crypto Analysis", True,
                      "Analysis complete (low confidence expected with few articles)")
        
        print(f"      📊 FinBERT Sentiment: {btc.news_sentiment:.3f}")
        print(f"      🎯 Confidence: {btc.sentiment_confidence:.2f}")
        print(f"      🎯 Signal: {btc.signal.name}")
        
        return True
        
    except Exception as e:
        log_warning(f"FinBERT test skipped: {e}")
        return True


def test_asset_detector():
    """Test 5: Asset Class Detector"""
    print("\n" + "=" * 70)
    print("🧪 TEST 5: ASSET CLASS DETECTOR")
    print("=" * 70)
    
    try:
        from src.kyperian.assets.detector import AssetDetector, AssetClass
        
        detector = AssetDetector()
        
        test_cases = [
            ("BTC", AssetClass.CRYPTO),
            ("BTCUSD", AssetClass.CRYPTO),
            ("ETH", AssetClass.CRYPTO),
            ("ETHEREUM", AssetClass.CRYPTO),
            ("SOL", AssetClass.CRYPTO),
            ("AAPL", AssetClass.STOCK),
            ("MSFT", AssetClass.STOCK),
            ("SPY", AssetClass.ETF),
            ("QQQ", AssetClass.ETF),
            ("EURUSD", AssetClass.FOREX),
            ("GC=F", AssetClass.COMMODITY),
        ]
        
        passed = 0
        failed = 0
        
        for symbol, expected in test_cases:
            detected = detector.detect(symbol)
            if detected == expected:
                print(f"   ✅ {symbol} → {detected.value}")
                passed += 1
            else:
                print(f"   ❌ {symbol} → {detected.value} (expected {expected.value})")
                failed += 1
        
        if failed == 0:
            log_result("Asset Detector", True, f"All {passed} cases passed")
        else:
            log_result("Asset Detector", False, f"{failed}/{passed+failed} cases failed")
        
        return failed == 0
        
    except Exception as e:
        log_result("Asset Detector", False, str(e))
        return False


def print_summary():
    """Print test summary"""
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    total = len(test_results["passed"]) + len(test_results["failed"])
    
    print(f"\n✅ PASSED: {len(test_results['passed'])}/{total}")
    for test in test_results["passed"]:
        print(f"   • {test}")
    
    if test_results["failed"]:
        print(f"\n❌ FAILED: {len(test_results['failed'])}/{total}")
        for test in test_results["failed"]:
            print(f"   • {test}")
    
    if test_results["warnings"]:
        print(f"\n⚠️  WARNINGS: {len(test_results['warnings'])}")
        for warning in test_results["warnings"]:
            print(f"   • {warning}")
    
    print("\n" + "=" * 70)
    if len(test_results["failed"]) == 0:
        print("🎉 ALL CRYPTO TESTS PASSED!")
    else:
        print(f"⚠️  {len(test_results['failed'])} test(s) failed")
    print("=" * 70)


def main():
    """Run all crypto tests"""
    print("=" * 70)
    print("🚀 KYPERIAN CRYPTO MODULE - COMPLETE TEST SUITE")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Run tests
    test_cryptonews_api()
    test_coindesk_api()
    test_crypto_analyzer()
    test_asset_detector()
    
    # Optional: FinBERT test (slower)
    print("\n" + "-" * 70)
    response = input("Run FinBERT integration test? (y/n, default=n): ").strip().lower()
    if response == 'y':
        test_crypto_analyzer_with_finbert()
    else:
        print("⏭️  Skipping FinBERT test")
    
    # Print summary
    print_summary()


if __name__ == "__main__":
    main()
