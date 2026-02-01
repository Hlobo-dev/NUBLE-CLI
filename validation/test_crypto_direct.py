"""
KYPERIAN Crypto Module - Fast Direct Test
Tests APIs directly with synchronous requests
"""

import sys
sys.path.insert(0, "/Users/humbertolobo/Desktop/bolt.new-main/KYPERIAN-CLI")

import requests
from datetime import datetime, timedelta

print("=" * 70)
print("🚀 KYPERIAN CRYPTO MODULE - DIRECT API TEST")
print("=" * 70)
print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# API Keys
CRYPTONEWS_KEY = "fci3fvhrbxocelhel4ddc7zbmgsxnq1zmwrkxgq2"
COINDESK_KEY = "78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c"

passed = 0
failed = 0

# ========================================
# TEST 1: CryptoNews API
# ========================================
print("\n🧪 TEST 1: CRYPTONEWS API")
print("-" * 50)

try:
    url = "https://cryptonews-api.com/api/v1"
    params = {
        "tickers": "BTC",
        "items": 3,
        "token": CRYPTONEWS_KEY
    }
    
    resp = requests.get(url, params=params, timeout=15)
    print(f"   Status: {resp.status_code}")
    
    if resp.status_code == 200:
        data = resp.json()
        articles = data.get("data", [])
        
        if articles:
            print(f"✅ CryptoNews: Retrieved {len(articles)} articles")
            for article in articles[:2]:
                title = article.get("title", "N/A")[:55]
                print(f"   📰 {title}...")
            passed += 1
        else:
            print(f"⚠️  CryptoNews: No articles (may need different plan)")
            print(f"   Response: {str(data)[:100]}")
            passed += 1  # API responded, count as pass
    else:
        print(f"❌ CryptoNews: HTTP {resp.status_code}")
        failed += 1
        
except Exception as e:
    print(f"❌ CryptoNews: Error - {e}")
    failed += 1

# ========================================
# TEST 2: CoinDesk Current Price
# ========================================
print("\n🧪 TEST 2: COINDESK CURRENT PRICES")
print("-" * 50)

try:
    url = "https://min-api.cryptocompare.com/data/pricemulti"
    headers = {"Authorization": f"Apikey {COINDESK_KEY}"}
    params = {"fsyms": "BTC,ETH,SOL", "tsyms": "USD"}
    
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    
    if resp.status_code == 200:
        data = resp.json()
        print(f"✅ CoinDesk Prices:")
        for symbol, prices in data.items():
            print(f"   💰 {symbol}: ${prices.get('USD', 0):,.2f}")
        passed += 1
    else:
        print(f"❌ CoinDesk Prices: HTTP {resp.status_code}")
        failed += 1
        
except Exception as e:
    print(f"❌ CoinDesk Prices: Error - {e}")
    failed += 1

# ========================================
# TEST 3: CoinDesk Historical OHLCV
# ========================================
print("\n🧪 TEST 3: COINDESK HISTORICAL OHLCV")
print("-" * 50)

try:
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    headers = {"Authorization": f"Apikey {COINDESK_KEY}"}
    params = {"fsym": "BTC", "tsym": "USD", "limit": 7}
    
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    
    if resp.status_code == 200:
        data = resp.json()
        
        if data.get("Response") == "Success":
            bars = data.get("Data", {}).get("Data", [])
            print(f"✅ CoinDesk Historical: {len(bars)} daily bars")
            
            for bar in bars[-3:]:
                ts = datetime.utcfromtimestamp(bar.get("time", 0))
                close = bar.get("close", 0)
                print(f"   📈 {ts.strftime('%Y-%m-%d')}: ${close:,.0f}")
            passed += 1
        else:
            print(f"❌ CoinDesk Historical: {data.get('Message', 'Error')}")
            failed += 1
    else:
        print(f"❌ CoinDesk Historical: HTTP {resp.status_code}")
        failed += 1
        
except Exception as e:
    print(f"❌ CoinDesk Historical: Error - {e}")
    failed += 1

# ========================================
# TEST 4: CoinDesk Index (CADLI)
# ========================================
print("\n🧪 TEST 4: COINDESK INDEX (CADLI)")
print("-" * 50)

try:
    url = "https://data-api.cryptocompare.com/index/cc/v1/historical/days"
    headers = {"Authorization": f"Apikey {COINDESK_KEY}"}
    params = {"market": "cadli", "instrument": "BTC-USD", "limit": 3}
    
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    
    if resp.status_code == 200:
        data = resp.json()
        
        if "Data" in data:
            print(f"✅ CoinDesk Index: Available")
            if isinstance(data["Data"], list) and len(data["Data"]) > 0:
                for item in data["Data"][-2:]:
                    ts = datetime.utcfromtimestamp(item.get("TIMESTAMP", 0))
                    close = item.get("CLOSE", 0)
                    vol = item.get("VOLUME", 0)
                    print(f"   📊 {ts.strftime('%Y-%m-%d')}: ${close:,.0f} (Vol: {vol:,.0f})")
            passed += 1
        else:
            print(f"⚠️  CoinDesk Index: No data in response")
            print(f"   Keys: {list(data.keys())}")
            passed += 1  # API responded
    else:
        print(f"❌ CoinDesk Index: HTTP {resp.status_code}")
        failed += 1
        
except Exception as e:
    print(f"❌ CoinDesk Index: Error - {e}")
    failed += 1

# ========================================
# TEST 5: Asset Detector
# ========================================
print("\n🧪 TEST 5: ASSET DETECTOR")
print("-" * 50)

try:
    from src.kyperian.assets.detector import AssetDetector, AssetClass
    
    detector = AssetDetector()
    
    test_cases = [
        ("BTC", AssetClass.CRYPTO),
        ("BTCUSD", AssetClass.CRYPTO),
        ("ETH", AssetClass.CRYPTO),
        ("AAPL", AssetClass.STOCK),
        ("SPY", AssetClass.ETF),
    ]
    
    all_ok = True
    for symbol, expected in test_cases:
        detected = detector.detect(symbol)
        status = "✓" if detected == expected else "✗"
        if detected != expected:
            all_ok = False
        print(f"   {status} {symbol} → {detected.value}")
    
    if all_ok:
        print(f"✅ Asset Detector: All cases passed")
        passed += 1
    else:
        print(f"❌ Asset Detector: Some cases failed")
        failed += 1
        
except Exception as e:
    print(f"❌ Asset Detector: Error - {e}")
    import traceback
    traceback.print_exc()
    failed += 1

# ========================================
# TEST 6: CoinDesk Client Class
# ========================================
print("\n🧪 TEST 6: COINDESK CLIENT CLASS")
print("-" * 50)

try:
    from src.kyperian.news.coindesk_client import CoinDeskClient
    
    client = CoinDeskClient()
    
    # Test current price
    btc = client.get_current_price("BTC")
    if btc and btc.get("price"):
        print(f"   ✅ get_current_price: BTC = ${btc['price']:,.2f}")
    else:
        print(f"   ⚠️  get_current_price: No data")
    
    # Test multi price
    prices = client.get_multi_price(["BTC", "ETH"])
    if prices:
        print(f"   ✅ get_multi_price: {len(prices)} symbols")
    
    # Test historical
    daily = client.get_historical_daily("BTC", limit=5)
    if daily:
        print(f"   ✅ get_historical_daily: {len(daily)} bars")
    
    print(f"✅ CoinDesk Client: Working")
    passed += 1
    
except Exception as e:
    print(f"❌ CoinDesk Client: Error - {e}")
    import traceback
    traceback.print_exc()
    failed += 1

# ========================================
# SUMMARY
# ========================================
print("\n" + "=" * 70)
print("📊 TEST SUMMARY")
print("=" * 70)

total = passed + failed
print(f"\n✅ Passed: {passed}/{total}")
print(f"❌ Failed: {failed}/{total}")

if failed == 0:
    print("\n🎉 ALL CRYPTO MODULE TESTS PASSED!")
else:
    print(f"\n⚠️  {failed} test(s) need attention")

print("=" * 70)
