#!/usr/bin/env python3
"""Simple CoinDesk API Test"""

import sys
sys.path.insert(0, "/Users/humbertolobo/Desktop/bolt.new-main/NUBLE-CLI")

import requests

print("=" * 60)
print("COINDESK PREMIUM API TEST")
print("=" * 60)

API_KEY = "78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c"
headers = {"Authorization": f"Apikey {API_KEY}"}

# Test 1: Current Price via CryptoCompare
print("\n1. Testing current price endpoint...")
url = "https://min-api.cryptocompare.com/data/price"
params = {"fsym": "BTC", "tsyms": "USD"}

try:
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if "USD" in data:
            print(f"   ✅ BTC Price: ${data['USD']:,.2f}")
        else:
            print(f"   Response: {data}")
    else:
        print(f"   Response: {resp.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Historical Daily
print("\n2. Testing historical daily endpoint...")
url = "https://min-api.cryptocompare.com/data/v2/histoday"
params = {"fsym": "BTC", "tsym": "USD", "limit": 7}

try:
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if data.get("Response") == "Success":
            bars = data.get("Data", {}).get("Data", [])
            print(f"   ✅ Retrieved {len(bars)} daily bars")
            if bars:
                last = bars[-1]
                print(f"   Latest: Close=${last.get('close', 0):,.2f}")
        else:
            print(f"   Response: {data.get('Message', 'Unknown error')}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Multi Price
print("\n3. Testing multi-price endpoint...")
url = "https://min-api.cryptocompare.com/data/pricemulti"
params = {"fsyms": "BTC,ETH,SOL", "tsyms": "USD"}

try:
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        for symbol, prices in data.items():
            print(f"   ✅ {symbol}: ${prices.get('USD', 0):,.2f}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: CoinDesk Index Endpoint
print("\n4. Testing CoinDesk index endpoint...")
url = "https://data-api.cryptocompare.com/index/cc/v1/historical/days"
params = {"market": "cadli", "instrument": "BTC-USD", "limit": 3}

try:
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    print(f"   Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        if "Data" in data:
            print(f"   ✅ Index data available")
            print(f"   Response keys: {list(data.keys())}")
        else:
            print(f"   Response: {str(data)[:200]}")
    else:
        print(f"   Response: {resp.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
