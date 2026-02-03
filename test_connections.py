#!/usr/bin/env python3
"""
NUBLE Connection & Webhook Test Suite

Tests all API connections and verifies TradingView/LuxAlgo webhook integration.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_status(name, status, details=""):
    icon = "✅" if status else "❌"
    print(f"  {icon} {name}: {details}")

def test_env_variables():
    """Test that all required environment variables are set."""
    print_header("1. Environment Variables Check")
    
    required = {
        'ANTHROPIC_API_KEY': 'AI Provider (Claude)',
        'POLYGON_API_KEY': 'Market Data',
        'STOCKNEWS_API_KEY': 'Stock News',
        'CRYPTONEWS_API_KEY': 'Crypto News',
        'COINDESK_API_KEY': 'CoinDesk',
        'AWS_ACCESS_KEY_ID': 'AWS Access',
        'AWS_SECRET_ACCESS_KEY': 'AWS Secret',
        'AWS_REGION': 'AWS Region',
    }
    
    all_ok = True
    for key, desc in required.items():
        value = os.getenv(key)
        if value and value != f"your_{key.lower()}_here":
            print_status(desc, True, f"{key} is set ({len(value)} chars)")
        else:
            print_status(desc, False, f"{key} NOT SET!")
            all_ok = False
    
    return all_ok

def test_anthropic():
    """Test Anthropic API connection."""
    print_header("2. Anthropic (Claude) API Test")
    
    try:
        import anthropic
        client = anthropic.Anthropic()
        
        # Quick test with minimal tokens
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[{"role": "user", "content": "Say 'connected'"}]
        )
        
        result = response.content[0].text
        print_status("Anthropic API", True, f"Response: {result[:50]}")
        return True
    except Exception as e:
        print_status("Anthropic API", False, str(e)[:60])
        return False

def test_polygon():
    """Test Polygon.io API connection."""
    print_header("3. Polygon.io Market Data Test")
    
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print_status("Polygon.io", False, "API key not set")
        return False
    
    try:
        import requests
        
        # Test with AAPL quote
        url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={api_key}"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if response.status_code == 200 and data.get('results'):
            result = data['results'][0]
            print_status("Polygon.io", True, f"AAPL Close: ${result.get('c', 'N/A')}")
            return True
        else:
            print_status("Polygon.io", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_status("Polygon.io", False, str(e)[:60])
        return False

def test_stocknews():
    """Test StockNews API connection."""
    print_header("4. StockNews API Test")
    
    api_key = os.getenv('STOCKNEWS_API_KEY')
    if not api_key:
        print_status("StockNews", False, "API key not set")
        return False
    
    try:
        import requests
        
        url = f"https://stocknewsapi.com/api/v1?tickers=AAPL&items=1&token={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get('data', []))
            print_status("StockNews", True, f"Got {count} news items")
            return True
        else:
            print_status("StockNews", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_status("StockNews", False, str(e)[:60])
        return False

def test_cryptonews():
    """Test CryptoNews API connection."""
    print_header("5. CryptoNews API Test")
    
    api_key = os.getenv('CRYPTONEWS_API_KEY')
    if not api_key:
        print_status("CryptoNews", False, "API key not set")
        return False
    
    try:
        import requests
        
        url = f"https://cryptonews-api.com/api/v1?tickers=BTC&items=1&token={api_key}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            count = len(data.get('data', []))
            print_status("CryptoNews", True, f"Got {count} crypto news items")
            return True
        else:
            print_status("CryptoNews", False, f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_status("CryptoNews", False, str(e)[:60])
        return False

def test_aws_connection():
    """Test AWS connection and DynamoDB access."""
    print_header("6. AWS & DynamoDB Test")
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        # Test STS (basic AWS auth)
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        account = identity.get('Account')
        print_status("AWS Authentication", True, f"Account: {account}")
        
        # Test DynamoDB
        dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        # List tables to verify access
        client = boto3.client('dynamodb', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        tables = client.list_tables()
        table_names = tables.get('TableNames', [])
        
        # Look for NUBLE tables
        nuble_tables = [t for t in table_names if 'nuble' in t.lower()]
        
        if nuble_tables:
            print_status("DynamoDB Tables", True, f"Found {len(nuble_tables)} NUBLE tables")
            for t in nuble_tables[:5]:
                print(f"      - {t}")
        else:
            print_status("DynamoDB Tables", False, "No NUBLE tables found - need to deploy infrastructure")
            print("      Run: cd infrastructure/aws && ./deploy.sh")
        
        return True
    except NoCredentialsError:
        print_status("AWS", False, "No credentials configured")
        return False
    except Exception as e:
        print_status("AWS", False, str(e)[:60])
        return False

def test_webhook_endpoint():
    """Check if webhook endpoint is deployed and accessible."""
    print_header("7. Webhook Endpoint Test")
    
    try:
        import boto3
        
        # Check API Gateway
        apigateway = boto3.client('apigatewayv2', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        apis = apigateway.get_apis()
        
        nuble_apis = [api for api in apis.get('Items', []) if 'nuble' in api.get('Name', '').lower()]
        
        if nuble_apis:
            for api in nuble_apis:
                api_id = api['ApiId']
                name = api['Name']
                endpoint = api.get('ApiEndpoint', 'N/A')
                print_status(f"API: {name}", True, f"Endpoint: {endpoint}")
                
                # Get routes
                routes = apigateway.get_routes(ApiId=api_id)
                webhook_routes = [r for r in routes.get('Items', []) if 'webhook' in r.get('RouteKey', '').lower()]
                
                for route in webhook_routes:
                    print(f"      📡 Webhook: {endpoint}{route['RouteKey'].replace('POST ', '')}")
            
            return True
        else:
            print_status("API Gateway", False, "No NUBLE API found - need to deploy")
            print("      Your TradingView webhooks need an API Gateway endpoint")
            print("      Run: cd infrastructure/aws && ./deploy.sh")
            return False
            
    except Exception as e:
        print_status("API Gateway", False, str(e)[:60])
        return False

def test_luxalgo_signals_table():
    """Check LuxAlgo signals in DynamoDB."""
    print_header("8. LuxAlgo Signals Storage Test")
    
    try:
        import boto3
        from datetime import datetime, timedelta
        
        dynamodb = boto3.resource('dynamodb', region_name=os.getenv('AWS_REGION', 'us-east-1'))
        
        # Try different table name patterns
        possible_tables = [
            'nuble-production-signals',
            'nuble-signals',
            'luxalgo-signals',
        ]
        
        signals_found = False
        for table_name in possible_tables:
            try:
                table = dynamodb.Table(table_name)
                
                # Scan for recent signals
                response = table.scan(Limit=5)
                items = response.get('Items', [])
                
                if items:
                    print_status(f"Table: {table_name}", True, f"Found {len(items)} signals")
                    for item in items[:3]:
                        symbol = item.get('symbol', 'N/A')
                        action = item.get('action', 'N/A')
                        tf = item.get('timeframe', 'N/A')
                        print(f"      📊 {symbol} {action} ({tf})")
                    signals_found = True
                    break
            except:
                continue
        
        if not signals_found:
            print_status("LuxAlgo Signals", False, "No signals found in DynamoDB")
            print("      Signals will appear after TradingView sends webhook alerts")
            print("      Make sure your TradingView alerts are configured with the webhook URL")
        
        return signals_found
        
    except Exception as e:
        print_status("LuxAlgo Signals", False, str(e)[:60])
        return False

def print_tradingview_setup():
    """Print TradingView webhook setup instructions."""
    print_header("📋 TradingView Webhook Setup")
    
    print("""
  To receive LuxAlgo signals in NUBLE, configure TradingView alerts:

  1. WEBHOOK URL (get from AWS API Gateway):
     https://YOUR-API-ID.execute-api.us-east-1.amazonaws.com/production/webhook

  2. ALERT MESSAGE TEMPLATE (copy this exactly):
     {
       "action": "{{strategy.order.action}}",
       "symbol": "{{ticker}}",
       "exchange": "{{exchange}}",
       "price": {{close}},
       "timeframe": "{{interval}}",
       "signal_type": "LuxAlgo Confirmation",
       "confirmations": 8,
       "trend_strength": 65,
       "time": "{{time}}"
     }

  3. For BUY signals:
     - action: "BUY"
     - signal_type: "Bullish Confirmation"

  4. For SELL signals:
     - action: "SELL"  
     - signal_type: "Bearish Confirmation"

  5. Recommended timeframes for LuxAlgo:
     - 4h (most reliable)
     - 1D (daily signals)
     - 1W (weekly signals)
""")

def main():
    """Run all connection tests."""
    print("\n" + "🔌"*30)
    print("     NUBLE CONNECTION & WEBHOOK TEST SUITE")
    print("🔌"*30)
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {}
    
    # Run tests
    results['env'] = test_env_variables()
    results['anthropic'] = test_anthropic()
    results['polygon'] = test_polygon()
    results['stocknews'] = test_stocknews()
    results['cryptonews'] = test_cryptonews()
    results['aws'] = test_aws_connection()
    results['webhook'] = test_webhook_endpoint()
    results['signals'] = test_luxalgo_signals_table()
    
    # Summary
    print_header("📊 SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n  Tests Passed: {passed}/{total}")
    print()
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name.upper()}")
    
    if not results['webhook']:
        print_tradingview_setup()
    
    if passed == total:
        print("\n  🎉 All systems operational!")
    else:
        print("\n  ⚠️  Some tests failed. Check the details above.")
    
    print("\n" + "="*60 + "\n")
    
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
