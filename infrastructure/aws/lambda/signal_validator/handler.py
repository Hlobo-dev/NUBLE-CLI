"""NUBLE Signal Validator Lambda Handler"""

import json
import os
import time
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict
from decimal import Decimal

import boto3
from botocore.config import Config

ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')
SIGNALS_TABLE = os.environ.get('DYNAMODB_SIGNALS_TABLE', f'nuble-{ENVIRONMENT}-signals')
DECISIONS_TABLE = os.environ.get('DYNAMODB_DECISIONS_TABLE', f'nuble-{ENVIRONMENT}-decisions')
EVENTBRIDGE_BUS = os.environ.get('EVENTBRIDGE_BUS_NAME', f'nuble-{ENVIRONMENT}-signals')

VALID_TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M']
TIMEFRAME_WEIGHTS = {'1m': 1, '5m': 2, '15m': 3, '30m': 4, '1h': 5, '4h': 6, '1D': 7, '1W': 8, '1M': 9}

BOTO_CONFIG = Config(connect_timeout=2, read_timeout=5, retries={'max_attempts': 2, 'mode': 'adaptive'})

dynamodb = boto3.resource('dynamodb', config=BOTO_CONFIG)
eventbridge = boto3.client('events', config=BOTO_CONFIG)
signals_table = dynamodb.Table(SIGNALS_TABLE)


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def json_response(status_code, body):
    return {
        'statusCode': status_code,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type,X-Api-Key',
            'Access-Control-Allow-Methods': 'GET,POST,OPTIONS',
        },
        'body': json.dumps(body, cls=DecimalEncoder)
    }


def normalize_timeframe(tf):
    if not tf:
        return '1h'
    tf_lower = tf.lower() # ('MIN', 'm').replace('HOUR', 'h').replace('DAY', 'D')
    mappings = {'1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m', '1h': '1h', '4h': '4h', '1d': '1D', '1w': '1W', 'd': '1D', 'w': '1W', 'daily': '1D', 'weekly': '1W'}
    return mappings.get(tf_lower, tf)


def validate_signal(signal):
    symbol = signal.get('symbol', '').upper().strip()
    action = signal.get('action', '').upper().strip()
    timeframe = normalize_timeframe(signal.get('timeframe', '1h'))
    
    if not symbol:
        return False, "Missing symbol", {}
    if action not in ['BUY', 'SELL', 'HOLD', 'CLOSE']:
        return False, f"Invalid action: {action}", {}
    if timeframe not in VALID_TIMEFRAMES:
        return False, f"Invalid timeframe: {timeframe}", {}
    
    return True, "", {
        'symbol': symbol,
        'action': action,
        'timeframe': timeframe,
        'price': Decimal(str(signal.get('price', 0))),
        'confidence': int(signal.get('confidence', 50)),
        'source': signal.get('source', 'tradingview'),
        'timestamp': int(time.time() * 1000),
    }


def store_signal(signal, signal_id):
    try:
        now = datetime.now(timezone.utc)
        timestamp_str = now.isoformat()  # GSI expects String for timestamp
        item = {
            'pk': f"SIGNAL#{signal['symbol']}",
            'sk': f"{signal['timeframe']}#{now.strftime('%Y%m%d%H%M%S')}#{signal_id}",
            'symbol': signal['symbol'],
            'timeframe': signal['timeframe'],
            'direction': signal['action'],  # Standardize to 'direction' for decision engine
            'action': signal['action'],
            'price': signal['price'],
            'confidence': signal['confidence'],
            'source': signal['source'],
            'timestamp': timestamp_str,  # Store as ISO string for GSI compatibility
            'timestamp_ms': signal['timestamp'],  # Keep numeric timestamp too
            'ttl': int(time.time()) + (7 * 24 * 3600),
            'gsi1pk': f"TF#{signal['timeframe']}",
            'gsi1sk': f"{signal['symbol']}#{now.strftime('%Y%m%d%H%M%S')}"
        }
        signals_table.put_item(Item=item)
        return True
    except Exception as e:
        print(f"Error storing signal: {e}")
        return False


def publish_event(signal, signal_id):
    try:
        eventbridge.put_events(Entries=[{
            'Source': 'nuble.signal.validator',
            'DetailType': 'SignalValidated',
            'Detail': json.dumps({
                'signal_id': signal_id,
                'symbol': signal['symbol'],
                'action': signal['action'],
                'timeframe': signal['timeframe'],
                'price': float(signal['price']),
                'timestamp': signal['timestamp']
            }),
            'EventBusName': EVENTBRIDGE_BUS
        }])
        return True
    except Exception as e:
        print(f"Error publishing: {e}")
        return False


def handle_webhook(event):
    start_time = time.time()
    request_id = event.get('requestContext', {}).get('requestId', 'unknown')
    
    try:
        body = event.get('body', '{}')
        signal_data = json.loads(body) if isinstance(body, str) else body
        
        is_valid, error_msg, validated = validate_signal(signal_data)
        if not is_valid:
            return json_response(400, {'success': False, 'error': error_msg})
        
        signal_id = hashlib.sha256(f"{validated['symbol']}:{validated['timeframe']}:{validated['action']}:{int(time.time()/60)}".encode()).hexdigest()[:16]
        
        stored = store_signal(validated, signal_id)
        published = publish_event(validated, signal_id)
        
        return json_response(200, {
            'success': True,
            'signal_id': signal_id,
            'symbol': validated['symbol'],
            'action': validated['action'],
            'timeframe': validated['timeframe'],
            'stored': stored,
            'published': published,
            'processing_time_ms': int((time.time() - start_time) * 1000)
        })
    except json.JSONDecodeError as e:
        return json_response(400, {'success': False, 'error': 'invalid_json', 'message': str(e)})
    except Exception as e:
        print(f"Error: {e}")
        return json_response(500, {'success': False, 'error': str(e)})


def handle_health(event):
    return json_response(200, {
        'status': 'healthy',
        'service': 'nuble-signal-validator',
        'environment': ENVIRONMENT,
        'timestamp': datetime.now(timezone.utc).isoformat()
    })


def get_recent_signals(symbol, limit=10):
    try:
        response = signals_table.query(
            KeyConditionExpression='pk = :pk',
            ExpressionAttributeValues={':pk': f"SIGNAL#{symbol}"},
            ScanIndexForward=False,
            Limit=limit
        )
        return response.get('Items', [])
    except Exception as e:
        print(f"Error: {e}")
        return []


def handle_signals(event):
    path_params = event.get('pathParameters', {}) or {}
    symbol = path_params.get('symbol', '').upper()
    if not symbol:
        return json_response(400, {'success': False, 'error': 'missing_symbol'})
    
    signals = get_recent_signals(symbol, 10)
    return json_response(200, {'success': True, 'symbol': symbol, 'signals': signals, 'count': len(signals)})


def lambda_handler(event, context):
    """Main Lambda handler."""
    raw_path = event.get('rawPath', '')
    http_method = event.get('requestContext', {}).get('http', {}).get('method', 'GET')
    
    print(f"Request: {http_method} {raw_path}")
    
    if 'health' in raw_path:
        return handle_health(event)
    elif 'webhook' in raw_path:
        return handle_webhook(event)
    elif 'signals' in raw_path:
        return handle_signals(event)
    elif http_method == 'OPTIONS':
        return json_response(200, {'message': 'OK'})
    else:
        return json_response(404, {'error': 'not_found', 'path': raw_path})
