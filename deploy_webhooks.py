#!/usr/bin/env python3
"""
NUBLE Quick AWS Deployment

Deploys the essential infrastructure for TradingView webhooks:
- DynamoDB table for signals
- Lambda function to receive webhooks
- API Gateway endpoint

This is a simplified deployment - for full features use the CloudFormation templates.
"""

import os
import sys
import json
import time
import zipfile
import io
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import boto3
from botocore.exceptions import ClientError

# Configuration
PROJECT_NAME = "nuble"
ENVIRONMENT = "production"
REGION = os.getenv("AWS_REGION", "us-east-1")
SIGNALS_TABLE = f"{PROJECT_NAME}-{ENVIRONMENT}-signals"

def print_step(step, msg):
    print(f"\n{'='*60}")
    print(f"  Step {step}: {msg}")
    print('='*60)

def create_dynamodb_table():
    """Create DynamoDB table for signals."""
    print_step(1, "Creating DynamoDB Signals Table")
    
    dynamodb = boto3.client('dynamodb', region_name=REGION)
    
    try:
        # Check if table exists
        dynamodb.describe_table(TableName=SIGNALS_TABLE)
        print(f"  ✅ Table {SIGNALS_TABLE} already exists")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceNotFoundException':
            raise
    
    # Create table
    print(f"  Creating table: {SIGNALS_TABLE}")
    
    try:
        dynamodb.create_table(
            TableName=SIGNALS_TABLE,
            KeySchema=[
                {'AttributeName': 'pk', 'KeyType': 'HASH'},
                {'AttributeName': 'sk', 'KeyType': 'RANGE'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'pk', 'AttributeType': 'S'},
                {'AttributeName': 'sk', 'AttributeType': 'S'},
                {'AttributeName': 'gsi1pk', 'AttributeType': 'S'},
                {'AttributeName': 'gsi1sk', 'AttributeType': 'S'}
            ],
            GlobalSecondaryIndexes=[
                {
                    'IndexName': 'gsi1',
                    'KeySchema': [
                        {'AttributeName': 'gsi1pk', 'KeyType': 'HASH'},
                        {'AttributeName': 'gsi1sk', 'KeyType': 'RANGE'}
                    ],
                    'Projection': {'ProjectionType': 'ALL'}
                }
            ],
            BillingMode='PAY_PER_REQUEST',
            Tags=[
                {'Key': 'Project', 'Value': 'NUBLE'},
                {'Key': 'Environment', 'Value': ENVIRONMENT}
            ]
        )
        
        # Wait for table to be active
        print("  Waiting for table to be active...")
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=SIGNALS_TABLE)
        
        # Enable TTL
        dynamodb.update_time_to_live(
            TableName=SIGNALS_TABLE,
            TimeToLiveSpecification={
                'Enabled': True,
                'AttributeName': 'ttl'
            }
        )
        
        print(f"  ✅ Table {SIGNALS_TABLE} created successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Error creating table: {e}")
        return False

def create_lambda_role():
    """Create IAM role for Lambda."""
    print_step(2, "Creating Lambda IAM Role")
    
    iam = boto3.client('iam', region_name=REGION)
    role_name = f"{PROJECT_NAME}-{ENVIRONMENT}-webhook-role"
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }
    
    try:
        iam.get_role(RoleName=role_name)
        print(f"  ✅ Role {role_name} already exists")
        return f"arn:aws:iam::{os.getenv('AWS_ACCOUNT_ID')}:role/{role_name}"
    except ClientError as e:
        if e.response['Error']['Code'] != 'NoSuchEntity':
            raise
    
    # Create role
    print(f"  Creating role: {role_name}")
    response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(trust_policy),
        Tags=[
            {'Key': 'Project', 'Value': 'NUBLE'},
            {'Key': 'Environment', 'Value': ENVIRONMENT}
        ]
    )
    role_arn = response['Role']['Arn']
    
    # Attach policies
    policies = [
        'arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
        'arn:aws:iam::aws:policy/AmazonDynamoDBFullAccess',
        'arn:aws:iam::aws:policy/AmazonEventBridgeFullAccess'
    ]
    
    for policy_arn in policies:
        iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
        print(f"  Attached: {policy_arn.split('/')[-1]}")
    
    # Wait for role propagation
    print("  Waiting for role propagation...")
    time.sleep(10)
    
    print(f"  ✅ Role created: {role_arn}")
    return role_arn

def create_lambda_function(role_arn):
    """Create Lambda function for webhook handling."""
    print_step(3, "Creating Lambda Webhook Function")
    
    lambda_client = boto3.client('lambda', region_name=REGION)
    function_name = f"{PROJECT_NAME}-{ENVIRONMENT}-webhook"
    
    # Check if function exists
    try:
        lambda_client.get_function(FunctionName=function_name)
        print(f"  ✅ Function {function_name} already exists")
        return lambda_client.get_function(FunctionName=function_name)['Configuration']['FunctionArn']
    except ClientError as e:
        if e.response['Error']['Code'] != 'ResourceNotFoundException':
            raise
    
    # Create inline code
    handler_code = '''
import json
import os
import time
import hashlib
from datetime import datetime, timezone
from decimal import Decimal

import boto3

SIGNALS_TABLE = os.environ.get('DYNAMODB_SIGNALS_TABLE', 'nuble-production-signals')
dynamodb = boto3.resource('dynamodb')
signals_table = dynamodb.Table(SIGNALS_TABLE)

def lambda_handler(event, context):
    try:
        # Parse body
        body = event.get('body', '{}')
        if isinstance(body, str):
            signal_data = json.loads(body)
        else:
            signal_data = body
        
        # Extract signal info
        symbol = signal_data.get('symbol', '').upper().strip()
        action = signal_data.get('action', '').upper().strip()
        timeframe = signal_data.get('timeframe', '4h')
        price = Decimal(str(signal_data.get('price', 0)))
        confidence = int(signal_data.get('confirmations', signal_data.get('confidence', 50)))
        
        if not symbol or action not in ['BUY', 'SELL', 'HOLD', 'CLOSE']:
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': 'Invalid signal data'})
            }
        
        # Generate signal ID
        now = datetime.now(timezone.utc)
        signal_id = hashlib.sha256(f"{symbol}:{timeframe}:{action}:{int(time.time()/60)}".encode()).hexdigest()[:16]
        
        # Store in DynamoDB
        item = {
            'pk': f"SIGNAL#{symbol}",
            'sk': f"{timeframe}#{now.strftime('%Y%m%d%H%M%S')}#{signal_id}",
            'signal_id': signal_id,
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'price': price,
            'confidence': confidence,
            'source': signal_data.get('source', 'tradingview'),
            'signal_type': signal_data.get('signal_type', 'LuxAlgo'),
            'timestamp': int(time.time() * 1000),
            'ttl': int(time.time()) + (7 * 24 * 3600),
            'gsi1pk': f"TF#{timeframe}",
            'gsi1sk': f"{symbol}#{now.strftime('%Y%m%d%H%M%S')}",
            'raw_data': json.dumps(signal_data)
        }
        
        signals_table.put_item(Item=item)
        
        print(f"Signal stored: {symbol} {action} ({timeframe}) - ID: {signal_id}")
        
        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({
                'success': True,
                'signal_id': signal_id,
                'symbol': symbol,
                'action': action,
                'timeframe': timeframe,
                'message': f'{action} signal for {symbol} received'
            })
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
            'body': json.dumps({'error': str(e)})
        }
'''
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('handler.py', handler_code)
    zip_buffer.seek(0)
    
    print(f"  Creating function: {function_name}")
    
    response = lambda_client.create_function(
        FunctionName=function_name,
        Runtime='python3.11',
        Role=role_arn,
        Handler='handler.lambda_handler',
        Code={'ZipFile': zip_buffer.read()},
        Timeout=30,
        MemorySize=256,
        Environment={
            'Variables': {
                'DYNAMODB_SIGNALS_TABLE': SIGNALS_TABLE,
                'ENVIRONMENT': ENVIRONMENT
            }
        },
        Tags={
            'Project': 'NUBLE',
            'Environment': ENVIRONMENT
        }
    )
    
    function_arn = response['FunctionArn']
    print(f"  ✅ Function created: {function_arn}")
    
    # Wait for function to be active
    time.sleep(5)
    
    return function_arn

def create_api_gateway(lambda_arn):
    """Create API Gateway for webhook endpoint."""
    print_step(4, "Creating API Gateway")
    
    apigateway = boto3.client('apigatewayv2', region_name=REGION)
    lambda_client = boto3.client('lambda', region_name=REGION)
    api_name = f"{PROJECT_NAME}-{ENVIRONMENT}-webhook-api"
    
    # Check if API exists
    apis = apigateway.get_apis()
    for api in apis.get('Items', []):
        if api['Name'] == api_name:
            print(f"  ✅ API {api_name} already exists")
            return api['ApiEndpoint']
    
    # Create HTTP API
    print(f"  Creating API: {api_name}")
    
    api_response = apigateway.create_api(
        Name=api_name,
        ProtocolType='HTTP',
        CorsConfiguration={
            'AllowOrigins': ['*'],
            'AllowMethods': ['POST', 'GET', 'OPTIONS'],
            'AllowHeaders': ['Content-Type', 'X-Api-Key']
        },
        Tags={'Project': 'NUBLE', 'Environment': ENVIRONMENT}
    )
    
    api_id = api_response['ApiId']
    api_endpoint = api_response['ApiEndpoint']
    
    # Create Lambda integration
    integration = apigateway.create_integration(
        ApiId=api_id,
        IntegrationType='AWS_PROXY',
        IntegrationUri=lambda_arn,
        PayloadFormatVersion='2.0',
        TimeoutInMillis=30000
    )
    
    integration_id = integration['IntegrationId']
    
    # Create routes
    routes = [
        ('POST /webhook', 'Main webhook endpoint'),
        ('POST /webhook/luxalgo', 'LuxAlgo webhook'),
        ('POST /webhook/mtf', 'Multi-timeframe webhook'),
        ('GET /health', 'Health check')
    ]
    
    for route_key, desc in routes:
        apigateway.create_route(
            ApiId=api_id,
            RouteKey=route_key,
            Target=f'integrations/{integration_id}'
        )
        print(f"  Created route: {route_key}")
    
    # Create stage
    apigateway.create_stage(
        ApiId=api_id,
        StageName=ENVIRONMENT,
        AutoDeploy=True
    )
    
    # Add Lambda permission
    function_name = lambda_arn.split(':')[-1]
    account_id = os.getenv('AWS_ACCOUNT_ID')
    
    try:
        lambda_client.add_permission(
            FunctionName=function_name,
            StatementId=f'apigateway-{api_id}',
            Action='lambda:InvokeFunction',
            Principal='apigateway.amazonaws.com',
            SourceArn=f'arn:aws:execute-api:{REGION}:{account_id}:{api_id}/*'
        )
    except ClientError as e:
        if 'already exists' not in str(e):
            raise
    
    webhook_url = f"{api_endpoint}/{ENVIRONMENT}/webhook"
    print(f"  ✅ API Gateway created!")
    print(f"\n  📡 WEBHOOK URL: {webhook_url}")
    
    return webhook_url

def test_webhook(webhook_url):
    """Send a test webhook to verify everything works."""
    print_step(5, "Testing Webhook")
    
    import requests
    
    test_signal = {
        "action": "BUY",
        "symbol": "BTCUSD",
        "price": 95000,
        "timeframe": "4h",
        "signal_type": "Test Signal",
        "confirmations": 8,
        "source": "deployment_test"
    }
    
    print(f"  Sending test signal to: {webhook_url}")
    print(f"  Payload: {json.dumps(test_signal)}")
    
    try:
        response = requests.post(webhook_url, json=test_signal, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"  ✅ Test successful!")
            print(f"  Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"  ❌ Test failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"  ❌ Test error: {e}")
        return False

def print_tradingview_config(webhook_url):
    """Print TradingView configuration."""
    print("\n" + "="*60)
    print("  🎯 TRADINGVIEW CONFIGURATION")
    print("="*60)
    
    print(f"""
  Your webhook is ready! Configure TradingView alerts:

  ╔══════════════════════════════════════════════════════════════╗
  ║  WEBHOOK URL (copy this):                                   ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  {webhook_url}
  ╚══════════════════════════════════════════════════════════════╝

  ALERT MESSAGE TEMPLATE (copy this for LuxAlgo):
  ──────────────────────────────────────────────
  {{
    "action": "BUY",
    "symbol": "{{{{ticker}}}}",
    "exchange": "{{{{exchange}}}}",
    "price": {{{{close}}}},
    "timeframe": "{{{{interval}}}}",
    "signal_type": "Bullish Confirmation",
    "confirmations": 8,
    "trend_strength": 65,
    "time": "{{{{time}}}}"
  }}

  For SELL signals, change:
  - "action": "SELL"
  - "signal_type": "Bearish Confirmation"

  RECOMMENDED SYMBOLS:
  ──────────────────────────────────────────────
  Crypto: BTCUSD, ETHUSD, SOLUSD
  Stocks: AAPL, NVDA, TSLA, AMD, MSFT

  RECOMMENDED TIMEFRAMES:
  ──────────────────────────────────────────────
  • 4h - Primary (best signal quality)
  • 1D - Daily confirmations
  • 1W - Weekly trend direction

  SETUP IN TRADINGVIEW:
  ──────────────────────────────────────────────
  1. Open chart with LuxAlgo indicator
  2. Right-click on chart → Add Alert
  3. Set Condition: LuxAlgo confirmation signals
  4. Enable "Webhook URL" and paste the URL above
  5. Set Alert Message to the template above
  6. Create alert for each symbol/timeframe combo
""")

def main():
    """Run the deployment."""
    print("\n" + "🚀"*30)
    print("     NUBLE AWS QUICK DEPLOYMENT")
    print("🚀"*30)
    print(f"\n  Project: {PROJECT_NAME}")
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Region: {REGION}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Create DynamoDB table
        if not create_dynamodb_table():
            return 1
        
        # Step 2: Create IAM role
        role_arn = create_lambda_role()
        if not role_arn:
            return 1
        
        # Step 3: Create Lambda function
        lambda_arn = create_lambda_function(role_arn)
        if not lambda_arn:
            return 1
        
        # Step 4: Create API Gateway
        webhook_url = create_api_gateway(lambda_arn)
        if not webhook_url:
            return 1
        
        # Step 5: Test the webhook
        test_webhook(webhook_url)
        
        # Print TradingView config
        print_tradingview_config(webhook_url)
        
        print("\n" + "="*60)
        print("  ✅ DEPLOYMENT COMPLETE!")
        print("="*60)
        print(f"\n  Webhook URL: {webhook_url}")
        print("\n  Run 'python test_connections.py' to verify everything.\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
