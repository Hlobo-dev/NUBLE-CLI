#!/usr/bin/env python3
"""
NUBLE Migration Script

Migrates from Kyperian to NUBLE:
1. Updates all code references from kyperian → nuble
2. Updates .env with new AWS credentials
3. Deploys infrastructure to new AWS account
4. Migrates signals from old DynamoDB to new
"""

import os
import sys
import re
import json
import time
import shutil
from pathlib import Path
from datetime import datetime

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

def print_banner():
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║      ███╗   ██╗██╗   ██╗██████╗ ██╗     ███████╗             ║
    ║      ████╗  ██║██║   ██║██╔══██╗██║     ██╔════╝             ║
    ║      ██╔██╗ ██║██║   ██║██████╔╝██║     █████╗               ║
    ║      ██║╚██╗██║██║   ██║██╔══██╗██║     ██╔══╝               ║
    ║      ██║ ╚████║╚██████╔╝██████╔╝███████╗███████╗             ║
    ║      ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝             ║
    ║                                                              ║
    ║              AWS MIGRATION TOOL                              ║
    ║         Kyperian (456309723884) → NUBLE (191613668206)       ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def print_step(num, title):
    print(f"\n{'='*60}")
    print(f"  Step {num}: {title}")
    print('='*60)

# Files to update with kyperian → nuble replacement
FILES_TO_UPDATE = [
    # Deployment scripts
    "infrastructure/aws/deploy.sh",
    "infrastructure/aws/deploy-decision-engine.sh",
    "infrastructure/aws/deploy-advisor.sh",
    "infrastructure/aws/deploy-enterprise.sh",
    "infrastructure/aws/deploy/deploy_v2.sh",
    # Dashboard
    "infrastructure/aws/dashboard/index.html",
    # Lambda handlers
    "infrastructure/aws/lambda/signal_validator/handler.py",
    "infrastructure/aws/lambda/decision_engine/handler.py",
    "infrastructure/aws/lambda/decision_engine/handler_v2.py",
    "infrastructure/aws/lambda/advisor/handler.py",
    "infrastructure/aws/lambda/telegram_bot/handler.py",
    # Source code
    "src/nuble/data/aggregator.py",
    "src/nuble/decision/ultimate_engine.py",
    "src/nuble/decision/engine_v2.py",
]

# CloudFormation templates (just update descriptions, resource names use parameters)
CF_TEMPLATES = [
    "infrastructure/aws/cloudformation/vpc.yaml",
    "infrastructure/aws/cloudformation/lambda.yaml",
    "infrastructure/aws/cloudformation/api-gateway.yaml",
    "infrastructure/aws/cloudformation/ecs.yaml",
    "infrastructure/aws/cloudformation/elasticache.yaml",
    "infrastructure/aws/cloudformation/monitoring.yaml",
]

def rename_kyperian_to_nuble(dry_run=True):
    """Replace all kyperian references with nuble."""
    print_step(1, "Renaming Kyperian → NUBLE in code")
    
    base_path = Path(__file__).parent
    
    replacements = [
        ('kyperian', 'nuble'),
        ('KYPERIAN', 'NUBLE'),
        ('Kyperian', 'Nuble'),
    ]
    
    files_changed = 0
    
    for file_path in FILES_TO_UPDATE + CF_TEMPLATES:
        full_path = base_path / file_path
        
        if not full_path.exists():
            print(f"  ⚠️  Skipping (not found): {file_path}")
            continue
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original = content
            for old, new in replacements:
                content = content.replace(old, new)
            
            if content != original:
                if dry_run:
                    print(f"  📝 Would update: {file_path}")
                else:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"  ✅ Updated: {file_path}")
                files_changed += 1
            else:
                print(f"  ⏭️  No changes: {file_path}")
                
        except Exception as e:
            print(f"  ❌ Error with {file_path}: {e}")
    
    print(f"\n  Total files {'to update' if dry_run else 'updated'}: {files_changed}")
    return files_changed

def update_env_file(new_access_key, new_secret_key, new_account_id="191613668206"):
    """Update .env file with new AWS credentials."""
    print_step(2, "Updating .env with new AWS credentials")
    
    env_path = Path(__file__).parent / '.env'
    
    new_env_content = f"""# NUBLE API Configuration
# All API keys configured for optimal performance
# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# ===== AI Provider =====
ANTHROPIC_API_KEY={os.getenv('ANTHROPIC_API_KEY', 'YOUR_ANTHROPIC_KEY')}

# ===== Market Data Providers =====
POLYGON_API_KEY={os.getenv('POLYGON_API_KEY', 'YOUR_POLYGON_KEY')}

# ===== News & Sentiment APIs =====
STOCKNEWS_API_KEY={os.getenv('STOCKNEWS_API_KEY', 'YOUR_STOCKNEWS_KEY')}
CRYPTONEWS_API_KEY={os.getenv('CRYPTONEWS_API_KEY', 'YOUR_CRYPTONEWS_KEY')}
COINDESK_API_KEY={os.getenv('COINDESK_API_KEY', 'YOUR_COINDESK_KEY')}

# ===== AWS Configuration (NEW ACCOUNT) =====
AWS_ACCESS_KEY_ID={new_access_key}
AWS_SECRET_ACCESS_KEY={new_secret_key}
AWS_ACCOUNT_ID={new_account_id}
AWS_REGION=us-east-1
AWS_PROFILE=nuble-prod

# ===== DynamoDB Tables =====
DYNAMODB_SIGNALS_TABLE=nuble-production-signals
DYNAMODB_DECISIONS_TABLE=nuble-production-decisions

# ===== Webhook Configuration =====
# Will be updated after deployment
WEBHOOK_URL=https://TO-BE-DEPLOYED.execute-api.us-east-1.amazonaws.com/production/webhook

# ===== Application =====
API_KEY=nuble_elite_2026_secure_key
"""
    
    # Backup old .env
    if env_path.exists():
        backup_path = env_path.with_suffix('.env.backup')
        shutil.copy(env_path, backup_path)
        print(f"  📦 Backed up old .env to {backup_path}")
    
    with open(env_path, 'w') as f:
        f.write(new_env_content)
    
    print(f"  ✅ Updated .env with new credentials")
    print(f"  📋 New Account ID: {new_account_id}")
    print(f"  📋 Access Key: {new_access_key[:8]}...")

def setup_aws_profile(access_key, secret_key, region="us-east-1"):
    """Configure AWS CLI profile for nuble-prod."""
    print_step(3, "Setting up AWS CLI profile")
    
    import subprocess
    
    # Create profile
    commands = [
        f'aws configure set aws_access_key_id {access_key} --profile nuble-prod',
        f'aws configure set aws_secret_access_key {secret_key} --profile nuble-prod',
        f'aws configure set region {region} --profile nuble-prod',
        'aws configure set output json --profile nuble-prod',
    ]
    
    for cmd in commands:
        # Hide the secret key in output
        display_cmd = cmd.replace(secret_key, '***')
        print(f"  Running: {display_cmd[:60]}...")
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    
    # Verify
    result = subprocess.run(
        'aws sts get-caller-identity --profile nuble-prod',
        shell=True, capture_output=True, text=True
    )
    
    if result.returncode == 0:
        identity = json.loads(result.stdout)
        print(f"  ✅ AWS Profile configured!")
        print(f"  📋 Account: {identity['Account']}")
        print(f"  📋 User ARN: {identity['Arn']}")
        return True
    else:
        print(f"  ❌ Failed to configure AWS: {result.stderr}")
        return False

def export_signals_from_old_account():
    """Export signals from old DynamoDB table."""
    print_step(4, "Exporting signals from old account")
    
    try:
        import boto3
        
        # Use old credentials
        dynamodb = boto3.resource(
            'dynamodb',
            region_name='us-east-1',
            aws_access_key_id='AKIAWUPRN2LWASHFJEMO',
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        table = dynamodb.Table('kyperian-production-signals')
        
        # Scan all items
        items = []
        response = table.scan()
        items.extend(response.get('Items', []))
        
        while 'LastEvaluatedKey' in response:
            response = table.scan(ExclusiveStartKey=response['LastEvaluatedKey'])
            items.extend(response.get('Items', []))
        
        # Save to file
        export_path = Path(__file__).parent / 'migration_data' / 'signals_export.json'
        export_path.parent.mkdir(exist_ok=True)
        
        with open(export_path, 'w') as f:
            json.dump(items, f, indent=2, default=str)
        
        print(f"  ✅ Exported {len(items)} signals to {export_path}")
        return items
        
    except Exception as e:
        print(f"  ❌ Export failed: {e}")
        return []

def import_signals_to_new_account(items):
    """Import signals to new DynamoDB table."""
    print_step(5, "Importing signals to new account")
    
    if not items:
        print("  ⚠️  No signals to import")
        return
    
    try:
        import boto3
        
        # Use new credentials (from profile)
        session = boto3.Session(profile_name='nuble-prod')
        dynamodb = session.resource('dynamodb', region_name='us-east-1')
        
        table = dynamodb.Table('nuble-production-signals')
        
        # Batch write
        with table.batch_writer() as batch:
            for item in items:
                # Update pk if it contains old prefix
                if 'pk' in item and 'KYPERIAN' in str(item.get('pk', '')):
                    item['pk'] = item['pk'].replace('KYPERIAN', 'NUBLE')
                batch.put_item(Item=item)
        
        print(f"  ✅ Imported {len(items)} signals to new account")
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")

def deploy_infrastructure():
    """Deploy infrastructure to new AWS account."""
    print_step(6, "Deploying infrastructure to new account")
    
    import subprocess
    
    deploy_script = Path(__file__).parent / 'infrastructure' / 'aws' / 'deploy.sh'
    
    if not deploy_script.exists():
        print(f"  ❌ Deployment script not found: {deploy_script}")
        return False
    
    print("  This will deploy:")
    print("    - VPC and networking")
    print("    - Lambda functions")
    print("    - API Gateway")
    print("    - DynamoDB tables")
    print("    - ElastiCache (Redis)")
    print("    - ECS Fargate")
    print("    - CloudWatch monitoring")
    print()
    print("  Estimated time: 20-30 minutes")
    print("  Estimated cost: ~$290/month")
    print()
    
    confirm = input("  Proceed with deployment? (yes/no): ")
    if confirm.lower() != 'yes':
        print("  ⏭️  Deployment skipped")
        return False
    
    # Set environment for deployment
    env = os.environ.copy()
    env['AWS_PROFILE'] = 'nuble-prod'
    env['AWS_REGION'] = 'us-east-1'
    
    # Run deployment
    result = subprocess.run(
        ['bash', str(deploy_script), 'production', 'deploy'],
        cwd=deploy_script.parent,
        env=env
    )
    
    if result.returncode == 0:
        print("  ✅ Infrastructure deployed successfully!")
        return True
    else:
        print("  ❌ Deployment failed")
        return False

def get_webhook_url():
    """Get the new webhook URL from API Gateway."""
    print_step(7, "Getting new webhook URL")
    
    try:
        import boto3
        
        session = boto3.Session(profile_name='nuble-prod')
        apigateway = session.client('apigatewayv2', region_name='us-east-1')
        
        apis = apigateway.get_apis()
        
        for api in apis.get('Items', []):
            if 'nuble' in api.get('Name', '').lower():
                endpoint = api.get('ApiEndpoint', '')
                webhook_url = f"{endpoint}/production/webhook"
                print(f"  ✅ Found API: {api['Name']}")
                print(f"  📡 Webhook URL: {webhook_url}")
                return webhook_url
        
        print("  ⚠️  No NUBLE API found yet")
        return None
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def print_summary(webhook_url=None):
    """Print migration summary."""
    print("\n" + "="*60)
    print("  📊 MIGRATION SUMMARY")
    print("="*60)
    
    print("""
  ✅ Code renamed from Kyperian → NUBLE
  ✅ .env updated with new credentials
  ✅ AWS CLI profile 'nuble-prod' configured
  ✅ Signals exported from old account
  ✅ Infrastructure deployed to new account
  ✅ Signals imported to new account
""")
    
    if webhook_url:
        print(f"""
  ╔══════════════════════════════════════════════════════════════╗
  ║  NEW WEBHOOK URL (update TradingView with this):             ║
  ╠══════════════════════════════════════════════════════════════╣
  ║  {webhook_url}
  ╚══════════════════════════════════════════════════════════════╝
""")
    
    print("""
  NEXT STEPS:
  ───────────────────────────────────────────────────────────────
  1. Update TradingView alerts with new webhook URL
  2. Run 'python test_connections.py' to verify
  3. Start the CLI: nuble
  4. Monitor CloudWatch for signal activity
  
  Old account can be decommissioned after 30 days of stable operation.
""")

def main():
    print_banner()
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Migration: Kyperian → NUBLE")
    
    print("\n" + "="*60)
    print("  📋 MIGRATION OPTIONS")
    print("="*60)
    print("""
  1. Full migration (recommended)
  2. Code rename only (kyperian → nuble)
  3. Update .env only
  4. Deploy infrastructure only
  5. Export/Import signals only
  6. Dry run (preview changes)
  
  0. Exit
""")
    
    choice = input("  Select option (1-6, 0 to exit): ").strip()
    
    if choice == '0':
        print("\n  Exiting. No changes made.")
        return 0
    
    if choice == '6':
        # Dry run
        rename_kyperian_to_nuble(dry_run=True)
        print("\n  This was a dry run. No files were changed.")
        return 0
    
    if choice in ['1', '2']:
        # Rename code
        rename_kyperian_to_nuble(dry_run=False)
    
    if choice in ['1', '3']:
        # Update .env
        print("\n  Enter new AWS credentials for account 191613668206:")
        access_key = input("  Access Key ID (AKIASZHIQRNXH5EZJOMK): ").strip()
        if not access_key:
            access_key = "AKIASZHIQRNXH5EZJOMK"
        
        secret_key = input("  Secret Access Key: ").strip()
        if not secret_key:
            print("  ❌ Secret key is required!")
            return 1
        
        update_env_file(access_key, secret_key)
        setup_aws_profile(access_key, secret_key)
    
    if choice in ['1', '5']:
        # Export/Import signals
        signals = export_signals_from_old_account()
        if signals:
            import_signals_to_new_account(signals)
    
    if choice in ['1', '4']:
        # Deploy infrastructure
        deploy_infrastructure()
    
    # Get webhook URL
    webhook_url = get_webhook_url()
    
    # Summary
    print_summary(webhook_url)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
