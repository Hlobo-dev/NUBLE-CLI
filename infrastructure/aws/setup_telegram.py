#!/usr/bin/env python3
"""
NUBLE ELITE - Telegram Bot Setup Helper
==========================================
This script helps you set up Telegram notifications for signal alerts.

Usage:
    python3 setup_telegram.py

Steps:
1. Create a bot with @BotFather on Telegram
2. Get your chat ID
3. Test the connection
4. Update AWS Lambda with credentials
"""

import json
import urllib.request
import urllib.error
import subprocess
import os
import sys

# Colors for terminal
class Colors:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header():
    print(f"""
{Colors.CYAN}╔════════════════════════════════════════════════════════════╗
║         NUBLE ELITE - Telegram Setup Helper              ║
╚════════════════════════════════════════════════════════════╝{Colors.END}
""")

def print_step(num, text):
    print(f"\n{Colors.BOLD}{Colors.CYAN}Step {num}: {text}{Colors.END}")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def send_telegram_message(token: str, chat_id: str, message: str) -> bool:
    """Send a test message to Telegram."""
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        data = json.dumps({
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'HTML',
        }).encode('utf-8')
        
        req = urllib.request.Request(
            url,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read())
            return result.get('ok', False)
            
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def get_chat_id_from_updates(token: str) -> str:
    """Get chat ID from recent updates."""
    try:
        url = f"https://api.telegram.org/bot{token}/getUpdates"
        
        with urllib.request.urlopen(url, timeout=10) as response:
            result = json.loads(response.read())
            
        if result.get('ok') and result.get('result'):
            for update in result['result']:
                if 'message' in update:
                    chat_id = update['message']['chat']['id']
                    chat_name = update['message']['chat'].get('first_name', 'Unknown')
                    print(f"  Found chat: {chat_name} (ID: {chat_id})")
                    return str(chat_id)
        
        return None
        
    except Exception as e:
        print_error(f"Error getting updates: {e}")
        return None

def update_lambda_env(token: str, chat_id: str) -> bool:
    """Update Lambda environment variables."""
    try:
        # Get current configuration
        result = subprocess.run([
            'aws', 'lambda', 'get-function-configuration',
            '--function-name', 'nuble-production-decision-engine',
            '--region', 'us-east-1',
            '--profile', 'nuble',
            '--output', 'json',
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print_error(f"Error getting Lambda config: {result.stderr}")
            return False
        
        config = json.loads(result.stdout)
        env_vars = config.get('Environment', {}).get('Variables', {})
        
        # Update with Telegram credentials
        env_vars['TELEGRAM_BOT_TOKEN'] = token
        env_vars['TELEGRAM_CHAT_ID'] = chat_id
        
        # Update Lambda
        result = subprocess.run([
            'aws', 'lambda', 'update-function-configuration',
            '--function-name', 'nuble-production-decision-engine',
            '--environment', f'Variables={json.dumps(env_vars)}',
            '--region', 'us-east-1',
            '--profile', 'nuble',
            '--output', 'json',
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print_error(f"Error updating Lambda: {result.stderr}")
            return False
        
        return True
        
    except Exception as e:
        print_error(f"Error: {e}")
        return False

def main():
    print_header()
    
    # Step 1: Create bot
    print_step(1, "Create a Telegram Bot")
    print("""
    1. Open Telegram and search for @BotFather
    2. Send /newbot command
    3. Follow the prompts to create your bot
    4. Copy the bot token (looks like: 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)
    """)
    
    token = input(f"{Colors.BOLD}Enter your bot token: {Colors.END}").strip()
    
    if not token or ':' not in token:
        print_error("Invalid token format. Should be like: 123456789:ABCdefGHI...")
        sys.exit(1)
    
    # Step 2: Get chat ID
    print_step(2, "Get Your Chat ID")
    print("""
    1. Open Telegram and find your new bot
    2. Send it any message (like "hello")
    3. Press Enter here to detect your chat ID
    """)
    
    input(f"{Colors.BOLD}Press Enter after sending a message to your bot...{Colors.END}")
    
    chat_id = get_chat_id_from_updates(token)
    
    if not chat_id:
        print_warning("Could not auto-detect chat ID. Please enter it manually.")
        print("  You can get it by visiting: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
        chat_id = input(f"{Colors.BOLD}Enter your chat ID: {Colors.END}").strip()
    
    if not chat_id:
        print_error("Chat ID is required")
        sys.exit(1)
    
    # Step 3: Test connection
    print_step(3, "Testing Connection")
    
    test_message = """
🚀 <b>NUBLE ELITE</b> - Connection Test

✅ Telegram notifications are now configured!

You will receive alerts when:
• All timeframes align (Weekly + Daily + 4H)
• High-confidence trading setups appear
• Veto conditions are triggered

🔔 System is ready for trading signals!
"""
    
    if send_telegram_message(token, chat_id, test_message):
        print_success("Test message sent successfully! Check your Telegram.")
    else:
        print_error("Failed to send test message. Please check your token and chat ID.")
        sys.exit(1)
    
    # Step 4: Update Lambda
    print_step(4, "Updating AWS Lambda")
    
    update = input(f"{Colors.BOLD}Update Lambda with Telegram credentials? (y/n): {Colors.END}").strip().lower()
    
    if update == 'y':
        if update_lambda_env(token, chat_id):
            print_success("Lambda environment variables updated!")
        else:
            print_warning("Could not update Lambda automatically.")
            print("\n  Run this command manually:")
            print(f"""
  aws lambda update-function-configuration \\
      --function-name nuble-production-decision-engine \\
      --environment 'Variables={{TELEGRAM_BOT_TOKEN={token},TELEGRAM_CHAT_ID={chat_id}}}' \\
      --region us-east-1 \\
      --profile nuble
            """)
    
    # Summary
    print(f"""
{Colors.GREEN}╔════════════════════════════════════════════════════════════╗
║                    SETUP COMPLETE!                          ║
╚════════════════════════════════════════════════════════════╝{Colors.END}

{Colors.BOLD}Your Telegram Credentials:{Colors.END}
  Bot Token: {token[:20]}...
  Chat ID:   {chat_id}

{Colors.BOLD}What Happens Now:{Colors.END}
  • The Decision Engine runs every 5 minutes
  • When signals align (1W + 1D + 4H same direction)
  • You'll receive a Telegram notification with:
    - Symbol and direction
    - Confidence level
    - Entry, stop loss, and target prices
    - Position size recommendation

{Colors.BOLD}Test It:{Colors.END}
  curl https://9vyvetp9c7.execute-api.us-east-1.amazonaws.com/production/check/BTCUSD

{Colors.CYAN}Happy Trading! 🚀{Colors.END}
""")

if __name__ == '__main__':
    main()
