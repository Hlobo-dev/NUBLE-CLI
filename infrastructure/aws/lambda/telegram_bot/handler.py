"""
KYPERIAN ADVISOR - TELEGRAM BOT HANDLER
=========================================

Handles incoming Telegram messages for conversational AI interaction.
This is the chat interface to the KYPERIAN ADVISOR.

Features:
- Natural language Q&A about portfolio
- Quick commands for instant info
- Trade confirmations
- Alert acknowledgments

Commands:
/start      - Welcome message
/help       - Show commands
/status     - System status
/portfolio  - Portfolio overview
/signals    - Current signals
/analyze    - Analyze a symbol
/digest     - Daily digest
/vix        - Current VIX

Author: KYPERIAN ELITE
Version: 5.0.0
"""

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import Dict, Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY', '')
ADVISOR_API_URL = os.environ.get('ADVISOR_API_URL', '')

OWNER_NAME = "Humberto"


def send_telegram_message(chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            return True
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
        return False


def send_typing_action(chat_id: str):
    """Send typing indicator"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendChatAction"
        data = {"chat_id": chat_id, "action": "typing"}
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except:
        pass


def call_advisor_api(endpoint: str, method: str = "GET", body: Dict = None) -> Dict:
    """Call the KYPERIAN ADVISOR API"""
    if not ADVISOR_API_URL:
        return {"error": "ADVISOR_API_URL not configured"}
    
    try:
        url = f"{ADVISOR_API_URL}{endpoint}"
        
        if method == "GET":
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url,
                data=json.dumps(body or {}).encode(),
                headers={"Content-Type": "application/json"},
                method=method,
            )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    
    except Exception as e:
        return {"error": str(e)}


def call_claude(prompt: str) -> str:
    """Call Claude for natural language processing"""
    if not ANTHROPIC_API_KEY:
        return "I'm having trouble connecting to my brain right now. Try again shortly!"
    
    try:
        url = "https://api.anthropic.com/v1/messages"
        
        system = f"""You are KYPERIAN, an elite AI wealth advisor for {OWNER_NAME}.
You're chatting via Telegram, so keep responses concise but helpful.
Use emojis where appropriate but don't overdo it.
If asked about specific trades or portfolio, provide actionable advice.
Current date: {datetime.now(timezone.utc).strftime('%B %d, %Y')}
"""
        
        data = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(data).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result["content"][0]["text"]
    
    except Exception as e:
        logger.error(f"Claude error: {e}")
        return f"I'm having a moment. Error: {e}"


def get_vix() -> tuple:
    """Get current VIX"""
    if not POLYGON_API_KEY:
        return 20.0, "NORMAL"
    
    try:
        url = f"https://api.polygon.io/v2/aggs/ticker/VIX/prev?apiKey={POLYGON_API_KEY}"
        req = urllib.request.Request(url, headers={"User-Agent": "KYPERIAN"})
        
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode())
            if data.get("results"):
                vix = data["results"][0].get("c", 20)
                if vix < 15:
                    state = "LOW"
                elif vix < 20:
                    state = "NORMAL"
                elif vix < 25:
                    state = "ELEVATED"
                elif vix < 30:
                    state = "HIGH"
                else:
                    state = "EXTREME"
                return vix, state
    except:
        pass
    
    return 20.0, "NORMAL"


def handle_command(command: str, args: str, chat_id: str) -> str:
    """Handle bot commands"""
    
    if command == "start":
        return f"""👋 Hey {OWNER_NAME}!

I'm <b>KYPERIAN</b>, your AI wealth advisor.

I'm here to help you make smarter trading decisions. I monitor markets 24/7, analyze signals, and keep you informed.

<b>Quick commands:</b>
/status - System status
/signals - Current tradeable signals
/analyze AAPL - Analyze a symbol
/vix - Current volatility
/digest - Daily summary
/help - All commands

Or just ask me anything about markets or your portfolio!

Let's build some wealth 📈"""

    elif command == "help":
        return """🤖 <b>KYPERIAN Commands</b>

<b>Quick Info:</b>
/status - System status
/vix - Current VIX level
/signals - Tradeable signals now

<b>Analysis:</b>
/analyze {SYMBOL} - Deep analysis
/check {SYMBOL} - Quick check
/digest - End of day summary

<b>Portfolio:</b>
/portfolio - Portfolio overview
/risk - Risk assessment

<b>Or just chat with me!</b>
Ask anything like:
• "Should I buy NVDA?"
• "What do you think about BTC?"
• "Is now a good time to be in the market?"

I'm powered by Claude AI and have real-time market data 🧠"""

    elif command == "status":
        vix, vix_state = get_vix()
        mode = "🛡️ DEFENSIVE" if vix > 30 else "✅ NORMAL"
        
        return f"""🤖 <b>KYPERIAN ADVISOR Status</b>

<b>Version:</b> 5.0.0
<b>Status:</b> ✅ Operational
<b>Mode:</b> {mode}

<b>Market Conditions:</b>
• VIX: {vix:.1f} ({vix_state})
• Market Hours: {'Open' if 9 <= datetime.now().hour < 16 else 'Closed'}

<b>Integrations:</b>
• Polygon.io: {'✅' if POLYGON_API_KEY else '❌'}
• Claude AI: {'✅' if ANTHROPIC_API_KEY else '❌'}
• Telegram: ✅

I'm monitoring 12 symbols across crypto, indices, and mega-cap stocks."""

    elif command == "vix":
        vix, state = get_vix()
        
        if state == "LOW":
            emoji = "😎"
            comment = "Markets are calm. Good conditions for trading."
        elif state == "NORMAL":
            emoji = "✅"
            comment = "Normal conditions. Proceed with standard risk."
        elif state == "ELEVATED":
            emoji = "⚠️"
            comment = "Getting a bit spicy. Consider tighter stops."
        elif state == "HIGH":
            emoji = "🔶"
            comment = "High volatility. Reduce position sizes."
        else:
            emoji = "🚨"
            comment = "Extreme volatility! Cash is a position."
        
        return f"""📊 <b>VIX: {vix:.1f}</b> ({state}) {emoji}

{comment}

<i>VIX measures expected 30-day volatility.
15 = Low, 20 = Normal, 30+ = High, 40+ = Extreme</i>"""

    elif command == "signals":
        result = call_advisor_api("/dashboard")
        
        if result.get("error"):
            return f"❌ Error fetching signals: {result['error']}"
        
        tradeable = result.get("summary", {}).get("tradeable_symbols", [])
        quick_view = result.get("quick_view", [])[:8]
        
        response = f"""📡 <b>Current Signals</b>

<b>Tradeable Now:</b> {len(tradeable)}
"""
        
        if tradeable:
            response += "\n🔥 <b>ACTIONABLE:</b>\n"
            for sym in tradeable[:3]:
                response += f"• {sym}\n"
        
        response += "\n<b>All Signals:</b>\n"
        for line in quick_view:
            response += f"{line}\n"
        
        if tradeable:
            response += f"\n<i>Use /analyze {tradeable[0]} for details</i>"
        
        return response

    elif command in ["analyze", "check"]:
        if not args:
            return "Usage: /analyze SYMBOL\n\nExample: /analyze AAPL"
        
        symbol = args.upper().strip()
        result = call_advisor_api(f"/analyze/{symbol}")
        
        if result.get("error"):
            return f"❌ Error: {result['error']}"
        
        a = result.get("analysis", {})
        
        direction_emoji = "📈" if a.get("direction") == "BUY" else "📉" if a.get("direction") == "SELL" else "➡️"
        trade_emoji = "🔥 TRADEABLE" if a.get("should_trade") else "⏸️ WAITING"
        
        response = f"""📊 <b>{a.get('symbol', symbol)} Analysis</b>

{direction_emoji} <b>{a.get('direction', 'N/A')}</b> | {a.get('strength', 'N/A')}
Confidence: <b>{a.get('confidence', 0):.1f}%</b>
Status: {trade_emoji}

<b>LuxAlgo Signals:</b>
• Weekly: {a.get('luxalgo', {}).get('weekly', {}).get('action', 'N/A')}
• Daily: {a.get('luxalgo', {}).get('daily', {}).get('action', 'N/A')}
• 4H: {a.get('luxalgo', {}).get('h4', {}).get('action', 'N/A')}
• Aligned: {'✅' if a.get('luxalgo', {}).get('aligned') else '❌'}

<b>Regime:</b> {a.get('regime', 'Unknown')} | VIX: {a.get('vix', 20):.1f}

<b>Technical:</b>
"""
        
        if a.get('market_data', {}).get('available'):
            md = a['market_data']
            response += f"• RSI: {md.get('rsi', 50):.1f}\n"
            response += f"• 1D Change: {md.get('change_1d', 0):.2f}%\n"
        
        if a.get('trade_setup'):
            ts = a['trade_setup']
            response += f"""
<b>Trade Setup:</b>
• Entry: ${ts['entry']:,.2f}
• Stop: ${ts['stop_loss']:,.2f} ({ts['stop_pct']:.1f}%)
• Target 1: ${ts['targets'][0]:,.2f}
• Size: {ts['position_pct']:.1f}% of portfolio
"""
        
        if a.get('veto'):
            response += f"\n⚠️ <b>VETO:</b> {a.get('veto_reason', 'Unknown')}"
        
        return response

    elif command == "digest":
        result = call_advisor_api("/digest")
        
        if result.get("error"):
            return f"❌ Error: {result['error']}"
        
        return result.get("digest", "Could not generate digest")

    elif command == "portfolio":
        return """📂 <b>Portfolio View</b>

<i>Portfolio integration coming soon!</i>

When connected, I'll show you:
• Current positions & P&L
• Risk exposure
• Asset allocation
• Performance vs benchmarks

For now, ask me about specific symbols or market conditions!"""

    elif command == "risk":
        vix, vix_state = get_vix()
        
        return f"""🛡️ <b>Risk Assessment</b>

<b>Market Volatility:</b>
• VIX: {vix:.1f} ({vix_state})
• Risk Level: {'🟢 Low' if vix < 20 else '🟡 Medium' if vix < 30 else '🔴 High'}

<b>Trading Rules Active:</b>
• Max Position: 8%
• Default Stop: 2.5%
• VIX > 30: Defensive mode
• VIX > 40: No new trades

<b>Current Mode:</b> {'🛡️ DEFENSIVE' if vix > 30 else '✅ NORMAL'}

<i>I adjust position sizing and stops based on current volatility.</i>"""

    else:
        return f"Unknown command: /{command}\n\nType /help for available commands."


def handle_message(message: str, chat_id: str) -> str:
    """Handle natural language messages"""
    
    # Check for quick patterns
    message_lower = message.lower()
    
    # Symbol lookup pattern
    if len(message) <= 5 and message.upper().isalpha():
        symbol = message.upper()
        result = call_advisor_api(f"/analyze/{symbol}")
        if not result.get("error"):
            # Return quick summary
            a = result.get("analysis", {})
            return f"""<b>{symbol}</b>: {a.get('direction', 'N/A')} ({a.get('strength', 'N/A')})
Confidence: {a.get('confidence', 0):.1f}%
Tradeable: {'✅' if a.get('should_trade') else '❌'}

<i>/analyze {symbol} for full details</i>"""
    
    # Get context for Claude
    signals_result = call_advisor_api("/dashboard")
    vix, vix_state = get_vix()
    
    context = f"""
User asked: "{message}"

Current market context:
- VIX: {vix} ({vix_state})
- Tradeable signals: {signals_result.get('summary', {}).get('tradeable_symbols', [])}
- Quick view: {signals_result.get('quick_view', [])[:5]}

Respond helpfully and concisely. If they're asking about a specific stock or trade, give specific recommendations.
"""
    
    return call_claude(context)


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle Telegram webhook events"""
    
    try:
        logger.info(f"Event: {json.dumps(event)[:500]}")
        
        # Parse body
        if 'body' in event:
            try:
                body = json.loads(event['body'])
            except:
                body = event
        else:
            body = event
        
        # Extract message
        message_data = body.get('message', {})
        
        if not message_data:
            return {'statusCode': 200, 'body': 'No message'}
        
        chat_id = str(message_data.get('chat', {}).get('id', ''))
        text = message_data.get('text', '')
        
        if not chat_id or not text:
            return {'statusCode': 200, 'body': 'Missing chat_id or text'}
        
        # Send typing indicator
        send_typing_action(chat_id)
        
        # Handle commands
        if text.startswith('/'):
            parts = text[1:].split(maxsplit=1)
            command = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            response = handle_command(command, args, chat_id)
        else:
            # Natural language
            response = handle_message(text, chat_id)
        
        # Send response
        send_telegram_message(chat_id, response)
        
        return {'statusCode': 200, 'body': 'OK'}
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        return {'statusCode': 200, 'body': f'Error: {e}'}


if __name__ == "__main__":
    # Test
    print(handle_command("start", "", "test"))
    print(handle_command("vix", "", "test"))
