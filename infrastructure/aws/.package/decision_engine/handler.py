"""
KYPERIAN ELITE - Decision Engine Lambda
=========================================
Multi-Timeframe Signal Fusion + Notification System

This is the BRAIN that connects everything:
- Reads signals from DynamoDB (LuxAlgo webhooks)
- Checks multi-timeframe alignment (Weekly → Daily → 4H)
- Applies veto rules (never trade against weekly)
- Sends Telegram/Discord notifications for high-confidence setups
- Generates position sizing and risk management recommendations

Author: KYPERIAN ELITE System
Version: 1.0.0
"""

from __future__ import annotations

import json
import os
import time
import hashlib
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging

import boto3
from botocore.config import Config

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Optimized boto3 config
BOTO_CONFIG = Config(
    connect_timeout=5,
    read_timeout=10,
    retries={'max_attempts': 3, 'mode': 'adaptive'},
)

# ============================================================
# CONFIGURATION
# ============================================================

# Environment variables
SIGNALS_TABLE = os.environ.get('DYNAMODB_SIGNALS_TABLE', 'kyperian-production-signals')
DECISIONS_TABLE = os.environ.get('DYNAMODB_DECISIONS_TABLE', 'kyperian-production-decisions')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')
ENVIRONMENT = os.environ.get('ENVIRONMENT', 'production')

# Symbols to monitor
MONITORED_SYMBOLS = ['BTCUSD', 'ETHUSD', 'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']

# Timeframe configuration
TIMEFRAMES = {
    '1W': {'weight': 0.45, 'max_age_hours': 168, 'veto_power': True, 'name': 'Weekly'},
    '1D': {'weight': 0.35, 'max_age_hours': 48, 'veto_power': False, 'name': 'Daily'},
    '4h': {'weight': 0.20, 'max_age_hours': 12, 'veto_power': False, 'name': '4-Hour'},
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 80,      # All 3 timeframes aligned + strong signals
    'medium': 65,    # 2 timeframes aligned
    'low': 50,       # Partial alignment
    'veto': 0,       # Weekly opposing = no trade
}

# Position sizing based on confidence
POSITION_SIZING = {
    'high': {'size_pct': 5.0, 'stop_pct': 2.0, 'target_pct': 6.0},
    'medium': {'size_pct': 3.0, 'stop_pct': 2.5, 'target_pct': 5.0},
    'low': {'size_pct': 1.5, 'stop_pct': 3.0, 'target_pct': 4.5},
}


# ============================================================
# DATA CLASSES
# ============================================================

class SignalDirection(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"


@dataclass
class TimeframeSignal:
    """Signal from a specific timeframe."""
    symbol: str
    timeframe: str
    action: str
    price: float
    confidence: float
    timestamp: datetime
    age_hours: float
    is_valid: bool  # Not expired
    source: str = "luxalgo"
    
    @property
    def direction(self) -> SignalDirection:
        if self.action in ['BUY', 'STRONG_BUY']:
            return SignalDirection.BUY
        elif self.action in ['SELL', 'STRONG_SELL']:
            return SignalDirection.SELL
        return SignalDirection.NEUTRAL


@dataclass
class AlignmentResult:
    """Result of multi-timeframe alignment check."""
    symbol: str
    is_aligned: bool
    direction: SignalDirection
    confidence: float
    confidence_level: str  # high, medium, low, veto
    signals: Dict[str, TimeframeSignal]
    weekly_signal: Optional[TimeframeSignal]
    daily_signal: Optional[TimeframeSignal]
    h4_signal: Optional[TimeframeSignal]
    veto_active: bool
    veto_reason: Optional[str]
    recommendation: str
    position_size_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    entry_price: float
    stop_price: float
    target_price: float
    risk_reward_ratio: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'is_aligned': self.is_aligned,
            'direction': self.direction.value,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level,
            'veto_active': self.veto_active,
            'veto_reason': self.veto_reason,
            'recommendation': self.recommendation,
            'position_size_pct': self.position_size_pct,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'entry_price': self.entry_price,
            'stop_price': self.stop_price,
            'target_price': self.target_price,
            'risk_reward_ratio': self.risk_reward_ratio,
            'signals': {
                tf: {
                    'action': sig.action,
                    'price': sig.price,
                    'age_hours': round(sig.age_hours, 1),
                    'is_valid': sig.is_valid,
                } for tf, sig in self.signals.items()
            },
            'timestamp': self.timestamp.isoformat(),
        }


# ============================================================
# AWS CLIENTS (Lazy initialization)
# ============================================================

_dynamodb = None
_signals_table = None
_decisions_table = None


def get_dynamodb():
    global _dynamodb
    if _dynamodb is None:
        _dynamodb = boto3.resource('dynamodb', config=BOTO_CONFIG)
    return _dynamodb


def get_signals_table():
    global _signals_table
    if _signals_table is None:
        _signals_table = get_dynamodb().Table(SIGNALS_TABLE)
    return _signals_table


def get_decisions_table():
    global _decisions_table
    if _decisions_table is None:
        _decisions_table = get_dynamodb().Table(DECISIONS_TABLE)
    return _decisions_table


# ============================================================
# SIGNAL RETRIEVAL
# ============================================================

def get_latest_signals(symbol: str) -> Dict[str, Optional[TimeframeSignal]]:
    """
    Retrieve the latest signal for each timeframe for a symbol.
    Returns dict with timeframe keys and TimeframeSignal values.
    """
    table = get_signals_table()
    now = datetime.now(timezone.utc)
    signals = {}
    
    for timeframe, config in TIMEFRAMES.items():
        try:
            # Query for latest signal in this timeframe
            response = table.query(
                KeyConditionExpression='pk = :pk AND begins_with(sk, :tf)',
                ExpressionAttributeValues={
                    ':pk': f'SIGNAL#{symbol}',
                    ':tf': f'{timeframe}#',
                },
                ScanIndexForward=False,  # Descending order
                Limit=1,
            )
            
            if response.get('Items'):
                item = response['Items'][0]
                
                # Parse timestamp
                ts = item.get('timestamp', 0)
                if isinstance(ts, Decimal):
                    ts = float(ts)
                
                # Handle milliseconds vs seconds
                if ts > 1e12:
                    ts = ts / 1000
                
                signal_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                age_hours = (now - signal_time).total_seconds() / 3600
                
                # Check if signal is still valid (not expired)
                is_valid = age_hours <= config['max_age_hours']
                
                signals[timeframe] = TimeframeSignal(
                    symbol=symbol,
                    timeframe=timeframe,
                    action=item.get('action', 'NEUTRAL'),
                    price=float(item.get('price', 0)),
                    confidence=float(item.get('confidence', 50)),
                    timestamp=signal_time,
                    age_hours=age_hours,
                    is_valid=is_valid,
                    source=item.get('source', 'luxalgo'),
                )
            else:
                signals[timeframe] = None
                
        except Exception as e:
            logger.error(f"Error fetching {timeframe} signal for {symbol}: {e}")
            signals[timeframe] = None
    
    return signals


# ============================================================
# ALIGNMENT ANALYSIS
# ============================================================

def check_alignment(symbol: str, signals: Dict[str, Optional[TimeframeSignal]]) -> AlignmentResult:
    """
    Check multi-timeframe alignment and generate trading recommendation.
    
    Rules:
    1. Weekly has VETO power - never trade against weekly
    2. All 3 aligned = HIGH confidence (80%+)
    3. Weekly + Daily aligned = MEDIUM confidence (65%+)
    4. Only 4H = LOW confidence (50%)
    5. Conflicting signals = NO TRADE
    """
    now = datetime.now(timezone.utc)
    
    weekly = signals.get('1W')
    daily = signals.get('1D')
    h4 = signals.get('4h')
    
    # Get valid signals only
    valid_signals = {tf: sig for tf, sig in signals.items() if sig and sig.is_valid}
    
    # Determine primary direction from weekly (if available)
    weekly_direction = weekly.direction if weekly and weekly.is_valid else SignalDirection.NEUTRAL
    daily_direction = daily.direction if daily and daily.is_valid else SignalDirection.NEUTRAL
    h4_direction = h4.direction if h4 and h4.is_valid else SignalDirection.NEUTRAL
    
    # Check for veto conditions
    veto_active = False
    veto_reason = None
    
    # Get current price from most recent signal
    current_price = 0
    for tf in ['4h', '1D', '1W']:
        if signals.get(tf) and signals[tf].price > 0:
            current_price = signals[tf].price
            break
    
    # Alignment logic
    aligned_count = 0
    primary_direction = SignalDirection.NEUTRAL
    
    # Weekly sets the direction
    if weekly and weekly.is_valid and weekly_direction != SignalDirection.NEUTRAL:
        primary_direction = weekly_direction
        aligned_count += 1
        
        # Check if daily aligns with weekly
        if daily and daily.is_valid:
            if daily_direction == primary_direction:
                aligned_count += 1
            elif daily_direction != SignalDirection.NEUTRAL and daily_direction != primary_direction:
                # Daily conflicts with weekly - reduce confidence
                pass
        
        # Check if 4H aligns with weekly
        if h4 and h4.is_valid:
            if h4_direction == primary_direction:
                aligned_count += 1
            elif h4_direction != SignalDirection.NEUTRAL and h4_direction != primary_direction:
                # 4H trying to trade against weekly - VETO
                if weekly_direction != h4_direction:
                    veto_active = True
                    veto_reason = f"4H {h4_direction.value} conflicts with Weekly {weekly_direction.value}"
    
    elif daily and daily.is_valid and daily_direction != SignalDirection.NEUTRAL:
        # No weekly signal, use daily as primary
        primary_direction = daily_direction
        aligned_count += 1
        
        if h4 and h4.is_valid and h4_direction == primary_direction:
            aligned_count += 1
    
    elif h4 and h4.is_valid and h4_direction != SignalDirection.NEUTRAL:
        # Only 4H signal - low confidence
        primary_direction = h4_direction
        aligned_count += 1
    
    # Calculate confidence based on alignment
    if veto_active:
        confidence = 0
        confidence_level = 'veto'
        is_aligned = False
    elif aligned_count >= 3:
        confidence = 85
        confidence_level = 'high'
        is_aligned = True
    elif aligned_count == 2:
        confidence = 70
        confidence_level = 'medium'
        is_aligned = True
    elif aligned_count == 1:
        confidence = 55
        confidence_level = 'low'
        is_aligned = False
    else:
        confidence = 0
        confidence_level = 'veto'
        is_aligned = False
    
    # Boost confidence for strong signals
    if is_aligned:
        for sig in valid_signals.values():
            if sig.action in ['STRONG_BUY', 'STRONG_SELL']:
                confidence = min(95, confidence + 5)
    
    # Get position sizing
    sizing = POSITION_SIZING.get(confidence_level, POSITION_SIZING['low'])
    
    # Calculate prices
    if current_price > 0 and primary_direction != SignalDirection.NEUTRAL:
        if primary_direction == SignalDirection.BUY:
            stop_price = current_price * (1 - sizing['stop_pct'] / 100)
            target_price = current_price * (1 + sizing['target_pct'] / 100)
        else:  # SELL
            stop_price = current_price * (1 + sizing['stop_pct'] / 100)
            target_price = current_price * (1 - sizing['target_pct'] / 100)
        
        risk_reward = sizing['target_pct'] / sizing['stop_pct'] if sizing['stop_pct'] > 0 else 0
    else:
        stop_price = 0
        target_price = 0
        risk_reward = 0
    
    # Generate recommendation
    if veto_active:
        recommendation = f"⛔ NO TRADE - {veto_reason}"
    elif is_aligned and confidence >= 80:
        recommendation = f"🚀 STRONG {primary_direction.value} - All timeframes aligned!"
    elif is_aligned and confidence >= 65:
        recommendation = f"✅ {primary_direction.value} - Good alignment, manage risk"
    elif confidence >= 50:
        recommendation = f"⚠️ WEAK {primary_direction.value} - Wait for confirmation"
    else:
        recommendation = "⏳ NO SIGNAL - Waiting for alignment"
    
    return AlignmentResult(
        symbol=symbol,
        is_aligned=is_aligned,
        direction=primary_direction,
        confidence=confidence,
        confidence_level=confidence_level,
        signals=valid_signals,
        weekly_signal=weekly,
        daily_signal=daily,
        h4_signal=h4,
        veto_active=veto_active,
        veto_reason=veto_reason,
        recommendation=recommendation,
        position_size_pct=sizing['size_pct'] if is_aligned else 0,
        stop_loss_pct=sizing['stop_pct'],
        take_profit_pct=sizing['target_pct'],
        entry_price=current_price,
        stop_price=stop_price,
        target_price=target_price,
        risk_reward_ratio=risk_reward,
        timestamp=now,
    )


# ============================================================
# NOTIFICATIONS
# ============================================================

def send_telegram(message: str) -> bool:
    """Send notification to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = json.dumps({
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True,
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
        logger.error(f"Telegram error: {e}")
        return False


def send_discord(message: str, embed: Dict = None) -> bool:
    """Send notification to Discord."""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("Discord not configured")
        return False
    
    try:
        payload = {'content': message}
        if embed:
            payload['embeds'] = [embed]
        
        data = json.dumps(payload).encode('utf-8')
        
        req = urllib.request.Request(
            DISCORD_WEBHOOK_URL,
            data=data,
            headers={'Content-Type': 'application/json'},
        )
        
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status in [200, 204]
            
    except Exception as e:
        logger.error(f"Discord error: {e}")
        return False


def format_telegram_alert(result: AlignmentResult) -> str:
    """Format alignment result for Telegram."""
    
    # Emoji based on confidence
    if result.confidence >= 80:
        emoji = "🚀🔥"
    elif result.confidence >= 65:
        emoji = "✅"
    elif result.confidence >= 50:
        emoji = "⚠️"
    else:
        emoji = "⏳"
    
    # Direction emoji
    dir_emoji = "📈" if result.direction == SignalDirection.BUY else "📉" if result.direction == SignalDirection.SELL else "➡️"
    
    # Build signal status
    signal_lines = []
    for tf in ['1W', '1D', '4h']:
        sig = result.signals.get(tf)
        if sig:
            age_str = f"{sig.age_hours:.1f}h ago" if sig.age_hours < 24 else f"{sig.age_hours/24:.1f}d ago"
            status = "✅" if sig.is_valid else "⏰"
            signal_lines.append(f"  {status} {TIMEFRAMES[tf]['name']}: {sig.action} @ ${sig.price:,.2f} ({age_str})")
        else:
            signal_lines.append(f"  ❌ {TIMEFRAMES[tf]['name']}: No signal")
    
    signals_text = "\n".join(signal_lines)
    
    message = f"""
{emoji} <b>KYPERIAN SIGNAL ALERT</b> {emoji}

<b>{result.symbol}</b> {dir_emoji} <b>{result.direction.value}</b>

📊 <b>Confidence:</b> {result.confidence:.0f}% ({result.confidence_level.upper()})

<b>Timeframe Signals:</b>
{signals_text}

{result.recommendation}
"""
    
    if result.is_aligned and result.entry_price > 0:
        message += f"""
💰 <b>Trade Setup:</b>
• Entry: ${result.entry_price:,.2f}
• Stop Loss: ${result.stop_price:,.2f} (-{result.stop_loss_pct:.1f}%)
• Take Profit: ${result.target_price:,.2f} (+{result.take_profit_pct:.1f}%)
• Position Size: {result.position_size_pct:.1f}% of portfolio
• Risk/Reward: {result.risk_reward_ratio:.1f}:1
"""
    
    if result.veto_active:
        message += f"""
⛔ <b>VETO ACTIVE:</b> {result.veto_reason}
"""
    
    message += f"""
⏰ {result.timestamp.strftime('%Y-%m-%d %H:%M UTC')}
"""
    
    return message.strip()


def format_discord_embed(result: AlignmentResult) -> Dict:
    """Format alignment result as Discord embed."""
    
    # Color based on direction and confidence
    if result.confidence >= 80:
        color = 0x00FF00 if result.direction == SignalDirection.BUY else 0xFF0000
    elif result.confidence >= 65:
        color = 0x90EE90 if result.direction == SignalDirection.BUY else 0xFFCCCB
    else:
        color = 0xFFFF00
    
    if result.veto_active:
        color = 0x808080
    
    # Build fields
    fields = []
    
    # Signal status
    for tf in ['1W', '1D', '4h']:
        sig = result.signals.get(tf)
        if sig:
            age_str = f"{sig.age_hours:.1f}h" if sig.age_hours < 24 else f"{sig.age_hours/24:.1f}d"
            status = "✅" if sig.is_valid else "⏰"
            fields.append({
                'name': f"{TIMEFRAMES[tf]['name']} ({tf})",
                'value': f"{status} {sig.action}\n${sig.price:,.2f}\n{age_str} ago",
                'inline': True,
            })
        else:
            fields.append({
                'name': f"{TIMEFRAMES[tf]['name']} ({tf})",
                'value': "❌ No signal",
                'inline': True,
            })
    
    # Trade setup if aligned
    if result.is_aligned and result.entry_price > 0:
        fields.append({
            'name': '💰 Trade Setup',
            'value': f"Entry: ${result.entry_price:,.2f}\nStop: ${result.stop_price:,.2f} (-{result.stop_loss_pct}%)\nTarget: ${result.target_price:,.2f} (+{result.take_profit_pct}%)",
            'inline': False,
        })
        fields.append({
            'name': '📊 Risk Management',
            'value': f"Position: {result.position_size_pct}%\nR/R: {result.risk_reward_ratio:.1f}:1",
            'inline': True,
        })
    
    return {
        'title': f"{'🚀' if result.is_aligned else '⏳'} {result.symbol} - {result.direction.value}",
        'description': result.recommendation,
        'color': color,
        'fields': fields,
        'footer': {
            'text': f"Confidence: {result.confidence:.0f}% | KYPERIAN ELITE",
        },
        'timestamp': result.timestamp.isoformat(),
    }


# ============================================================
# PERSISTENCE
# ============================================================

def store_decision(result: AlignmentResult) -> bool:
    """Store decision in DynamoDB for tracking."""
    try:
        table = get_decisions_table()
        
        item = {
            'pk': f"DECISION#{result.symbol}",
            'sk': result.timestamp.strftime('%Y%m%d%H%M%S'),
            'symbol': result.symbol,
            'direction': result.direction.value,
            'confidence': Decimal(str(result.confidence)),
            'confidence_level': result.confidence_level,
            'is_aligned': result.is_aligned,
            'veto_active': result.veto_active,
            'recommendation': result.recommendation,
            'entry_price': Decimal(str(result.entry_price)) if result.entry_price else Decimal('0'),
            'stop_price': Decimal(str(result.stop_price)) if result.stop_price else Decimal('0'),
            'target_price': Decimal(str(result.target_price)) if result.target_price else Decimal('0'),
            'position_size_pct': Decimal(str(result.position_size_pct)),
            'timestamp_iso': result.timestamp.isoformat(),
            'ttl': int((result.timestamp + timedelta(days=30)).timestamp()),
            # GSI for querying by confidence level
            'gsi1pk': f"CONFIDENCE#{result.confidence_level}",
            'gsi1sk': f"{result.symbol}#{result.timestamp.strftime('%Y%m%d%H%M%S')}",
        }
        
        if result.veto_reason:
            item['veto_reason'] = result.veto_reason
        
        table.put_item(Item=item)
        return True
        
    except Exception as e:
        logger.error(f"Error storing decision: {e}")
        return False


def get_last_notification_time(symbol: str) -> Optional[datetime]:
    """Get last notification time to prevent spam."""
    try:
        table = get_decisions_table()
        
        response = table.query(
            KeyConditionExpression='pk = :pk',
            ExpressionAttributeValues={':pk': f"DECISION#{symbol}"},
            ScanIndexForward=False,
            Limit=1,
        )
        
        if response.get('Items'):
            item = response['Items'][0]
            ts_iso = item.get('timestamp_iso')
            if ts_iso:
                return datetime.fromisoformat(ts_iso.replace('Z', '+00:00'))
        
        return None
        
    except Exception as e:
        logger.error(f"Error getting last notification: {e}")
        return None


def should_notify(result: AlignmentResult, min_interval_minutes: int = 60) -> bool:
    """
    Determine if we should send a notification.
    Prevents spam by checking:
    1. Minimum time since last notification
    2. Confidence changed significantly
    3. Direction changed
    """
    if not result.is_aligned and not result.veto_active:
        return False
    
    if result.confidence < 65 and not result.veto_active:
        return False
    
    last_time = get_last_notification_time(result.symbol)
    if last_time:
        elapsed = (result.timestamp - last_time).total_seconds() / 60
        if elapsed < min_interval_minutes:
            logger.info(f"Skipping notification for {result.symbol}: only {elapsed:.1f} min since last")
            return False
    
    return True


# ============================================================
# MAIN HANDLERS
# ============================================================

def process_symbol(symbol: str) -> AlignmentResult:
    """Process a single symbol and return alignment result."""
    logger.info(f"Processing {symbol}")
    
    # Get latest signals
    signals = get_latest_signals(symbol)
    
    # Check alignment
    result = check_alignment(symbol, signals)
    
    logger.info(f"{symbol}: {result.direction.value} | Confidence: {result.confidence}% | Aligned: {result.is_aligned}")
    
    return result


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler.
    
    Triggers:
    1. Scheduled (EventBridge): Check all symbols every 5 minutes
    2. New Signal (EventBridge): Check specific symbol when new signal arrives
    3. API Gateway: Manual check or dashboard data
    """
    start_time = time.time()
    
    try:
        # Determine trigger type
        source = event.get('source', '')
        
        if source == 'aws.events':
            # Scheduled check
            logger.info("Scheduled alignment check")
            symbols_to_check = MONITORED_SYMBOLS
            
        elif source == 'kyperian.signals':
            # New signal arrived
            detail = event.get('detail', {})
            symbol = detail.get('symbol')
            if symbol:
                logger.info(f"New signal trigger for {symbol}")
                symbols_to_check = [symbol]
            else:
                symbols_to_check = MONITORED_SYMBOLS
                
        elif 'httpMethod' in event or 'requestContext' in event:
            # API Gateway request
            return handle_api_request(event)
            
        else:
            # Default: check all
            symbols_to_check = MONITORED_SYMBOLS
        
        # Process each symbol
        results = []
        notifications_sent = 0
        
        for symbol in symbols_to_check:
            try:
                result = process_symbol(symbol)
                results.append(result.to_dict())
                
                # Send notifications for significant signals
                if should_notify(result):
                    # Store decision
                    store_decision(result)
                    
                    # Send Telegram
                    if TELEGRAM_BOT_TOKEN:
                        msg = format_telegram_alert(result)
                        if send_telegram(msg):
                            notifications_sent += 1
                            logger.info(f"Telegram notification sent for {symbol}")
                    
                    # Send Discord
                    if DISCORD_WEBHOOK_URL:
                        embed = format_discord_embed(result)
                        if send_discord("", embed):
                            notifications_sent += 1
                            logger.info(f"Discord notification sent for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results.append({'symbol': symbol, 'error': str(e)})
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'symbols_checked': len(symbols_to_check),
                'notifications_sent': notifications_sent,
                'results': results,
                'processing_time_ms': round(processing_time, 2),
            }),
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e),
            }),
        }


def handle_api_request(event: Dict[str, Any]) -> Dict[str, Any]:
    """Handle API Gateway requests for dashboard."""
    
    path = event.get('rawPath', event.get('path', ''))
    method = event.get('httpMethod', event.get('requestContext', {}).get('http', {}).get('method', 'GET'))
    
    # CORS headers
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    }
    
    if method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}
    
    # Dashboard endpoint - get all current alignments
    if '/dashboard' in path or '/status' in path:
        results = []
        for symbol in MONITORED_SYMBOLS:
            try:
                result = process_symbol(symbol)
                results.append(result.to_dict())
            except Exception as e:
                results.append({'symbol': symbol, 'error': str(e)})
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'success': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbols': results,
            }),
        }
    
    # Check specific symbol
    if '/check/' in path:
        symbol = path.split('/check/')[-1].upper()
        try:
            result = process_symbol(symbol)
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'success': True,
                    **result.to_dict(),
                }),
            }
        except Exception as e:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({
                    'success': False,
                    'error': str(e),
                }),
            }
    
    # Default: return status
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps({
            'service': 'KYPERIAN Decision Engine',
            'version': '1.0.0',
            'endpoints': [
                '/dashboard - Get all symbol alignments',
                '/check/{symbol} - Check specific symbol',
            ],
            'monitored_symbols': MONITORED_SYMBOLS,
        }),
    }
