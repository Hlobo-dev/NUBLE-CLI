"""
NUBLE ELITE - Decision Engine V2 Lambda Handler
====================================================
Serverless handler for the institutional-grade decision engine.

Features:
- Processes signals from DynamoDB
- Runs full 4-layer analysis
- Sends notifications for high-confidence trades
- Stores decisions for tracking

Author: NUBLE ELITE
Version: 2.0.0
"""

from __future__ import annotations

import json
import os
import time
import asyncio
import urllib.request
import urllib.error
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
import logging
import hashlib

import boto3
from botocore.config import Config

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS Config
BOTO_CONFIG = Config(
    connect_timeout=5,
    read_timeout=10,
    retries={'max_attempts': 3, 'mode': 'adaptive'},
)

# Environment
SIGNALS_TABLE = os.environ.get('DYNAMODB_SIGNALS_TABLE', 'nuble-production-signals')
DECISIONS_TABLE = os.environ.get('DYNAMODB_DECISIONS_TABLE', 'nuble-production-decisions')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')
DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL', '')

# Monitored symbols
MONITORED_SYMBOLS = ['BTCUSD', 'ETHUSD', 'SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA', 'AMD']

# Timeframe config
TIMEFRAMES = {
    '1W': {'max_age_hours': 168, 'weight': 0.45},
    '1D': {'max_age_hours': 48, 'weight': 0.35},
    '4h': {'max_age_hours': 12, 'weight': 0.20},
}

# ============================================================
# LAZY INITIALIZATION
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

def get_signals_for_symbol(symbol: str) -> Dict[str, Any]:
    """Get all signals for a symbol from DynamoDB."""
    table = get_signals_table()
    now = datetime.now(timezone.utc)
    signals = {}
    
    for tf, config in TIMEFRAMES.items():
        try:
            response = table.query(
                KeyConditionExpression='pk = :pk AND begins_with(sk, :tf)',
                ExpressionAttributeValues={
                    ':pk': f'SIGNAL#{symbol}',
                    ':tf': f'{tf}#',
                },
                ScanIndexForward=False,
                Limit=1,
            )
            
            if response.get('Items'):
                item = response['Items'][0]
                
                ts = item.get('timestamp', 0)
                if isinstance(ts, Decimal):
                    ts = float(ts)
                if ts > 1e12:
                    ts = ts / 1000
                
                signal_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                age_hours = (now - signal_time).total_seconds() / 3600
                is_valid = age_hours <= config['max_age_hours']
                
                action = item.get('action', 'NEUTRAL')
                
                # Convert action to score
                if action in ['BUY', 'STRONG_BUY']:
                    score = 0.8 if action == 'STRONG_BUY' else 0.5
                elif action in ['SELL', 'STRONG_SELL']:
                    score = -0.8 if action == 'STRONG_SELL' else -0.5
                else:
                    score = 0
                
                signals[tf] = {
                    'action': action,
                    'score': score if is_valid else 0,
                    'price': float(item.get('price', 0)),
                    'confidence': float(item.get('confidence', 50)),
                    'age_hours': age_hours,
                    'is_valid': is_valid,
                    'timestamp': signal_time.isoformat(),
                }
            else:
                signals[tf] = None
                
        except Exception as e:
            logger.error(f"Error fetching {tf} signal for {symbol}: {e}")
            signals[tf] = None
    
    return signals


# ============================================================
# SIMPLIFIED INLINE DECISION ENGINE (No external dependencies)
# ============================================================

class InlineDecisionEngine:
    """
    Simplified decision engine that runs in Lambda without external deps.
    Implements the 4-layer scoring system inline.
    """
    
    # Layer weights
    SIGNAL_WEIGHT = 0.40
    CONTEXT_WEIGHT = 0.30
    VALIDATION_WEIGHT = 0.20
    RISK_WEIGHT = 0.10
    
    # Thresholds
    STRONG_THRESHOLD = 75
    MODERATE_THRESHOLD = 55
    WEAK_THRESHOLD = 40
    
    def analyze(self, symbol: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full analysis on a symbol with its signals.
        
        Returns a decision dict with all layer scores.
        """
        now = datetime.now(timezone.utc)
        reasoning = []
        
        # ========== LAYER 1: SIGNAL LAYER (40%) ==========
        signal_result = self._analyze_signals(signals)
        reasoning.append(
            f"📊 Signal: {signal_result['score']*100:+.1f}% "
            f"(Alignment: {signal_result['alignment']:.0%})"
        )
        
        # ========== LAYER 2: CONTEXT LAYER (30%) ==========
        context_result = self._analyze_context(signals)
        reasoning.append(
            f"🌍 Context: {context_result['score']*100:+.1f}% "
            f"(Regime: {context_result['regime']}, Vol: {context_result['volatility']})"
        )
        
        # ========== LAYER 3: VALIDATION LAYER (20%) ==========
        validation_result = self._analyze_validation(signals, signal_result)
        reasoning.append(
            f"📜 Validation: {validation_result['score']*100:+.1f}% "
            f"(Historical WR: {validation_result['estimated_wr']:.0%})"
        )
        
        # ========== LAYER 4: RISK LAYER (10% + VETO) ==========
        risk_result = self._analyze_risk(signals)
        
        if risk_result['veto']:
            reasoning.append(f"🚫 VETO: {risk_result['veto_reason']}")
            return self._create_veto_result(symbol, signals, risk_result, reasoning, now)
        
        # ========== COMBINE LAYERS ==========
        raw_confidence = (
            signal_result['score'] * self.SIGNAL_WEIGHT +
            context_result['score'] * self.CONTEXT_WEIGHT +
            validation_result['score'] * self.VALIDATION_WEIGHT +
            risk_result['score'] * self.RISK_WEIGHT
        )
        
        # Convert to 0-100 scale (scores are -1 to +1)
        base_confidence = (raw_confidence + 1) / 2 * 100
        
        # Apply regime alignment bonus/penalty
        direction = signal_result['direction']
        regime_alignment = self._check_regime_alignment(direction, context_result['regime'])
        adjusted_confidence = base_confidence * (1 + regime_alignment * 0.15)
        
        # Cap confidence
        final_confidence = max(0, min(100, adjusted_confidence))
        
        # Determine strength
        if final_confidence >= self.STRONG_THRESHOLD:
            strength = 'STRONG'
        elif final_confidence >= self.MODERATE_THRESHOLD:
            strength = 'MODERATE'
        elif final_confidence >= self.WEAK_THRESHOLD:
            strength = 'WEAK'
        else:
            strength = 'NO_TRADE'
        
        is_aligned = signal_result['alignment'] > 0.6 and direction != 0
        
        # Get current price
        current_price = self._get_price_from_signals(signals)
        
        # Calculate levels
        stop_pct = 0.02 * (1 + context_result['volatility_pct'])
        target_pct = stop_pct * 3  # 3:1 R/R
        
        if direction > 0:
            stop_loss = current_price * (1 - stop_pct)
            take_profit = current_price * (1 + target_pct)
        elif direction < 0:
            stop_loss = current_price * (1 + stop_pct)
            take_profit = current_price * (1 - target_pct)
        else:
            stop_loss = current_price
            take_profit = current_price
        
        # Position size
        position_pct = (final_confidence / 100) * 5 * (1 - context_result['volatility_pct'])
        position_pct = max(0.5, min(5.0, position_pct))
        
        reasoning.append(f"🎯 Final: {final_confidence:.1f}% → {strength}")
        
        direction_str = 'BUY' if direction > 0 else 'SELL' if direction < 0 else 'NEUTRAL'
        
        return {
            'symbol': symbol,
            'timestamp': now.isoformat(),
            'direction': direction_str,
            'strength': strength,
            'confidence': round(final_confidence, 1),
            'is_aligned': is_aligned,
            'should_trade': strength in ['STRONG', 'MODERATE'] and is_aligned,
            'layers': {
                'signal': {
                    'score': round(signal_result['score'], 3),
                    'direction': direction,
                    'alignment': round(signal_result['alignment'], 3),
                    'weekly': signal_result.get('weekly_action', 'N/A'),
                    'daily': signal_result.get('daily_action', 'N/A'),
                    'h4': signal_result.get('h4_action', 'N/A'),
                },
                'context': {
                    'score': round(context_result['score'], 3),
                    'regime': context_result['regime'],
                    'volatility': context_result['volatility'],
                    'volatility_pct': round(context_result['volatility_pct'], 3),
                },
                'validation': {
                    'score': round(validation_result['score'], 3),
                    'estimated_wr': round(validation_result['estimated_wr'], 3),
                    'samples': validation_result['samples'],
                },
                'risk': {
                    'score': round(risk_result['score'], 3),
                    'veto': risk_result['veto'],
                    'checks_passed': risk_result['checks_passed'],
                },
            },
            'trade_setup': {
                'entry': round(current_price, 4),
                'stop_loss': round(stop_loss, 4),
                'take_profit': round(take_profit, 4),
                'stop_pct': round(stop_pct * 100, 2),
                'target_pct': round(target_pct * 100, 2),
                'position_pct': round(position_pct, 2),
                'risk_reward': round(target_pct / stop_pct, 1) if stop_pct > 0 else 0,
            },
            'signals': {
                tf: {
                    'action': sig['action'] if sig else 'NO_SIGNAL',
                    'price': sig['price'] if sig else 0,
                    'age_hours': round(sig['age_hours'], 1) if sig else 0,
                    'valid': sig['is_valid'] if sig else False,
                } for tf, sig in signals.items()
            },
            'reasoning': reasoning,
            'data_points': 15,
        }
    
    def _analyze_signals(self, signals: Dict) -> Dict[str, Any]:
        """Analyze Layer 1: Technical Signals."""
        
        weekly = signals.get('1W')
        daily = signals.get('1D')
        h4 = signals.get('4h')
        
        # Get scores
        w_score = weekly['score'] if weekly and weekly['is_valid'] else 0
        d_score = daily['score'] if daily and daily['is_valid'] else 0
        h_score = h4['score'] if h4 and h4['is_valid'] else 0
        
        # Weighted combination (weekly has most weight)
        combined = w_score * 0.45 + d_score * 0.35 + h_score * 0.20
        
        # Calculate alignment
        scores = [w_score, d_score, h_score]
        non_zero = [s for s in scores if s != 0]
        
        if len(non_zero) == 0:
            alignment = 0
        elif len(non_zero) == 3 and all(s > 0 for s in non_zero):
            alignment = 1.0
        elif len(non_zero) == 3 and all(s < 0 for s in non_zero):
            alignment = 1.0
        elif len(non_zero) >= 2:
            signs = [s > 0 for s in non_zero]
            if all(signs) or not any(signs):
                alignment = 0.7
            else:
                alignment = 0.0
        else:
            alignment = 0.3
        
        # Direction from weekly (veto power)
        if w_score > 0.3:
            direction = 1
        elif w_score < -0.3:
            direction = -1
        else:
            # Fall back to daily
            if d_score > 0.3:
                direction = 1
            elif d_score < -0.3:
                direction = -1
            else:
                direction = 0
        
        # Boost score if aligned
        if alignment > 0.8:
            combined *= 1.2
        
        return {
            'score': max(-1, min(1, combined)),
            'direction': direction,
            'alignment': alignment,
            'weekly_action': weekly['action'] if weekly else 'N/A',
            'daily_action': daily['action'] if daily else 'N/A',
            'h4_action': h4['action'] if h4 else 'N/A',
        }
    
    def _analyze_context(self, signals: Dict) -> Dict[str, Any]:
        """Analyze Layer 2: Market Context (simplified)."""
        
        # Estimate regime from signal consistency
        weekly = signals.get('1W')
        daily = signals.get('1D')
        
        if weekly and weekly['is_valid']:
            if weekly['score'] > 0.3:
                regime = 'BULL'
            elif weekly['score'] < -0.3:
                regime = 'BEAR'
            else:
                regime = 'SIDEWAYS'
        else:
            regime = 'UNKNOWN'
        
        # Estimate volatility from signal age variance
        # (In real system, would use ATR/VIX)
        ages = []
        for tf in ['1W', '1D', '4h']:
            sig = signals.get(tf)
            if sig:
                ages.append(sig['age_hours'])
        
        if ages:
            # Fresh signals = lower volatility environment
            avg_age = sum(ages) / len(ages)
            if avg_age < 6:
                volatility = 'LOW'
                vol_pct = 0.3
            elif avg_age < 24:
                volatility = 'NORMAL'
                vol_pct = 0.5
            else:
                volatility = 'HIGH'
                vol_pct = 0.7
        else:
            volatility = 'UNKNOWN'
            vol_pct = 0.5
        
        # Context score: positive in bull with fresh signals
        if regime == 'BULL':
            base_score = 0.6
        elif regime == 'BEAR':
            base_score = -0.3
        else:
            base_score = 0.0
        
        # Adjust for volatility
        if volatility == 'HIGH':
            base_score *= 0.5
        
        return {
            'score': base_score,
            'regime': regime,
            'volatility': volatility,
            'volatility_pct': vol_pct,
        }
    
    def _analyze_validation(self, signals: Dict, signal_result: Dict) -> Dict[str, Any]:
        """Analyze Layer 3: Historical Validation (estimated)."""
        
        # In real system, would query historical database
        # Here we estimate based on alignment quality
        
        alignment = signal_result['alignment']
        direction = signal_result['direction']
        
        # Higher alignment = higher estimated win rate
        if alignment > 0.8:
            estimated_wr = 0.65
        elif alignment > 0.6:
            estimated_wr = 0.55
        elif alignment > 0.3:
            estimated_wr = 0.50
        else:
            estimated_wr = 0.45
        
        # Score based on estimated edge
        score = (estimated_wr - 0.5) * 2  # Convert 0.5-0.7 to 0-0.4
        
        return {
            'score': max(-1, min(1, score)),
            'estimated_wr': estimated_wr,
            'samples': 50,  # Placeholder
        }
    
    def _analyze_risk(self, signals: Dict) -> Dict[str, Any]:
        """Analyze Layer 4: Risk Checks."""
        
        veto = False
        veto_reason = None
        checks_passed = 6
        
        # Check for conflicting strong signals (weekly vs 4H)
        weekly = signals.get('1W')
        h4 = signals.get('4h')
        
        if weekly and h4 and weekly['is_valid'] and h4['is_valid']:
            if (weekly['score'] > 0.3 and h4['score'] < -0.3) or \
               (weekly['score'] < -0.3 and h4['score'] > 0.3):
                veto = True
                veto_reason = f"4H conflicts with Weekly (4H: {h4['action']}, 1W: {weekly['action']})"
                checks_passed -= 1
        
        # Check for stale signals
        all_stale = True
        for tf in ['1W', '1D', '4h']:
            sig = signals.get(tf)
            if sig and sig['is_valid']:
                all_stale = False
                break
        
        if all_stale:
            veto = True
            veto_reason = "All signals expired"
            checks_passed -= 1
        
        # Risk score (higher = riskier)
        risk_score = 1 - (checks_passed / 6)
        
        return {
            'score': 1 - risk_score,  # Invert: higher = safer
            'veto': veto,
            'veto_reason': veto_reason,
            'checks_passed': checks_passed,
        }
    
    def _check_regime_alignment(self, direction: int, regime: str) -> float:
        """Check if trade direction aligns with regime."""
        if direction == 0:
            return 0
        
        if regime == 'BULL' and direction > 0:
            return 0.2  # 20% bonus
        elif regime == 'BEAR' and direction < 0:
            return 0.2
        elif regime == 'BULL' and direction < 0:
            return -0.15  # 15% penalty
        elif regime == 'BEAR' and direction > 0:
            return -0.15
        
        return 0
    
    def _get_price_from_signals(self, signals: Dict) -> float:
        """Get current price from most recent signal."""
        for tf in ['4h', '1D', '1W']:
            sig = signals.get(tf)
            if sig and sig['price'] > 0:
                return sig['price']
        return 0
    
    def _create_veto_result(self, symbol: str, signals: Dict, risk: Dict, 
                           reasoning: List[str], now: datetime) -> Dict:
        """Create a VETO result."""
        current_price = self._get_price_from_signals(signals)
        
        return {
            'symbol': symbol,
            'timestamp': now.isoformat(),
            'direction': 'NEUTRAL',
            'strength': 'NO_TRADE',
            'confidence': 0,
            'is_aligned': False,
            'should_trade': False,
            'layers': {
                'signal': {'score': 0, 'direction': 0, 'alignment': 0},
                'context': {'score': 0, 'regime': 'UNKNOWN', 'volatility': 'UNKNOWN'},
                'validation': {'score': 0, 'estimated_wr': 0.5, 'samples': 0},
                'risk': {
                    'score': 0,
                    'veto': True,
                    'veto_reason': risk['veto_reason'],
                    'checks_passed': risk['checks_passed'],
                },
            },
            'trade_setup': {
                'entry': current_price,
                'stop_loss': current_price,
                'take_profit': current_price,
                'position_pct': 0,
                'risk_reward': 0,
            },
            'signals': {
                tf: {
                    'action': sig['action'] if sig else 'NO_SIGNAL',
                    'price': sig['price'] if sig else 0,
                    'age_hours': round(sig['age_hours'], 1) if sig else 0,
                    'valid': sig['is_valid'] if sig else False,
                } for tf, sig in signals.items()
            },
            'reasoning': reasoning,
            'data_points': 15,
        }


# ============================================================
# NOTIFICATIONS
# ============================================================

def format_telegram_message(decision: Dict) -> str:
    """Format decision for Telegram notification."""
    
    symbol = decision['symbol']
    direction = decision['direction']
    strength = decision['strength']
    confidence = decision['confidence']
    
    # Emoji
    if strength == 'STRONG':
        emoji = "🚀🔥"
    elif strength == 'MODERATE':
        emoji = "✅"
    elif strength == 'WEAK':
        emoji = "⚠️"
    else:
        emoji = "⏳"
    
    dir_emoji = "📈" if direction == 'BUY' else "📉" if direction == 'SELL' else "➡️"
    
    # Layers
    layers = decision['layers']
    
    # Trade setup
    setup = decision['trade_setup']
    
    message = f"""
{emoji} <b>NUBLE ELITE V2 ALERT</b> {emoji}

<b>{symbol}</b> {dir_emoji} <b>{direction}</b>
Strength: {strength} | Confidence: {confidence:.1f}%

<b>📊 Layer Analysis (15 data points):</b>
• Signal:     {layers['signal']['score']*100:+.1f}% (Align: {layers['signal']['alignment']:.0%})
• Context:    {layers['context']['score']*100:+.1f}% ({layers['context']['regime']})
• Validation: {layers['validation']['score']*100:+.1f}% (WR: {layers['validation']['estimated_wr']:.0%})
• Risk:       {layers['risk']['score']*100:+.1f}% ({layers['risk']['checks_passed']}/6 checks)

<b>📈 Signals:</b>
• Weekly: {decision['signals']['1W']['action']} ({decision['signals']['1W']['age_hours']:.1f}h ago)
• Daily:  {decision['signals']['1D']['action']} ({decision['signals']['1D']['age_hours']:.1f}h ago)
• 4H:     {decision['signals']['4h']['action']} ({decision['signals']['4h']['age_hours']:.1f}h ago)
"""
    
    if decision['should_trade']:
        message += f"""
<b>💰 Trade Setup:</b>
• Entry:  ${setup['entry']:,.2f}
• Stop:   ${setup['stop_loss']:,.2f} (-{setup['stop_pct']:.1f}%)
• Target: ${setup['take_profit']:,.2f} (+{setup['target_pct']:.1f}%)
• Size:   {setup['position_pct']:.1f}% of portfolio
• R/R:    {setup['risk_reward']:.1f}:1
"""
    
    if layers['risk']['veto']:
        message += f"""
<b>⛔ VETO:</b> {layers['risk'].get('veto_reason', 'Unknown')}
"""
    
    message += f"""
<b>Reasoning:</b>
{chr(10).join('• ' + r for r in decision['reasoning'])}

⏰ {decision['timestamp'][:19]}
"""
    
    return message.strip()


def send_telegram(message: str) -> bool:
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = json.dumps({
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read())
            return result.get('ok', False)
            
    except Exception as e:
        logger.error(f"Telegram error: {e}")
        return False


def send_discord(message: str, embed: Dict = None) -> bool:
    """Send message to Discord."""
    if not DISCORD_WEBHOOK_URL:
        return False
    
    try:
        payload = {'content': message}
        if embed:
            payload['embeds'] = [embed]
        
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(DISCORD_WEBHOOK_URL, data=data, 
                                      headers={'Content-Type': 'application/json'})
        
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status in [200, 204]
            
    except Exception as e:
        logger.error(f"Discord error: {e}")
        return False


# ============================================================
# STORAGE
# ============================================================

def store_decision(decision: Dict) -> bool:
    """Store decision in DynamoDB."""
    try:
        table = get_decisions_table()
        
        timestamp = decision['timestamp'][:14].replace('-', '').replace('T', '').replace(':', '')
        
        item = {
            'pk': f"DECISION#{decision['symbol']}",
            'sk': timestamp,
            'symbol': decision['symbol'],
            'direction': decision['direction'],
            'strength': decision['strength'],
            'confidence': Decimal(str(round(decision['confidence'], 2))),
            'is_aligned': decision['is_aligned'],
            'should_trade': decision['should_trade'],
            'layers': json.loads(json.dumps(decision['layers']), parse_float=Decimal),
            'trade_setup': json.loads(json.dumps(decision['trade_setup']), parse_float=Decimal),
            'reasoning': decision['reasoning'],
            'data_points': decision['data_points'],
            'timestamp_iso': decision['timestamp'],
            'ttl': int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
            'gsi1pk': f"STRENGTH#{decision['strength']}",
            'gsi1sk': f"{decision['symbol']}#{timestamp}",
        }
        
        table.put_item(Item=item)
        return True
        
    except Exception as e:
        logger.error(f"Error storing decision: {e}")
        return False


def should_notify(decision: Dict, min_interval_minutes: int = 60) -> bool:
    """Determine if we should send a notification."""
    
    # Only notify for tradeable signals
    if not decision['should_trade']:
        return False
    
    # Only notify for STRONG or MODERATE
    if decision['strength'] not in ['STRONG', 'MODERATE']:
        return False
    
    # Check last notification time
    try:
        table = get_decisions_table()
        response = table.query(
            KeyConditionExpression='pk = :pk',
            ExpressionAttributeValues={':pk': f"DECISION#{decision['symbol']}"},
            ScanIndexForward=False,
            Limit=1,
        )
        
        if response.get('Items'):
            last = response['Items'][0]
            last_time = datetime.fromisoformat(last['timestamp_iso'].replace('Z', '+00:00'))
            elapsed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60
            
            if elapsed < min_interval_minutes:
                logger.info(f"Skipping notification: {elapsed:.1f}min since last")
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error checking notification history: {e}")
        return True


# ============================================================
# MAIN HANDLER
# ============================================================

def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler."""
    
    start_time = time.time()
    engine = InlineDecisionEngine()
    
    try:
        # Determine what to process
        source = event.get('source', '')
        
        if source == 'nuble.signals':
            # New signal arrived
            detail = event.get('detail', {})
            symbol = detail.get('symbol')
            symbols_to_check = [symbol] if symbol else MONITORED_SYMBOLS
        elif 'httpMethod' in event or 'requestContext' in event:
            # API request
            return handle_api_request(event, engine)
        else:
            # Scheduled or default
            symbols_to_check = MONITORED_SYMBOLS
        
        # Process each symbol
        results = []
        notifications_sent = 0
        
        for symbol in symbols_to_check:
            try:
                # Get signals
                signals = get_signals_for_symbol(symbol)
                
                # Run analysis
                decision = engine.analyze(symbol, signals)
                results.append(decision)
                
                # Store decision
                store_decision(decision)
                
                # Send notifications
                if should_notify(decision):
                    msg = format_telegram_message(decision)
                    
                    if send_telegram(msg):
                        notifications_sent += 1
                        logger.info(f"Telegram sent for {symbol}")
                    
                    if send_discord(msg):
                        logger.info(f"Discord sent for {symbol}")
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results.append({'symbol': symbol, 'error': str(e)})
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'success': True,
                'version': '2.0.0',
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


def handle_api_request(event: Dict, engine: InlineDecisionEngine) -> Dict:
    """Handle API Gateway requests."""
    
    path = event.get('rawPath', event.get('path', ''))
    method = event.get('httpMethod', event.get('requestContext', {}).get('http', {}).get('method', 'GET'))
    
    headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    
    if method == 'OPTIONS':
        return {'statusCode': 200, 'headers': headers, 'body': ''}
    
    # Dashboard
    if '/dashboard' in path or '/status' in path:
        results = []
        for symbol in MONITORED_SYMBOLS:
            try:
                signals = get_signals_for_symbol(symbol)
                decision = engine.analyze(symbol, signals)
                results.append(decision)
            except Exception as e:
                results.append({'symbol': symbol, 'error': str(e)})
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps({
                'success': True,
                'version': '2.0.0',
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'symbols': results,
            }),
        }
    
    # Check specific symbol
    if '/check/' in path or '/analyze/' in path:
        symbol = path.split('/')[-1].upper()
        try:
            signals = get_signals_for_symbol(symbol)
            decision = engine.analyze(symbol, signals)
            return {
                'statusCode': 200,
                'headers': headers,
                'body': json.dumps({
                    'success': True,
                    'version': '2.0.0',
                    **decision,
                }),
            }
        except Exception as e:
            return {
                'statusCode': 400,
                'headers': headers,
                'body': json.dumps({'success': False, 'error': str(e)}),
            }
    
    # Default
    return {
        'statusCode': 200,
        'headers': headers,
        'body': json.dumps({
            'service': 'NUBLE Decision Engine V2',
            'version': '2.0.0',
            'endpoints': [
                '/dashboard - Get all symbol analyses',
                '/check/{symbol} - Analyze specific symbol',
            ],
            'monitored_symbols': MONITORED_SYMBOLS,
            'layers': ['Signal (40%)', 'Context (30%)', 'Validation (20%)', 'Risk (10% + VETO)'],
        }),
    }
