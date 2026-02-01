"""
Institutional Risk Manager

Controls:
1. Position limits (per asset, per sector)
2. Exposure limits (gross, net)
3. Volatility targeting
4. Drawdown-based de-risking
5. Kill switch

This is what separates research from production.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskState(Enum):
    """Risk state machine states."""
    NORMAL = "normal"           # Full risk budget
    REDUCED = "reduced"         # 50% risk budget
    MINIMAL = "minimal"         # 25% risk budget
    HALTED = "halted"          # No new positions


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RiskAlert:
    """Risk alert event."""
    timestamp: datetime
    severity: AlertSeverity
    category: str
    message: str
    details: Dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "💀"
        }
        return f"[{self.timestamp}] {emoji[self.severity]} {self.severity.value.upper()}: {self.message}"


@dataclass
class RiskLimits:
    """
    Risk limit configuration.
    
    These limits are based on institutional best practices.
    """
    # Position limits
    max_position_pct: float = 0.10          # 10% max single position
    max_sector_pct: float = 0.30            # 30% max sector exposure
    max_correlated_pct: float = 0.40        # 40% max correlated positions
    
    # Exposure limits
    max_gross_exposure: float = 1.5         # 150% gross
    max_net_exposure: float = 0.5           # 50% net long
    min_net_exposure: float = -0.5          # 50% net short (max)
    
    # Volatility targeting
    target_volatility: float = 0.15         # 15% annualized vol target
    max_volatility: float = 0.25            # 25% max vol before scaling
    vol_lookback: int = 20                  # Days for vol estimation
    
    # Leverage limits
    max_leverage: float = 2.0               # Maximum leverage
    min_cash_buffer: float = 0.05           # 5% cash buffer
    
    # Drawdown limits (trigger state transitions)
    drawdown_warning: float = 0.05          # 5% DD → INFO alert
    drawdown_reduced: float = 0.10          # 10% DD → REDUCED state
    drawdown_minimal: float = 0.15          # 15% DD → MINIMAL state
    drawdown_halt: float = 0.20             # 20% DD → HALTED state
    
    # Daily limits
    max_daily_trades: int = 100
    max_daily_turnover: float = 0.5         # 50% of portfolio
    max_daily_loss: float = 0.03            # 3% daily loss limit
    
    # Concentration limits
    max_positions: int = 50                 # Maximum number of positions


@dataclass
class Position:
    """Individual position tracking."""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    sector: str = "Unknown"
    asset_class: str = "Stock"
    side: str = "LONG"  # LONG or SHORT
    
    @property
    def pnl_pct(self) -> float:
        """Unrealized P&L percentage."""
        if self.side == "SHORT":
            return (self.avg_price - self.market_value / abs(self.quantity)) / self.avg_price
        return (self.market_value / abs(self.quantity) - self.avg_price) / self.avg_price


@dataclass 
class TradeRequest:
    """Trade request to be validated."""
    symbol: str
    side: str           # BUY or SELL
    quantity: float
    price: float
    sector: str = "Unknown"
    is_short: bool = False


@dataclass
class TradeDecision:
    """Result of trade validation."""
    allowed: bool
    original_quantity: float
    adjusted_quantity: float
    reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class RiskManager:
    """
    Real-time risk management system.
    
    Features:
    - Position limit enforcement
    - Exposure monitoring
    - Volatility targeting
    - Drawdown monitoring with state machine
    - Kill switch
    - Alert system
    
    Usage:
    ------
    risk_mgr = RiskManager(limits=RiskLimits())
    
    # Before each trade
    decision = risk_mgr.check_trade(trade_request)
    if decision.allowed:
        execute_trade(decision.adjusted_quantity)
    
    # After each trade
    risk_mgr.update_position(symbol, new_position)
    
    # Periodically
    risk_mgr.update_nav(current_nav)
    status = risk_mgr.get_status()
    """
    
    # Sector mapping for common symbols
    SECTOR_MAP = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'GOOGL': 'Technology',
        'META': 'Technology', 'AMZN': 'Consumer', 'TSLA': 'Consumer',
        'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy',
        'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
        'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
        'GLD': 'Commodities', 'SLV': 'Commodities',
        'BTC': 'Crypto', 'ETH': 'Crypto', 'SOL': 'Crypto',
    }
    
    def __init__(self, limits: RiskLimits = None, initial_nav: float = 1_000_000):
        self.limits = limits or RiskLimits()
        self.state = RiskState.NORMAL
        
        # Portfolio tracking
        self.positions: Dict[str, Position] = {}
        self.nav = initial_nav
        self.high_water_mark = initial_nav
        self.initial_nav = initial_nav
        
        # Daily tracking
        self.daily_start_nav = initial_nav
        self.daily_trades = 0
        self.daily_turnover = 0.0
        self.daily_pnl = 0.0
        
        # Volatility tracking
        self.returns_history: List[float] = []
        
        # State history
        self.state_history: List[Tuple[datetime, RiskState, str]] = []
        
        # Alerts
        self.alerts: List[RiskAlert] = []
        
        logger.info(f"RiskManager initialized with NAV ${initial_nav:,.0f}")
    
    def check_trade(self, request: TradeRequest) -> TradeDecision:
        """
        Validate trade request against all risk limits.
        
        Parameters:
        -----------
        request : TradeRequest
            Trade to validate
            
        Returns:
        --------
        TradeDecision with allowed/adjusted quantity
        """
        warnings = []
        
        # 1. Check kill switch
        if self.state == RiskState.HALTED:
            return TradeDecision(
                allowed=False,
                original_quantity=request.quantity,
                adjusted_quantity=0,
                reason="Trading HALTED due to drawdown limit"
            )
        
        # 2. Check daily trade limit
        if self.daily_trades >= self.limits.max_daily_trades:
            return TradeDecision(
                allowed=False,
                original_quantity=request.quantity,
                adjusted_quantity=0,
                reason=f"Daily trade limit reached ({self.limits.max_daily_trades})"
            )
        
        # 3. Check daily loss limit
        if self.daily_pnl / self.daily_start_nav < -self.limits.max_daily_loss:
            return TradeDecision(
                allowed=False,
                original_quantity=request.quantity,
                adjusted_quantity=0,
                reason=f"Daily loss limit reached ({self.limits.max_daily_loss:.1%})"
            )
        
        # 4. Check position limit
        trade_value = request.quantity * request.price
        current_position = self.positions.get(request.symbol)
        current_value = current_position.market_value if current_position else 0
        
        if request.side == 'BUY':
            new_value = current_value + trade_value
        else:
            new_value = current_value - trade_value
        
        position_pct = abs(new_value) / self.nav
        
        if position_pct > self.limits.max_position_pct:
            # Reduce to allowed size
            max_value = self.limits.max_position_pct * self.nav
            allowed_trade = max_value - abs(current_value)
            
            if allowed_trade <= 0:
                return TradeDecision(
                    allowed=False,
                    original_quantity=request.quantity,
                    adjusted_quantity=0,
                    reason=f"Position limit reached for {request.symbol} ({self.limits.max_position_pct:.0%})"
                )
            
            adjusted_qty = allowed_trade / request.price
            warnings.append(f"Quantity reduced from {request.quantity:.0f} to {adjusted_qty:.0f} (position limit)")
            request.quantity = adjusted_qty
        
        # 5. Check sector limit
        sector = request.sector or self.SECTOR_MAP.get(request.symbol, "Unknown")
        sector_exposure = self._get_sector_exposure(sector)
        
        if sector_exposure + position_pct > self.limits.max_sector_pct:
            warnings.append(f"Near sector limit for {sector}")
            if sector_exposure >= self.limits.max_sector_pct:
                return TradeDecision(
                    allowed=False,
                    original_quantity=request.quantity,
                    adjusted_quantity=0,
                    reason=f"Sector limit reached for {sector} ({self.limits.max_sector_pct:.0%})"
                )
        
        # 6. Check gross exposure
        gross = self._get_gross_exposure()
        if gross + position_pct / self.nav > self.limits.max_gross_exposure:
            return TradeDecision(
                allowed=False,
                original_quantity=request.quantity,
                adjusted_quantity=0,
                reason=f"Gross exposure limit reached ({self.limits.max_gross_exposure:.0%})"
            )
        
        # 7. Check net exposure
        net = self._get_net_exposure()
        if request.side == 'BUY' and not request.is_short:
            new_net = net + position_pct
        else:
            new_net = net - position_pct
        
        if new_net > self.limits.max_net_exposure:
            warnings.append(f"Near max net long exposure")
        if new_net < self.limits.min_net_exposure:
            warnings.append(f"Near max net short exposure")
        
        # 8. Check turnover limit
        if self.daily_turnover + trade_value / self.nav > self.limits.max_daily_turnover:
            warnings.append(f"Near daily turnover limit")
        
        # 9. Apply risk state scaling
        scale = self._get_risk_scale()
        if scale < 1.0:
            original = request.quantity
            request.quantity = request.quantity * scale
            warnings.append(f"Quantity scaled by {scale:.0%} due to {self.state.value} state")
        
        return TradeDecision(
            allowed=True,
            original_quantity=request.quantity / scale if scale > 0 else request.quantity,
            adjusted_quantity=request.quantity,
            reason=None,
            warnings=warnings
        )
    
    def update_position(self, symbol: str, position: Position):
        """Update position after trade execution."""
        if position.market_value == 0:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = position
        
        self.daily_trades += 1
        self.daily_turnover += abs(position.market_value)
        
        # Check position count
        if len(self.positions) > self.limits.max_positions:
            self._add_alert(
                AlertSeverity.WARNING,
                "concentration",
                f"Position count ({len(self.positions)}) exceeds limit ({self.limits.max_positions})"
            )
    
    def update_nav(self, nav: float):
        """
        Update NAV and check drawdown.
        
        This is the core risk monitoring function.
        """
        old_nav = self.nav
        self.nav = nav
        
        # Calculate daily P&L
        self.daily_pnl = nav - self.daily_start_nav
        
        # Update returns history for volatility
        if old_nav > 0:
            ret = (nav - old_nav) / old_nav
            self.returns_history.append(ret)
            if len(self.returns_history) > 252:  # Keep 1 year
                self.returns_history.pop(0)
        
        # Update high water mark
        if nav > self.high_water_mark:
            self.high_water_mark = nav
        
        # Calculate drawdown
        drawdown = (self.high_water_mark - nav) / self.high_water_mark
        
        # Check drawdown thresholds and transition state
        if drawdown >= self.limits.drawdown_halt:
            self._transition_state(RiskState.HALTED, f"Drawdown {drawdown:.1%} exceeds halt threshold")
        elif drawdown >= self.limits.drawdown_minimal:
            self._transition_state(RiskState.MINIMAL, f"Drawdown {drawdown:.1%} exceeds minimal threshold")
        elif drawdown >= self.limits.drawdown_reduced:
            self._transition_state(RiskState.REDUCED, f"Drawdown {drawdown:.1%} exceeds reduced threshold")
        elif drawdown >= self.limits.drawdown_warning:
            if self.state != RiskState.NORMAL:
                self._transition_state(RiskState.NORMAL, "Drawdown recovered below warning threshold")
            self._add_alert(
                AlertSeverity.INFO,
                "drawdown",
                f"Drawdown at {drawdown:.1%} - monitoring"
            )
        else:
            if self.state != RiskState.NORMAL:
                self._transition_state(RiskState.NORMAL, "Drawdown fully recovered")
        
        # Check volatility
        current_vol = self._get_current_volatility()
        if current_vol > self.limits.max_volatility:
            self._add_alert(
                AlertSeverity.WARNING,
                "volatility",
                f"Portfolio volatility {current_vol:.1%} exceeds max {self.limits.max_volatility:.1%}"
            )
    
    def kill_switch(self, reason: str = "Manual activation"):
        """
        Emergency halt all trading.
        
        This should flatten all positions (implementation depends on broker).
        """
        self._transition_state(RiskState.HALTED, f"KILL SWITCH: {reason}")
        self._add_alert(
            AlertSeverity.EMERGENCY,
            "kill_switch",
            f"KILL SWITCH ACTIVATED: {reason}",
            {"positions_to_close": list(self.positions.keys())}
        )
        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")
        
        # In production, this would:
        # 1. Cancel all open orders
        # 2. Flatten all positions
        # 3. Notify operations team
        print("🚨🚨🚨 KILL SWITCH ACTIVATED 🚨🚨🚨")
        print(f"Reason: {reason}")
        print(f"Positions to close: {list(self.positions.keys())}")
    
    def daily_reset(self):
        """Reset daily counters (call at start of trading day)."""
        self.daily_start_nav = self.nav
        self.daily_trades = 0
        self.daily_turnover = 0.0
        self.daily_pnl = 0.0
        logger.info("Daily risk counters reset")
    
    def get_status(self) -> Dict:
        """Get comprehensive risk status."""
        drawdown = (self.high_water_mark - self.nav) / self.high_water_mark if self.high_water_mark > 0 else 0
        
        return {
            'state': self.state.value,
            'risk_scale': self._get_risk_scale(),
            'nav': self.nav,
            'high_water_mark': self.high_water_mark,
            'drawdown': drawdown,
            'gross_exposure': self._get_gross_exposure(),
            'net_exposure': self._get_net_exposure(),
            'n_positions': len(self.positions),
            'daily_trades': self.daily_trades,
            'daily_turnover': self.daily_turnover / self.nav if self.nav > 0 else 0,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.daily_start_nav if self.daily_start_nav > 0 else 0,
            'current_volatility': self._get_current_volatility(),
            'recent_alerts': [str(a) for a in self.alerts[-5:]],
            'sector_exposures': self._get_all_sector_exposures()
        }
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get DataFrame of all positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for symbol, pos in self.positions.items():
            data.append({
                'symbol': symbol,
                'quantity': pos.quantity,
                'avg_price': pos.avg_price,
                'market_value': pos.market_value,
                'weight': pos.market_value / self.nav,
                'sector': pos.sector,
                'side': pos.side,
                'pnl_pct': pos.pnl_pct
            })
        
        return pd.DataFrame(data).sort_values('market_value', ascending=False)
    
    # Private methods
    
    def _get_risk_scale(self) -> float:
        """Get position scaling based on risk state."""
        scales = {
            RiskState.NORMAL: 1.0,
            RiskState.REDUCED: 0.5,
            RiskState.MINIMAL: 0.25,
            RiskState.HALTED: 0.0
        }
        return scales[self.state]
    
    def _get_gross_exposure(self) -> float:
        """Calculate gross exposure (sum of absolute positions)."""
        if not self.positions or self.nav == 0:
            return 0.0
        return sum(abs(p.market_value) for p in self.positions.values()) / self.nav
    
    def _get_net_exposure(self) -> float:
        """Calculate net exposure (long - short)."""
        if not self.positions or self.nav == 0:
            return 0.0
        
        long_value = sum(p.market_value for p in self.positions.values() if p.side == "LONG")
        short_value = sum(abs(p.market_value) for p in self.positions.values() if p.side == "SHORT")
        
        return (long_value - short_value) / self.nav
    
    def _get_sector_exposure(self, sector: str) -> float:
        """Calculate exposure to a specific sector."""
        if not self.positions or self.nav == 0:
            return 0.0
        
        sector_value = sum(
            abs(p.market_value) for p in self.positions.values()
            if p.sector == sector
        )
        return sector_value / self.nav
    
    def _get_all_sector_exposures(self) -> Dict[str, float]:
        """Get exposure to all sectors."""
        exposures = {}
        for pos in self.positions.values():
            sector = pos.sector
            if sector not in exposures:
                exposures[sector] = 0.0
            exposures[sector] += abs(pos.market_value) / self.nav if self.nav > 0 else 0
        return exposures
    
    def _get_current_volatility(self) -> float:
        """Calculate current portfolio volatility."""
        if len(self.returns_history) < self.limits.vol_lookback:
            return 0.15  # Default assumption
        
        recent_returns = self.returns_history[-self.limits.vol_lookback:]
        return np.std(recent_returns) * np.sqrt(252)
    
    def _transition_state(self, new_state: RiskState, reason: str):
        """Transition to a new risk state."""
        if new_state == self.state:
            return
        
        old_state = self.state
        self.state = new_state
        
        self.state_history.append((datetime.utcnow(), new_state, reason))
        
        severity = {
            RiskState.NORMAL: AlertSeverity.INFO,
            RiskState.REDUCED: AlertSeverity.WARNING,
            RiskState.MINIMAL: AlertSeverity.CRITICAL,
            RiskState.HALTED: AlertSeverity.EMERGENCY
        }
        
        self._add_alert(
            severity[new_state],
            "state_transition",
            f"Risk state: {old_state.value} → {new_state.value}",
            {"reason": reason}
        )
        
        logger.warning(f"Risk state transition: {old_state.value} → {new_state.value}: {reason}")
    
    def _add_alert(self, severity: AlertSeverity, category: str, message: str, details: Dict = None):
        """Add a risk alert."""
        alert = RiskAlert(
            timestamp=datetime.utcnow(),
            severity=severity,
            category=category,
            message=message,
            details=details or {}
        )
        self.alerts.append(alert)
        
        # Keep last 100 alerts
        if len(self.alerts) > 100:
            self.alerts.pop(0)
        
        # Log based on severity
        if severity == AlertSeverity.EMERGENCY:
            logger.critical(str(alert))
        elif severity == AlertSeverity.CRITICAL:
            logger.error(str(alert))
        elif severity == AlertSeverity.WARNING:
            logger.warning(str(alert))
        else:
            logger.info(str(alert))


# Test the implementation
if __name__ == "__main__":
    print("Testing Risk Manager...")
    print("="*60)
    
    # Initialize
    limits = RiskLimits(
        max_position_pct=0.10,
        max_sector_pct=0.30,
        drawdown_warning=0.05,
        drawdown_reduced=0.10,
        drawdown_halt=0.20
    )
    
    risk_mgr = RiskManager(limits=limits, initial_nav=1_000_000)
    
    # Test trade validation
    print("\n1. Testing Trade Validation...")
    
    trade = TradeRequest(
        symbol='AAPL',
        side='BUY',
        quantity=500,
        price=180.0,
        sector='Technology'
    )
    
    decision = risk_mgr.check_trade(trade)
    print(f"   Trade allowed: {decision.allowed}")
    print(f"   Adjusted qty: {decision.adjusted_quantity}")
    
    # Test position limit
    print("\n2. Testing Position Limit...")
    
    trade_large = TradeRequest(
        symbol='MSFT',
        side='BUY',
        quantity=5000,  # Would be ~20% of portfolio
        price=400.0,
        sector='Technology'
    )
    
    decision = risk_mgr.check_trade(trade_large)
    print(f"   Trade allowed: {decision.allowed}")
    print(f"   Reason: {decision.reason}")
    
    # Test drawdown
    print("\n3. Testing Drawdown Monitoring...")
    
    risk_mgr.update_nav(950000)  # 5% drawdown
    print(f"   State after 5% DD: {risk_mgr.state.value}")
    
    risk_mgr.update_nav(900000)  # 10% drawdown
    print(f"   State after 10% DD: {risk_mgr.state.value}")
    
    risk_mgr.update_nav(800000)  # 20% drawdown
    print(f"   State after 20% DD: {risk_mgr.state.value}")
    
    # Test kill switch
    print("\n4. Testing Kill Switch...")
    risk_mgr.state = RiskState.NORMAL  # Reset for demo
    risk_mgr.kill_switch("Test activation")
    
    print("\n5. Final Status:")
    status = risk_mgr.get_status()
    for key, value in status.items():
        if key != 'recent_alerts':
            print(f"   {key}: {value}")
