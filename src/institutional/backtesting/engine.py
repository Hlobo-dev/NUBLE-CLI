"""
Institutional-Grade Backtesting Framework
==========================================

Production backtesting engine with:
- Event-driven architecture
- Walk-forward optimization
- Transaction cost modeling
- Slippage simulation
- Performance attribution
- Risk analytics

Architecture:
    Historical Data → Event Queue → Strategy → Broker → Portfolio
    
References:
- Ernest Chan, "Quantitative Trading"
- Lopez de Prado, "Advances in Financial Machine Learning"
- Marcos Lopez de Prado, "The 10 Reasons Most Machine Learning Funds Fail"
"""

import logging
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, Generator, List, 
    Optional, Set, Tuple, Union
)
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class EventType(Enum):
    """Event types for event-driven backtest."""
    MARKET = "market"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


@dataclass
class MarketEvent:
    """Market data event."""
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None


@dataclass
class Order:
    """Trade order."""
    id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0


@dataclass
class Fill:
    """Order fill/execution."""
    order_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float
    slippage: float


@dataclass
class Position:
    """Portfolio position."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return abs(self.quantity) * self.avg_price
        
    @property
    def is_long(self) -> bool:
        return self.quantity > 0
        
    @property
    def is_short(self) -> bool:
        return self.quantity < 0
        
    @property
    def is_flat(self) -> bool:
        return self.quantity == 0


@dataclass
class BacktestConfig:
    """
    Backtesting configuration.
    """
    # Capital
    initial_capital: float = 100000
    
    # Transaction costs
    commission_rate: float = 0.001  # 10 bps
    slippage_model: str = "fixed"  # fixed, volume, volatility
    slippage_bps: float = 5  # 5 bps
    
    # Risk
    max_position_pct: float = 0.1  # 10% max per position
    max_leverage: float = 1.0  # No leverage by default
    
    # Execution
    fill_at_open: bool = False  # False = fill at close
    allow_shorting: bool = True
    fractional_shares: bool = True
    
    # Walk-forward
    train_window_days: int = 252  # 1 year training
    test_window_days: int = 63  # 3 months testing
    min_train_samples: int = 100
    
    # Validation
    warmup_period: int = 60  # Days to warm up indicators


class TransactionCostModel:
    """
    Transaction cost and slippage modeling.
    
    Models:
    - Fixed: Constant slippage in basis points
    - Volume: Slippage proportional to % of volume
    - Volatility: Slippage proportional to volatility
    - Square root: Market impact model (institutional)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def calculate_slippage(
        self,
        order: Order,
        market: MarketEvent,
        avg_volume: float = 1000000
    ) -> float:
        """
        Calculate slippage for order.
        
        Args:
            order: The order
            market: Current market data
            avg_volume: Average daily volume
            
        Returns:
            Slippage as price adjustment
        """
        if self.config.slippage_model == "fixed":
            return self._fixed_slippage(order, market)
        elif self.config.slippage_model == "volume":
            return self._volume_slippage(order, market, avg_volume)
        elif self.config.slippage_model == "volatility":
            return self._volatility_slippage(order, market)
        elif self.config.slippage_model == "sqrt":
            return self._sqrt_impact(order, market, avg_volume)
        else:
            return self._fixed_slippage(order, market)
            
    def _fixed_slippage(self, order: Order, market: MarketEvent) -> float:
        """Fixed basis points slippage."""
        direction = 1 if order.side == OrderSide.BUY else -1
        return market.close * (self.config.slippage_bps / 10000) * direction
        
    def _volume_slippage(
        self, 
        order: Order, 
        market: MarketEvent,
        avg_volume: float
    ) -> float:
        """Volume-dependent slippage."""
        participation_rate = order.quantity / max(avg_volume, 1)
        
        # Slippage increases with participation
        bps = self.config.slippage_bps * (1 + participation_rate * 10)
        
        direction = 1 if order.side == OrderSide.BUY else -1
        return market.close * (bps / 10000) * direction
        
    def _volatility_slippage(self, order: Order, market: MarketEvent) -> float:
        """Volatility-based slippage."""
        # Use high-low as volatility proxy
        volatility = (market.high - market.low) / market.close
        
        bps = self.config.slippage_bps * (1 + volatility * 100)
        
        direction = 1 if order.side == OrderSide.BUY else -1
        return market.close * (bps / 10000) * direction
        
    def _sqrt_impact(
        self,
        order: Order,
        market: MarketEvent,
        avg_volume: float
    ) -> float:
        """
        Square-root market impact model.
        
        Impact = σ * sqrt(Q/V)
        
        Reference: Almgren-Chriss model
        """
        volatility = (market.high - market.low) / market.close
        participation = order.quantity / max(avg_volume, 1)
        
        impact = volatility * np.sqrt(participation)
        
        direction = 1 if order.side == OrderSide.BUY else -1
        return market.close * impact * direction
        
    def calculate_commission(self, order: Order, fill_price: float) -> float:
        """Calculate commission."""
        notional = order.quantity * fill_price
        return notional * self.config.commission_rate


class Portfolio:
    """
    Portfolio state management.
    
    Tracks:
    - Positions
    - Cash
    - P&L
    - Risk metrics
    """
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        
        # History
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.returns: List[float] = []
        self.trades: List[Dict[str, Any]] = []
        
        # Metrics
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
    @property
    def equity(self) -> float:
        """Total portfolio value."""
        positions_value = sum(
            pos.quantity * pos.avg_price 
            for pos in self.positions.values()
        )
        return self.cash + positions_value
        
    @property
    def gross_exposure(self) -> float:
        """Gross exposure (sum of absolute positions)."""
        return sum(
            abs(pos.quantity) * pos.avg_price 
            for pos in self.positions.values()
        )
        
    @property
    def net_exposure(self) -> float:
        """Net exposure (long - short)."""
        return sum(
            pos.quantity * pos.avg_price 
            for pos in self.positions.values()
        )
        
    @property
    def leverage(self) -> float:
        """Current leverage."""
        return self.gross_exposure / max(self.equity, 1)
        
    def update_position(self, fill: Fill):
        """Update position from fill."""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            
        pos = self.positions[symbol]
        
        if fill.side == OrderSide.BUY:
            # Buying
            if pos.quantity >= 0:
                # Adding to long
                total_cost = pos.quantity * pos.avg_price + fill.quantity * fill.price
                pos.quantity += fill.quantity
                pos.avg_price = total_cost / pos.quantity if pos.quantity != 0 else 0
            else:
                # Covering short
                if fill.quantity <= abs(pos.quantity):
                    # Partial cover
                    pos.realized_pnl += (pos.avg_price - fill.price) * fill.quantity
                    pos.quantity += fill.quantity
                else:
                    # Cover and go long
                    cover_qty = abs(pos.quantity)
                    pos.realized_pnl += (pos.avg_price - fill.price) * cover_qty
                    long_qty = fill.quantity - cover_qty
                    pos.quantity = long_qty
                    pos.avg_price = fill.price
        else:
            # Selling
            if pos.quantity <= 0:
                # Adding to short
                total_cost = abs(pos.quantity) * pos.avg_price + fill.quantity * fill.price
                pos.quantity -= fill.quantity
                pos.avg_price = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0
            else:
                # Closing long
                if fill.quantity <= pos.quantity:
                    # Partial close
                    pos.realized_pnl += (fill.price - pos.avg_price) * fill.quantity
                    pos.quantity -= fill.quantity
                else:
                    # Close and go short
                    close_qty = pos.quantity
                    pos.realized_pnl += (fill.price - pos.avg_price) * close_qty
                    short_qty = fill.quantity - close_qty
                    pos.quantity = -short_qty
                    pos.avg_price = fill.price
                    
        # Update cash
        if fill.side == OrderSide.BUY:
            self.cash -= fill.quantity * fill.price + fill.commission
        else:
            self.cash += fill.quantity * fill.price - fill.commission
            
        # Record trade
        self.trades.append({
            'timestamp': fill.timestamp,
            'symbol': symbol,
            'side': fill.side.value,
            'quantity': fill.quantity,
            'price': fill.price,
            'commission': fill.commission,
            'slippage': fill.slippage,
        })
        
    def mark_to_market(self, prices: Dict[str, float], timestamp: datetime):
        """Mark positions to market."""
        for symbol, pos in self.positions.items():
            if symbol in prices and pos.quantity != 0:
                current_price = prices[symbol]
                if pos.is_long:
                    pos.unrealized_pnl = (current_price - pos.avg_price) * pos.quantity
                else:
                    pos.unrealized_pnl = (pos.avg_price - current_price) * abs(pos.quantity)
                    
        # Record equity
        current_equity = self.equity
        self.equity_curve.append((timestamp, current_equity))
        
        # Update drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)
        
        # Calculate return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2][1]
            ret = (current_equity - prev_equity) / prev_equity
            self.returns.append(ret)


class Strategy(ABC):
    """
    Abstract base strategy.
    
    Subclass and implement generate_signals() for custom strategies.
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self._warmup_complete = False
        self._history: Dict[str, pd.DataFrame] = {}
        
    @abstractmethod
    def generate_signals(
        self,
        market_data: Dict[str, MarketEvent]
    ) -> Dict[str, Tuple[OrderSide, float]]:
        """
        Generate trading signals.
        
        Args:
            market_data: Dict of symbol -> MarketEvent
            
        Returns:
            Dict of symbol -> (side, target_weight)
            target_weight is the target portfolio weight [-1, 1]
        """
        pass
        
    def update_history(self, market: MarketEvent):
        """Update internal price history."""
        symbol = market.symbol
        
        if symbol not in self._history:
            self._history[symbol] = pd.DataFrame(
                columns=['open', 'high', 'low', 'close', 'volume']
            )
            
        row = pd.DataFrame([{
            'open': market.open,
            'high': market.high,
            'low': market.low,
            'close': market.close,
            'volume': market.volume,
        }], index=[market.timestamp])
        
        self._history[symbol] = pd.concat([self._history[symbol], row])
        
    def get_history(self, symbol: str, lookback: int = 100) -> pd.DataFrame:
        """Get price history for symbol."""
        if symbol in self._history:
            return self._history[symbol].tail(lookback)
        return pd.DataFrame()


class ModelStrategy(Strategy):
    """
    Strategy that uses ML model predictions.
    """
    
    def __init__(
        self,
        symbols: List[str],
        model,
        feature_extractor: Callable,
        threshold: float = 0.5
    ):
        super().__init__(symbols)
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        
    def generate_signals(
        self,
        market_data: Dict[str, MarketEvent]
    ) -> Dict[str, Tuple[OrderSide, float]]:
        """Generate signals from model predictions."""
        signals = {}
        
        for symbol, market in market_data.items():
            self.update_history(market)
            
            # Get history
            history = self.get_history(symbol)
            if len(history) < 60:
                continue
                
            # Extract features
            features = self.feature_extractor(history)
            
            # Get prediction
            prediction = self.model.predict(features)
            
            if prediction > self.threshold:
                signals[symbol] = (OrderSide.BUY, min(prediction, 1.0))
            elif prediction < -self.threshold:
                signals[symbol] = (OrderSide.SELL, min(abs(prediction), 1.0))
                
        return signals


class Broker:
    """
    Simulated broker for order execution.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.cost_model = TransactionCostModel(config)
        
        self._pending_orders: List[Order] = []
        self._order_counter = 0
        
    def submit_order(self, order: Order) -> str:
        """Submit order for execution."""
        self._order_counter += 1
        order.id = f"ORD-{self._order_counter:06d}"
        self._pending_orders.append(order)
        return order.id
        
    def process_orders(
        self,
        market_data: Dict[str, MarketEvent]
    ) -> List[Fill]:
        """Process pending orders against market data."""
        fills = []
        remaining_orders = []
        
        for order in self._pending_orders:
            if order.symbol not in market_data:
                remaining_orders.append(order)
                continue
                
            market = market_data[order.symbol]
            fill = self._try_fill(order, market)
            
            if fill:
                fills.append(fill)
            elif order.status == OrderStatus.PENDING:
                remaining_orders.append(order)
                
        self._pending_orders = remaining_orders
        return fills
        
    def _try_fill(self, order: Order, market: MarketEvent) -> Optional[Fill]:
        """Try to fill order against market."""
        # Determine fill price
        if order.order_type == OrderType.MARKET:
            if self.config.fill_at_open:
                base_price = market.open
            else:
                base_price = market.close
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY:
                if market.low <= order.price:
                    base_price = min(order.price, market.close)
                else:
                    return None
            else:
                if market.high >= order.price:
                    base_price = max(order.price, market.close)
                else:
                    return None
        elif order.order_type == OrderType.STOP:
            if order.side == OrderSide.BUY:
                if market.high >= order.stop_price:
                    base_price = max(order.stop_price, market.close)
                else:
                    return None
            else:
                if market.low <= order.stop_price:
                    base_price = min(order.stop_price, market.close)
                else:
                    return None
        else:
            base_price = market.close
            
        # Calculate slippage
        slippage = self.cost_model.calculate_slippage(order, market)
        fill_price = base_price + slippage
        
        # Calculate commission
        commission = self.cost_model.calculate_commission(order, fill_price)
        
        # Create fill
        fill = Fill(
            order_id=order.id,
            timestamp=market.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            slippage=slippage,
        )
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.commission = commission
        order.slippage = slippage
        
        return fill
        
    def cancel_order(self, order_id: str):
        """Cancel pending order."""
        for order in self._pending_orders:
            if order.id == order_id:
                order.status = OrderStatus.CANCELLED
                break
                
        self._pending_orders = [
            o for o in self._pending_orders 
            if o.id != order_id
        ]


class BacktestEngine:
    """
    Main backtesting engine.
    
    Usage:
        engine = BacktestEngine(config)
        results = engine.run(strategy, data)
        print(results.summary())
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.portfolio = Portfolio(self.config.initial_capital)
        self.broker = Broker(self.config)
        
        self._data: Dict[str, pd.DataFrame] = {}
        self._current_date: Optional[datetime] = None
        
    def load_data(self, data: Dict[str, pd.DataFrame]):
        """
        Load historical data.
        
        Args:
            data: Dict of symbol -> DataFrame with OHLCV columns
        """
        for symbol, df in data.items():
            # Ensure datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Missing column {col} in {symbol}")
                    
            self._data[symbol] = df.sort_index()
            
    def run(
        self,
        strategy: Strategy,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> 'BacktestResults':
        """
        Run backtest.
        
        Args:
            strategy: Trading strategy
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            BacktestResults object
        """
        if not self._data:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Determine date range
        all_dates = set()
        for df in self._data.values():
            all_dates.update(df.index.tolist())
            
        dates = sorted(all_dates)
        
        if start_date:
            dates = [d for d in dates if d >= start_date]
        if end_date:
            dates = [d for d in dates if d <= end_date]
            
        # Warmup
        warmup_end = min(self.config.warmup_period, len(dates) // 4)
        
        logger.info(f"Running backtest from {dates[warmup_end]} to {dates[-1]}")
        
        # Main loop
        for i, date in enumerate(dates):
            self._current_date = date
            
            # Get market data for this date
            market_data = {}
            prices = {}
            
            for symbol, df in self._data.items():
                if date in df.index:
                    row = df.loc[date]
                    market_data[symbol] = MarketEvent(
                        timestamp=date,
                        symbol=symbol,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume'],
                        vwap=row.get('vwap'),
                    )
                    prices[symbol] = row['close']
                    
            # Process pending orders
            fills = self.broker.process_orders(market_data)
            for fill in fills:
                self.portfolio.update_position(fill)
                
            # Skip warmup period for signals
            if i < warmup_end:
                self.portfolio.mark_to_market(prices, date)
                continue
                
            # Generate signals
            signals = strategy.generate_signals(market_data)
            
            # Convert signals to orders
            for symbol, (side, weight) in signals.items():
                if symbol not in market_data:
                    continue
                    
                self._execute_signal(symbol, side, weight, market_data[symbol])
                
            # Mark to market
            self.portfolio.mark_to_market(prices, date)
            
        return BacktestResults(self.portfolio, self.config)
        
    def _execute_signal(
        self,
        symbol: str,
        side: OrderSide,
        weight: float,
        market: MarketEvent
    ):
        """Convert signal to order and submit."""
        # Calculate target position
        target_value = self.portfolio.equity * weight * self.config.max_position_pct
        
        current_pos = self.portfolio.positions.get(symbol)
        current_value = 0
        
        if current_pos:
            current_value = current_pos.quantity * market.close
            
        # Calculate order size
        if side == OrderSide.BUY:
            order_value = target_value - current_value
        else:
            order_value = -target_value - current_value
            
        if abs(order_value) < 100:  # Minimum order size
            return
            
        quantity = abs(order_value / market.close)
        
        if not self.config.fractional_shares:
            quantity = int(quantity)
            
        if quantity <= 0:
            return
            
        # Check shorting allowed
        if not self.config.allow_shorting and order_value < 0:
            return
            
        # Create and submit order
        order = Order(
            id="",
            timestamp=market.timestamp,
            symbol=symbol,
            side=OrderSide.BUY if order_value > 0 else OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
        )
        
        self.broker.submit_order(order)


@dataclass
class BacktestResults:
    """
    Backtest results and analytics.
    """
    portfolio: Portfolio
    config: BacktestConfig
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.portfolio.returns:
            return {'error': 'No returns calculated'}
            
        returns = np.array(self.portfolio.returns)
        
        # Calculate metrics
        total_return = (self.portfolio.equity / self.config.initial_capital - 1) * 100
        cagr = self._calculate_cagr()
        volatility = np.std(returns) * np.sqrt(252) * 100
        sharpe = self._calculate_sharpe()
        sortino = self._calculate_sortino()
        max_dd = self.portfolio.max_drawdown * 100
        calmar = cagr / max_dd if max_dd > 0 else 0
        
        # Trade statistics
        n_trades = len(self.portfolio.trades)
        
        return {
            'total_return_pct': round(total_return, 2),
            'cagr_pct': round(cagr, 2),
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe, 3),
            'sortino_ratio': round(sortino, 3),
            'max_drawdown_pct': round(max_dd, 2),
            'calmar_ratio': round(calmar, 3),
            'n_trades': n_trades,
            'final_equity': round(self.portfolio.equity, 2),
            'initial_capital': self.config.initial_capital,
        }
        
    def _calculate_cagr(self) -> float:
        """Calculate CAGR."""
        if not self.portfolio.equity_curve:
            return 0.0
            
        start = self.portfolio.equity_curve[0]
        end = self.portfolio.equity_curve[-1]
        
        years = (end[0] - start[0]).days / 365.25
        if years <= 0:
            return 0.0
            
        return ((end[1] / start[1]) ** (1 / years) - 1) * 100
        
    def _calculate_sharpe(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio."""
        returns = np.array(self.portfolio.returns)
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
        
    def _calculate_sortino(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio."""
        returns = np.array(self.portfolio.returns)
        if len(returns) == 0:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252
        downside = returns[returns < 0]
        
        if len(downside) == 0 or downside.std() == 0:
            return 0.0
            
        return np.sqrt(252) * excess_returns.mean() / downside.std()
        
    def equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        df = pd.DataFrame(
            self.portfolio.equity_curve,
            columns=['timestamp', 'equity']
        )
        df.set_index('timestamp', inplace=True)
        return df
        
    def trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame."""
        return pd.DataFrame(self.portfolio.trades)
        
    def monthly_returns(self) -> pd.Series:
        """Get monthly returns."""
        equity = self.equity_curve_df()
        return equity['equity'].resample('M').last().pct_change().dropna()
        
    def drawdown_series(self) -> pd.Series:
        """Get drawdown time series."""
        equity = self.equity_curve_df()['equity']
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        return drawdown


class WalkForwardOptimizer:
    """
    Walk-forward optimization for strategy validation.
    
    Process:
    1. Split data into train/test folds
    2. Optimize parameters on training data
    3. Validate on out-of-sample test data
    4. Roll forward and repeat
    
    Reference: Robert Pardo, "The Evaluation and Optimization of Trading Strategies"
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def generate_folds(
        self,
        data: pd.DataFrame
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test folds.
        
        Returns:
            List of (train_data, test_data) tuples
        """
        folds = []
        
        dates = data.index.unique().sort_values()
        n_dates = len(dates)
        
        train_days = self.config.train_window_days
        test_days = self.config.test_window_days
        
        i = 0
        while i + train_days + test_days <= n_dates:
            train_end = i + train_days
            test_end = train_end + test_days
            
            train_dates = dates[i:train_end]
            test_dates = dates[train_end:test_end]
            
            train_data = data[data.index.isin(train_dates)]
            test_data = data[data.index.isin(test_dates)]
            
            if len(train_data) >= self.config.min_train_samples:
                folds.append((train_data, test_data))
                
            i += test_days  # Roll forward by test window
            
        return folds
        
    def run(
        self,
        strategy_factory: Callable[..., Strategy],
        data: Dict[str, pd.DataFrame],
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        Args:
            strategy_factory: Factory function that creates strategy given params
            data: Historical data
            param_grid: Parameter grid to search
            
        Returns:
            Optimization results
        """
        # Combine all data
        all_data = pd.concat(data.values())
        
        # Generate folds
        folds = self.generate_folds(all_data)
        
        logger.info(f"Walk-forward: {len(folds)} folds")
        
        # Results storage
        fold_results = []
        
        for fold_idx, (train_data, test_data) in enumerate(folds):
            logger.info(f"Fold {fold_idx + 1}/{len(folds)}")
            
            # Get unique dates for this fold
            train_symbols = {}
            test_symbols = {}
            
            for symbol, df in data.items():
                train_mask = df.index.isin(train_data.index)
                test_mask = df.index.isin(test_data.index)
                
                if train_mask.any():
                    train_symbols[symbol] = df[train_mask]
                if test_mask.any():
                    test_symbols[symbol] = df[test_mask]
                    
            # Grid search on training data
            best_params = None
            best_sharpe = float('-inf')
            
            for params in self._param_combinations(param_grid):
                # Create strategy with these params
                strategy = strategy_factory(**params)
                
                # Backtest on training data
                engine = BacktestEngine(self.config)
                engine.load_data(train_symbols)
                results = engine.run(strategy)
                
                sharpe = results.summary().get('sharpe_ratio', 0)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = params
                    
            # Test with best params
            if best_params:
                strategy = strategy_factory(**best_params)
                engine = BacktestEngine(self.config)
                engine.load_data(test_symbols)
                test_results = engine.run(strategy)
                
                fold_results.append({
                    'fold': fold_idx,
                    'train_sharpe': best_sharpe,
                    'test_sharpe': test_results.summary().get('sharpe_ratio', 0),
                    'best_params': best_params,
                    'test_return': test_results.summary().get('total_return_pct', 0),
                })
                
        # Aggregate results
        return {
            'folds': fold_results,
            'avg_train_sharpe': np.mean([f['train_sharpe'] for f in fold_results]),
            'avg_test_sharpe': np.mean([f['test_sharpe'] for f in fold_results]),
            'sharpe_decay': 1 - np.mean([f['test_sharpe'] for f in fold_results]) / 
                           max(np.mean([f['train_sharpe'] for f in fold_results]), 0.01),
        }
        
    def _param_combinations(
        self, 
        param_grid: Dict[str, List[Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate parameter combinations."""
        if not param_grid:
            yield {}
            return
            
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        from itertools import product
        
        for combo in product(*values):
            yield dict(zip(keys, combo))


class MonteCarloSimulator:
    """
    Monte Carlo simulation for risk analysis.
    
    Methods:
    - Bootstrap resampling
    - Block bootstrap (preserves autocorrelation)
    - Parametric simulation
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
        
    def bootstrap(
        self,
        returns: np.ndarray,
        n_days: int = 252
    ) -> np.ndarray:
        """
        Simple bootstrap simulation.
        
        Returns:
            Array of shape (n_simulations, n_days)
        """
        n = len(returns)
        simulated = np.zeros((self.n_simulations, n_days))
        
        for i in range(self.n_simulations):
            indices = np.random.randint(0, n, size=n_days)
            simulated[i] = returns[indices]
            
        return simulated
        
    def block_bootstrap(
        self,
        returns: np.ndarray,
        n_days: int = 252,
        block_size: int = 22  # Monthly blocks
    ) -> np.ndarray:
        """
        Block bootstrap preserving autocorrelation.
        """
        n = len(returns)
        n_blocks = n_days // block_size + 1
        simulated = np.zeros((self.n_simulations, n_days))
        
        for i in range(self.n_simulations):
            sim_returns = []
            
            for _ in range(n_blocks):
                # Random block start
                start = np.random.randint(0, max(n - block_size, 1))
                block = returns[start:start + block_size]
                sim_returns.extend(block)
                
            simulated[i] = sim_returns[:n_days]
            
        return simulated
        
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Value at Risk."""
        cumulative_returns = np.cumprod(1 + returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        var = np.percentile(final_returns, (1 - confidence) * 100)
        return var
        
    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence: float = 0.95
    ) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        cumulative_returns = np.cumprod(1 + returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        var = np.percentile(final_returns, (1 - confidence) * 100)
        cvar = final_returns[final_returns <= var].mean()
        
        return cvar
        
    def probability_of_loss(
        self,
        returns: np.ndarray,
        threshold: float = 0.0
    ) -> float:
        """Calculate probability of loss beyond threshold."""
        cumulative_returns = np.cumprod(1 + returns, axis=1) - 1
        final_returns = cumulative_returns[:, -1]
        
        return (final_returns < threshold).mean()
