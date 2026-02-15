#!/usr/bin/env python3
"""
NUBLE Paper Trading Simulator

This module simulates real trading with paper money to validate
the system before deploying real capital.

Key Features:
- Realistic execution simulation
- Transaction cost modeling
- Daily P&L tracking
- Performance comparison to backtest
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
from pathlib import Path
import random

import numpy as np
import pandas as pd

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    from config.production_config import ProductionConfig, get_config
except ImportError:
    # Fallback if not in path
    ProductionConfig = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"


class OrderSide(Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """Represents a trading order."""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    fill_time: Optional[datetime] = None
    order_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000,9999)}"


@dataclass
class Position:
    """Represents a portfolio position."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity
    
    @property
    def pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class TradeRecord:
    """Record of an executed trade."""
    symbol: str
    side: str
    quantity: float
    price: float
    cost: float
    timestamp: datetime
    pnl: float = 0.0


class PaperTrader:
    """
    Paper trading simulator for strategy validation.
    
    Simulates realistic trading execution without real money.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost_bps: float = 10,
        slippage_bps: float = 5,
        config: Optional['ProductionConfig'] = None
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.config = config
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[TradeRecord] = []
        
        # Daily tracking
        self.daily_values: Dict[str, float] = {}
        self.daily_pnl: Dict[str, float] = {}
        
        # Performance metrics
        self.high_water_mark = initial_capital
        self.max_drawdown = 0.0
        
        # Logging
        self.trade_log: List[Dict] = []
        
    @property
    def total_value(self) -> float:
        """Total portfolio value."""
        position_value = sum(p.market_value for p in self.positions.values())
        return self.cash + position_value
    
    @property
    def current_drawdown(self) -> float:
        """Current drawdown from high water mark."""
        if self.high_water_mark == 0:
            return 0.0
        return (self.high_water_mark - self.total_value) / self.high_water_mark
    
    def update_prices(self, prices: Dict[str, float]):
        """Update position prices with current market data."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price
        
        # Update high water mark
        if self.total_value > self.high_water_mark:
            self.high_water_mark = self.total_value
        
        # Update max drawdown
        current_dd = self.current_drawdown
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
    
    def place_order(self, order: Order) -> Order:
        """Place an order (simulated execution)."""
        logger.info(f"Placing order: {order.side.value} {order.quantity:.2f} {order.symbol}")
        
        self.orders.append(order)
        
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            self._execute_order(order)
        
        return order
    
    def _execute_order(self, order: Order):
        """Execute an order with simulated slippage and costs."""
        # Get base price (would come from real data in production)
        if order.symbol in self.positions:
            base_price = self.positions[order.symbol].current_price
        else:
            # Default price for simulation
            base_price = 100.0
        
        # Apply slippage
        slippage = base_price * (self.slippage_bps / 10000)
        if order.side == OrderSide.BUY:
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage
        
        # Calculate costs
        trade_value = order.quantity * fill_price
        transaction_cost = trade_value * (self.transaction_cost_bps / 10000)
        total_cost = trade_value + transaction_cost
        
        # Check if we have enough cash for buys
        if order.side == OrderSide.BUY:
            if total_cost > self.cash:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: Insufficient cash (${self.cash:.2f} < ${total_cost:.2f})")
                return
        
        # Execute the trade
        order.status = OrderStatus.FILLED
        order.fill_price = fill_price
        order.fill_time = datetime.now()
        
        # Update positions
        if order.side == OrderSide.BUY:
            self._add_position(order.symbol, order.quantity, fill_price)
            self.cash -= total_cost
        else:
            pnl = self._remove_position(order.symbol, order.quantity, fill_price)
            self.cash += trade_value - transaction_cost
        
        # Record trade
        trade = TradeRecord(
            symbol=order.symbol,
            side=order.side.value,
            quantity=order.quantity,
            price=fill_price,
            cost=transaction_cost,
            timestamp=datetime.now(),
            pnl=pnl if order.side == OrderSide.SELL else 0.0
        )
        self.trades.append(trade)
        
        logger.info(f"Filled: {order.side.value} {order.quantity:.2f} {order.symbol} @ ${fill_price:.2f}")
    
    def _add_position(self, symbol: str, quantity: float, price: float):
        """Add to a position."""
        if symbol in self.positions:
            # Average into existing position
            pos = self.positions[symbol]
            total_value = (pos.quantity * pos.avg_cost) + (quantity * price)
            total_quantity = pos.quantity + quantity
            pos.avg_cost = total_value / total_quantity
            pos.quantity = total_quantity
            pos.current_price = price
        else:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                current_price=price
            )
    
    def _remove_position(self, symbol: str, quantity: float, price: float) -> float:
        """Remove from a position. Returns realized P&L."""
        if symbol not in self.positions:
            logger.warning(f"No position in {symbol} to sell")
            return 0.0
        
        pos = self.positions[symbol]
        pnl = (price - pos.avg_cost) * min(quantity, pos.quantity)
        
        pos.quantity -= quantity
        if pos.quantity <= 0:
            del self.positions[symbol]
        
        return pnl
    
    def record_daily(self, date: str):
        """Record daily portfolio value."""
        value = self.total_value
        self.daily_values[date] = value
        
        # Calculate daily P&L
        dates = sorted(self.daily_values.keys())
        if len(dates) > 1:
            prev_date = dates[-2]
            prev_value = self.daily_values[prev_date]
            self.daily_pnl[date] = value - prev_value
        else:
            self.daily_pnl[date] = value - self.initial_capital
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary."""
        if not self.daily_values:
            return {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'total_trades': 0,
                'win_rate': 0.0,
            }
        
        # Calculate returns
        returns = []
        dates = sorted(self.daily_values.keys())
        for i in range(1, len(dates)):
            prev_val = self.daily_values[dates[i-1]]
            curr_val = self.daily_values[dates[i]]
            if prev_val > 0:
                returns.append((curr_val - prev_val) / prev_val)
        
        returns = np.array(returns)
        
        # Calculate Sharpe
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Calculate win rate
        winning_trades = [t for t in self.trades if t.pnl > 0]
        total_closed = len([t for t in self.trades if t.side == 'sell'])
        win_rate = len(winning_trades) / max(total_closed, 1)
        
        # Total return
        total_return = (self.total_value - self.initial_capital) / self.initial_capital
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'total_value': self.total_value,
            'cash': self.cash,
            'positions': len(self.positions),
            'realized_pnl': sum(t.pnl for t in self.trades),
            'transaction_costs': sum(t.cost for t in self.trades),
        }
    
    def save_state(self, filepath: str):
        """Save trader state to file."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': {s: asdict(p) for s, p in self.positions.items()},
            'daily_values': self.daily_values,
            'performance': self.get_performance_summary(),
            'trades': [
                {
                    'symbol': t.symbol,
                    'side': t.side,
                    'quantity': t.quantity,
                    'price': t.price,
                    'cost': t.cost,
                    'pnl': t.pnl,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in self.trades[-100:]  # Last 100 trades
            ]
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load trader state from file."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.initial_capital = state['initial_capital']
        self.cash = state['cash']
        self.daily_values = state['daily_values']
        
        # Rebuild positions
        for symbol, pos_data in state['positions'].items():
            self.positions[symbol] = Position(**pos_data)
        
        logger.info(f"State loaded from {filepath}")


def run_paper_trading_demo():
    """Run a demo of paper trading."""
    print("="*60)
    print("NUBLE Paper Trading Demo")
    print("="*60)
    
    # Initialize trader
    trader = PaperTrader(
        initial_capital=100000,
        transaction_cost_bps=10,
        slippage_bps=5
    )
    
    print(f"\nInitial Capital: ${trader.initial_capital:,.2f}")
    
    # Simulated price data
    symbols = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMD']
    prices = {
        'AAPL': 185.0,
        'MSFT': 420.0,
        'NVDA': 850.0,
        'GOOGL': 165.0,
        'AMD': 175.0
    }
    
    # Update prices
    trader.update_prices(prices)
    
    # Simulate trading over 10 days
    print("\n📈 Simulating 10 days of trading...\n")
    
    for day in range(1, 11):
        date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
        
        # Simulate price movements
        for symbol in prices:
            change = random.uniform(-0.03, 0.04)  # -3% to +4%
            prices[symbol] *= (1 + change)
        
        trader.update_prices(prices)
        
        # Random trade based on "signal"
        if random.random() > 0.5:
            symbol = random.choice(symbols)
            side = random.choice([OrderSide.BUY, OrderSide.SELL])
            
            # Calculate quantity
            if side == OrderSide.BUY:
                # Buy up to 10% of portfolio
                max_value = trader.cash * 0.1
                quantity = max_value / prices[symbol]
            else:
                # Sell existing position
                if symbol in trader.positions:
                    quantity = trader.positions[symbol].quantity * 0.5
                else:
                    continue
            
            order = Order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=OrderType.MARKET
            )
            order = trader.place_order(order)
        
        # Record daily value
        trader.record_daily(date)
        
        print(f"Day {day}: Portfolio Value: ${trader.total_value:,.2f} | "
              f"Cash: ${trader.cash:,.2f} | "
              f"Positions: {len(trader.positions)}")
    
    # Final summary
    print("\n" + "="*60)
    print("TRADING SUMMARY")
    print("="*60)
    
    summary = trader.get_performance_summary()
    
    print(f"\n📊 Performance:")
    print(f"  Total Return: {summary['total_return']:+.2%}")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {summary['max_drawdown']:.2%}")
    
    print(f"\n📈 Activity:")
    print(f"  Total Trades: {summary['total_trades']}")
    print(f"  Win Rate: {summary['win_rate']:.0%}")
    print(f"  Transaction Costs: ${summary['transaction_costs']:.2f}")
    
    print(f"\n💰 Portfolio:")
    print(f"  Final Value: ${summary['total_value']:,.2f}")
    print(f"  Cash: ${summary['cash']:,.2f}")
    print(f"  Positions: {summary['positions']}")
    print(f"  Realized P&L: ${summary['realized_pnl']:,.2f}")
    
    # Show positions
    if trader.positions:
        print("\n📋 Current Positions:")
        for symbol, pos in trader.positions.items():
            print(f"  {symbol}: {pos.quantity:.2f} shares @ ${pos.avg_cost:.2f} "
                  f"(Current: ${pos.current_price:.2f}, P&L: {pos.pnl_pct:+.2%})")
    
    # Save state
    import os
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    save_path = os.path.join(_root, "data", "paper_trading_state.json")
    trader.save_state(save_path)
    print(f"\n💾 State saved to: {save_path}")
    
    print("\n" + "="*60)
    print("Demo complete. Ready for production paper trading!")
    print("="*60)


if __name__ == "__main__":
    run_paper_trading_demo()
