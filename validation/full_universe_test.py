"""
Full Universe Validation Test

The OOS test only used 3 symbols (AAPL, MSFT, NVDA).
This validates across ALL 22 symbols in the downloaded dataset.

Key questions:
1. What is the survivorship bias in our symbol selection?
2. Which symbols fail and why?
3. Is the strategy robust across different sectors/volatilities?

This is CRITICAL for institutional deployment.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


@dataclass
class SymbolResult:
    """Results for a single symbol."""
    symbol: str
    sector: str
    
    # Data quality
    has_data: bool
    data_start: Optional[datetime] = None
    data_end: Optional[datetime] = None
    n_bars: int = 0
    pct_missing: float = 0.0
    
    # Strategy metrics
    sharpe: float = np.nan
    returns: float = np.nan
    max_drawdown: float = np.nan
    win_rate: float = np.nan
    n_trades: int = 0
    
    # Validation metrics
    pbo: float = np.nan
    is_viable: bool = False
    failure_reason: Optional[str] = None


class FullUniverseValidator:
    """
    Validate strategy across full symbol universe.
    
    Tests all 22 symbols downloaded from Polygon.io.
    """
    
    # Full symbol universe
    SYMBOLS = [
        # Tech
        'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'GOOGL', 'META', 'AMZN', 'TSLA',
        # Financials
        'JPM', 'BAC', 'GS',
        # Healthcare
        'JNJ', 'PFE', 'UNH',
        # Energy
        'XOM', 'CVX',
        # ETFs
        'SPY', 'QQQ', 'IWM', 'DIA',
        # Commodities
        'GLD'
    ]
    
    SECTOR_MAP = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'NVDA': 'Technology',
        'AMD': 'Technology', 'INTC': 'Technology', 'GOOGL': 'Technology',
        'META': 'Technology', 'AMZN': 'Consumer', 'TSLA': 'Consumer',
        'JPM': 'Financials', 'BAC': 'Financials', 'GS': 'Financials',
        'JNJ': 'Healthcare', 'PFE': 'Healthcare', 'UNH': 'Healthcare',
        'XOM': 'Energy', 'CVX': 'Energy',
        'SPY': 'Index', 'QQQ': 'Index', 'IWM': 'Index', 'DIA': 'Index',
        'GLD': 'Commodities'
    }
    
    def __init__(self, data_dir: str = None):
        """Initialize validator."""
        if data_dir is None:
            # Default to expected location
            self.data_dir = Path(__file__).parent.parent / 'data' / 'polygon'
        else:
            self.data_dir = Path(data_dir)
        
        self.results: Dict[str, SymbolResult] = {}
        
    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load data for a single symbol."""
        # Try different file patterns
        patterns = [
            f"{symbol}.csv",
            f"{symbol}_daily.csv",
            f"{symbol}_1d.csv",
        ]
        
        for pattern in patterns:
            filepath = self.data_dir / pattern
            if filepath.exists():
                df = pd.read_csv(filepath)
                
                # Standardize column names
                df.columns = [c.lower() for c in df.columns]
                
                # Parse date
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                elif 'timestamp' in df.columns:
                    df['date'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('date')
                
                # Ensure required columns
                required = ['open', 'high', 'low', 'close', 'volume']
                if all(c in df.columns for c in required):
                    return df.sort_index()
        
        return None
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        """Check data quality metrics."""
        if df is None or len(df) == 0:
            return {
                'is_valid': False,
                'reason': 'No data'
            }
        
        # Basic checks
        n_bars = len(df)
        
        if n_bars < 252:  # Less than 1 year
            return {
                'is_valid': False,
                'reason': f'Insufficient data: {n_bars} bars (need 252+)'
            }
        
        # Check for gaps
        df = df.sort_index()
        dates = pd.DatetimeIndex(df.index)
        
        # Calculate business day gaps
        expected_days = pd.bdate_range(dates[0], dates[-1])
        pct_missing = 1 - len(df) / len(expected_days) if len(expected_days) > 0 else 0
        
        if pct_missing > 0.10:  # More than 10% missing
            return {
                'is_valid': False,
                'reason': f'Too many gaps: {pct_missing:.1%} missing'
            }
        
        # Check for stale prices
        close_changes = df['close'].pct_change().abs()
        pct_zero_moves = (close_changes < 0.0001).sum() / len(close_changes)
        
        if pct_zero_moves > 0.50:  # More than 50% zero moves (illiquid)
            return {
                'is_valid': False,
                'reason': f'Illiquid: {pct_zero_moves:.1%} days with no price change'
            }
        
        # Check for extreme moves (data errors)
        pct_extreme = (close_changes > 0.50).sum() / len(close_changes)  # 50%+ moves
        if pct_extreme > 0.01:  # More than 1%
            return {
                'is_valid': False,
                'reason': f'Data quality: {pct_extreme:.1%} extreme moves (>50%)'
            }
        
        # Check for reasonable volume
        avg_volume = df['volume'].mean()
        if avg_volume < 100000:
            return {
                'is_valid': False,
                'reason': f'Low liquidity: avg volume {avg_volume:,.0f}'
            }
        
        return {
            'is_valid': True,
            'n_bars': n_bars,
            'start': df.index[0],
            'end': df.index[-1],
            'pct_missing': pct_missing,
            'avg_volume': avg_volume
        }
    
    def compute_simple_strategy(self, df: pd.DataFrame) -> Dict:
        """
        Compute simple momentum strategy for validation.
        
        This isn't the full ML strategy - it's a simplified version
        to test data viability across symbols.
        """
        if df is None or len(df) < 252:
            return {'is_valid': False, 'reason': 'Insufficient data'}
        
        # Simple momentum strategy
        returns = df['close'].pct_change()
        momentum_20 = df['close'].pct_change(20)
        volatility = returns.rolling(20).std()
        
        # Signal: long if positive momentum, vol-weighted
        signal = np.sign(momentum_20) / (volatility + 0.01)
        signal = signal / signal.abs().rolling(252).mean()  # Normalize
        signal = signal.clip(-2, 2)  # Cap at 2x
        
        # Strategy returns
        strat_returns = signal.shift(1) * returns
        strat_returns = strat_returns.dropna()
        
        if len(strat_returns) < 126:  # 6 months
            return {'is_valid': False, 'reason': 'Not enough valid signals'}
        
        # Metrics
        sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252) if strat_returns.std() > 0 else 0
        total_return = (1 + strat_returns).prod() - 1
        
        # Drawdown
        cum_returns = (1 + strat_returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdown = (cum_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # Win rate
        win_rate = (strat_returns > 0).mean()
        
        # Trade count (simplified)
        signal_changes = signal.diff().abs() > 0.5
        n_trades = signal_changes.sum()
        
        return {
            'is_valid': True,
            'sharpe': sharpe,
            'returns': total_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'n_trades': int(n_trades),
            'n_days': len(strat_returns)
        }
    
    def validate_symbol(self, symbol: str) -> SymbolResult:
        """Full validation for a single symbol."""
        result = SymbolResult(
            symbol=symbol,
            sector=self.SECTOR_MAP.get(symbol, 'Unknown')
        )
        
        # Load data
        df = self.load_symbol_data(symbol)
        
        if df is None:
            result.has_data = False
            result.failure_reason = "No data file found"
            return result
        
        result.has_data = True
        
        # Check data quality
        quality = self.check_data_quality(df)
        
        if not quality['is_valid']:
            result.failure_reason = quality['reason']
            return result
        
        result.data_start = quality['start']
        result.data_end = quality['end']
        result.n_bars = quality['n_bars']
        result.pct_missing = quality['pct_missing']
        
        # Run strategy
        strategy = self.compute_simple_strategy(df)
        
        if not strategy['is_valid']:
            result.failure_reason = strategy['reason']
            return result
        
        result.sharpe = strategy['sharpe']
        result.returns = strategy['returns']
        result.max_drawdown = strategy['max_drawdown']
        result.win_rate = strategy['win_rate']
        result.n_trades = strategy['n_trades']
        
        # Determine viability
        # Threshold: Sharpe > 0, DD < 50%, at least 50 trades
        if strategy['sharpe'] > 0 and strategy['max_drawdown'] < 0.50 and strategy['n_trades'] >= 50:
            result.is_viable = True
        else:
            reasons = []
            if strategy['sharpe'] <= 0:
                reasons.append(f"Low Sharpe ({strategy['sharpe']:.2f})")
            if strategy['max_drawdown'] >= 0.50:
                reasons.append(f"High DD ({strategy['max_drawdown']:.1%})")
            if strategy['n_trades'] < 50:
                reasons.append(f"Low trade count ({strategy['n_trades']})")
            result.failure_reason = "; ".join(reasons)
        
        return result
    
    def validate_universe(self) -> pd.DataFrame:
        """Validate all symbols in universe."""
        print("="*70)
        print("FULL UNIVERSE VALIDATION TEST")
        print("="*70)
        print(f"\nValidating {len(self.SYMBOLS)} symbols...")
        print(f"Data directory: {self.data_dir}")
        print()
        
        for symbol in self.SYMBOLS:
            print(f"  Testing {symbol}...", end=" ")
            result = self.validate_symbol(symbol)
            self.results[symbol] = result
            
            if result.is_viable:
                print(f"✅ Sharpe={result.sharpe:.2f}, Return={result.returns:.1%}")
            elif result.has_data:
                print(f"⚠️ {result.failure_reason}")
            else:
                print(f"❌ {result.failure_reason}")
        
        return self.get_summary()
    
    def get_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of all results."""
        data = []
        for symbol, result in self.results.items():
            data.append({
                'symbol': symbol,
                'sector': result.sector,
                'has_data': result.has_data,
                'is_viable': result.is_viable,
                'n_bars': result.n_bars,
                'sharpe': result.sharpe,
                'returns': result.returns,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'n_trades': result.n_trades,
                'failure_reason': result.failure_reason
            })
        
        return pd.DataFrame(data)
    
    def print_report(self):
        """Print comprehensive validation report."""
        df = self.get_summary()
        
        print("\n" + "="*70)
        print("UNIVERSE VALIDATION REPORT")
        print("="*70)
        
        # Overall stats
        n_total = len(df)
        n_has_data = df['has_data'].sum()
        n_viable = df['is_viable'].sum()
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total symbols: {n_total}")
        print(f"   With data: {n_has_data} ({n_has_data/n_total:.0%})")
        print(f"   Viable: {n_viable} ({n_viable/n_total:.0%})")
        print(f"   Pass rate: {n_viable/n_has_data:.0%}" if n_has_data > 0 else "   Pass rate: N/A")
        
        # Viable symbols
        viable = df[df['is_viable']]
        if len(viable) > 0:
            print(f"\n✅ VIABLE SYMBOLS ({len(viable)}):")
            for _, row in viable.iterrows():
                print(f"   {row['symbol']:6s} [{row['sector']:12s}] Sharpe={row['sharpe']:.2f}, Return={row['returns']:.1%}, DD={row['max_drawdown']:.1%}")
            
            print(f"\n   Average Sharpe: {viable['sharpe'].mean():.2f}")
            print(f"   Average Return: {viable['returns'].mean():.1%}")
            print(f"   Average MaxDD: {viable['max_drawdown'].mean():.1%}")
        
        # Failed symbols
        failed = df[~df['is_viable']]
        if len(failed) > 0:
            print(f"\n❌ FAILED SYMBOLS ({len(failed)}):")
            for _, row in failed.iterrows():
                reason = row['failure_reason'] if pd.notna(row['failure_reason']) else "Unknown"
                print(f"   {row['symbol']:6s} [{row['sector']:12s}] {reason}")
        
        # Sector breakdown
        print(f"\n📁 SECTOR BREAKDOWN:")
        for sector in df['sector'].unique():
            sector_df = df[df['sector'] == sector]
            sector_viable = sector_df['is_viable'].sum()
            sector_total = len(sector_df)
            sector_avg_sharpe = sector_df[sector_df['is_viable']]['sharpe'].mean()
            
            status = "✅" if sector_viable > 0 else "⚠️"
            sharpe_str = f"Avg Sharpe={sector_avg_sharpe:.2f}" if not pd.isna(sector_avg_sharpe) else ""
            print(f"   {status} {sector:12s}: {sector_viable}/{sector_total} viable {sharpe_str}")
        
        # Survivorship bias warning
        print(f"\n⚠️ SURVIVORSHIP BIAS CHECK:")
        if n_viable < n_total * 0.5:
            print(f"   CAUTION: Only {n_viable/n_total:.0%} of symbols are viable!")
            print(f"   The OOS test may overstate strategy performance.")
            print(f"   Production should include failure detection logic.")
        else:
            print(f"   {n_viable/n_total:.0%} pass rate is reasonable.")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if n_viable >= 10:
            print(f"   1. Use top {min(10, n_viable)} symbols by Sharpe for production")
        if len(failed) > 0:
            no_data = failed[~failed['has_data']]
            if len(no_data) > 0:
                print(f"   2. Download data for: {', '.join(no_data['symbol'].tolist())}")
        
        # Combined Sharpe if we traded all viable
        if len(viable) > 0:
            # Simplified combined Sharpe (assumes equal weight, 0.5 correlation)
            avg_sharpe = viable['sharpe'].mean()
            n = len(viable)
            combined_sharpe = avg_sharpe * np.sqrt(n) / np.sqrt(1 + (n-1) * 0.5)
            print(f"   3. Combined portfolio Sharpe estimate: {combined_sharpe:.2f}")
            print(f"      (assumes equal weight, 0.5 avg correlation)")
        
        print("\n" + "="*70)
        
        return df


def main():
    """Run full universe validation."""
    # Try to find data directory
    possible_dirs = [
        Path(__file__).parent.parent / 'data' / 'polygon',
        Path(__file__).parent.parent / 'data',
        Path.home() / 'polygon_data',
    ]
    
    data_dir = None
    for d in possible_dirs:
        if d.exists():
            # Check if any CSV files exist
            if list(d.glob('*.csv')):
                data_dir = d
                break
    
    if data_dir is None:
        print("⚠️ No data directory found!")
        print("   Please ensure polygon data is downloaded to one of:")
        for d in possible_dirs:
            print(f"     - {d}")
        print("\n   Creating mock test with synthetic data...")
        data_dir = Path(__file__).parent / 'test_data'
        create_mock_data(data_dir)
    
    validator = FullUniverseValidator(data_dir=str(data_dir))
    results = validator.validate_universe()
    validator.print_report()
    
    # Save results
    output_path = Path(__file__).parent / 'universe_validation_results.csv'
    results.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    return results


def create_mock_data(data_dir: Path):
    """Create mock data for testing when real data unavailable."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock data for a few symbols
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    
    for symbol in ['AAPL', 'MSFT', 'NVDA', 'SPY']:
        np.random.seed(hash(symbol) % 2**32)
        
        returns = np.random.normal(0.0005, 0.02, len(dates))
        price = 100 * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'date': dates,
            'open': price * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
            'high': price * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
            'low': price * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
            'close': price,
            'volume': np.random.uniform(1e6, 1e8, len(dates))
        })
        
        df.to_csv(data_dir / f'{symbol}.csv', index=False)
    
    print(f"   Created mock data in {data_dir}")


if __name__ == "__main__":
    main()
