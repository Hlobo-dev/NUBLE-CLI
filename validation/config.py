"""
VALIDATION CONFIGURATION
=========================
Central configuration for the rigorous validation framework.
"""

import os
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class WalkForwardConfig:
    """Walk-forward validation parameters."""
    train_size: int = 756      # ~3 years
    test_size: int = 63        # ~3 months
    purge_size: int = 5        # Days to purge
    embargo_size: int = 5      # Days to embargo


@dataclass
class CPCVConfig:
    """CPCV validation parameters."""
    n_splits: int = 6          # Number of groups
    n_test_groups: int = 2     # Groups for testing
    purge_size: int = 5        # Days to purge
    embargo_size: int = 5      # Days to embargo


@dataclass
class ValidationConfig:
    """Master configuration for validation framework."""
    
    # Sub-configs
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    cpcv: CPCVConfig = field(default_factory=CPCVConfig)
    
    # Polygon.io API (you have the most expensive subscription!)
    # Default to your key, but can be overridden by environment variable
    polygon_api_key: str = field(default_factory=lambda: os.getenv('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY'))
    
    # Data paths
    data_dir: Path = field(default_factory=lambda: Path('data'))
    historical_dir: Path = field(default_factory=lambda: Path('data/historical'))
    train_dir: Path = field(default_factory=lambda: Path('data/train'))
    test_dir: Path = field(default_factory=lambda: Path('data/test'))
    
    # Temporal split (CRITICAL - no peeking at test data!)
    train_start: str = "2015-01-01"
    train_end: str = "2022-12-31"
    test_start: str = "2023-01-01"
    test_end: str = "2026-01-31"
    
    # Symbols to validate
    symbols: List[str] = field(default_factory=lambda: [
        # Primary test targets
        "SLV", "TSLA",
        # Broad market
        "SPY", "QQQ",
        # Commodities
        "GLD", "USO",
        # Bonds
        "TLT", "IEF",
        # Sectors
        "XLF", "XLE", "XLK",
        # Large caps
        "AAPL", "NVDA", "AMD", "MSFT", "GOOGL", "AMZN", "META",
        # Financials
        "JPM", "BAC",
        # Energy
        "XOM", "CVX",
    ])
    
    # Walk-forward parameters
    wf_train_size: int = 252 * 3  # 3 years of trading days
    wf_test_size: int = 63        # 3 months (1 quarter)
    wf_step_size: int = 63        # Step forward by 1 quarter
    wf_purge_size: int = 5        # Days to purge between train/test
    wf_embargo_size: int = 5      # Days to embargo after test
    
    # Transaction costs (realistic)
    transaction_cost: float = 0.001  # 0.1% (10 bps round-trip)
    slippage: float = 0.0005         # 0.05% slippage
    risk_free_rate: float = 0.02     # 2% annual risk-free rate
    
    # Model parameters
    n_estimators: int = 100
    max_depth: int = 5
    random_state: int = 42
    
    # Results directory
    results_dir: Path = field(default_factory=lambda: Path('data/results'))
    
    # Expected realistic ranges (for sanity checks)
    max_realistic_sharpe: float = 2.0  # Anything above is suspicious
    min_realistic_win_rate: float = 0.51
    max_realistic_win_rate: float = 0.58
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for dir_path in [self.data_dir, self.historical_dir, self.train_dir, self.test_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)


# Global config instance
CONFIG = ValidationConfig()
