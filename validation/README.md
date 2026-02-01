# RIGOROUS VALIDATION FRAMEWORK

## Overview

This validation framework implements institutional-grade backtesting following Lopez de Prado's methodology from "Advances in Financial Machine Learning" (2018).

## Key Insight

**If Sharpe > 3.0, there are bugs.**

- Renaissance Medallion (best hedge fund ever): Sharpe 2-3
- Top quant funds: Sharpe 1.5-2.5
- Realistic target: Sharpe 1.0-1.5
- Anything above 3.0 indicates lookahead bias or other issues

## Components

### 1. Data Acquisition (`data_downloader.py`)
- Uses your Polygon.io paid subscription
- Downloads full historical data 2015-2026
- Strict train/test split:
  - Train: 2015-01-01 to 2022-12-31 (8 years)
  - Test: 2023-01-01 to 2026-01-31 (3+ years, NEVER touched during development)

### 2. Lookahead Bias Audit (`lookahead_audit.py`)
- Checks Triple Barrier volatility uses only past data
- Checks labels determined at barrier touch, not earlier
- Checks fractional differentiation doesn't use future data
- Checks CV splits have proper purging

### 3. Walk-Forward Validation (`walk_forward.py`)
- Expanding window training
- Purge gap (5 days) between train and test
- Embargo period (5 days) after test
- Transaction costs (0.1% round-trip) included
- Calculates:
  - Sharpe, Sortino, Calmar ratios
  - Max drawdown
  - Win rate, profit factor

### 4. CPCV with PBO (`cpcv.py`)
- Combinatorial Purged Cross-Validation
- Tests ALL possible train/test combinations
- Probability of Backtest Overfitting (PBO):
  - PBO > 0.5 = Strategy is likely overfit
  - PBO < 0.3 = Reasonable generalization
- Deflated Sharpe Ratio (adjusts for multiple testing)

### 5. Orchestrator (`orchestrator.py`)
- Runs complete validation pipeline
- Integrates with Phase 1+2 components

## Usage

```python
from validation.orchestrator import ValidationOrchestrator

# Create with your pipelines
orchestrator = ValidationOrchestrator(
    feature_pipeline=my_features,
    label_pipeline=my_labels,
    model_factory=my_model
)

# Run validation (don't run final test until ready!)
results = orchestrator.run_complete_validation(
    download_data=True,
    run_audit=True,
    run_walk_forward=True,
    run_cpcv=True,
    run_final_test=False  # Only True at the VERY END
)
```

## Quick Test Results

Testing with simple Random Forest + basic features on 5 symbols:

| Symbol | Sharpe | Return |
|--------|--------|--------|
| SPY    | -0.21  | -16.3% |
| QQQ    | -0.21  | -24.1% |
| TSLA   | 0.27   | -3.3%  |
| AAPL   | 0.70   | 163.4% |
| GLD    | -0.88  | -27.4% |

**Aggregate: Sharpe = -0.07 ± 0.53**

This is EXACTLY what we expect from a naive strategy - it doesn't beat the market after transaction costs.

## Directory Structure

```
data/
├── historical/    # Raw data from Polygon.io
├── train/         # Training data (2015-2022)
├── test/          # Test data (2023-2026) - DO NOT TOUCH!
└── results/       # Validation results

validation/
├── __init__.py
├── config.py           # Central configuration
├── data_downloader.py  # Polygon.io data fetching
├── lookahead_audit.py  # Bias detection
├── walk_forward.py     # Walk-forward validation
├── cpcv.py            # CPCV + PBO + Deflated Sharpe
├── orchestrator.py    # Complete pipeline
└── run_validation.py  # Integration script
```

## Configuration

Key settings in `config.py`:

```python
# Training period
train_start = "2015-01-01"
train_end = "2022-12-31"

# Test period (NEVER touch during development!)
test_start = "2023-01-01"
test_end = "2026-01-31"

# Walk-forward settings
train_size = 756  # ~3 years
test_size = 63    # ~3 months
purge_size = 5    # Days purged
embargo_size = 5  # Days embargoed

# Transaction costs
transaction_cost = 0.001  # 0.1% round-trip
```

## Next Steps

1. **Integrate Phase 1+2 Pipeline**
   - Use TripleBarrierLabeler for realistic labels
   - Use FractionalDifferentiator for stationary features
   - Use HMMRegimeDetector for regime features
   - Use MetaLabeler for bet sizing

2. **Run Walk-Forward Validation**
   - If Sharpe > 2.0, check for bugs
   - If Sharpe 1.0-2.0, proceed carefully
   - If Sharpe < 1.0, improve strategy

3. **Run CPCV with PBO**
   - If PBO > 0.30, reduce model complexity
   - If DSR not significant, increase sample size

4. **Final OOS Test**
   - Only run ONCE at the very end
   - This is your true performance estimate
   - If it differs significantly from walk-forward, you overfit

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey et al. (2014). "The Probability of Backtest Overfitting"
- Bailey & López de Prado (2014). "The Deflated Sharpe Ratio"
