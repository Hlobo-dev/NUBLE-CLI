# AFML ML TRADING SYSTEM - PROGRESS REPORT

## Current Status: Priority 3 (Sample Weighting) In Progress

### Completed

✅ **Phase 1+2 Implementation**
- Triple Barrier Labeling with volatility-adjusted thresholds
- Fractional Differentiation for stationarity
- HMM Regime Detection (Bull/Bear markets)
- Meta-Labeling for bet sizing

✅ **Validation Framework** (8 files created)
- Walk-forward validation with purge/embargo
- CPCV with PBO analysis
- Naive strategy benchmarks
- Transaction cost modeling (0.1% round-trip)

✅ **Data Infrastructure**
- 22 symbols downloaded from Polygon.io
- Training period: 2015-2022 (~2014 rows per symbol)
- Test period: 2023-2026 (NEVER TOUCHED until final)

✅ **Priority 1: ML Primary Signal** 
- Created `ml_primary_signal.py` with 66 features
- Feature categories: Momentum, Volatility, Technical, Volume
- All features have ZERO lookahead bias

✅ **Priority 2: Regime-Adaptive Pipeline**
- Created `regime_adaptive_validation.py`
- Separate models for Bull vs Bear regimes
- HMM correctly detecting regimes:
  - SPY: Bull 1182 days (Vol=9.9%), Bear 226 days (Vol=40.3%)

### Results Progression

| Phase | Average Sharpe | Notes |
|-------|----------------|-------|
| Original Phase 1+2 | -0.12 | No edge |
| With ML Primary Signal | -0.12 | Same baseline |
| Regime-Adaptive | **+0.06** | Improvement! |
| Sample Weighted | In Progress | |

### Per-Symbol Results (Regime-Adaptive)

| Symbol | Sharpe | Return | vs Naive |
|--------|--------|--------|----------|
| SPY | +0.23 | +5.6% | +0.77 |
| QQQ | -0.09 | -9.5% | +0.40 |
| AAPL | -0.70 | -35.0% | -0.48 |
| TSLA | +0.79 | +89.0% | -0.07 |
| **AVG** | **+0.06** | | **+0.15** |

### In Progress

⏳ **Priority 3: AFML Sample Weighting**
- Created `sample_weights.py` with uniqueness-based weights
- Updated MetaLabeler to accept sample_weight parameter
- Issue discovered: Primary signals all zero due to probability threshold
- Need to fix primary signal → meta-label integration

### Pending

⏳ **Priority 4: Final OOS Test**
- Blocked until: Sharpe > 0.5, PBO < 0.5
- Test data in `/data/test/{SYMBOL}.csv`
- Currently not qualified

### Key Files

```
src/institutional/
├── models/
│   ├── primary/
│   │   └── ml_primary_signal.py    # 66-feature ML primary signal
│   └── meta/
│       └── meta_labeler.py          # Updated with sample_weight
├── regime/
│   └── hmm_detector.py              # HMM regime detection
├── validation/
│   └── sample_weights.py            # AFML sample weighting (Ch 4)

validation/
├── regime_adaptive_validation.py    # Working regime-adaptive pipeline
├── sample_weighted_validation2.py   # Sample weighting (needs fix)
└── cpcv.py                          # CPCV with PBO analysis
```

### Technical Notes

**Issue Found During Sample Weighting Integration:**
- MetaLabeler's `create_meta_labels()` filters where `primary_signals != 0`
- Primary signals were all 0 because probability threshold too tight (0.55/0.45)
- Model outputs probabilities close to 0.5, so no signals generated
- Result: Meta-labeler gets 0 samples

**Solution Needed:**
1. Lower primary signal threshold (e.g., >0.51 for long, <0.49 for short)
2. Or use sign of probability - 0.5 as continuous signal
3. Or remove probability-based filtering, use direct classification

### API Credentials

- **Polygon.io**: `JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY` (PAID subscription)

### Environment

- Python 3.14 with venv at `.venv`
- PyTorch 2.10.0 with MPS (Apple Silicon)
- Key packages: numba, hmmlearn, scikit-learn, pyarrow

### Validation Thresholds (Per User Instructions)

- **Sharpe > 3.0** = RED FLAG (catching bugs correctly)
- **Target realistic Sharpe**: 0.5-1.5
- **PBO < 0.5** before OOS testing
- Renaissance Medallion (best ever): Sharpe 2-3

### Next Steps

1. **Fix primary signal threshold** to generate non-zero signals
2. **Complete sample weighting validation** 
3. **Run CPCV** on improved pipeline
4. If Sharpe > 0.5 and PBO < 0.5: **Final OOS test**
