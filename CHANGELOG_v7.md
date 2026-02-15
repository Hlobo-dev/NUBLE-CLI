# CHANGELOG — NUBLE v7.0.0

## v7.0.0 — Institutional Upgrade (Phase 1 + Phase 2)

### Added

#### Phase 1 — New Modules
- `src/nuble/data/polygon_universe.py` (~560 lines) — Full US stock universe daily OHLCV via Polygon grouped daily bars. Supports backfill with resume, single-ticker API fallback, active universe filtering (>$5, >$1M volume).
- `src/nuble/data/sec_edgar.py` (~766 lines) — SEC EDGAR XBRL parser. Downloads quarterly/annual filings for any ticker, computes 33 fundamental ratios (profitability, liquidity, leverage, efficiency, valuation, growth), provides quality scores.
- `src/nuble/data/fred_macro.py` (~306 lines) — FRED macro data pipeline. 8 raw series (GDP, CPI, unemployment, etc.) + 4 derived indicators (term spread, credit spread, real rates) + 3 regime classifications (expansion/recession/neutral).
- `src/nuble/ml/universal_model.py` (~740 lines) — ONE LightGBM model trained on ALL stocks. 27 stock-agnostic features (RSI, ROC, volatility, moving average crossovers, etc.), triple-barrier labeling, time-based train/val/test split with purge gap, 4 quality gates (IC > 0.01, accuracy > 40%, features > 20, no single feature > 30%).

#### Phase 2 — Scripts & Infrastructure
- `scripts/backfill_universe.py` (~190 lines) — Backfill script with resume capability, progress tracking with ETA, post-backfill validation, argparse (--quick/--days/--start/--end).
- `scripts/train_universal.py` (~340 lines) — Training script with adaptive panel building, API fallback for missing stocks, comprehensive evaluation report, prediction testing on 5 symbols.
- `src/nuble/ml/model_manager.py` (~280 lines) — Model lifecycle manager: freshness checks (stale after 7 days), health verification (loadable, quality gates), background retraining (non-blocking, threaded), CLI/API integration.

### Changed

#### Phase 1 — Integration
- `src/nuble/ml/predictor.py` — Universal model as primary prediction source; falls through to per-ticker models if universal unavailable.
- `src/nuble/ml/__init__.py` — Exports for UniversalTechnicalModel, ModelManager, get_model_manager.
- `src/nuble/agents/fundamental_analyst.py` — SEC EDGAR XBRL integration for fundamental analysis.
- `src/nuble/agents/macro_analyst.py` — FRED macro integration for macro regime analysis.
- `src/nuble/agents/orchestrator.py` — Model type logging (universal vs per-ticker).
- `src/nuble/manager.py` — Model type display, model freshness check (once per session).
- `src/nuble/decision/enrichment_engine.py` — XBRL ratios + FRED regimes + quality score enrichment.
- `src/nuble/data/__init__.py` — Exports for 3 new data modules.

#### Phase 2 — Fixes & Improvements
- `src/nuble/data/polygon_universe.py` — Fixed Parquet schema mismatch (timestamp[ms] vs timestamp[s]) in `_append_to_parquet()` via `pa.unify_schemas()`. Added default Polygon API key.
- `src/nuble/cli.py` — Added ML model health to `/status` command via ModelManager.
- `pyproject.toml` — Added missing dependencies: numpy, pandas (with version pins), optional deps for ML (lightgbm, shap, scikit-learn, scipy, joblib, statsmodels, pyarrow, torch) and data (fredapi, boto3).

### Removed
- `src/nuble/core/unified_orchestrator.py` (992 lines) — Unused alternative orchestrator, superseded by `agents/orchestrator.py`.
- `src/nuble/core/tools.py` (415 lines) — Tool registry for unused core orchestrator.
- `src/nuble/core/tool_handlers.py` (894 lines) — Tool handlers for unused core orchestrator.
- `src/nuble/core/memory.py` (499 lines) — Memory module for unused core orchestrator, superseded by root-level `memory.py`.
- `tests/test_integration_unified.py` (287 lines) — Tests for removed core modules.

**Total dead code removed: 3,087 lines**

### Training Metrics — Production (2,000 stocks)
- Stocks used: 1,834 (131 skipped — insufficient history)
- Total samples: 645,985
- Training split: 452,189 train / 96,888 val / 96,888 test (purge gap = 10)
- Features: 27
- Date range: 2024-08-07 → 2026-01-22
- **Test IC: 0.0300** (threshold > 0.01) ✅
- **Test Accuracy: 45.2%** (threshold > 40%, random = 33%) ✅
- **Features used: 27** (threshold > 20) ✅
- **Max feature importance: 21.4%** (threshold < 30%) ✅
- All 4 quality gates passed ✅
- Best iteration: 176 (early stopped at patience 100)
- Training time: 3.1 min total (2.8 min panel build + 14s LightGBM)
- Model size: 3,392 KB
- Top 5 features: vol_regime (22.9%), month_sin (12.5%), month_cos (11.5%), vol_of_vol_63 (6.2%), garman_klass_vol (5.8%)
- Label distribution: SHORT 39%, NEUTRAL 15%, LONG 46%

### Data Backfill — Production
- Dates: 499 (2024-02-12 → 2026-02-06)
- Total rows: 4,229,755
- Active stocks: 5,662 (filtered from 10,935)
- Size: 133.4 MB across 25 Parquet files
- Source: Polygon Premium grouped daily bars
