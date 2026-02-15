"""
WRDS ML Predictor v2 — Multi-Tier Ensemble Bridge
===================================================
Bridges the trained multi-tier LightGBM ensemble into the APEX agent pipeline.

BEFORE (v1): Single lgb_latest_model.txt → IC ≈ 0.014 → no tier awareness
AFTER  (v2): 4 tier-specific models → IC 0.027-0.096 → routes by market cap

This module:
1. Loads 4 per-tier LightGBM models (mega/large/mid/small)
2. Loads the GKX panel and ticker↔PERMNO mapping
3. Classifies stocks by market cap into tiers
4. Routes each prediction to the correct tier model
5. Computes cross-sectional rank WITHIN tier (not across all stocks)
6. Applies tier-specific ensemble weights and strategies
7. Falls back to S3DataManager when local data unavailable
8. Provides get_top_picks(), get_tier_predictions(), get_market_regime()

Tier Configuration (System B — DO NOT CHANGE):
  Mega  (>$10B):   IC=0.027, weight=14.8%, strategy=raw
  Large ($2-10B):   IC=0.019, weight=10.5%, strategy=hedged
  Mid   ($500M-2B): IC=0.038, weight=21.3%, strategy=raw
  Small (<$500M):   IC=0.096, weight=53.4%, strategy=vix_scaled

Usage:
    from nuble.ml.wrds_predictor import get_wrds_predictor
    predictor = get_wrds_predictor()
    signal = predictor.predict("AAPL")   # Routes to mega-cap model
    signal = predictor.predict("PLTR")   # Routes to mid/large model
    top = predictor.get_top_picks(20)
    regime = predictor.get_market_regime()
"""

import os
import json
import logging
import gc
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Project paths ─────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "lightgbm")
_RESULTS_DIR = os.path.join(_PROJECT_ROOT, "wrds_pipeline", "phase3", "results")

# ── Tier configuration (System B production — Grade A+) ──────────────
TIER_CONFIG = {
    "mega": {
        "label": "Mega-Cap (>$10B)",
        "min_lmc": 9.21,
        "max_lmc": np.inf,
        "min_mcap_millions": 10000,
        "max_mcap_millions": np.inf,
        "ic": 0.027,
        "weight": 0.148,
        "strategy": "raw",
    },
    "large": {
        "label": "Large-Cap ($2-10B)",
        "min_lmc": 7.60,
        "max_lmc": 9.21,
        "min_mcap_millions": 2000,
        "max_mcap_millions": 10000,
        "ic": 0.019,
        "weight": 0.105,
        "strategy": "hedged",
    },
    "mid": {
        "label": "Mid-Cap ($500M-2B)",
        "min_lmc": 6.21,
        "max_lmc": 7.60,
        "min_mcap_millions": 500,
        "max_mcap_millions": 2000,
        "ic": 0.038,
        "weight": 0.213,
        "strategy": "raw",
    },
    "small": {
        "label": "Small-Cap (<$500M)",
        "min_lmc": 4.61,
        "max_lmc": 6.21,
        "min_mcap_millions": 100,
        "max_mcap_millions": 500,
        "ic": 0.096,
        "weight": 0.534,
        "strategy": "vix_scaled",
    },
}

# VIX exposure map for small-cap strategy
VIX_EXPOSURE = {
    "low":    1.0,
    "normal": 0.8,
    "high":   0.5,
    "crisis": 0.2,
}


def _get_data_manager():
    """Lazy import S3DataManager to avoid circular dependencies."""
    try:
        from nuble.data.s3_data_manager import get_data_manager
        return get_data_manager()
    except Exception as e:
        logger.debug(f"S3DataManager not available: {e}")
        return None


class WRDSPredictor:
    """
    Production multi-tier ML predictor powered by WRDS institutional data.

    Uses 4 tier-specific LightGBM models trained on:
    - GKX panel: 3.76M stock-months × 540 features (1926-2024)
    - Curated feature selection: 51-69 features per tier
    - IC-first training methodology (step6d/step6h)
    - Huber loss with strong regularization

    Compatible with APEX orchestrator interface.
    """

    def __init__(self, data_dir: str = None, models_dir: str = None):
        self.data_dir = data_dir or os.path.abspath(_DATA_DIR)
        self.models_dir = models_dir or os.path.abspath(_MODELS_DIR)
        self._dm = _get_data_manager()

        self._models: Dict[str, Any] = {}
        self._feature_names: Dict[str, list] = {}

        self._panel: Optional[pd.DataFrame] = None
        self._latest_date = None
        self._ticker_to_permno: Dict[str, int] = {}
        self._permno_to_ticker: Dict[int, str] = {}

        self._ready = False
        self._load_attempted = False

    def _ensure_loaded(self):
        """Lazy load: only load on first use."""
        if not self._load_attempted:
            self._load_attempted = True
            self._load()

    def _load(self):
        """Load all tier models, panel, and ticker map."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning("LightGBM not installed — WRDS predictor unavailable")
            return

        self._load_models(lgb)

        if not self._models:
            self._load_single_model(lgb)

        if not self._models:
            logger.warning("No tier models or single model found")
            return

        self._load_panel()

        if self._panel is None:
            logger.warning("GKX panel not available")
            return

        self._build_ticker_map()

        self._ready = True
        logger.info(f"✅ WRDS Predictor v2 ready: "
                    f"{len(self._models)} tier models, "
                    f"{len(self._ticker_to_permno)} tickers, "
                    f"latest={self._latest_date}")

    def _load_models(self, lgb):
        """Load per-tier LightGBM models and their feature lists."""
        registry_path = os.path.join(self.models_dir, "tier_model_registry.json")
        registry = {}
        if os.path.exists(registry_path):
            with open(registry_path) as f:
                registry = json.load(f)

        for tier_name in TIER_CONFIG:
            model_path = os.path.join(self.models_dir, f"lgb_{tier_name}.txt")

            if not os.path.exists(model_path) and self._dm:
                try:
                    downloaded = self._dm.load_model(f"lightgbm/lgb_{tier_name}.txt")
                    model_path = str(downloaded)
                except Exception:
                    pass

            if os.path.exists(model_path):
                try:
                    model = lgb.Booster(model_file=model_path)
                    self._models[tier_name] = model

                    tier_info = registry.get("models", {}).get(tier_name, {})
                    if "features" in tier_info:
                        self._feature_names[tier_name] = tier_info["features"]
                    else:
                        self._feature_names[tier_name] = model.feature_name()

                    logger.info(f"  {tier_name} model: {model.num_trees()} trees, "
                                f"{len(self._feature_names[tier_name])} features")
                except Exception as e:
                    logger.warning(f"  Failed to load {tier_name} model: {e}")

        if self._models:
            logger.info(f"Loaded {len(self._models)} tier models: "
                        f"{list(self._models.keys())}")

    def _load_single_model(self, lgb):
        """Fallback: load single lgb_latest_model.txt for all tiers."""
        candidates = [
            os.path.join(self.data_dir, "lgb_latest_model.txt"),
            os.path.join(self.models_dir, "lgb_latest_model.txt"),
        ]
        for path in candidates:
            if os.path.exists(path):
                try:
                    model = lgb.Booster(model_file=path)
                    for tier_name in TIER_CONFIG:
                        self._models[tier_name] = model
                        self._feature_names[tier_name] = model.feature_name()
                    logger.info(f"Fallback: single model for all tiers: {path}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load single model {path}: {e}")

        if self._dm:
            try:
                downloaded = self._dm.load_model("lightgbm/lgb_latest_model.txt")
                model = lgb.Booster(model_file=str(downloaded))
                for tier_name in TIER_CONFIG:
                    self._models[tier_name] = model
                    self._feature_names[tier_name] = model.feature_name()
                logger.info("Fallback: single model from S3")
            except Exception:
                pass

    def _load_panel(self):
        """Load GKX panel (last 24 months only for memory)."""
        panel_path = os.path.join(self.data_dir, "gkx_panel.parquet")
        panel_loaded = False

        if os.path.exists(panel_path):
            try:
                full_panel = pd.read_parquet(panel_path)
                panel_loaded = True
            except Exception as e:
                logger.warning(f"Failed to read GKX panel: {e}")
        elif self._dm:
            try:
                full_panel = self._dm.load_parquet("gkx_panel.parquet")
                panel_loaded = True
            except Exception as e:
                logger.warning(f"GKX panel not in S3: {e}")

        if not panel_loaded:
            return

        full_panel["date"] = pd.to_datetime(full_panel["date"])
        cutoff = full_panel["date"].max() - pd.DateOffset(months=24)
        self._panel = full_panel[full_panel["date"] >= cutoff].copy()
        self._latest_date = self._panel["date"].max()

        del full_panel
        gc.collect()

        logger.info(f"GKX panel: {len(self._panel):,} rows, "
                    f"{self._panel['permno'].nunique()} stocks, "
                    f"latest={self._latest_date}")

    def _build_ticker_map(self):
        """Build bidirectional ticker ↔ PERMNO mapping."""
        map_path = os.path.join(self.data_dir, "ticker_permno_map.parquet")
        if os.path.exists(map_path):
            try:
                tmap = pd.read_parquet(map_path)
                for _, row in tmap.iterrows():
                    if pd.notna(row.get("ticker")) and pd.notna(row.get("permno")):
                        ticker = str(row["ticker"]).strip().upper()
                        permno = int(row["permno"])
                        self._ticker_to_permno[ticker] = permno
                        self._permno_to_ticker[permno] = ticker
                logger.info(f"Ticker map from ticker_permno_map: "
                            f"{len(self._ticker_to_permno)} tickers")
            except Exception as e:
                logger.warning(f"Failed to load ticker map: {e}")

        if not self._ticker_to_permno:
            alt_path = os.path.join(self.data_dir, "permno_ticker_map.parquet")
            if os.path.exists(alt_path):
                try:
                    pmap = pd.read_parquet(alt_path)
                    for _, row in pmap.iterrows():
                        if pd.notna(row.get("ticker")) and pd.notna(row.get("permno")):
                            ticker = str(row["ticker"]).strip().upper()
                            permno = int(row["permno"])
                            self._ticker_to_permno[ticker] = permno
                            self._permno_to_ticker[permno] = ticker
                except Exception:
                    pass

        if not self._ticker_to_permno:
            crsp_path = os.path.join(self.data_dir, "crsp_monthly.parquet")
            if os.path.exists(crsp_path):
                try:
                    crsp = pd.read_parquet(crsp_path, columns=["permno", "date", "ticker"])
                    crsp["date"] = pd.to_datetime(crsp["date"])
                    latest = crsp.sort_values("date").groupby("permno").last()
                    for permno, row in latest.iterrows():
                        if pd.notna(row.get("ticker")):
                            ticker = str(row["ticker"]).strip().upper()
                            self._ticker_to_permno[ticker] = int(permno)
                            self._permno_to_ticker[int(permno)] = ticker
                    del crsp, latest
                except Exception:
                    pass

        logger.info(f"Ticker map: {len(self._ticker_to_permno)} tickers")

    def _classify_tier(self, log_market_cap: float) -> str:
        """Classify a stock into its market cap tier."""
        if log_market_cap >= 9.21:
            return "mega"
        elif log_market_cap >= 7.60:
            return "large"
        elif log_market_cap >= 6.21:
            return "mid"
        elif log_market_cap >= 4.61:
            return "small"
        else:
            return "micro"

    def _get_vix_exposure(self, vix: float) -> float:
        """Get VIX-based exposure multiplier for small-cap strategy."""
        if vix < 15:
            return VIX_EXPOSURE["low"]
        elif vix < 25:
            return VIX_EXPOSURE["normal"]
        elif vix < 35:
            return VIX_EXPOSURE["high"]
        else:
            return VIX_EXPOSURE["crisis"]

    def _resolve_permno(self, ticker: str) -> Optional[int]:
        """Convert ticker to PERMNO."""
        ticker = ticker.upper().strip()
        if ticker in self._ticker_to_permno:
            return self._ticker_to_permno[ticker]
        for t, p in self._ticker_to_permno.items():
            if ticker == t or (len(ticker) >= 2 and ticker in t):
                return p
        return None

    @property
    def is_ready(self) -> bool:
        self._ensure_loaded()
        return self._ready

    def predict(self, ticker: str) -> Dict[str, Any]:
        """
        Get ML prediction for a stock using the correct tier model.

        Returns dict compatible with APEX orchestrator:
            - ticker, permno, tier
            - raw_score, ensemble_score (tier-weighted)
            - cross_sectional_rank (within tier, not global)
            - decile (1-10, within tier)
            - signal: STRONG_BUY / BUY / HOLD / SELL / STRONG_SELL
            - confidence (0-1, based on z-score within tier)
            - direction: BULLISH / BEARISH / NEUTRAL
            - market_cap_millions, sector
            - strategy, tier_weight
            - top_drivers (feature importance × values)
            - data staleness warning
        """
        self._ensure_loaded()

        if not self._ready:
            return {
                "ticker": ticker.upper(),
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": "WRDS predictor not ready",
                "model_type": "wrds_multi_tier_lgb",
            }

        permno = self._resolve_permno(ticker)
        if permno is None:
            return {
                "ticker": ticker.upper(),
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": f"Ticker {ticker} not found in WRDS universe "
                         f"({len(self._ticker_to_permno)} tickers available)",
                "model_type": "wrds_multi_tier_lgb",
            }

        stock_data = self._panel[self._panel["permno"] == permno].sort_values("date")
        if len(stock_data) == 0:
            return {
                "ticker": ticker.upper(),
                "permno": int(permno),
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": f"No feature data for {ticker} (PERMNO {permno}) in panel",
                "model_type": "wrds_multi_tier_lgb",
            }

        row = stock_data.iloc[-1]
        latest_date = row["date"]

        # Determine tier by market cap
        lmc = row.get("log_market_cap", np.nan)
        if pd.isna(lmc):
            mcap = row.get("market_cap", row.get("mktcap", 0))
            if mcap > 0:
                lmc = np.log(max(mcap, 1))
            else:
                lmc = 7.0

        tier = self._classify_tier(lmc)
        if tier == "micro":
            tier = "small"

        model = self._models.get(tier)
        if model is None:
            return {
                "ticker": ticker.upper(),
                "permno": int(permno),
                "tier": tier,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "error": f"No model loaded for tier {tier}",
                "model_type": "wrds_multi_tier_lgb",
            }

        feature_names = self._feature_names.get(tier, model.feature_name())

        features = np.array([
            float(row[f]) if f in row.index and pd.notna(row[f]) else np.nan
            for f in feature_names
        ]).reshape(1, -1)
        features = np.nan_to_num(features, nan=0.0)

        feature_coverage = sum(1 for f in feature_names
                               if f in row.index and pd.notna(row[f])) / len(feature_names)

        raw_score = float(model.predict(features)[0])

        # Cross-sectional ranking WITHIN the same tier
        tier_mask = self._panel["date"] == self._latest_date
        all_latest = self._panel[tier_mask].copy()

        lmc_col = "log_market_cap"
        if lmc_col not in all_latest.columns:
            mcap_col = "market_cap" if "market_cap" in all_latest.columns else "mktcap"
            all_latest[lmc_col] = np.log(all_latest[mcap_col].clip(lower=1))

        all_latest["_tier"] = all_latest[lmc_col].apply(self._classify_tier)
        all_latest.loc[all_latest["_tier"] == "micro", "_tier"] = "small"
        tier_stocks = all_latest[all_latest["_tier"] == tier]

        if len(tier_stocks) > 10:
            tier_feature_data = np.array([
                [float(r.get(f, np.nan)) if pd.notna(r.get(f, np.nan))
                 else np.nan for f in feature_names]
                for _, r in tier_stocks.iterrows()
            ])
            tier_feature_data = np.nan_to_num(tier_feature_data, nan=0.0)
            tier_scores = model.predict(tier_feature_data)

            rank_pct = float((tier_scores < raw_score).mean() * 100)
            decile = min(10, max(1, int(rank_pct / 10) + 1))
            score_std = np.std(tier_scores)
            score_mean = np.mean(tier_scores)
        else:
            rank_pct = 50.0
            decile = 5
            score_std = 1.0
            score_mean = 0.0

        # Signal classification
        if rank_pct >= 80:
            signal = "STRONG_BUY"
            direction = "BULLISH"
        elif rank_pct >= 60:
            signal = "BUY"
            direction = "BULLISH"
        elif rank_pct >= 40:
            signal = "HOLD"
            direction = "NEUTRAL"
        elif rank_pct >= 20:
            signal = "SELL"
            direction = "BEARISH"
        else:
            signal = "STRONG_SELL"
            direction = "BEARISH"

        # Confidence based on z-score within tier distribution
        z_score = abs(raw_score - score_mean) / (score_std + 1e-8)
        confidence = min(1.0, z_score / 3.0)

        if feature_coverage < 0.5:
            confidence *= 0.5
        elif feature_coverage < 0.8:
            confidence *= 0.8

        # Feature importance / top drivers
        top_drivers = []
        if hasattr(model, "feature_importance"):
            try:
                importances = model.feature_importance(importance_type="gain")
                feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
                total_imp = sum(importances) + 1e-8
                for fname, imp in feat_imp[:5]:
                    val = row.get(fname, np.nan)
                    top_drivers.append({
                        "feature": fname,
                        "importance_pct": round(float(imp / total_imp * 100), 2),
                        "value": round(float(val), 4) if pd.notna(val) else None,
                    })
            except Exception:
                pass

        # Sector
        sector = None
        for col in ["gsector", "siccd"]:
            if col in row.index and pd.notna(row[col]):
                sector = int(row[col])
                break

        # Market cap in millions
        mcap_raw = row.get("market_cap", row.get("mktcap", np.exp(lmc)))
        mcap_millions = float(mcap_raw) if mcap_raw < 1e8 else float(mcap_raw / 1e6)

        ensemble_score = raw_score * TIER_CONFIG[tier]["weight"]
        staleness_days = (pd.Timestamp.now() - self._latest_date).days

        return {
            "ticker": ticker.upper(),
            "permno": int(permno),
            "tier": tier,
            "raw_score": round(raw_score, 6),
            "ensemble_score": round(ensemble_score, 6),
            "cross_sectional_rank": round(rank_pct, 1),
            "decile": decile,
            "signal": signal,
            "direction": direction,
            "confidence": round(confidence, 3),
            "predicted_return": round(raw_score, 6),
            "predicted_return_pct": round(raw_score * 100, 2),
            "decile_rank": decile,
            "percentile": round(rank_pct, 1),
            "market_cap_millions": round(mcap_millions, 1),
            "sector": sector,
            "strategy": TIER_CONFIG[tier]["strategy"],
            "tier_weight": TIER_CONFIG[tier]["weight"],
            "feature_coverage": round(feature_coverage, 3),
            "features_used": len(feature_names),
            "universe_size": len(tier_stocks),
            "top_drivers": top_drivers,
            "latest_date": str(self._latest_date.date()) if self._latest_date else None,
            "data_staleness_days": staleness_days,
            "data_date": str(latest_date.date()),
            "model_type": "wrds_multi_tier_lgb",
            "model_ic": TIER_CONFIG[tier]["ic"],
        }

    def predict_batch(self, tickers: list) -> Dict[str, Dict[str, Any]]:
        """Predict for multiple tickers efficiently."""
        return {t: self.predict(t) for t in tickers}

    def get_top_picks(self, n: int = 20, tier: Optional[str] = None) -> List[Dict]:
        """Get top N stock picks, optionally filtered by tier."""
        self._ensure_loaded()
        if not self._ready:
            return []

        latest = self._panel[self._panel["date"] == self._latest_date].copy()

        lmc_col = "log_market_cap"
        if lmc_col not in latest.columns:
            mcap_col = "market_cap" if "market_cap" in latest.columns else "mktcap"
            latest[lmc_col] = np.log(latest[mcap_col].clip(lower=1))

        latest["_tier"] = latest[lmc_col].apply(self._classify_tier)
        latest.loc[latest["_tier"] == "micro", "_tier"] = "small"

        if tier:
            latest = latest[latest["_tier"] == tier]

        results = []
        tiers_to_score = [tier] if tier else ["mega", "large", "mid", "small"]

        for tier_name in tiers_to_score:
            tier_stocks = latest[latest["_tier"] == tier_name]
            model = self._models.get(tier_name)
            if model is None or len(tier_stocks) == 0:
                continue

            feature_names = self._feature_names.get(tier_name, model.feature_name())

            tier_features = np.array([
                [float(r.get(f, np.nan)) if pd.notna(r.get(f, np.nan))
                 else np.nan for f in feature_names]
                for _, r in tier_stocks.iterrows()
            ])
            tier_features = np.nan_to_num(tier_features, nan=0.0)
            tier_scores = model.predict(tier_features)

            for i, (idx, row) in enumerate(tier_stocks.iterrows()):
                permno = int(row["permno"])
                ticker_str = self._permno_to_ticker.get(permno, f"PERMNO_{permno}")
                score = float(tier_scores[i])
                rank_pct = float((tier_scores < score).mean() * 100)
                decile = min(10, max(1, int(rank_pct / 10) + 1))

                if rank_pct >= 80:
                    signal = "STRONG_BUY"
                elif rank_pct >= 60:
                    signal = "BUY"
                elif rank_pct >= 40:
                    signal = "HOLD"
                elif rank_pct >= 20:
                    signal = "SELL"
                else:
                    signal = "STRONG_SELL"

                mcap_raw = row.get("market_cap", row.get("mktcap", 0))
                results.append({
                    "ticker": ticker_str,
                    "permno": permno,
                    "tier": tier_name,
                    "raw_score": round(score, 6),
                    "cross_sectional_rank": round(rank_pct, 1),
                    "decile": decile,
                    "signal": signal,
                    "market_cap_millions": round(float(mcap_raw), 1) if mcap_raw < 1e8
                                          else round(float(mcap_raw / 1e6), 1),
                })

        results.sort(key=lambda x: x["raw_score"], reverse=True)
        return results[:n]

    def get_universe_snapshot(self) -> List[Dict]:
        """Get predictions for ALL stocks in the latest month."""
        return self.get_top_picks(n=99999)

    def get_tier_predictions(self, tier: str) -> List[Dict]:
        """Get predictions for all stocks in a specific tier."""
        return self.get_top_picks(n=99999, tier=tier)

    def get_market_regime(self) -> Dict:
        """Get current macro regime information from the panel."""
        self._ensure_loaded()
        if not self._ready:
            return {"regime": "unknown"}

        latest_rows = self._panel[self._panel["date"] == self._latest_date]
        if len(latest_rows) == 0:
            return {"regime": "unknown"}

        latest = latest_rows.iloc[0]
        regime = {}

        macro_cols = {
            "vix": "vix", "tbl": "tbl", "lty": "lty",
            "fed_funds_rate": "fed_funds_rate",
            "cpi": "cpi", "cpi_yoy": "cpi_yoy",
            "breakeven_10y": "breakeven_10y",
            "term_spread_10y2y": "term_spread",
            "corp_spread_bbb": "credit_spread",
            "leading_index": "leading_index",
            "consumer_sentiment": "consumer_sentiment",
        }
        for panel_col, output_name in macro_cols.items():
            if panel_col in latest.index and pd.notna(latest[panel_col]):
                regime[output_name] = round(float(latest[panel_col]), 4)

        vix = regime.get("vix", 20)
        regime["vix_exposure"] = self._get_vix_exposure(vix)

        if vix > 35:
            regime["regime"] = "crisis"
        elif vix > 25:
            regime["regime"] = "stress"
        elif regime.get("term_spread", 0) < 0:
            regime["regime"] = "late_cycle"
        else:
            regime["regime"] = "normal"

        regime["latest_date"] = str(self._latest_date.date()) if self._latest_date else None
        return regime

    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata and performance summary."""
        self._ensure_loaded()
        info = {
            "ready": self._ready,
            "model_type": "Multi-Tier LightGBM (System B)",
            "data_source": "WRDS (3.76M stock-months, 1926-2024)",
            "tier_models": {},
        }
        if self._ready:
            info["n_tickers_mapped"] = len(self._ticker_to_permno)
            info["latest_date"] = str(self._latest_date.date()) if self._latest_date else None
            info["n_stocks_latest"] = int(
                self._panel[self._panel["date"] == self._latest_date]["permno"].nunique()
            ) if self._panel is not None else 0

            for tier_name, model in self._models.items():
                info["tier_models"][tier_name] = {
                    "n_trees": model.num_trees(),
                    "n_features": len(self._feature_names.get(tier_name, [])),
                    "ic": TIER_CONFIG[tier_name]["ic"],
                    "weight": TIER_CONFIG[tier_name]["weight"],
                    "strategy": TIER_CONFIG[tier_name]["strategy"],
                }
        return info


# ── Singleton ─────────────────────────────────────────────────────────
_wrds_predictor_instance = None


def get_wrds_predictor(**kwargs) -> WRDSPredictor:
    """Get or create the singleton WRDSPredictor instance."""
    global _wrds_predictor_instance
    if _wrds_predictor_instance is None:
        _wrds_predictor_instance = WRDSPredictor(**kwargs)
    return _wrds_predictor_instance
