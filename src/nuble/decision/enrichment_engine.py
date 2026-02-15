"""
Statistical Enrichment Engine — Institutional-grade data enrichment.

This engine does NOT make trading decisions. It computes statistical context
that Claude cannot derive from raw numbers, then presents everything
to Claude in a format optimized for expert reasoning.

What it computes:
- Percentile ranks (where does current value sit in its own history?)
- Z-scores (how many standard deviations from mean?)
- Rates of change (is the indicator accelerating or decelerating?)
- Anomaly detection (any metric > 2σ from norm gets flagged)
- Cross-source divergences (price vs sentiment, technicals vs fundamentals)
- Conditional context (what has historically happened in similar setups?)
- Information completeness (what % of data sources are reporting?)
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class EnrichedMetric:
    """A single data point with full statistical context."""
    name: str
    value: float
    unit: str                         # "ratio", "percent", "dollars", "index", "days"

    # Statistical context (computed from historical data)
    percentile_90d: Optional[float] = None    # Where this value sits in 90-day range (0-100)
    z_score_20d: Optional[float] = None       # Std devs from 20-day mean
    rate_of_change_5d: Optional[float] = None # 5-day rate of change

    # Historical range
    min_90d: Optional[float] = None
    max_90d: Optional[float] = None
    mean_90d: Optional[float] = None

    # Anomaly flag
    is_anomaly: bool = False          # True if |z_score| > 2.0
    anomaly_description: str = ""


@dataclass
class Divergence:
    """A detected divergence between two data sources."""
    source_a: str
    source_b: str
    description: str
    severity: str            # "strong", "moderate", "weak"
    implication: str         # What this divergence historically means


@dataclass
class Anomaly:
    """A statistically unusual observation."""
    metric_name: str
    current_value: float
    z_score: float
    description: str
    historical_context: str  # What has happened in similar situations


@dataclass
class ConflictV2:
    """A conflict between data sources, with quantitative measurement."""
    sources: List[str]
    description: str
    severity: str                     # "critical", "moderate", "minor"
    agreement_score: float            # -1 (complete disagreement) to +1 (complete agreement)
    resolution_guidance: str          # Domain-specific guidance for Claude


@dataclass
class ConsensusMetrics:
    """Mathematical consensus measurement across all data sources."""
    direction_agreement: float        # 0-1, what % of sources agree on direction
    weighted_direction_agreement: float  # Same but weighted by source confidence
    bullish_sources: List[str]
    bearish_sources: List[str]
    neutral_sources: List[str]
    dominant_direction: str           # "BULLISH", "BEARISH", "NEUTRAL", "SPLIT"
    conviction_level: str             # "high" (>80% agree), "moderate" (60-80%), "low" (40-60%), "divided" (<40%)


@dataclass
class EnrichedIntelligence:
    """The complete output of the enrichment engine. This is what Claude receives."""
    symbol: str
    timestamp: str

    # Price context
    price: Optional[EnrichedMetric] = None
    volume: Optional[EnrichedMetric] = None

    # Technical context (enriched, not scored)
    technicals: Dict[str, EnrichedMetric] = field(default_factory=dict)

    # Sentiment context (enriched)
    sentiment: Dict[str, EnrichedMetric] = field(default_factory=dict)

    # Fundamental context (enriched)
    fundamentals: Dict[str, Any] = field(default_factory=dict)

    # Macro context (enriched)
    macro: Dict[str, Any] = field(default_factory=dict)

    # Risk metrics (enriched)
    risk: Dict[str, Any] = field(default_factory=dict)

    # LuxAlgo signals (raw, preserved)
    luxalgo: Dict[str, Any] = field(default_factory=dict)

    # ML predictions (raw, preserved)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)

    # Statistical findings
    anomalies: List[Anomaly] = field(default_factory=list)
    divergences: List[Divergence] = field(default_factory=list)
    conflicts: List[ConflictV2] = field(default_factory=list)
    consensus: Optional[ConsensusMetrics] = None

    # Data quality
    sources_reporting: int = 0
    sources_total: int = 8
    data_coverage_pct: float = 0.0
    stale_sources: List[str] = field(default_factory=list)

    # The formatted brief for Claude (the key output)
    intelligence_brief: str = ""


# ═══════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════

class EnrichmentEngine:
    """
    Statistical Enrichment Engine.

    Takes raw agent outputs + SharedDataLayer data and produces
    statistically-enriched intelligence for Claude.

    Usage:
        engine = EnrichmentEngine()
        enriched = await engine.enrich(
            symbol="TSLA",
            shared_data=shared_data_layer,
            agent_outputs=agent_results_dict,
            lambda_data=lambda_dict,
        )
        # enriched.intelligence_brief → feed this to Claude
    """

    async def enrich(
        self,
        symbol: str,
        shared_data: Any,
        agent_outputs: Optional[Dict[str, Any]] = None,
        lambda_data: Optional[Dict[str, Any]] = None,
    ) -> EnrichedIntelligence:
        """Main entry point. Enriches all available data for a symbol."""

        result = EnrichedIntelligence(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
        )
        agent_outputs = agent_outputs or {}

        # ─── STEP 1: Enrich Price & Volume ───
        await self._enrich_price_volume(symbol, shared_data, result)

        # ─── STEP 2: Enrich Technicals ───
        await self._enrich_technicals(symbol, shared_data, result)

        # ─── STEP 3: Enrich Sentiment ───
        self._enrich_sentiment(agent_outputs, result)

        # ─── STEP 4: Enrich Fundamentals ───
        self._enrich_fundamentals(agent_outputs, result)

        # ─── STEP 5: Enrich Macro ───
        await self._enrich_macro(shared_data, agent_outputs, result)

        # ─── STEP 6: Enrich Risk ───
        self._enrich_risk(agent_outputs, result)

        # ─── STEP 7: Preserve LuxAlgo (raw — no enrichment needed) ───
        self._preserve_luxalgo(lambda_data, result)

        # ─── STEP 8: Preserve ML Predictions (raw) ───
        self._preserve_ml(agent_outputs, lambda_data, result)

        # ─── STEP 8.5: Add v2 ML Prediction (F4 pipeline) ───
        await self._add_ml_v2_prediction(symbol, shared_data, result)

        # ─── STEP 9: Detect Anomalies ───
        self._detect_anomalies(result)

        # ─── STEP 10: Detect Divergences ───
        self._detect_divergences(result)

        # ─── STEP 11: Detect Conflicts ───
        self._detect_conflicts(agent_outputs, result)

        # ─── STEP 12: Compute Consensus ───
        self._compute_consensus(agent_outputs, result)

        # ─── STEP 13: Compute Data Coverage ───
        self._compute_data_coverage(agent_outputs, result)

        # ─── STEP 14: Build Intelligence Brief ───
        result.intelligence_brief = self._build_intelligence_brief(result)

        return result

    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Price & Volume Enrichment
    # ═══════════════════════════════════════════════════════════════

    async def _enrich_price_volume(self, symbol: str, shared_data, result: EnrichedIntelligence):
        """Enrich current price and volume with historical statistical context."""
        try:
            quote = await shared_data.get_quote(symbol)
            historical = await shared_data.get_historical(symbol, 90)

            if not quote or not historical:
                return

            q_results = quote.get('results', [])
            if not q_results:
                return
            q = q_results[0]

            current_price = q.get('c', 0)
            current_volume = q.get('v', 0)

            hist_results = historical.get('results', [])
            if not hist_results or len(hist_results) < 20:
                # Not enough history for enrichment, just store raw values
                result.price = EnrichedMetric(
                    name="price", value=current_price, unit="dollars"
                )
                return

            closes = np.array([r['c'] for r in hist_results])
            volumes = np.array([r['v'] for r in hist_results], dtype=float)

            # Price enrichment
            result.price = self._enrich_single_metric(
                name="price", value=current_price, unit="dollars",
                history=closes
            )

            # Volume enrichment
            if current_volume > 0:
                result.volume = self._enrich_single_metric(
                    name="volume", value=float(current_volume), unit="shares",
                    history=volumes
                )

        except Exception as e:
            logger.warning(f"Price/volume enrichment failed for {symbol}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Technical Enrichment
    # ═══════════════════════════════════════════════════════════════

    async def _enrich_technicals(self, symbol: str, shared_data, result: EnrichedIntelligence):
        """Enrich technical indicators with percentile ranks and z-scores."""
        try:
            historical = await shared_data.get_historical(symbol, 90)
            if not historical or not historical.get('results') or len(historical['results']) < 30:
                return

            hist = historical['results']
            closes = np.array([r['c'] for r in hist])
            highs = np.array([r['h'] for r in hist])
            lows = np.array([r['l'] for r in hist])
            volumes = np.array([r['v'] for r in hist], dtype=float)

            # ── RSI with history ──
            rsi_data = await shared_data.get_rsi(symbol)
            if rsi_data and rsi_data.get('results', {}).get('values'):
                current_rsi = rsi_data['results']['values'][0].get('value')
                if current_rsi is not None:
                    # Compute RSI history from closes for percentile/z-score
                    rsi_history = self._compute_rsi_series(closes, period=14)
                    result.technicals['rsi_14'] = self._enrich_single_metric(
                        name="RSI(14)", value=float(current_rsi), unit="index",
                        history=rsi_history
                    )

            # ── MACD with history ──
            macd_data = await shared_data.get_macd(symbol)
            if macd_data and macd_data.get('results', {}).get('values'):
                macd_vals = macd_data['results']['values'][0]
                histogram = macd_vals.get('histogram')
                macd_value = macd_vals.get('value')
                signal = macd_vals.get('signal')

                if histogram is not None:
                    # Compute MACD histogram history
                    hist_series = self._compute_macd_histogram_series(closes)
                    result.technicals['macd_histogram'] = self._enrich_single_metric(
                        name="MACD Histogram", value=float(histogram), unit="ratio",
                        history=hist_series
                    )
                if macd_value is not None:
                    result.technicals['macd_line'] = EnrichedMetric(
                        name="MACD Line", value=float(macd_value), unit="ratio"
                    )
                if signal is not None:
                    result.technicals['macd_signal'] = EnrichedMetric(
                        name="MACD Signal", value=float(signal), unit="ratio"
                    )

            # ── SMA positions ──
            for window in [20, 50, 200]:
                sma_data = await shared_data.get_sma(symbol, window)
                if sma_data and sma_data.get('results', {}).get('values'):
                    sma_val = sma_data['results']['values'][0].get('value')
                    if sma_val is not None:
                        # Price relative to SMA (enriched)
                        distance_pct = (closes[-1] / sma_val - 1) * 100
                        # Compute history of price-to-SMA ratios
                        sma_series = self._compute_sma_series(closes, window)
                        if len(sma_series) > 0:
                            ratio_series = closes[-len(sma_series):] / sma_series
                            result.technicals[f'price_vs_sma{window}'] = self._enrich_single_metric(
                                name=f"Price/SMA({window})",
                                value=float(closes[-1] / sma_val),
                                unit="ratio",
                                history=ratio_series
                            )
                            result.technicals[f'price_vs_sma{window}'].rate_of_change_5d = round(distance_pct, 2)

                        result.technicals[f'sma_{window}'] = EnrichedMetric(
                            name=f"SMA({window})", value=round(float(sma_val), 2), unit="dollars"
                        )

            # ── Momentum (multi-period) ──
            for period in [5, 10, 20]:
                if len(closes) > period:
                    mom = (closes[-1] / closes[-period - 1] - 1) * 100
                    # Compute momentum history
                    mom_series = np.array([
                        (closes[i] / closes[i - period] - 1) * 100
                        for i in range(period, len(closes))
                    ])
                    result.technicals[f'momentum_{period}d'] = self._enrich_single_metric(
                        name=f"Momentum({period}d)", value=float(mom), unit="percent",
                        history=mom_series
                    )

            # ── Volatility (realized, annualized) ──
            if len(closes) >= 21:
                returns = np.diff(closes) / closes[:-1]
                vol_20d = float(np.std(returns[-20:]) * np.sqrt(252)) * 100

                # Compute rolling 20-day vol for enrichment
                vol_series = np.array([
                    np.std(returns[max(0, i - 19):i + 1]) * np.sqrt(252) * 100
                    for i in range(19, len(returns))
                ])
                result.technicals['realized_vol_20d'] = self._enrich_single_metric(
                    name="Realized Vol (20d, annualized)", value=float(vol_20d), unit="percent",
                    history=vol_series
                )

            # ── Average True Range ──
            if len(hist) >= 15:
                tr = np.maximum(
                    highs[1:] - lows[1:],
                    np.maximum(
                        np.abs(highs[1:] - closes[:-1]),
                        np.abs(lows[1:] - closes[:-1])
                    )
                )
                atr_14 = float(np.mean(tr[-14:]))
                atr_pct = (atr_14 / closes[-1]) * 100

                # ATR history
                atr_series = np.array([
                    np.mean(tr[max(0, i - 13):i + 1])
                    for i in range(13, len(tr))
                ])
                result.technicals['atr_14'] = self._enrich_single_metric(
                    name="ATR(14)", value=float(atr_14), unit="dollars",
                    history=atr_series
                )
                result.technicals['atr_14_pct'] = EnrichedMetric(
                    name="ATR(14) as % of Price", value=round(atr_pct, 2), unit="percent"
                )

        except Exception as e:
            logger.warning(f"Technical enrichment failed for {symbol}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Sentiment Enrichment
    # ═══════════════════════════════════════════════════════════════

    def _enrich_sentiment(self, agent_outputs: Dict, result: EnrichedIntelligence):
        """Enrich sentiment data from NewsAnalyst and other agents."""
        try:
            news_data = agent_outputs.get('news_analyst', {})
            if not news_data:
                return

            # Article-level sentiment aggregation
            sym_news = news_data.get('symbol_news', {})
            all_articles = []
            for sym_data in sym_news.values():
                if isinstance(sym_data, dict):
                    all_articles.extend(sym_data.get('articles', []))

            if all_articles:
                pos = sum(1 for a in all_articles if a.get('sentiment') in ('Positive', 'Bullish'))
                neg = sum(1 for a in all_articles if a.get('sentiment') in ('Negative', 'Bearish'))
                neu = len(all_articles) - pos - neg
                total = max(1, len(all_articles))

                # Net sentiment ratio: -1 (all negative) to +1 (all positive)
                net_ratio = (pos - neg) / total

                result.sentiment['news_sentiment_ratio'] = EnrichedMetric(
                    name="News Sentiment (net ratio)",
                    value=round(net_ratio, 3),
                    unit="ratio"
                )
                result.sentiment['news_article_count'] = EnrichedMetric(
                    name="News Article Count",
                    value=float(total),
                    unit="count"
                )
                result.sentiment['news_breakdown'] = EnrichedMetric(
                    name="Sentiment Breakdown",
                    value=float(pos),  # Use positive count as primary value
                    unit="count"
                )
                # Store breakdown details
                result.sentiment['news_breakdown'].anomaly_description = (
                    f"Positive: {pos}, Negative: {neg}, Neutral: {neu} of {total}"
                )

            # Fear & Greed Index
            market_sentiment = news_data.get('market_sentiment', {})
            fgi = market_sentiment.get('fear_greed_index')
            if fgi is not None and isinstance(fgi, (int, float)):
                result.sentiment['fear_greed_index'] = EnrichedMetric(
                    name="Fear & Greed Index", value=float(fgi), unit="index"
                )

            # StockNews quantitative sentiment stats
            for sym, sym_data in sym_news.items():
                if isinstance(sym_data, dict):
                    quant_sent = sym_data.get('quantitative_sentiment', {})
                    if quant_sent and isinstance(quant_sent, dict):
                        for key, val in quant_sent.items():
                            if isinstance(val, (int, float)):
                                result.sentiment[f'stocknews_{key}'] = EnrichedMetric(
                                    name=f"StockNews {key}", value=float(val), unit="index"
                                )

            # Analyst ratings
            ratings = news_data.get('analyst_ratings', {})
            if ratings and isinstance(ratings, dict):
                result.sentiment['analyst_ratings'] = EnrichedMetric(
                    name="Analyst Ratings Data Available",
                    value=1.0, unit="boolean"
                )
                # Store raw ratings data in details
                result.fundamentals['analyst_ratings_raw'] = ratings

            # Earnings calendar
            earnings = news_data.get('earnings_calendar', {})
            if earnings:
                result.fundamentals['earnings_calendar'] = earnings

        except Exception as e:
            logger.warning(f"Sentiment enrichment failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Fundamental Enrichment
    # ═══════════════════════════════════════════════════════════════

    def _enrich_fundamentals(self, agent_outputs: Dict, result: EnrichedIntelligence):
        """Preserve and organize fundamental data from FundamentalAnalyst."""
        try:
            fund_data = agent_outputs.get('fundamental_analyst', {})
            if not fund_data:
                return

            # Pass through all fundamental data — Claude knows how to interpret these
            for key in ['company_overview', 'quarterly_financials', 'annual_financials',
                        'dividends', 'valuation', 'growth_metrics', 'financial_health',
                        'sec_filing_insights', 'current_price', 'price_trend']:
                if key in fund_data:
                    result.fundamentals[key] = fund_data[key]

            # ── SEC EDGAR XBRL ratios (Gu-Kelly-Xiu 40 fundamentals) ──
            if 'xbrl_ratios' in fund_data:
                ratios = fund_data['xbrl_ratios']
                result.fundamentals['xbrl_ratios'] = ratios
                # Surface key cross-sectional ratios for anomaly detection
                # Ratios are in a flat dict (earnings_yield, book_to_market, etc.)
                for ratio_name in ['earnings_to_price', 'book_to_market', 'cashflow_to_price',
                                   'dividend_yield', 'sales_to_price']:
                    val = ratios.get(ratio_name)
                    if val is not None and np.isfinite(val):
                        result.fundamentals[f'xbrl_{ratio_name}'] = val

            if 'fundamental_quality' in fund_data:
                quality = fund_data['fundamental_quality']
                result.fundamentals['fundamental_quality'] = quality
                score = quality.get('score')
                if score is not None:
                    result.fundamentals['quality_score_enriched'] = EnrichedMetric(
                        name="fundamental_quality_score",
                        value=float(score),
                        unit="score_0_100",
                        is_anomaly=(score < 25 or score > 90),
                        anomaly_description=(
                            f"Quality score {score:.0f}/100 ({quality.get('grade', '?')}) — "
                            f"{'exceptionally high' if score > 90 else 'dangerously low'}"
                            if (score < 25 or score > 90) else ""
                        ),
                    )

        except Exception as e:
            logger.warning(f"Fundamental enrichment failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Macro Enrichment
    # ═══════════════════════════════════════════════════════════════

    async def _enrich_macro(self, shared_data, agent_outputs: Dict, result: EnrichedIntelligence):
        """Enrich macro context with statistical depth."""
        try:
            macro_data = agent_outputs.get('macro_analyst', {})

            # VIX enrichment with historical context
            vix_quote = await shared_data.get_quote('VIX')
            vix_hist = await shared_data.get_historical('VIX', 90)

            if vix_quote and vix_quote.get('results'):
                vix_val = vix_quote['results'][0].get('c')
                if vix_val is not None:
                    vix_history = None
                    if vix_hist and vix_hist.get('results') and len(vix_hist['results']) >= 20:
                        vix_history = np.array([r['c'] for r in vix_hist['results']])

                    result.macro['vix'] = self._enrich_single_metric(
                        name="VIX", value=float(vix_val), unit="index",
                        history=vix_history
                    )

            # Pass through all macro data
            if macro_data:
                for key in ['sector_performance', 'market_breadth', 'yield_curve',
                            'dollar', 'commodities', 'macro_news', 'sentiment']:
                    if key in macro_data:
                        result.macro[key] = macro_data[key]

                # ── FRED macro data (Gu-Kelly-Xiu 8 macro variables) ──
                fred_data = macro_data.get('fred_macro')
                if fred_data:
                    result.macro['fred_macro'] = fred_data

                    # Surface regime indicators as enriched metrics
                    regimes = fred_data.get('regimes', {})
                    if 'yield_curve' in regimes:
                        yc = regimes['yield_curve']
                        result.macro['yield_curve_fred'] = yc
                        # Flag inverted yield curve as anomaly
                        if yc == 'inverted':
                            result.anomalies.append(Anomaly(
                                metric_name="yield_curve_regime",
                                current_value=fred_data.get('current', {}).get('term_spread', -1.0),
                                z_score=-2.5,  # Inversion is rare
                                description="Yield curve is INVERTED (10Y-3M < 0)",
                                historical_context="Yield curve inversions have preceded every US recession since 1970 with 12-18 month lead time.",
                            ))

                    if 'credit_cycle' in regimes:
                        result.macro['credit_cycle'] = regimes['credit_cycle']
                    if 'monetary_policy' in regimes:
                        result.macro['monetary_policy'] = regimes['monetary_policy']

                # Pass through FRED-derived yield curve and regime fields
                for key in ['yield_curve_fred', 'credit_cycle', 'monetary_policy']:
                    if key in macro_data and key not in result.macro:
                        result.macro[key] = macro_data[key]

        except Exception as e:
            logger.warning(f"Macro enrichment failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Risk Enrichment
    # ═══════════════════════════════════════════════════════════════

    def _enrich_risk(self, agent_outputs: Dict, result: EnrichedIntelligence):
        """Preserve risk manager output."""
        try:
            risk_data = agent_outputs.get('risk_manager', {})
            if risk_data:
                result.risk = risk_data  # Pass through everything
        except Exception as e:
            logger.warning(f"Risk enrichment failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    # STEP 7 & 8: Preserve LuxAlgo and ML (raw)
    # ═══════════════════════════════════════════════════════════════

    def _preserve_luxalgo(self, lambda_data: Optional[Dict], result: EnrichedIntelligence):
        """Preserve LuxAlgo signals without modification."""
        if not lambda_data:
            return
        # LuxAlgo may be nested in decision_engine or at top level
        luxalgo = lambda_data.get('luxalgo', {})
        if luxalgo:
            result.luxalgo = {
                'weekly_action': luxalgo.get('weekly', luxalgo.get('weekly_action')),
                'daily_action': luxalgo.get('daily', luxalgo.get('daily_action')),
                'h4_action': luxalgo.get('h4', luxalgo.get('h4_action')),
                'direction': luxalgo.get('direction'),
                'aligned': luxalgo.get('aligned', False),
                'valid_count': luxalgo.get('valid_count', 0),
                'score': luxalgo.get('score'),
            }
        else:
            # Try top-level keys (from lambda_data format)
            result.luxalgo = {
                'weekly_action': lambda_data.get('luxalgo_weekly_action'),
                'daily_action': lambda_data.get('luxalgo_daily_action'),
                'h4_action': lambda_data.get('luxalgo_h4_action'),
                'direction': lambda_data.get('luxalgo_direction'),
                'aligned': lambda_data.get('luxalgo_aligned', False),
                'valid_count': lambda_data.get('luxalgo_valid_count', 0),
                'score': lambda_data.get('luxalgo_score'),
            }

    def _preserve_ml(self, agent_outputs: Dict, lambda_data: Optional[Dict], result: EnrichedIntelligence):
        """Preserve ML predictions without modification."""
        quant = agent_outputs.get('quant_analyst', {})
        if quant:
            signals = quant.get('ml_signals', quant.get('signals', {}))
            if signals:
                result.ml_predictions['quant_analyst'] = signals

        if lambda_data:
            for key in ['ml_direction', 'ml_confidence', 'ml_price_target',
                        'ml_signals', 'ml_predictions']:
                if key in lambda_data:
                    result.ml_predictions[key] = lambda_data[key]

    async def _add_ml_v2_prediction(self, symbol: str, shared_data, result: EnrichedIntelligence):
        """
        Run the v2 ML predictor (F4) if a trained model exists for this symbol.

        Adds the full prediction dict (with SHAP explanations) to
        result.ml_predictions['v2_prediction'].
        """
        try:
            from ..ml.predictor import get_predictor
            import pandas as pd

            predictor = get_predictor()
            if not predictor.has_model(symbol):
                return

            # Fetch OHLCV for feature computation
            historical = await shared_data.get_historical(symbol, 120)
            if not historical or not historical.get('results') or len(historical['results']) < 60:
                return

            bars = historical['results']
            df = pd.DataFrame(bars)
            if 't' in df.columns:
                df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.set_index('date').sort_index()
            rename = {}
            for src, dst in [('o', 'open'), ('h', 'high'), ('l', 'low'), ('c', 'close'), ('v', 'volume')]:
                if src in df.columns:
                    rename[src] = dst
            if rename:
                df = df.rename(columns=rename)
            cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
            if len(cols) < 5:
                return
            df = df[cols].dropna()

            import asyncio
            pred = await asyncio.get_event_loop().run_in_executor(
                None, lambda: predictor.predict(symbol, df)
            )

            if pred and pred.get('confidence', 0) > 0:
                result.ml_predictions['v2_prediction'] = pred
                model_type = pred.get('model_type', 'per-ticker')
                logger.info(
                    "Enrichment: v2 ML prediction for %s → %s (%.0f%%) [%s model]",
                    symbol, pred.get('direction'), pred.get('confidence', 0) * 100,
                    model_type,
                )
        except ImportError:
            pass  # v2 ML module not installed — silent skip
        except Exception as exc:
            logger.warning("v2 ML prediction failed for %s: %s", symbol, exc)

    # ═══════════════════════════════════════════════════════════════
    # STEP 9: Anomaly Detection
    # ═══════════════════════════════════════════════════════════════

    def _detect_anomalies(self, result: EnrichedIntelligence):
        """Scan all enriched metrics for statistical anomalies (|z-score| > 2.0)."""
        all_metrics = {}
        all_metrics.update(result.technicals)
        all_metrics.update(result.sentiment)
        for key, val in result.macro.items():
            if isinstance(val, EnrichedMetric):
                all_metrics[key] = val

        if result.price:
            all_metrics['price'] = result.price
        if result.volume:
            all_metrics['volume'] = result.volume

        for key, metric in all_metrics.items():
            if isinstance(metric, EnrichedMetric) and metric.z_score_20d is not None:
                if abs(metric.z_score_20d) > 2.0:
                    metric.is_anomaly = True

                    direction = "above" if metric.z_score_20d > 0 else "below"
                    mean_str = f"{metric.mean_90d:.2f}" if metric.mean_90d is not None else "N/A"
                    min_str = f"{metric.min_90d:.2f}" if metric.min_90d is not None else "N/A"
                    max_str = f"{metric.max_90d:.2f}" if metric.max_90d is not None else "N/A"

                    metric.anomaly_description = (
                        f"{metric.name} is {abs(metric.z_score_20d):.1f}σ {direction} "
                        f"its 20-day mean ({mean_str}). "
                        f"Current: {metric.value:.2f}, 90d range: [{min_str}, {max_str}]"
                    )

                    pct_str = f"{metric.percentile_90d:.0f}th" if metric.percentile_90d is not None else "N/A"
                    roc_str = f"{metric.rate_of_change_5d}" if metric.rate_of_change_5d is not None else "N/A"

                    result.anomalies.append(Anomaly(
                        metric_name=metric.name,
                        current_value=metric.value,
                        z_score=metric.z_score_20d,
                        description=metric.anomaly_description,
                        historical_context=(
                            f"At the {pct_str} percentile of its 90-day range. "
                            f"5-day rate of change: {roc_str}"
                        ),
                    ))

    # ═══════════════════════════════════════════════════════════════
    # STEP 10: Divergence Detection
    # ═══════════════════════════════════════════════════════════════

    def _detect_divergences(self, result: EnrichedIntelligence):
        """Detect divergences between related metrics."""

        # Divergence 1: Price momentum vs RSI momentum
        mom_5 = result.technicals.get('momentum_5d')
        rsi = result.technicals.get('rsi_14')
        if (mom_5 and rsi
                and isinstance(mom_5, EnrichedMetric) and isinstance(rsi, EnrichedMetric)):
            if mom_5.rate_of_change_5d is not None and rsi.rate_of_change_5d is not None:
                # Price going up but RSI going down = bearish divergence
                if mom_5.value > 0 and rsi.rate_of_change_5d < -5:
                    result.divergences.append(Divergence(
                        source_a="Price Momentum",
                        source_b="RSI",
                        description="Price is rising but RSI is declining (bearish divergence)",
                        severity="moderate",
                        implication="Bearish RSI divergence suggests weakening momentum. "
                                    "Historically associated with pullbacks within 5-10 trading days."
                    ))
                elif mom_5.value < 0 and rsi.rate_of_change_5d > 5:
                    result.divergences.append(Divergence(
                        source_a="Price Momentum",
                        source_b="RSI",
                        description="Price is falling but RSI is rising (bullish divergence)",
                        severity="moderate",
                        implication="Bullish RSI divergence suggests selling pressure is weakening. "
                                    "Historically associated with reversals within 5-10 trading days."
                    ))

        # Divergence 2: Sentiment vs Price
        sent_ratio = result.sentiment.get('news_sentiment_ratio')
        if (sent_ratio and mom_5
                and isinstance(sent_ratio, EnrichedMetric) and isinstance(mom_5, EnrichedMetric)):
            if sent_ratio.value > 0.3 and mom_5.value < -2:
                result.divergences.append(Divergence(
                    source_a="News Sentiment",
                    source_b="Price Action",
                    description="Sentiment is positive but price is declining",
                    severity="moderate",
                    implication="Positive sentiment with negative price action can indicate either "
                                "a buying opportunity (sentiment leads) or complacency (price leads)."
                ))
            elif sent_ratio.value < -0.3 and mom_5.value > 2:
                result.divergences.append(Divergence(
                    source_a="News Sentiment",
                    source_b="Price Action",
                    description="Sentiment is negative but price is rising",
                    severity="moderate",
                    implication="Negative sentiment with positive price action — classic 'wall of worry' "
                                "pattern. Can indicate strong underlying demand despite pessimism."
                ))

        # Divergence 3: Volume vs Price
        if (result.volume and mom_5
                and isinstance(result.volume, EnrichedMetric)
                and isinstance(mom_5, EnrichedMetric)):
            if result.volume.z_score_20d is not None:
                if mom_5.value > 2 and result.volume.z_score_20d < -1:
                    result.divergences.append(Divergence(
                        source_a="Price",
                        source_b="Volume",
                        description="Price is rising on below-average volume",
                        severity="weak",
                        implication="Low-volume rallies tend to be less sustainable. "
                                    "Watch for volume confirmation on continued moves."
                    ))
                elif abs(mom_5.value) < 1 and result.volume.z_score_20d > 2:
                    result.divergences.append(Divergence(
                        source_a="Volume",
                        source_b="Price",
                        description=f"Volume spike ({result.volume.z_score_20d:.1f}σ above average) "
                                    f"with minimal price change",
                        severity="strong",
                        implication="High volume without price movement often precedes a breakout. "
                                    "Suggests accumulation or distribution — watch direction of next move."
                    ))

    # ═══════════════════════════════════════════════════════════════
    # STEP 11: Conflict Detection
    # ═══════════════════════════════════════════════════════════════

    def _extract_agent_directions(self, agent_outputs: Dict) -> Dict[str, str]:
        """Extract directional signals from each agent's output."""
        directions = {}

        # MarketAnalyst direction
        market = agent_outputs.get('market_analyst', {})
        for sym, data in market.items():
            if isinstance(data, dict):
                sig = data.get('signal', {})
                if isinstance(sig, dict) and sig.get('direction'):
                    directions['market_analyst'] = self._normalize_direction(sig['direction'])

        # QuantAnalyst direction
        quant = agent_outputs.get('quant_analyst', {})
        ml_signals = quant.get('ml_signals', quant.get('signals', {}))
        if isinstance(ml_signals, dict):
            for sym, data in ml_signals.items():
                if isinstance(data, dict) and data.get('primary_signal'):
                    directions['quant_analyst'] = self._normalize_direction(data['primary_signal'])

        # NewsAnalyst direction (from sentiment)
        news = agent_outputs.get('news_analyst', {})
        if news:
            sym_news = news.get('symbol_news', {})
            for sym_data in sym_news.values():
                if isinstance(sym_data, dict):
                    articles = sym_data.get('articles', [])
                    pos = sum(1 for a in articles if a.get('sentiment') in ('Positive', 'Bullish'))
                    neg = sum(1 for a in articles if a.get('sentiment') in ('Negative', 'Bearish'))
                    if pos > neg * 1.5:
                        directions['news_analyst'] = 'BULLISH'
                    elif neg > pos * 1.5:
                        directions['news_analyst'] = 'BEARISH'
                    else:
                        directions['news_analyst'] = 'NEUTRAL'

        # RiskManager direction
        risk = agent_outputs.get('risk_manager', {})
        if risk:
            risk_score = risk.get('risk_score', {})
            overall = risk_score.get('overall', 50) if isinstance(risk_score, dict) else 50
            if overall > 70:
                directions['risk_manager'] = 'BEARISH'
            elif overall < 30:
                directions['risk_manager'] = 'BULLISH'
            else:
                directions['risk_manager'] = 'NEUTRAL'

        # MacroAnalyst direction
        macro = agent_outputs.get('macro_analyst', {})
        if macro:
            vol = macro.get('volatility', {})
            vix = vol.get('vix') if isinstance(vol, dict) else None
            if vix is not None:
                if vix < 18:
                    directions['macro_analyst'] = 'BULLISH'
                elif vix > 28:
                    directions['macro_analyst'] = 'BEARISH'
                else:
                    directions['macro_analyst'] = 'NEUTRAL'

        return directions

    def _detect_conflicts(self, agent_outputs: Dict, result: EnrichedIntelligence):
        """Detect conflicts between agent data sources."""
        agent_directions = self._extract_agent_directions(agent_outputs)

        # Detect pairwise conflicts
        agents = list(agent_directions.keys())
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                a, b = agents[i], agents[j]
                dir_a, dir_b = agent_directions[a], agent_directions[b]

                if dir_a == 'NEUTRAL' or dir_b == 'NEUTRAL':
                    continue

                if dir_a != dir_b:
                    pair = {a, b}
                    # Determine severity based on which agents disagree
                    if pair & {'market_analyst', 'quant_analyst'} and pair & {'risk_manager'}:
                        severity = "critical"
                        resolution = (
                            "Technical/quantitative signals disagree with risk assessment. "
                            "This suggests opportunity exists but with elevated risk. "
                            "Consider reduced position sizing or waiting for confirmation."
                        )
                    elif pair == {'market_analyst', 'news_analyst'}:
                        severity = "moderate"
                        resolution = (
                            "Technicals and news sentiment disagree. News can lead or lag price. "
                            "If news is newly negative with positive technicals, watch for potential reversal. "
                            "If technicals just turned while sentiment hasn't caught up, technicals may be leading."
                        )
                    elif pair & {'macro_analyst'}:
                        severity = "moderate"
                        resolution = (
                            "Macro environment disagrees with stock-specific signals. "
                            "Strong stock-specific catalysts can override macro headwinds, but macro "
                            "typically dominates during high-VIX regimes."
                        )
                    else:
                        severity = "minor"
                        resolution = f"{a} says {dir_a} while {b} says {dir_b}."

                    result.conflicts.append(ConflictV2(
                        sources=[a, b],
                        description=f"{a}={dir_a} vs {b}={dir_b}",
                        severity=severity,
                        agreement_score=-1.0,
                        resolution_guidance=resolution,
                    ))

    # ═══════════════════════════════════════════════════════════════
    # STEP 12: Consensus
    # ═══════════════════════════════════════════════════════════════

    def _compute_consensus(self, agent_outputs: Dict, result: EnrichedIntelligence):
        """Compute mathematical consensus across all agents."""
        directions: Dict[str, Tuple[str, float]] = {}

        # MarketAnalyst
        market = agent_outputs.get('market_analyst', {})
        for sym, data in market.items():
            if isinstance(data, dict):
                sig = data.get('signal', {})
                if isinstance(sig, dict) and sig.get('direction'):
                    directions['market_analyst'] = (
                        self._normalize_direction(sig['direction']),
                        sig.get('confidence', 0.5)
                    )

        # QuantAnalyst
        quant = agent_outputs.get('quant_analyst', {})
        ml_signals = quant.get('ml_signals', quant.get('signals', {}))
        if isinstance(ml_signals, dict):
            for sym, data in ml_signals.items():
                if isinstance(data, dict) and data.get('primary_signal'):
                    directions['quant_analyst'] = (
                        self._normalize_direction(data['primary_signal']),
                        data.get('confidence', 0.5)
                    )

        # NewsAnalyst
        news = agent_outputs.get('news_analyst', {})
        if news:
            sym_news = news.get('symbol_news', {})
            for sym_data in sym_news.values():
                if isinstance(sym_data, dict):
                    articles = sym_data.get('articles', [])
                    pos = sum(1 for a in articles if a.get('sentiment') in ('Positive', 'Bullish'))
                    neg = sum(1 for a in articles if a.get('sentiment') in ('Negative', 'Bearish'))
                    total = max(1, len(articles))
                    if pos > neg * 1.5:
                        directions['news_analyst'] = ('BULLISH', min(0.85, 0.4 + total * 0.03))
                    elif neg > pos * 1.5:
                        directions['news_analyst'] = ('BEARISH', min(0.85, 0.4 + total * 0.03))
                    else:
                        directions['news_analyst'] = ('NEUTRAL', 0.4)

        # RiskManager
        risk = agent_outputs.get('risk_manager', {})
        if risk:
            rs = risk.get('risk_score', {})
            overall = rs.get('overall', 50) if isinstance(rs, dict) else 50
            if overall > 70:
                directions['risk_manager'] = ('BEARISH', 0.75)
            elif overall < 30:
                directions['risk_manager'] = ('BULLISH', 0.75)
            else:
                directions['risk_manager'] = ('NEUTRAL', 0.6)

        # MacroAnalyst
        macro = agent_outputs.get('macro_analyst', {})
        if macro:
            vol = macro.get('volatility', {})
            vix = vol.get('vix') if isinstance(vol, dict) else None
            if vix is not None:
                if vix < 18:
                    directions['macro_analyst'] = ('BULLISH', 0.65)
                elif vix > 28:
                    directions['macro_analyst'] = ('BEARISH', 0.75)
                else:
                    directions['macro_analyst'] = ('NEUTRAL', 0.5)

        if not directions:
            result.consensus = ConsensusMetrics(
                direction_agreement=0, weighted_direction_agreement=0,
                bullish_sources=[], bearish_sources=[], neutral_sources=[],
                dominant_direction="UNKNOWN", conviction_level="none"
            )
            return

        bullish = [name for name, (d, c) in directions.items() if d == 'BULLISH']
        bearish = [name for name, (d, c) in directions.items() if d == 'BEARISH']
        neutral = [name for name, (d, c) in directions.items() if d == 'NEUTRAL']

        total = len(directions)
        max_count = max(len(bullish), len(bearish), len(neutral))
        direction_agreement = max_count / total

        # Weighted agreement
        bullish_weight = sum(c for name, (d, c) in directions.items() if d == 'BULLISH')
        bearish_weight = sum(c for name, (d, c) in directions.items() if d == 'BEARISH')
        total_weight = sum(c for name, (d, c) in directions.items())

        if bullish_weight > bearish_weight:
            dominant = "BULLISH"
            weighted_agreement = bullish_weight / total_weight if total_weight else 0
        elif bearish_weight > bullish_weight:
            dominant = "BEARISH"
            weighted_agreement = bearish_weight / total_weight if total_weight else 0
        else:
            dominant = "SPLIT" if bullish_weight == bearish_weight and bullish_weight > 0 else "NEUTRAL"
            weighted_agreement = 0.5

        if direction_agreement >= 0.8:
            conviction = "high"
        elif direction_agreement >= 0.6:
            conviction = "moderate"
        elif direction_agreement >= 0.4:
            conviction = "low"
        else:
            conviction = "divided"

        result.consensus = ConsensusMetrics(
            direction_agreement=round(direction_agreement, 3),
            weighted_direction_agreement=round(weighted_agreement, 3),
            bullish_sources=bullish,
            bearish_sources=bearish,
            neutral_sources=neutral,
            dominant_direction=dominant,
            conviction_level=conviction,
        )

    # ═══════════════════════════════════════════════════════════════
    # STEP 13: Data Coverage
    # ═══════════════════════════════════════════════════════════════

    def _compute_data_coverage(self, agent_outputs: Dict, result: EnrichedIntelligence):
        """Compute what % of data sources are reporting."""
        expected = ['market_analyst', 'news_analyst', 'risk_manager', 'fundamental_analyst',
                    'quant_analyst', 'macro_analyst', 'portfolio_optimizer', 'crypto_specialist']
        reporting = [a for a in expected if a in agent_outputs and agent_outputs[a]]

        result.sources_reporting = len(reporting)
        result.sources_total = len(expected)
        result.data_coverage_pct = len(reporting) / len(expected) * 100

    # ═══════════════════════════════════════════════════════════════
    # STEP 14: Build Intelligence Brief
    # ═══════════════════════════════════════════════════════════════

    def _build_intelligence_brief(self, r: EnrichedIntelligence) -> str:
        """
        Build the final intelligence brief for Claude.

        This is the KEY output. It's what Claude reads to make its judgment.
        Every number is preserved. Every anomaly is flagged.
        Claude has all the context it needs to reason expertly.
        """
        lines = []
        lines.append(f"═══ STATISTICAL INTELLIGENCE BRIEF: {r.symbol} ═══")
        lines.append(f"Generated: {r.timestamp}")
        lines.append(f"Data Coverage: {r.sources_reporting}/{r.sources_total} agents ({r.data_coverage_pct:.0f}%)")
        lines.append("")

        # ── CONSENSUS ──
        if r.consensus:
            c = r.consensus
            lines.append(f"MULTI-AGENT CONSENSUS: {c.dominant_direction} "
                         f"(conviction: {c.conviction_level}, agreement: {c.direction_agreement:.0%})")
            if c.bullish_sources:
                lines.append(f"  Bullish: {', '.join(c.bullish_sources)}")
            if c.bearish_sources:
                lines.append(f"  Bearish: {', '.join(c.bearish_sources)}")
            if c.neutral_sources:
                lines.append(f"  Neutral: {', '.join(c.neutral_sources)}")
            lines.append("")

        # ── ANOMALIES (highest priority — these are unusual and actionable) ──
        if r.anomalies:
            lines.append(f"⚠️ STATISTICAL ANOMALIES DETECTED ({len(r.anomalies)}):")
            for a in sorted(r.anomalies, key=lambda x: abs(x.z_score), reverse=True):
                lines.append(f"  • {a.description}")
                lines.append(f"    Context: {a.historical_context}")
            lines.append("")

        # ── DIVERGENCES ──
        if r.divergences:
            lines.append(f"DIVERGENCES DETECTED ({len(r.divergences)}):")
            for d in r.divergences:
                lines.append(f"  [{d.severity.upper()}] {d.description}")
                lines.append(f"    Implication: {d.implication}")
            lines.append("")

        # ── CONFLICTS ──
        if r.conflicts:
            lines.append(f"AGENT CONFLICTS ({len(r.conflicts)}):")
            for c in sorted(r.conflicts, key=lambda x: {'critical': 0, 'moderate': 1, 'minor': 2}.get(x.severity, 3)):
                lines.append(f"  [{c.severity.upper()}] {c.description}")
                lines.append(f"    Guidance: {c.resolution_guidance}")
            lines.append("")

        # ── PRICE & VOLUME ──
        if r.price:
            lines.append(f"PRICE: ${r.price.value:.2f}")
            self._append_metric_context(lines, r.price, indent="  ")
        if r.volume:
            lines.append(f"VOLUME: {r.volume.value:,.0f}")
            self._append_metric_context(lines, r.volume, indent="  ")
        lines.append("")

        # ── TECHNICALS ──
        if r.technicals:
            lines.append("TECHNICAL INDICATORS (with statistical context):")
            for key, metric in sorted(r.technicals.items()):
                if isinstance(metric, EnrichedMetric):
                    anomaly_flag = " ⚠️ ANOMALY" if metric.is_anomaly else ""
                    lines.append(f"  {metric.name}: {metric.value:.4g} {metric.unit}{anomaly_flag}")
                    self._append_metric_context(lines, metric, indent="    ")
            lines.append("")

        # ── SENTIMENT ──
        if r.sentiment:
            lines.append("SENTIMENT DATA:")
            for key, metric in r.sentiment.items():
                if isinstance(metric, EnrichedMetric):
                    lines.append(f"  {metric.name}: {metric.value:.4g} {metric.unit}")
                    if metric.anomaly_description:
                        lines.append(f"    {metric.anomaly_description}")
            lines.append("")

        # ── LUXALGO SIGNALS ──
        if r.luxalgo and r.luxalgo.get('valid_count', 0) > 0:
            lx = r.luxalgo
            lines.append(f"LUXALGO PREMIUM SIGNALS:")
            lines.append(f"  Weekly: {lx.get('weekly_action', 'N/A')} | "
                         f"Daily: {lx.get('daily_action', 'N/A')} | "
                         f"4H: {lx.get('h4_action', 'N/A')}")
            lines.append(f"  Aligned: {'YES ✅' if lx.get('aligned') else 'NO'} | "
                         f"Valid timeframes: {lx.get('valid_count', 0)}/3")
            lines.append("")

        # ── ML PREDICTIONS ──
        if r.ml_predictions:
            lines.append("ML PREDICTIONS:")
            # v2 prediction (F4 pipeline — highest quality)
            v2 = r.ml_predictions.get('v2_prediction')
            if v2 and isinstance(v2, dict):
                direction = v2.get('direction', 'N/A')
                confidence = v2.get('confidence', 0)
                lines.append(f"  F4 Pipeline: {direction} ({confidence:.0%})")
                proba = v2.get('probabilities', {})
                if proba:
                    proba_str = ', '.join(f"{k.replace('proba_', '')}={v:.0%}" for k, v in proba.items())
                    lines.append(f"    Probabilities: {proba_str}")
                explanation = v2.get('explanation', {})
                top_feats = explanation.get('top_features', [])
                if top_feats:
                    lines.append("    SHAP drivers:")
                    for feat in top_feats[:5]:
                        fname = feat.get('feature', '?')
                        fval = feat.get('shap_value', feat.get('value', 0))
                        arrow = "↑" if fval > 0 else "↓"
                        lines.append(f"      {arrow} {fname}: {fval:+.4f}")
                model_info = v2.get('model_info', {})
                if model_info.get('cv_mean_ic'):
                    lines.append(f"    CV IC: {model_info['cv_mean_ic']:.4f}")
            # Legacy predictions
            for key, val in r.ml_predictions.items():
                if key == 'v2_prediction':
                    continue
                if isinstance(val, dict):
                    for sym, sig in val.items():
                        if isinstance(sig, dict):
                            lines.append(f"  {sym}: {sig.get('primary_signal', 'N/A')} "
                                         f"(confidence: {sig.get('confidence', 'N/A')}, "
                                         f"model: {sig.get('model_used', 'N/A')})")
                elif key in ('ml_direction', 'ml_confidence'):
                    lines.append(f"  Lambda {key}: {val}")
            lines.append("")

        # ── FUNDAMENTALS ──
        if r.fundamentals:
            lines.append("FUNDAMENTAL DATA:")
            for key, val in r.fundamentals.items():
                if key in ('valuation', 'growth_metrics', 'financial_health') and isinstance(val, dict):
                    lines.append(f"  {key}:")
                    for k, v in val.items():
                        if v is not None:
                            lines.append(f"    {k}: {v}")
                elif key == 'company_overview' and isinstance(val, dict):
                    name = val.get('name', '')
                    sector = val.get('sic_description', '')
                    if name:
                        lines.append(f"  Company: {name} ({sector})")
            lines.append("")

        # ── MACRO CONTEXT ──
        has_macro = False
        if r.macro:
            lines.append("MACRO CONTEXT:")
            vix = r.macro.get('vix')
            if isinstance(vix, EnrichedMetric):
                anomaly_flag = " ⚠️ ANOMALY" if vix.is_anomaly else ""
                lines.append(f"  VIX: {vix.value:.1f}{anomaly_flag}")
                self._append_metric_context(lines, vix, indent="    ")
                has_macro = True

            for key in ['sector_performance', 'yield_curve', 'dollar', 'market_breadth']:
                val = r.macro.get(key)
                if val and isinstance(val, dict):
                    lines.append(f"  {key}: {self._compact_dict(val)}")
                    has_macro = True
            if has_macro:
                lines.append("")

        # ── RISK ──
        if r.risk:
            lines.append("RISK ASSESSMENT:")
            if isinstance(r.risk, dict):
                rs = r.risk.get('risk_score', {})
                if isinstance(rs, dict):
                    lines.append(f"  Overall Risk Score: {rs.get('overall', 'N/A')}/100")
                var = r.risk.get('var_analysis', {})
                if isinstance(var, dict) and var.get('var_95'):
                    lines.append(f"  VaR (95%): {var['var_95']}")
                dd = r.risk.get('drawdown', {})
                if isinstance(dd, dict) and dd.get('max_drawdown'):
                    lines.append(f"  Max Drawdown: {dd['max_drawdown']}")
            lines.append("")

        lines.append(f"═══ END INTELLIGENCE BRIEF ═══")

        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════

    def _enrich_single_metric(
        self, name: str, value: float, unit: str,
        history: Optional[np.ndarray] = None
    ) -> EnrichedMetric:
        """Enrich a single metric with statistical context from its history."""
        metric = EnrichedMetric(name=name, value=value, unit=unit)

        if history is not None and len(history) >= 20:
            history = history[np.isfinite(history)]  # Remove NaN/Inf
            if len(history) >= 20:
                # Percentile rank in 90-day (or available) history
                metric.percentile_90d = float(np.sum(history <= value) / len(history) * 100)

                # Z-score from 20-day mean/std
                recent = history[-20:]
                mean_20d = float(np.mean(recent))
                std_20d = float(np.std(recent))
                if std_20d > 1e-10:
                    metric.z_score_20d = float((value - mean_20d) / std_20d)

                # Rate of change (5-day)
                if len(history) >= 5:
                    metric.rate_of_change_5d = float(value - history[-5])

                # Historical range
                metric.min_90d = float(np.min(history))
                metric.max_90d = float(np.max(history))
                metric.mean_90d = float(np.mean(history))

                # Anomaly detection
                if metric.z_score_20d is not None and abs(metric.z_score_20d) > 2.0:
                    metric.is_anomaly = True

        return metric

    def _compute_rsi_series(self, closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute RSI for the entire price series."""
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        rsi_values = []
        if len(gains) < period:
            return np.array([])

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100.0 - (100.0 / (1.0 + rs)))

        return np.array(rsi_values)

    def _compute_macd_histogram_series(self, closes: np.ndarray) -> np.ndarray:
        """Compute MACD histogram for the entire series."""
        if len(closes) < 35:
            return np.array([])

        ema_12 = self._ema(closes, 12)
        ema_26 = self._ema(closes, 26)

        # Align lengths
        min_len = min(len(ema_12), len(ema_26))
        macd_line = ema_12[-min_len:] - ema_26[-min_len:]

        signal = self._ema(macd_line, 9)
        min_len2 = min(len(macd_line), len(signal))
        histogram = macd_line[-min_len2:] - signal[-min_len2:]

        return histogram

    def _compute_sma_series(self, closes: np.ndarray, window: int) -> np.ndarray:
        """Compute SMA series."""
        if len(closes) < window:
            return np.array([])
        return np.convolve(closes, np.ones(window) / window, mode='valid')

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA series."""
        if len(data) < period:
            return np.array([])

        multiplier = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[period - 1] = np.mean(data[:period])

        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]

        return ema[period - 1:]

    def _normalize_direction(self, direction: str) -> str:
        """Normalize direction strings."""
        d = direction.upper()
        if d in ('BUY', 'BULLISH', 'LONG', 'UP', 'STRONG_BUY'):
            return 'BULLISH'
        elif d in ('SELL', 'BEARISH', 'SHORT', 'DOWN', 'STRONG_SELL'):
            return 'BEARISH'
        return 'NEUTRAL'

    def _append_metric_context(self, lines: List[str], metric: EnrichedMetric, indent: str = "  "):
        """Append statistical context lines for a metric."""
        parts = []
        if metric.percentile_90d is not None:
            parts.append(f"90d percentile: {metric.percentile_90d:.0f}th")
        if metric.z_score_20d is not None:
            parts.append(f"z-score: {metric.z_score_20d:+.2f}σ")
        if metric.rate_of_change_5d is not None:
            parts.append(f"5d Δ: {metric.rate_of_change_5d:+.4g}")
        if metric.min_90d is not None and metric.max_90d is not None:
            parts.append(f"90d range: [{metric.min_90d:.4g}, {metric.max_90d:.4g}]")

        if parts:
            lines.append(f"{indent}{' | '.join(parts)}")

    def _compact_dict(self, d: Dict, max_items: int = 5) -> str:
        """Compact dict representation for the brief."""
        items = []
        for k, v in list(d.items())[:max_items]:
            if isinstance(v, dict):
                # Show one level deep
                sub = ', '.join(f"{sk}={sv}" for sk, sv in list(v.items())[:3] if sv is not None)
                items.append(f"{k}({sub})")
            elif isinstance(v, float):
                items.append(f"{k}={v:.4g}")
            elif v is not None:
                items.append(f"{k}={v}")
        return "; ".join(items)


__all__ = ['EnrichmentEngine', 'EnrichedIntelligence', 'EnrichedMetric',
           'Anomaly', 'Divergence', 'ConflictV2', 'ConsensusMetrics']
