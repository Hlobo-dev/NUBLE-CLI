#!/usr/bin/env python3
"""
NUBLE Macro Analyst Agent — ELITE TIER

Comprehensive macroeconomic analysis using multiple real data sources:
- Polygon.io VIX — Volatility regime monitoring
- Polygon.io Sector ETFs — Sector rotation & performance
- Polygon.io Treasury proxies (SHY/IEF/TLT) — Yield curve analysis
- Polygon.io Dollar (UUP/UDN) — Dollar strength
- Polygon.io Commodities (GLD, USO) — Inflation signals
- Polygon.io News — Macro-relevant headlines
- Polygon.io SMA — Sector trend analysis
- Alternative.me Fear & Greed — Sentiment overlay
- StockNews API — Macro event detection
"""

import os
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)

# ── FRED Macro Data integration (Phase 1 upgrade) ────────────
_fred_macro = None


def _get_fred_macro():
    """Lazy-load FRED macro data singleton."""
    global _fred_macro
    if _fred_macro is None:
        try:
            from ..data.fred_macro import FREDMacroData
            _fred_macro = FREDMacroData()
            logger.info("FRED macro data pipeline initialized")
        except Exception as e:
            logger.debug("FRED macro data not available: %s", e)
    return _fred_macro


class MacroAnalystAgent(SpecializedAgent):
    """
    Macro Analyst Agent — ELITE Macroeconomic Analysis
    
    REAL DATA from 8+ sources:
    1. Polygon VIX — Volatility regime
    2. Polygon 11 Sector ETFs — Sector rotation, leadership, relative strength
    3. Polygon Treasury proxies (SHY/IEF/TLT) — Yield curve shape
    4. Polygon Dollar (UUP) — Currency risk
    5. Polygon Commodities (GLD/USO) — Inflation/safe-haven demand
    6. Polygon Market Indices (SPY/QQQ/IWM/DIA) — Breadth
    7. Polygon News — Macro headlines & event risk
    8. Polygon SMA — Sector trend scoring
    9. Alternative.me Fear & Greed — Sentiment overlay
    10. StockNews API — Macro event detection
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        self.stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Macro Analyst",
            "description": "Elite macroeconomic analysis from 8+ real data sources",
            "capabilities": [
                "volatility_regime", "sector_rotation", "market_breadth",
                "yield_curve", "dollar_strength", "commodity_signals",
                "macro_news", "risk_assessment", "cross_asset_analysis"
            ],
            "data_sources": [
                "Polygon VIX", "Polygon Sectors", "Polygon Treasury Proxies",
                "Polygon Dollar", "Polygon Commodities", "Polygon News",
                "Polygon SMA", "Alternative.me FGI", "StockNews API"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute macro analysis with real data."""
        start = datetime.now()
        
        try:
            shared = self._get_shared_data(task)
            data = {
                'volatility_regime': await self._analyze_volatility(shared),
                'sector_performance': await self._get_sector_performance(shared),
                'market_breadth': await self._analyze_market_breadth(shared),
                'yield_curve': await self._analyze_yield_curve(shared),
                'dollar_strength': await self._analyze_dollar(shared),
                'commodity_signals': await self._analyze_commodities(shared),
                'macro_news': await self._get_macro_news(shared),
                'sentiment_overlay': await self._get_sentiment(shared),
                'risk_assessment': await self._assess_macro_risk(shared),
                'market_impact': await self._assess_impact(shared)
            }
            
            # === FRED Macro Data (Phase 1 upgrade) ===
            try:
                fred = _get_fred_macro()
                if fred and fred.is_available:
                    fred_data = fred.get_current()
                    if fred_data:
                        data['fred_macro'] = fred_data
                        # Enrich yield curve with real FRED data
                        regimes = fred_data.get('regimes', {})
                        if regimes.get('yield_curve'):
                            data['yield_curve_fred'] = regimes['yield_curve']
                        if regimes.get('credit_cycle'):
                            data['credit_cycle'] = regimes['credit_cycle']
                        if regimes.get('monetary_policy'):
                            data['monetary_policy'] = regimes['monetary_policy']
                        logger.info("FRED macro data added: %d raw indicators, %d regimes",
                                    len(fred_data.get('raw', {})), len(regimes))
            except Exception as e:
                logger.debug("FRED macro integration failed: %s", e)
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.MACRO_ANALYST,
                success=True,
                data=data,
                confidence=0.80,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.MACRO_ANALYST,
                success=False,
                data={},
                confidence=0,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000),
                error=str(e)
            )
    
    def _polygon_get(self, url: str, params: Dict = None) -> Dict:
        """Helper for Polygon API calls."""
        p = {'apiKey': self.polygon_key}
        if params:
            p.update(params)
        try:
            resp = requests.get(url, params=p, timeout=12)
            if resp.status_code == 200:
                return resp.json()
        except Exception as e:
            logger.warning(f"Polygon call failed: {e}")
        return {}
    
    async def _get_prev_close(self, ticker: str, shared=None) -> Dict:
        """Get previous close data from Polygon."""
        data = (await shared.get_quote(ticker)) if shared else self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev")
        results = data.get('results', [])
        if results:
            r = results[0]
            change_pct = ((r['c'] - r['o']) / r['o']) * 100 if r['o'] else 0
            return {'close': r['c'], 'open': r['o'], 'high': r['h'], 'low': r['l'],
                    'volume': r.get('v', 0), 'change_pct': round(change_pct, 2)}
        return {}
    
    async def _analyze_volatility(self, shared=None) -> Dict:
        """Analyze volatility regime from real VIX data."""
        vix_data = await self._get_prev_close('VIX', shared)
        
        if not vix_data:
            return {'error': 'VIX data unavailable'}
        
        vix = vix_data['close']
        
        if vix > 35:
            regime, stance = 'CRISIS', 'EXTREME FEAR'
        elif vix > 25:
            regime, stance = 'HIGH_VOL', 'ELEVATED FEAR'
        elif vix > 20:
            regime, stance = 'NORMAL', 'CAUTIOUS'
        elif vix > 15:
            regime, stance = 'LOW_VOL', 'COMPLACENT'
        else:
            regime, stance = 'VERY_LOW_VOL', 'EXTREME COMPLACENCY'
        
        return {
            'vix': round(vix, 2),
            'vix_change': vix_data.get('change_pct', 0),
            'regime': regime,
            'market_stance': stance,
            'data_source': 'polygon_live'
        }
    
    async def _get_sector_performance(self, shared=None) -> Dict:
        """Get real sector ETF performance with SMA trend scoring."""
        sector_etfs = {
            'Technology': 'XLK', 'Healthcare': 'XLV', 'Financials': 'XLF',
            'Energy': 'XLE', 'Consumer Discretionary': 'XLY', 'Consumer Staples': 'XLP',
            'Industrials': 'XLI', 'Materials': 'XLB', 'Utilities': 'XLU',
            'Real Estate': 'XLRE', 'Communication Services': 'XLC',
        }
        
        performance = {}
        for sector, etf in sector_etfs.items():
            data = await self._get_prev_close(etf, shared)
            if data:
                entry = {
                    'etf': etf,
                    'price': data['close'],
                    'change_pct': data.get('change_pct', 0),
                    'volume': data.get('volume', 0),
                }
                
                # Add SMA trend from Polygon server-side indicator
                sma_data = (await shared.get_sma(etf, 50)) if shared else self._polygon_get(
                    f"https://api.polygon.io/v1/indicators/sma/{etf}",
                    {'timespan': 'day', 'window': 50, 'series_type': 'close', 'order': 'desc', 'limit': 1}
                )
                sma_results = sma_data.get('results', {}).get('values', [])
                if sma_results:
                    sma_50 = sma_results[0].get('value', 0)
                    entry['sma_50'] = round(sma_50, 2)
                    entry['above_sma50'] = data['close'] > sma_50
                    entry['pct_vs_sma50'] = round(((data['close'] - sma_50) / sma_50) * 100, 2)
                
                performance[sector] = entry
        
        if performance:
            sorted_sectors = sorted(performance.items(), key=lambda x: x[1].get('change_pct', 0), reverse=True)
            leaders = [s[0] for s in sorted_sectors[:3]]
            laggards = [s[0] for s in sorted_sectors[-3:]]
            
            # Risk-on vs risk-off scoring
            offensive = ['Technology', 'Consumer Discretionary', 'Financials', 'Communication Services']
            defensive = ['Utilities', 'Consumer Staples', 'Healthcare', 'Real Estate']
            
            off_avg = np.mean([performance[s]['change_pct'] for s in offensive if s in performance])
            def_avg = np.mean([performance[s]['change_pct'] for s in defensive if s in performance])
            rotation = 'RISK_ON' if off_avg > def_avg else 'RISK_OFF'
            
            above_sma = sum(1 for s in performance.values() if s.get('above_sma50', False))
            
            return {
                'sectors': performance,
                'leaders': leaders,
                'laggards': laggards,
                'rotation_signal': rotation,
                'offensive_avg_pct': round(float(off_avg), 2),
                'defensive_avg_pct': round(float(def_avg), 2),
                'sectors_above_sma50': above_sma,
                'total_sectors': len(performance),
                'data_source': 'polygon_live'
            }
        
        return {'error': 'Sector data unavailable'}
    
    async def _analyze_market_breadth(self, shared=None) -> Dict:
        """Analyze market breadth using major indices."""
        indices = {
            'SPY': 'S&P 500', 'QQQ': 'Nasdaq 100',
            'IWM': 'Russell 2000', 'DIA': 'Dow Jones',
        }
        
        breadth = {}
        for ticker, name in indices.items():
            data = await self._get_prev_close(ticker, shared)
            if data:
                breadth[name] = {
                    'ticker': ticker,
                    'price': data['close'],
                    'change_pct': data.get('change_pct', 0),
                }
        
        if len(breadth) >= 2:
            changes = [v.get('change_pct', 0) for v in breadth.values() if isinstance(v, dict)]
            advancing = sum(1 for c in changes if c > 0)
            avg_change = np.mean(changes)
            
            if advancing == len(changes) and avg_change > 0.5:
                signal = 'STRONG_BULLISH'
            elif advancing >= len(changes) * 0.75:
                signal = 'BULLISH'
            elif advancing <= len(changes) * 0.25:
                signal = 'BEARISH'
            elif advancing == 0 and avg_change < -0.5:
                signal = 'STRONG_BEARISH'
            else:
                signal = 'MIXED'
            
            breadth['breadth_signal'] = signal
            breadth['advancing'] = advancing
            breadth['declining'] = len(changes) - advancing
            breadth['data_source'] = 'polygon_live'
        
        return breadth if breadth else {'error': 'Market breadth data unavailable'}
    
    async def _analyze_yield_curve(self, shared=None) -> Dict:
        """Analyze yield curve using treasury bond ETF proxies."""
        shy = await self._get_prev_close('SHY', shared)   # 1-3yr treasuries (short end)
        ief = await self._get_prev_close('IEF', shared)   # 7-10yr treasuries (mid)
        tlt = await self._get_prev_close('TLT', shared)   # 20+yr treasuries (long end)
        
        if not shy or not tlt:
            return {'error': 'Treasury data unavailable', 'data_source': 'polygon_error'}
        
        # When long bonds rally (TLT up) vs short (SHY), the curve is flattening/inverting
        # Use relative performance as proxy for curve shape
        tlt_change = tlt.get('change_pct', 0)
        shy_change = shy.get('change_pct', 0)
        ief_change = ief.get('change_pct', 0) if ief else 0
        
        spread = tlt_change - shy_change  # positive = long bonds outperforming = flattening
        
        if spread > 0.5:
            curve_signal = 'FLATTENING'
            implication = 'Flight to safety / growth concerns'
        elif spread < -0.5:
            curve_signal = 'STEEPENING'
            implication = 'Growth expectations rising / inflation concerns'
        else:
            curve_signal = 'STABLE'
            implication = 'No significant curve shift'
        
        return {
            'shy_price': shy['close'],
            'shy_change_pct': shy_change,
            'ief_price': ief.get('close') if ief else None,
            'ief_change_pct': ief_change,
            'tlt_price': tlt['close'],
            'tlt_change_pct': tlt_change,
            'curve_spread': round(spread, 2),
            'curve_signal': curve_signal,
            'implication': implication,
            'data_source': 'polygon_live'
        }
    
    async def _analyze_dollar(self, shared=None) -> Dict:
        """Analyze US dollar strength."""
        uup = await self._get_prev_close('UUP', shared)  # Dollar bullish ETF
        
        if not uup:
            return {'error': 'Dollar data unavailable'}
        
        change = uup.get('change_pct', 0)
        if change > 0.3:
            signal = 'STRENGTHENING'
            impact = 'Headwind for earnings, commodities, EM'
        elif change < -0.3:
            signal = 'WEAKENING'
            impact = 'Tailwind for earnings, commodities, EM'
        else:
            signal = 'STABLE'
            impact = 'Neutral dollar impact'
        
        return {
            'uup_price': uup['close'],
            'change_pct': change,
            'signal': signal,
            'impact': impact,
            'data_source': 'polygon_live'
        }
    
    async def _analyze_commodities(self, shared=None) -> Dict:
        """Analyze commodity signals for inflation/risk."""
        gld = await self._get_prev_close('GLD', shared)  # Gold
        uso = await self._get_prev_close('USO', shared)  # Oil
        
        result = {'data_source': 'polygon_live'}
        
        if gld:
            gld_change = gld.get('change_pct', 0)
            result['gold'] = {
                'price': gld['close'],
                'change_pct': gld_change,
                'signal': 'SAFE_HAVEN_DEMAND' if gld_change > 0.5 else 'RISK_ON' if gld_change < -0.5 else 'NEUTRAL'
            }
        
        if uso:
            uso_change = uso.get('change_pct', 0)
            result['oil'] = {
                'price': uso['close'],
                'change_pct': uso_change,
                'signal': 'INFLATION_RISK' if uso_change > 1 else 'DEFLATION_SIGNAL' if uso_change < -1 else 'STABLE'
            }
        
        return result
    
    async def _get_macro_news(self, shared=None) -> Dict:
        """Get macro-relevant news from Polygon & StockNews PRO (ALL premium endpoints)."""
        headlines = []
        
        # Polygon news — broad market
        data = self._polygon_get("https://api.polygon.io/v2/reference/news", {
            'ticker': 'SPY', 'limit': 5, 'order': 'desc'
        })
        for article in data.get('results', []):
            headlines.append({
                'title': article.get('title', ''),
                'source': article.get('publisher', {}).get('name', ''),
                'date': article.get('published_utc', ''),
                'tickers': [t.get('ticker', '') for t in article.get('tickers', [])],
                'data_source': 'polygon_news'
            })
        
        # StockNews PRO — General Market News (Fed, CPI, macro — premium section=general)
        general_market = []
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/category", params={
                    'section': 'general', 'items': 8, 'sortby': 'rank',
                    'extra-fields': 'id,eventid,rankscore',
                    'page': 1, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    for article in resp.json().get('data', []):
                        general_market.append({
                            'title': article.get('title', ''),
                            'sentiment': article.get('sentiment', ''),
                            'source': article.get('source_name', ''),
                            'rank_score': article.get('rankscore', ''),
                            'date': article.get('date', ''),
                            'data_source': 'stocknews_general_market'
                        })
                        headlines.append({
                            'title': article.get('title', ''),
                            'sentiment': article.get('sentiment', ''),
                            'source': article.get('source_name', ''),
                            'data_source': 'stocknews_general_market'
                        })
            except Exception:
                pass
        
        # StockNews PRO — Trending Headlines (market-moving stories)
        trending = []
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/trending-headlines", params={
                    'page': 1, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    for h in resp.json().get('data', [])[:5]:
                        trending.append({
                            'title': h.get('title', ''),
                            'description': (h.get('text', '')[:200]) if h.get('text') else '',
                            'source': h.get('source_name', ''),
                            'date': h.get('date', ''),
                        })
            except Exception:
                pass
        
        # StockNews PRO — Breaking Events (macro-relevant)
        breaking_events = []
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/events", params={
                    'page': 1, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    for e in resp.json().get('data', [])[:5]:
                        breaking_events.append({
                            'title': e.get('title', ''),
                            'event_id': e.get('eventid', ''),
                            'source': e.get('source_name', ''),
                            'date': e.get('date', ''),
                        })
            except Exception:
                pass
        
        # StockNews PRO — Top Mentioned Tickers (market attention)
        top_mentioned = []
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/top-mention", params={
                    'date': 'last7days', 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    top_mentioned = resp.json().get('data', [])[:10]
            except Exception:
                pass
        
        # StockNews PRO — Sundown Digest (daily macro summary)
        sundown = []
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/sundown-digest", params={
                    'page': 1, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    for d in resp.json().get('data', [])[:2]:
                        sundown.append({
                            'title': d.get('title', ''),
                            'text': d.get('text', ''),
                            'date': d.get('date', ''),
                        })
            except Exception:
                pass
        
        # Detect macro themes from ALL collected headlines
        themes = set()
        for h in headlines:
            title = h.get('title', '').lower()
            theme_map = {
                'RATE_HIKE': ['rate hike', 'rate increase', 'hawkish', 'tighten'],
                'RATE_CUT': ['rate cut', 'rate decrease', 'dovish', 'easing'],
                'INFLATION': ['inflation', 'cpi', 'pce', 'price increase'],
                'RECESSION': ['recession', 'slowdown', 'contraction', 'gdp decline'],
                'GEOPOLITICAL': ['war', 'tariff', 'sanction', 'geopolitical', 'conflict'],
                'EARNINGS': ['earnings season', 'quarterly results', 'beat expectations'],
                'LABOR': ['jobs', 'unemployment', 'nonfarm', 'labor market', 'payroll'],
                'TRADE': ['trade war', 'tariff', 'import', 'export', 'trade deal'],
                'BANKING': ['bank crisis', 'bank failure', 'svb', 'credit crunch'],
                'FISCAL': ['government shutdown', 'debt ceiling', 'deficit', 'spending bill'],
            }
            for theme, keywords in theme_map.items():
                if any(k in title for k in keywords):
                    themes.add(theme)
        
        return {
            'headlines': headlines[:8],
            'general_market_news': general_market[:5],
            'trending_headlines': trending,
            'breaking_events': breaking_events,
            'top_mentioned_tickers': top_mentioned,
            'sundown_digest': sundown,
            'detected_themes': list(themes),
            'data_source': 'polygon_news + stocknews_pro_full'
        }
    
    async def _get_sentiment(self, shared=None) -> Dict:
        """Get market sentiment from Fear & Greed Index."""
        if shared:
            fng_data = await shared.get_fear_greed()
            if fng_data and fng_data.get('data'):
                fng = fng_data['data']
                current = int(fng[0].get('value', 50))
                trend = [int(d.get('value', 50)) for d in fng]
                return {
                    'fear_greed_index': current,
                    'label': fng[0].get('value_classification', 'Unknown'),
                    'trend_7d': trend,
                    'direction': 'IMPROVING' if trend[0] > trend[-1] else 'DETERIORATING',
                    'data_source': 'alternative_me_live'
                }
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
            if resp.status_code == 200:
                fng_data = resp.json().get('data', [])
                if fng_data:
                    current = int(fng_data[0].get('value', 50))
                    trend = [int(d.get('value', 50)) for d in fng_data]
                    return {
                        'fear_greed_index': current,
                        'label': fng_data[0].get('value_classification', 'Unknown'),
                        'trend_7d': trend,
                        'direction': 'IMPROVING' if trend[0] > trend[-1] else 'DETERIORATING',
                        'data_source': 'alternative_me_live'
                    }
        except Exception:
            pass
        return {'error': 'Sentiment data unavailable'}
    
    async def _assess_macro_risk(self, shared=None) -> Dict:
        """Assess overall macro risk from real indicators."""
        risks = []
        risk_score = 0
        
        # VIX level
        vix_data = await self._get_prev_close('VIX', shared)
        if vix_data:
            vix = vix_data['close']
            if vix > 30:
                risks.append(f'VIX at {vix:.1f} — extreme volatility')
                risk_score += 3
            elif vix > 20:
                risks.append(f'VIX at {vix:.1f} — elevated volatility')
                risk_score += 1
        
        # Gold (safe haven demand)
        gld_data = await self._get_prev_close('GLD', shared)
        if gld_data and gld_data.get('change_pct', 0) > 1:
            risks.append(f"Gold up {gld_data['change_pct']:.1f}% — safe haven demand")
            risk_score += 1
        
        # TLT (flight to safety)
        tlt_data = await self._get_prev_close('TLT', shared)
        if tlt_data and tlt_data.get('change_pct', 0) > 1:
            risks.append(f"Long bonds up {tlt_data['change_pct']:.1f}% — flight to safety")
            risk_score += 1
        
        # Small caps divergence
        spy_data = await self._get_prev_close('SPY', shared)
        iwm_data = await self._get_prev_close('IWM', shared)
        if spy_data and iwm_data:
            divergence = (spy_data.get('change_pct', 0) - iwm_data.get('change_pct', 0))
            if abs(divergence) > 1.5:
                risks.append(f'Large/small cap divergence {divergence:.1f}% — breadth concern')
                risk_score += 1
        
        # Dollar shock
        uup = await self._get_prev_close('UUP', shared)
        if uup and abs(uup.get('change_pct', 0)) > 0.5:
            risks.append(f"Dollar move {uup['change_pct']:.1f}% — currency risk")
            risk_score += 1
        
        if not risks:
            risks.append('No significant macro risk signals detected')
        
        if risk_score >= 5:
            overall = 'HIGH'
        elif risk_score >= 3:
            overall = 'ELEVATED'
        elif risk_score >= 1:
            overall = 'MODERATE'
        else:
            overall = 'LOW'
        
        return {
            'overall_risk': overall,
            'risk_score': risk_score,
            'key_risks': risks,
            'data_source': 'polygon_calculated'
        }
    
    async def _assess_impact(self, shared=None) -> Dict:
        """Assess market impact across asset classes from real data."""
        spy = await self._get_prev_close('SPY', shared)
        tlt = await self._get_prev_close('TLT', shared)
        uup = await self._get_prev_close('UUP', shared)
        gld = await self._get_prev_close('GLD', shared)
        vix = await self._get_prev_close('VIX', shared)
        
        def direction(data, key='change_pct'):
            if not data:
                return 'UNKNOWN'
            val = data.get(key, 0)
            if val > 0.3:
                return 'POSITIVE'
            elif val < -0.3:
                return 'NEGATIVE'
            return 'NEUTRAL'
        
        return {
            'equity_impact': direction(spy),
            'bond_impact': direction(tlt),
            'dollar_impact': direction(uup),
            'gold_impact': direction(gld),
            'volatility_outlook': 'HIGH' if vix and vix.get('close', 0) > 25 else 'LOW' if vix and vix.get('close', 0) < 15 else 'MODERATE',
            'data_source': 'polygon_live'
        }


__all__ = ['MacroAnalystAgent']
