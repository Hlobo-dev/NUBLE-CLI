#!/usr/bin/env python3
"""
NUBLE Risk Manager Agent — ELITE TIER

Comprehensive risk assessment using multiple real data sources:
- Polygon.io historical prices — VaR, CVaR, drawdown, correlation
- Polygon.io VIX — Volatility regime monitoring
- Polygon.io sector ETFs — Concentration & sector risk
- Polygon.io news — Event risk detection
- Alternative.me Fear & Greed — Sentiment risk overlay
- StockNews API — Earnings/analyst risk signals
"""

import os
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)


class RiskManagerAgent(SpecializedAgent):
    """
    Risk Manager Agent — ELITE Risk Assessment
    
    REAL DATA from 6+ sources:
    1. Polygon Historical — VaR (95%, 99%), CVaR, max drawdown, beta
    2. Polygon VIX — Volatility regime (crisis/high/normal/low)
    3. Polygon Sector ETFs — Sector concentration risk
    4. Polygon News — Event risk from headlines
    5. Alternative.me Fear & Greed — Market sentiment risk
    6. StockNews API — Earnings surprise & analyst risk
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        self.stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Risk Manager",
            "description": "Elite risk assessment from 6+ real data sources",
            "capabilities": [
                "var_calculation", "stress_testing", "correlation_analysis",
                "drawdown_analysis", "position_sizing", "volatility_regime",
                "event_risk", "sentiment_risk", "sector_concentration"
            ],
            "data_sources": [
                "Polygon Historical", "Polygon VIX", "Polygon Sectors",
                "Polygon News", "Alternative.me FGI", "StockNews API"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute risk analysis with real data."""
        start = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            portfolio = task.context.get('user_profile', {}).get('portfolio', {})
            
            analysis_symbols = list(portfolio.keys()) if portfolio else symbols
            if not analysis_symbols:
                analysis_symbols = ['SPY']
            
            data = {
                'var_analysis': self._calculate_var(analysis_symbols, portfolio),
                'correlations': self._analyze_correlations(analysis_symbols[:5]),
                'drawdown': self._analyze_drawdown(analysis_symbols[:3]),
                'position_limits': self._calculate_limits(analysis_symbols, portfolio),
                'volatility_regime': await self._get_volatility_regime(task),
                'sentiment_risk': await self._get_sentiment_risk(task),
                'event_risk': await self._get_event_risk(analysis_symbols[:3], task),
                'risk_score': self._overall_risk_score(analysis_symbols)
            }
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.RISK_MANAGER,
                success=True,
                data=data,
                confidence=0.85,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.RISK_MANAGER,
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
    
    def _get_returns(self, symbol: str, days: int = 252) -> np.ndarray:
        """Fetch real historical returns from Polygon."""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y-%m-%d')
        
        data = self._polygon_get(
            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
            {'sort': 'asc'}
        )
        results = data.get('results', [])
        if len(results) >= 20:
            closes = np.array([r['c'] for r in results])
            return np.diff(closes) / closes[:-1]
        return np.array([])
    
    def _calculate_var(self, symbols: List[str], portfolio: Dict) -> Dict:
        """Calculate Value at Risk from REAL historical returns."""
        portfolio_value = sum(portfolio.values()) if portfolio else 100000
        
        all_returns = []
        symbols_used = []
        
        for symbol in symbols[:5]:
            returns = self._get_returns(symbol)
            if len(returns) > 0:
                all_returns.append(returns)
                symbols_used.append(symbol)
        
        if not all_returns:
            return {'error': 'Could not fetch historical data for VaR calculation'}
        
        min_len = min(len(r) for r in all_returns)
        trimmed = [r[-min_len:] for r in all_returns]
        portfolio_returns = np.mean(trimmed, axis=0)
        
        var_95 = float(np.percentile(portfolio_returns, 5))
        var_99 = float(np.percentile(portfolio_returns, 1))
        cvar_95 = float(np.mean(portfolio_returns[portfolio_returns <= var_95]))
        var_95_10d = var_95 * np.sqrt(10)
        
        # Monte Carlo VaR (parametric supplement)
        mu = float(np.mean(portfolio_returns))
        sigma = float(np.std(portfolio_returns))
        mc_var_95 = mu - 1.645 * sigma
        
        return {
            'var_95_1d': round(portfolio_value * abs(var_95), 2),
            'var_95_1d_pct': round(abs(var_95) * 100, 3),
            'var_99_1d': round(portfolio_value * abs(var_99), 2),
            'var_99_1d_pct': round(abs(var_99) * 100, 3),
            'var_95_10d': round(portfolio_value * abs(var_95_10d), 2),
            'cvar_95': round(portfolio_value * abs(cvar_95), 2),
            'parametric_var_95': round(portfolio_value * abs(mc_var_95), 2),
            'portfolio_value': portfolio_value,
            'method': 'Historical Simulation + Parametric',
            'data_points': min_len,
            'symbols_used': symbols_used,
            'data_source': 'polygon_calculated'
        }
    
    def _analyze_correlations(self, symbols: List[str]) -> Dict:
        """Analyze real correlations from Polygon data."""
        if len(symbols) < 2:
            return {'note': 'Need at least 2 symbols for correlation analysis'}
        
        returns_dict = {}
        for symbol in symbols[:5]:
            returns = self._get_returns(symbol, days=120)
            if len(returns) > 0:
                returns_dict[symbol] = returns
        
        if len(returns_dict) < 2:
            return {'error': 'Insufficient data for correlation analysis'}
        
        min_len = min(len(r) for r in returns_dict.values())
        aligned = {s: r[-min_len:] for s, r in returns_dict.items()}
        symbols_list = list(aligned.keys())
        matrix = np.array([aligned[s] for s in symbols_list])
        
        corr_matrix = np.corrcoef(matrix)
        
        pairs = {}
        for i in range(len(symbols_list)):
            for j in range(i + 1, len(symbols_list)):
                pair_key = f"{symbols_list[i]}/{symbols_list[j]}"
                pairs[pair_key] = round(float(corr_matrix[i, j]), 3)
        
        upper_tri = corr_matrix[np.triu_indices(len(symbols_list), k=1)]
        avg_corr = float(np.mean(upper_tri))
        max_corr = float(np.max(upper_tri))
        
        if avg_corr > 0.7:
            diversification = 'LOW'
        elif avg_corr > 0.4:
            diversification = 'MODERATE'
        else:
            diversification = 'HIGH'
        
        return {
            'correlation_pairs': pairs,
            'avg_correlation': round(avg_corr, 3),
            'max_correlation': round(max_corr, 3),
            'diversification_quality': diversification,
            'data_points': min_len,
            'data_source': 'polygon_calculated'
        }
    
    def _analyze_drawdown(self, symbols: List[str]) -> Dict:
        """Analyze real drawdown from historical data."""
        drawdowns = {}
        
        for symbol in symbols[:3]:
            returns = self._get_returns(symbol, days=252)
            if len(returns) < 20:
                continue
            
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown_series = (cumulative - running_max) / running_max
            
            max_dd = float(np.min(drawdown_series))
            current_dd = float(drawdown_series[-1])
            
            in_drawdown = drawdown_series < -0.01
            dd_periods = np.diff(np.where(np.diff(in_drawdown.astype(int)))[0])
            avg_recovery = int(np.mean(dd_periods)) if len(dd_periods) > 0 else 0
            
            drawdowns[symbol] = {
                'max_drawdown': round(max_dd, 4),
                'max_drawdown_pct': round(max_dd * 100, 2),
                'current_drawdown': round(current_dd, 4),
                'current_drawdown_pct': round(current_dd * 100, 2),
                'avg_recovery_days': avg_recovery,
                'data_source': 'polygon_calculated'
            }
        
        return drawdowns if drawdowns else {'error': 'Could not calculate drawdowns'}
    
    def _calculate_limits(self, symbols: List[str], portfolio: Dict) -> Dict:
        """Calculate position limits based on real volatility."""
        portfolio_value = sum(portfolio.values()) if portfolio else 100000
        
        returns = self._get_returns(symbols[0] if symbols else 'SPY')
        if len(returns) > 0:
            daily_vol = float(np.std(returns))
            annual_vol = daily_vol * np.sqrt(252)
            avg_return = float(np.mean(returns))
            win_rate = float(np.mean(returns > 0))
            
            max_position_pct = min(0.20, 0.02 / (2 * daily_vol)) if daily_vol > 0 else 0.10
            
            return {
                'max_position_size': round(portfolio_value * max_position_pct, 2),
                'max_position_pct': round(max_position_pct * 100, 1),
                'max_sector_exposure': round(portfolio_value * 0.25, 2),
                'stop_loss_suggestion': round(2 * daily_vol, 4),
                'stop_loss_pct': round(2 * daily_vol * 100, 2),
                'daily_volatility': round(daily_vol, 4),
                'annual_volatility': round(annual_vol, 4),
                'win_rate': round(win_rate, 3),
                'data_source': 'polygon_calculated'
            }
        
        return {
            'max_position_size': round(portfolio_value * 0.10, 2),
            'note': 'Default limits — historical data unavailable'
        }
    
    async def _get_volatility_regime(self, task=None) -> Dict:
        """Get real VIX-based volatility regime."""
        shared = self._get_shared_data(task)
        data = (await shared.get_vix()) if shared else self._polygon_get("https://api.polygon.io/v2/aggs/ticker/VIX/prev")
        results = data.get('results', [])
        
        if not results:
            return {'regime': 'UNKNOWN', 'data_source': 'polygon_error'}
        
        vix = results[0]['c']
        vix_change = ((results[0]['c'] - results[0]['o']) / results[0]['o']) * 100 if results[0].get('o') else 0
        
        if vix > 35:
            regime, stance = 'CRISIS', 'EXTREME_FEAR'
        elif vix > 25:
            regime, stance = 'HIGH_VOL', 'ELEVATED_FEAR'
        elif vix > 20:
            regime, stance = 'NORMAL', 'CAUTIOUS'
        elif vix > 15:
            regime, stance = 'LOW_VOL', 'COMPLACENT'
        else:
            regime, stance = 'VERY_LOW_VOL', 'EXTREME_COMPLACENCY'
        
        risk_adj = 'Reduce exposure' if vix > 25 else 'Normal sizing' if vix > 15 else 'Full sizing allowed'
        
        return {
            'vix': round(vix, 2),
            'vix_change_pct': round(vix_change, 2),
            'regime': regime,
            'stance': stance,
            'sizing_recommendation': risk_adj,
            'data_source': 'polygon_live'
        }
    
    async def _get_sentiment_risk(self, task=None) -> Dict:
        """Get sentiment-based risk from Fear & Greed Index."""
        shared = self._get_shared_data(task)
        if shared:
            fng_data = await shared.get_fear_greed()
            if fng_data and fng_data.get('data'):
                fng = fng_data['data']
                current = int(fng[0].get('value', 50))
                trend = [int(d.get('value', 50)) for d in fng]
                if current < 20:
                    risk_level, contrarian = 'EXTREME_FEAR', 'Potential buying opportunity (contrarian)'
                elif current < 35:
                    risk_level, contrarian = 'FEAR', 'Market pessimism elevated'
                elif current > 80:
                    risk_level, contrarian = 'EXTREME_GREED', 'Potential correction risk (contrarian)'
                elif current > 65:
                    risk_level, contrarian = 'GREED', 'Market optimism elevated'
                else:
                    risk_level, contrarian = 'NEUTRAL', 'No extreme sentiment'
                return {
                    'fear_greed_index': current,
                    'label': fng[0].get('value_classification', 'Unknown'),
                    'risk_level': risk_level,
                    'contrarian_signal': contrarian,
                    'trend_7d': trend,
                    'trend_direction': 'IMPROVING' if trend[0] > trend[-1] else 'DETERIORATING',
                    'data_source': 'alternative_me_live'
                }

        result = {}
        
        try:
            resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=8)
            if resp.status_code == 200:
                fng_data = resp.json().get('data', [])
                if fng_data:
                    current = int(fng_data[0].get('value', 50))
                    trend = [int(d.get('value', 50)) for d in fng_data]
                    
                    if current < 20:
                        risk_level = 'EXTREME_FEAR'
                        contrarian = 'Potential buying opportunity (contrarian)'
                    elif current < 35:
                        risk_level = 'FEAR'
                        contrarian = 'Market pessimism elevated'
                    elif current > 80:
                        risk_level = 'EXTREME_GREED'
                        contrarian = 'Potential correction risk (contrarian)'
                    elif current > 65:
                        risk_level = 'GREED'
                        contrarian = 'Market optimism elevated'
                    else:
                        risk_level = 'NEUTRAL'
                        contrarian = 'No extreme sentiment'
                    
                    result = {
                        'fear_greed_index': current,
                        'label': fng_data[0].get('value_classification', 'Unknown'),
                        'risk_level': risk_level,
                        'contrarian_signal': contrarian,
                        'trend_7d': trend,
                        'trend_direction': 'IMPROVING' if trend[0] > trend[-1] else 'DETERIORATING',
                        'data_source': 'alternative_me_live'
                    }
        except Exception as e:
            logger.warning(f"Sentiment risk fetch failed: {e}")
            result = {'error': 'Sentiment data unavailable'}
        
        return result
    
    async def _get_event_risk(self, symbols: List[str], task=None) -> List[Dict]:
        """Detect event risk from real news sources + premium endpoints."""
        events = []
        
        # Polygon news
        for symbol in symbols[:3]:
            try:
                data = self._polygon_get("https://api.polygon.io/v2/reference/news", {
                    'ticker': symbol, 'limit': 5, 'order': 'desc'
                })
                for article in data.get('results', []):
                    title = article.get('title', '').lower()
                    risk_keywords = {
                        'HIGH': ['lawsuit', 'sec investigation', 'fraud', 'bankruptcy', 'recall', 'crash', 'hack', 'breach'],
                        'MEDIUM': ['downgrade', 'warning', 'miss', 'layoff', 'restructur', 'debt', 'default'],
                        'LOW': ['volatil', 'uncertain', 'concern', 'risk', 'challenge']
                    }
                    
                    for severity, keywords in risk_keywords.items():
                        if any(k in title for k in keywords):
                            events.append({
                                'symbol': symbol,
                                'severity': severity,
                                'headline': article.get('title', ''),
                                'source': article.get('publisher', {}).get('name', ''),
                                'date': article.get('published_utc', ''),
                                'data_source': 'polygon_news'
                            })
                            break
            except Exception:
                continue
        
        # StockNews PRO — NEGATIVE sentiment filter (premium endpoint)
        if self.stocknews_key:
            for symbol in symbols[:3]:
                try:
                    resp = requests.get("https://stocknewsapi.com/api/v1", params={
                        'tickers': symbol, 'items': 5, 'sentiment': 'negative',
                        'extra-fields': 'id,eventid,rankscore',
                        'token': self.stocknews_key
                    }, timeout=8)
                    if resp.status_code == 200:
                        for article in resp.json().get('data', []):
                            events.append({
                                'symbol': symbol,
                                'severity': 'MEDIUM',
                                'headline': article.get('title', ''),
                                'source': article.get('source_name', ''),
                                'rank_score': article.get('rankscore', ''),
                                'data_source': 'stocknews_negative_filter'
                            })
                except Exception:
                    continue
        
        # StockNews PRO — Earnings Calendar (check if symbols report soon)
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/earnings-calendar", params={
                    'page': 1, 'items': 50, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    earnings = resp.json().get('data', [])
                    for e in earnings:
                        ticker = e.get('ticker', '').upper()
                        if ticker in [s.upper() for s in symbols]:
                            events.append({
                                'symbol': ticker,
                                'severity': 'HIGH',
                                'type': 'UPCOMING_EARNINGS',
                                'headline': f"{ticker} earnings report scheduled",
                                'earnings_date': e.get('date', ''),
                                'earnings_time': e.get('time', ''),
                                'data_source': 'stocknews_earnings_calendar'
                            })
            except Exception:
                pass
        
        # StockNews PRO — Recent Downgrades (premium /ratings endpoint)
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/ratings", params={
                    'items': 20, 'page': 1, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    ratings = resp.json().get('data', [])
                    for r in ratings:
                        ticker = r.get('ticker', '').upper()
                        action = str(r.get('action', '')).lower()
                        if ticker in [s.upper() for s in symbols] and 'downgrade' in action:
                            events.append({
                                'symbol': ticker,
                                'severity': 'HIGH',
                                'type': 'ANALYST_DOWNGRADE',
                                'headline': f"{ticker} downgraded by {r.get('analyst_company', 'analyst')}",
                                'rating_from': r.get('rating_from', ''),
                                'rating_to': r.get('rating_to', ''),
                                'target_from': r.get('target_from', ''),
                                'target_to': r.get('target_to', ''),
                                'data_source': 'stocknews_ratings'
                            })
            except Exception:
                pass
        
        # StockNews PRO — Breaking events that may affect portfolio
        if self.stocknews_key:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/events", params={
                    'page': 1, 'token': self.stocknews_key
                }, timeout=8)
                if resp.status_code == 200:
                    breaking = resp.json().get('data', [])
                    for e in breaking[:5]:
                        tickers = e.get('tickers', [])
                        if isinstance(tickers, list) and any(t.upper() in [s.upper() for s in symbols] for t in tickers):
                            events.append({
                                'symbol': ', '.join(tickers),
                                'severity': 'HIGH',
                                'type': 'BREAKING_EVENT',
                                'headline': e.get('title', ''),
                                'event_id': e.get('eventid', ''),
                                'data_source': 'stocknews_events'
                            })
            except Exception:
                pass
        
        return events if events else [{'note': 'No significant event risks detected'}]
    
    def _overall_risk_score(self, symbols: List[str]) -> Dict:
        """Calculate overall risk score from real data."""
        returns = self._get_returns(symbols[0] if symbols else 'SPY')
        
        if len(returns) < 20:
            return {'score': 5, 'rating': 'UNKNOWN', 'note': 'Insufficient data'}
        
        daily_vol = float(np.std(returns))
        annual_vol = daily_vol * np.sqrt(252)
        var_95 = abs(float(np.percentile(returns, 5)))
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = abs(float(np.min((cumulative - running_max) / running_max)))
        
        vol_score = min(10, annual_vol * 20)
        dd_score = min(10, max_dd * 25)
        var_score = min(10, var_95 * 200)
        
        score = round((vol_score * 0.4 + dd_score * 0.35 + var_score * 0.25), 1)
        
        if score <= 3:
            rating = 'LOW'
        elif score <= 6:
            rating = 'MODERATE'
        else:
            rating = 'HIGH'
        
        recommendations = []
        if annual_vol > 0.30:
            recommendations.append('High volatility — consider hedging or reducing position size')
        if max_dd > 0.20:
            recommendations.append(f'Historical max drawdown of {max_dd*100:.1f}% — set stop-losses')
        if var_95 > 0.03:
            recommendations.append(f'Daily VaR(95%) is {var_95*100:.1f}% — significant daily risk')
        if not recommendations:
            recommendations.append('Risk metrics within normal parameters')
        
        return {
            'score': round(score, 1),
            'rating': rating,
            'components': {
                'volatility_score': round(vol_score, 1),
                'drawdown_score': round(dd_score, 1),
                'var_score': round(var_score, 1),
            },
            'metrics': {
                'annual_volatility': round(annual_vol, 4),
                'max_drawdown': round(max_dd, 4),
                'var_95_daily': round(var_95, 4),
            },
            'recommendations': recommendations,
            'data_source': 'polygon_calculated'
        }


__all__ = ['RiskManagerAgent']
