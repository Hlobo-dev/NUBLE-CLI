#!/usr/bin/env python3
"""
NUBLE Fundamental Analyst Agent — ELITE TIER

The most comprehensive fundamental analysis specialist using:
- Polygon.io /vX/reference/financials — Real SEC quarterly financials
- Polygon.io /v3/reference/tickers — Company details, market cap, SIC
- Polygon.io /stocks/v1/dividends — Dividend history & yield
- Polygon.io /v2/reference/news — Company-specific news with sentiment
- Polygon.io /v2/aggs/ticker/prev — Current price for live valuation
- Polygon.io /v1/indicators/sma — Server-side moving averages for price context
- StockNews API — Additional analyst & news sentiment
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)


class FundamentalAnalystAgent(SpecializedAgent):
    """
    Fundamental Analyst Agent — Valuation & Financials Expert
    
    REAL DATA from 7 endpoints:
    1. Polygon Financials API (income statement, balance sheet, cash flow)
    2. Polygon Ticker Details (company info, market cap, sector)
    3. Polygon Dividends (yield, payout history, frequency)
    4. Polygon News (company-specific headlines with sentiment)
    5. Polygon Previous Close (live price for valuation)
    6. Polygon SMA (price trend context for fundamental overlay)
    7. StockNews API (analyst ratings, earnings sentiment)
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        self.stocknews_key = os.environ.get('STOCKNEWS_API_KEY', 'zzad9pmlwttixx0fnsenstctzgdk7ysx0ctkgrk0')
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Fundamental Analyst",
            "description": "Valuation and financial analysis from real SEC filings, dividends, and news",
            "capabilities": [
                "financial_statements", "valuation_metrics", "profitability_analysis",
                "growth_analysis", "balance_sheet_health", "dividend_analysis",
                "earnings_quality", "news_sentiment", "peer_comparison"
            ],
            "data_sources": [
                "Polygon Financials", "Polygon Dividends", "Polygon News",
                "Polygon Ticker Details", "StockNews API"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute fundamental analysis with real data."""
        start = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            
            analyses = {}
            for symbol in symbols[:3]:
                analyses[symbol] = self._analyze_fundamentals(symbol)
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.FUNDAMENTAL_ANALYST,
                success=True,
                data=analyses,
                confidence=0.85,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.FUNDAMENTAL_ANALYST,
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
            logger.warning(f"Polygon call failed {url}: {e}")
        return {}
    
    def _analyze_fundamentals(self, symbol: str) -> Dict:
        """Full fundamental analysis from multiple real data sources."""
        symbol = symbol.upper().replace('$', '')
        result = {'symbol': symbol, 'data_source': 'polygon_multi_endpoint', 'analysis_date': datetime.now().isoformat()}
        
        # === 1. Company Details (market cap, sector, description) ===
        details_data = self._polygon_get(f"https://api.polygon.io/v3/reference/tickers/{symbol}")
        details = details_data.get('results', {})
        if details:
            result['company'] = {
                'name': details.get('name', symbol),
                'market_cap': details.get('market_cap'),
                'share_class_shares_outstanding': details.get('share_class_shares_outstanding'),
                'description': (details.get('description', '')[:300] + '...') if details.get('description') else None,
                'sic_description': details.get('sic_description'),
                'sic_code': details.get('sic_code'),
                'homepage_url': details.get('homepage_url'),
                'primary_exchange': details.get('primary_exchange'),
                'locale': details.get('locale'),
                'type': details.get('type'),
            }
        
        # === 2. Current Price (for live valuation ratios) ===
        prev_data = self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev")
        prev_results = prev_data.get('results', [])
        current_price = prev_results[0]['c'] if prev_results else None
        if prev_results:
            r = prev_results[0]
            result['price'] = {
                'current': r['c'],
                'open': r['o'],
                'high': r['h'],
                'low': r['l'],
                'volume': r.get('v', 0),
                'vwap': r.get('vw'),
                'change_pct': round(((r['c'] - r['o']) / r['o']) * 100, 2) if r['o'] else 0,
            }
        
        # === 3. Financial Statements (last 4 quarters + 4 annual) ===
        fin_data = self._polygon_get("https://api.polygon.io/vX/reference/financials", {
            'ticker': symbol, 'limit': 5, 'timeframe': 'quarterly', 'order': 'desc'
        })
        financials = fin_data.get('results', [])
        
        # Also get annual for TTM calculations
        annual_data = self._polygon_get("https://api.polygon.io/vX/reference/financials", {
            'ticker': symbol, 'limit': 2, 'timeframe': 'annual', 'order': 'desc'
        })
        annual_financials = annual_data.get('results', [])
        
        if financials:
            latest = financials[0]
            income = latest.get('financials', {}).get('income_statement', {})
            balance = latest.get('financials', {}).get('balance_sheet', {})
            cashflow = latest.get('financials', {}).get('cash_flow_statement', {})
            
            # Extract real values
            revenue = income.get('revenues', {}).get('value', 0)
            net_income = income.get('net_income_loss', {}).get('value', 0)
            gross_profit = income.get('gross_profit', {}).get('value', 0)
            operating_income = income.get('operating_income_loss', {}).get('value', 0)
            eps_basic = income.get('basic_earnings_per_share', {}).get('value')
            eps_diluted = income.get('diluted_earnings_per_share', {}).get('value')
            
            total_assets = balance.get('assets', {}).get('value', 0)
            total_liabilities = balance.get('liabilities', {}).get('value', 0)
            equity = balance.get('equity', {}).get('value', 0)
            current_assets = balance.get('current_assets', {}).get('value', 0)
            current_liabilities = balance.get('current_liabilities', {}).get('value', 0)
            long_term_debt = balance.get('long_term_debt', {}).get('value', 0)
            cash = balance.get('cash', {}).get('value', 0) or balance.get('cash_and_cash_equivalents', {}).get('value', 0)
            
            operating_cf = cashflow.get('net_cash_flow_from_operating_activities', {}).get('value', 0)
            investing_cf = cashflow.get('net_cash_flow_from_investing_activities', {}).get('value', 0)
            financing_cf = cashflow.get('net_cash_flow_from_financing_activities', {}).get('value', 0)
            capex = cashflow.get('capital_expenditure', {}).get('value', 0)
            
            # Free Cash Flow
            free_cash_flow = operating_cf + capex if capex < 0 else operating_cf - abs(capex)
            
            result['profitability'] = {
                'gross_margin': round(gross_profit / revenue, 4) if revenue else None,
                'operating_margin': round(operating_income / revenue, 4) if revenue else None,
                'net_margin': round(net_income / revenue, 4) if revenue else None,
                'roe': round(net_income / equity, 4) if equity else None,
                'roa': round(net_income / total_assets, 4) if total_assets else None,
                'roic': round(operating_income / (equity + long_term_debt), 4) if (equity + long_term_debt) else None,
            }
            
            result['income_statement'] = {
                'revenue': revenue,
                'gross_profit': gross_profit,
                'operating_income': operating_income,
                'net_income': net_income,
                'eps_basic': eps_basic,
                'eps_diluted': eps_diluted,
                'fiscal_period': latest.get('fiscal_period'),
                'fiscal_year': latest.get('fiscal_year'),
                'filing_date': latest.get('filing_date'),
            }
            
            result['balance_sheet'] = {
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'total_equity': equity,
                'cash_and_equivalents': cash,
                'long_term_debt': long_term_debt,
                'current_ratio': round(current_assets / current_liabilities, 2) if current_liabilities else None,
                'quick_ratio': round((current_assets - balance.get('inventory', {}).get('value', 0)) / current_liabilities, 2) if current_liabilities else None,
                'debt_to_equity': round(total_liabilities / equity, 2) if equity else None,
                'net_debt': long_term_debt - cash if long_term_debt and cash else None,
            }
            
            result['cash_flow'] = {
                'operating_cash_flow': operating_cf,
                'investing_cash_flow': investing_cf,
                'financing_cash_flow': financing_cf,
                'free_cash_flow': free_cash_flow,
                'fcf_margin': round(free_cash_flow / revenue, 4) if revenue else None,
                'capex': capex,
            }
            
            # === TTM calculations from last 4 quarters ===
            if len(financials) >= 4:
                ttm_revenue = sum(
                    f.get('financials', {}).get('income_statement', {}).get('revenues', {}).get('value', 0)
                    for f in financials[:4]
                )
                ttm_net_income = sum(
                    f.get('financials', {}).get('income_statement', {}).get('net_income_loss', {}).get('value', 0)
                    for f in financials[:4]
                )
                ttm_fcf = sum(
                    f.get('financials', {}).get('cash_flow_statement', {}).get('net_cash_flow_from_operating_activities', {}).get('value', 0)
                    + f.get('financials', {}).get('cash_flow_statement', {}).get('capital_expenditure', {}).get('value', 0)
                    for f in financials[:4]
                )
                result['ttm'] = {
                    'revenue': ttm_revenue,
                    'net_income': ttm_net_income,
                    'free_cash_flow': ttm_fcf,
                }
            
            # === Growth: QoQ and YoY ===
            if len(financials) >= 2:
                prior = financials[1]
                prior_income = prior.get('financials', {}).get('income_statement', {})
                prior_revenue = prior_income.get('revenues', {}).get('value', 0)
                prior_net = prior_income.get('net_income_loss', {}).get('value', 0)
                
                result['growth'] = {
                    'revenue_growth_qoq': round((revenue - prior_revenue) / abs(prior_revenue), 4) if prior_revenue else None,
                    'earnings_growth_qoq': round((net_income - prior_net) / abs(prior_net), 4) if prior_net else None,
                }
            
            if len(financials) >= 5:
                yoy = financials[4]
                yoy_income = yoy.get('financials', {}).get('income_statement', {})
                yoy_revenue = yoy_income.get('revenues', {}).get('value', 0)
                yoy_net = yoy_income.get('net_income_loss', {}).get('value', 0)
                
                result.setdefault('growth', {})
                result['growth']['revenue_growth_yoy'] = round((revenue - yoy_revenue) / abs(yoy_revenue), 4) if yoy_revenue else None
                result['growth']['earnings_growth_yoy'] = round((net_income - yoy_net) / abs(yoy_net), 4) if yoy_net else None
            
            # === Valuation Ratios (live) ===
            market_cap = result.get('company', {}).get('market_cap')
            ttm_data = result.get('ttm', {})
            if market_cap:
                ttm_rev = ttm_data.get('revenue', revenue * 4)
                ttm_ni = ttm_data.get('net_income', net_income * 4)
                ttm_fcf_val = ttm_data.get('free_cash_flow', free_cash_flow * 4)
                
                result['valuation'] = {
                    'pe_ratio': round(market_cap / ttm_ni, 1) if ttm_ni and ttm_ni > 0 else None,
                    'price_to_sales': round(market_cap / ttm_rev, 1) if ttm_rev else None,
                    'price_to_book': round(market_cap / equity, 1) if equity and equity > 0 else None,
                    'price_to_fcf': round(market_cap / ttm_fcf_val, 1) if ttm_fcf_val and ttm_fcf_val > 0 else None,
                    'ev_to_ebitda': None,  # Would need enterprise value
                }
                
                # Enterprise Value
                if market_cap and long_term_debt is not None and cash is not None:
                    ev = market_cap + long_term_debt - cash
                    ebitda_est = operating_income * 4  # Rough quarterly annualization
                    result['valuation']['enterprise_value'] = ev
                    if ebitda_est > 0:
                        result['valuation']['ev_to_ebitda'] = round(ev / ebitda_est, 1)
                
                # Earnings Yield & FCF Yield
                if current_price and eps_diluted:
                    result['valuation']['earnings_yield'] = round((eps_diluted * 4) / current_price * 100, 2)
                if current_price and ttm_fcf_val and result.get('company', {}).get('share_class_shares_outstanding'):
                    shares = result['company']['share_class_shares_outstanding']
                    fcf_per_share = ttm_fcf_val / shares if shares else 0
                    result['valuation']['fcf_yield'] = round(fcf_per_share / current_price * 100, 2) if current_price else None
            
            # === Earnings Quality Score ===
            if operating_cf and net_income:
                accrual_ratio = (net_income - operating_cf) / total_assets if total_assets else 0
                quality = 'HIGH' if operating_cf > net_income else 'MODERATE' if operating_cf > net_income * 0.7 else 'LOW'
                result['earnings_quality'] = {
                    'cash_conversion': round(operating_cf / net_income, 2) if net_income else None,
                    'accrual_ratio': round(accrual_ratio, 4),
                    'quality_rating': quality,
                    'fcf_to_net_income': round(free_cash_flow / net_income, 2) if net_income else None,
                }
        else:
            result['error'] = f'No financial data available for {symbol}'
        
        # === 4. Dividend Analysis ===
        result['dividends'] = self._get_dividend_data(symbol, current_price)
        
        # === 5. Company News from Polygon ===
        result['recent_news'] = self._get_polygon_news(symbol)
        
        # === 6. StockNews API sentiment ===
        result['news_sentiment'] = self._get_stocknews_sentiment(symbol)
        
        # === 7. Price trend context (SMA from Polygon) ===
        result['price_trend'] = self._get_price_trend(symbol)
        
        return result
    
    def _get_dividend_data(self, symbol: str, current_price: float = None) -> Dict:
        """Get real dividend data from Polygon."""
        try:
            data = self._polygon_get("https://api.polygon.io/stocks/v1/dividends", {
                'ticker': symbol, 'limit': 8, 'sort': 'ex_dividend_date.desc'
            })
            divs = data.get('results', [])
            
            if not divs:
                return {'pays_dividend': False, 'data_source': 'polygon_dividends'}
            
            latest = divs[0]
            frequency = latest.get('frequency', 4)
            annual_dividend = latest.get('cash_amount', 0) * (frequency if frequency else 4)
            
            dividend_yield = None
            if current_price and annual_dividend:
                dividend_yield = round(annual_dividend / current_price * 100, 2)
            
            # Dividend growth (compare to year-ago)
            growth = None
            if len(divs) >= 5:
                old_amount = divs[4].get('cash_amount', 0)
                new_amount = latest.get('cash_amount', 0)
                if old_amount > 0:
                    growth = round((new_amount - old_amount) / old_amount * 100, 2)
            
            return {
                'pays_dividend': True,
                'latest_amount': latest.get('cash_amount'),
                'frequency': frequency,
                'annual_dividend_est': round(annual_dividend, 4) if annual_dividend else None,
                'dividend_yield': dividend_yield,
                'ex_dividend_date': latest.get('ex_dividend_date'),
                'pay_date': latest.get('pay_date'),
                'dividend_growth_yoy': growth,
                'distribution_type': latest.get('distribution_type'),
                'data_source': 'polygon_dividends'
            }
        except Exception as e:
            logger.warning(f"Dividend fetch failed for {symbol}: {e}")
            return {'pays_dividend': None, 'error': str(e)}
    
    def _get_polygon_news(self, symbol: str) -> List[Dict]:
        """Get company-specific news from Polygon."""
        try:
            data = self._polygon_get(f"https://api.polygon.io/v2/reference/news", {
                'ticker': symbol, 'limit': 5, 'order': 'desc'
            })
            articles = data.get('results', [])
            return [{
                'title': a.get('title', ''),
                'publisher': a.get('publisher', {}).get('name', ''),
                'published': a.get('published_utc', ''),
                'tickers': a.get('tickers', []),
                'description': (a.get('description', '')[:200] + '...') if a.get('description') else None,
                'data_source': 'polygon_news'
            } for a in articles[:5]]
        except Exception as e:
            logger.warning(f"Polygon news failed for {symbol}: {e}")
            return []
    
    def _get_stocknews_sentiment(self, symbol: str) -> Dict:
        """Get sentiment from StockNews PRO — FULL premium endpoints."""
        if not self.stocknews_key:
            return {'error': 'StockNews API key not configured'}
        
        result = {}
        
        # 1. StockNews PRO — /stat endpoint for quantitative sentiment (7 & 30 days)
        for period, label in [('last7days', '7d'), ('last30days', '30d')]:
            try:
                resp = requests.get("https://stocknewsapi.com/api/v1/stat", params={
                    'tickers': symbol, 'date': period, 'page': 1,
                    'token': self.stocknews_key
                }, timeout=10)
                if resp.status_code == 200:
                    stat_data = resp.json().get('data', {})
                    if stat_data:
                        result[f'sentiment_stats_{label}'] = {
                            'data': stat_data,
                            'period': period,
                            'data_source': 'stocknews_stat'
                        }
            except Exception:
                pass
        
        # 2. StockNews PRO — Ticker news with rank score + sentiment
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1", params={
                'tickers': symbol, 'items': 10, 'sortby': 'rank',
                'extra-fields': 'id,eventid,rankscore',
                'token': self.stocknews_key
            }, timeout=10)
            
            if resp.status_code == 200:
                articles = resp.json().get('data', [])
                if articles:
                    pos = neg = neu = 0
                    for a in articles:
                        sent = str(a.get('sentiment', '')).lower()
                        text = str(a.get('title', '')).lower()
                        if sent in ('positive', 'bullish') or any(w in text for w in ['beat', 'upgrade', 'surge', 'record', 'strong']):
                            pos += 1
                        elif sent in ('negative', 'bearish') or any(w in text for w in ['miss', 'downgrade', 'decline', 'warning', 'weak']):
                            neg += 1
                        else:
                            neu += 1
                    
                    total = pos + neg + neu
                    score = (pos - neg) / total if total else 0
                    
                    result['ticker_sentiment'] = {
                        'score': round(score, 2),
                        'label': 'BULLISH' if score > 0.2 else 'BEARISH' if score < -0.2 else 'NEUTRAL',
                        'positive': pos, 'negative': neg, 'neutral': neu,
                        'article_count': total,
                        'data_source': 'stocknews_pro'
                    }
        except Exception as e:
            logger.warning(f"StockNews sentiment failed for {symbol}: {e}")
        
        # 3. StockNews PRO — /ratings for analyst upgrades/downgrades on this symbol
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/ratings", params={
                'items': 20, 'page': 1, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                ratings = resp.json().get('data', [])
                symbol_ratings = []
                for r in ratings:
                    if r.get('ticker', '').upper() == symbol.upper():
                        symbol_ratings.append({
                            'action': r.get('action', ''),
                            'rating_from': r.get('rating_from', ''),
                            'rating_to': r.get('rating_to', ''),
                            'target_from': r.get('target_from', ''),
                            'target_to': r.get('target_to', ''),
                            'analyst': r.get('analyst', ''),
                            'analyst_company': r.get('analyst_company', ''),
                            'date': r.get('date', ''),
                        })
                if symbol_ratings:
                    result['analyst_ratings'] = {
                        'ratings': symbol_ratings,
                        'data_source': 'stocknews_ratings'
                    }
        except Exception:
            pass
        
        # 4. StockNews PRO — /earnings-calendar to check upcoming earnings
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1/earnings-calendar", params={
                'page': 1, 'items': 50, 'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                earnings = resp.json().get('data', [])
                for e in earnings:
                    if e.get('ticker', '').upper() == symbol.upper():
                        result['upcoming_earnings'] = {
                            'date': e.get('date', ''),
                            'time': e.get('time', ''),
                            'data_source': 'stocknews_earnings_calendar'
                        }
                        break
        except Exception:
            pass
        
        # 5. StockNews PRO — Earnings topic news (press releases about earnings)
        try:
            resp = requests.get("https://stocknewsapi.com/api/v1", params={
                'tickers': symbol, 'items': 5, 'topic': 'earnings',
                'extra-fields': 'id,eventid,rankscore',
                'token': self.stocknews_key
            }, timeout=10)
            if resp.status_code == 200:
                earnings_news = resp.json().get('data', [])
                if earnings_news:
                    result['earnings_news'] = {
                        'articles': [{
                            'title': a.get('title', ''),
                            'sentiment': a.get('sentiment', ''),
                            'source': a.get('source_name', ''),
                            'date': a.get('date', ''),
                            'rank_score': a.get('rankscore', ''),
                        } for a in earnings_news[:5]],
                        'data_source': 'stocknews_earnings_topic'
                    }
        except Exception:
            pass
        
        if not result:
            result = {'error': 'StockNews data unavailable', 'data_source': 'stocknews_error'}
        
        return result
    
    def _get_price_trend(self, symbol: str) -> Dict:
        """Get SMA data from Polygon for fundamental trend overlay."""
        trend = {}
        
        for window in [50, 200]:
            try:
                data = self._polygon_get(f"https://api.polygon.io/v1/indicators/sma/{symbol}", {
                    'timespan': 'day', 'window': window, 'series_type': 'close',
                    'order': 'desc', 'limit': 1
                })
                values = data.get('results', {}).get('values', [])
                if values:
                    trend[f'sma_{window}'] = round(values[0].get('value', 0), 2)
            except Exception:
                pass
        
        # Determine trend
        prev_data = self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev")
        prev_results = prev_data.get('results', [])
        if prev_results and trend:
            price = prev_results[0]['c']
            sma50 = trend.get('sma_50')
            sma200 = trend.get('sma_200')
            
            if sma50 and sma200:
                if price > sma50 > sma200:
                    trend['signal'] = 'STRONG_UPTREND'
                elif price > sma50:
                    trend['signal'] = 'UPTREND'
                elif price < sma50 < sma200:
                    trend['signal'] = 'STRONG_DOWNTREND'
                elif price < sma50:
                    trend['signal'] = 'DOWNTREND'
                else:
                    trend['signal'] = 'SIDEWAYS'
                
                # Golden/Death cross
                if sma50 > sma200:
                    trend['cross'] = 'GOLDEN_CROSS'
                elif sma50 < sma200:
                    trend['cross'] = 'DEATH_CROSS'
        
        trend['data_source'] = 'polygon_sma_api'
        return trend


__all__ = ['FundamentalAnalystAgent']
