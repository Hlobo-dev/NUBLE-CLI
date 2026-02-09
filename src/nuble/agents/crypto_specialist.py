#!/usr/bin/env python3
"""
NUBLE Crypto Specialist Agent — ELITE TIER

Comprehensive cryptocurrency analysis from multiple real data sources:
- Polygon.io — Price, volume, historical data for crypto pairs
- Polygon.io RSI/SMA/MACD — Server-side technical indicators
- CoinGecko — Coin-specific data (ATH, ATL, market cap rank, supply, DeFi TVL)
- CoinGecko Global — Total market cap, BTC dominance
- Alternative.me — Crypto Fear & Greed Index
- CryptoNews API — Crypto-specific news & sentiment
"""

import os
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging

from .base import SpecializedAgent, AgentTask, AgentResult, AgentType

logger = logging.getLogger(__name__)

# Polygon crypto ticker mapping
CRYPTO_TICKERS = {
    'BTC': 'X:BTCUSD', 'ETH': 'X:ETHUSD', 'SOL': 'X:SOLUSD',
    'XRP': 'X:XRPUSD', 'ADA': 'X:ADAUSD', 'DOGE': 'X:DOGEUSD',
    'AVAX': 'X:AVAXUSD', 'DOT': 'X:DOTUSD', 'MATIC': 'X:MATICUSD',
    'LINK': 'X:LINKUSD', 'UNI': 'X:UNIUSD', 'AAVE': 'X:AAVEUSD',
    'ATOM': 'X:ATOMUSD', 'NEAR': 'X:NEARUSD',
}

# CoinGecko ID mapping (free, no key required)
COINGECKO_IDS = {
    'BTC': 'bitcoin', 'ETH': 'ethereum', 'SOL': 'solana',
    'XRP': 'ripple', 'ADA': 'cardano', 'DOGE': 'dogecoin',
    'AVAX': 'avalanche-2', 'DOT': 'polkadot', 'MATIC': 'matic-network',
    'LINK': 'chainlink', 'UNI': 'uniswap', 'AAVE': 'aave',
    'ATOM': 'cosmos', 'NEAR': 'near',
}


class CryptoSpecialistAgent(SpecializedAgent):
    """
    Crypto Specialist Agent — ELITE Crypto & DeFi Expert
    
    REAL DATA from 6+ sources:
    1. Polygon.io — Price, volume, VWAP for crypto pairs
    2. Polygon.io Indicators — Server-side RSI, SMA, MACD for crypto
    3. CoinGecko /coins/{id} — Coin-specific (ATH, ATL, supply, market cap rank)
    4. CoinGecko /global — Total market cap, BTC dominance, DeFi TVL
    5. Alternative.me — Crypto Fear & Greed Index (7-day trend)
    6. CryptoNews API — Crypto-specific news with sentiment labels
    """
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.polygon_key = os.environ.get('POLYGON_API_KEY', 'JHKwAdyIOeExkYOxh3LwTopmqqVVFeBY')
        self.crypto_news_key = os.environ.get('CRYPTONEWS_API_KEY', os.environ.get('CRYPTO_NEWS_KEY', 'fci3fvhrbxocelhel4ddc7zbmgsxnq1zmwrkxgq2'))
        self.coindesk_key = os.environ.get('COINDESK_API_KEY', '78b5a8d834762d6baf867a6a465d8bbf401fbee0bbe4384940572b9cb1404f6c')
    
    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "name": "Crypto Specialist",
            "description": "Elite crypto & DeFi analysis from 6+ real data sources",
            "capabilities": [
                "on_chain_analytics", "defi_protocols", "whale_tracking",
                "market_overview", "cross_chain", "coin_fundamentals",
                "server_side_indicators", "crypto_news_sentiment"
            ],
            "data_sources": [
                "Polygon Price/Volume", "Polygon RSI/SMA/MACD",
                "CoinGecko Coins", "CoinGecko Global", "Alternative.me FGI",
                "CryptoNews API"
            ]
        }
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute crypto analysis with real data."""
        start = datetime.now()
        
        try:
            symbols = task.context.get('symbols', [])
            crypto_symbols = self._filter_crypto(symbols)
            
            shared = self._get_shared_data(task)
            data = {
                'market_overview': await self._get_market_overview(shared),
                'price_data': await self._get_price_data(crypto_symbols, shared),
                'coin_fundamentals': await self._get_coin_fundamentals(crypto_symbols, shared),
                'technicals': await self._get_crypto_technicals(crypto_symbols, shared),
                'sentiment': await self._get_crypto_sentiment(shared),
                'defi_overview': await self._get_defi_overview(shared)
            }
            
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.CRYPTO_SPECIALIST,
                success=True,
                data=data,
                confidence=0.82,
                execution_time_ms=int((datetime.now() - start).total_seconds() * 1000)
            )
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                agent_type=AgentType.CRYPTO_SPECIALIST,
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
    
    def _filter_crypto(self, symbols: List[str]) -> List[str]:
        """Filter for crypto symbols."""
        crypto_list = set(CRYPTO_TICKERS.keys())
        return [s.upper() for s in symbols if s.upper() in crypto_list] or ['BTC', 'ETH']
    
    async def _get_market_overview(self, shared=None) -> Dict:
        """Get crypto market overview from CoinGecko."""
        if shared:
            data = await shared.get_coingecko_global()
            if data and data.get('data'):
                d = data['data']
                total_mcap = d.get('total_market_cap', {}).get('usd', 0)
                total_vol = d.get('total_volume', {}).get('usd', 0)
                btc_dom = d.get('market_cap_percentage', {}).get('btc', 0)
                eth_dom = d.get('market_cap_percentage', {}).get('eth', 0)
                mcap_change = d.get('market_cap_change_percentage_24h_usd', 0)
                result = {
                    'total_market_cap': f"${total_mcap / 1e12:.2f}T",
                    'total_market_cap_raw': total_mcap,
                    'btc_dominance': round(btc_dom, 1),
                    'eth_dominance': round(eth_dom, 1),
                    'total_volume_24h': f"${total_vol / 1e9:.0f}B",
                    'market_cap_change_24h': round(mcap_change, 2),
                    'active_cryptocurrencies': d.get('active_cryptocurrencies', 0),
                    'data_source': 'coingecko_live'
                }
                fng_data = await shared.get_fear_greed()
                if fng_data and fng_data.get('data'):
                    result['fear_greed_index'] = int(fng_data['data'][0].get('value', 0))
                    result['fear_greed_label'] = fng_data['data'][0].get('value_classification', 'Unknown')
                return result
        try:
            resp = requests.get("https://api.coingecko.com/api/v3/global",
                                timeout=10, headers={'Accept': 'application/json'})
            
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                total_mcap = data.get('total_market_cap', {}).get('usd', 0)
                total_vol = data.get('total_volume', {}).get('usd', 0)
                btc_dom = data.get('market_cap_percentage', {}).get('btc', 0)
                eth_dom = data.get('market_cap_percentage', {}).get('eth', 0)
                mcap_change = data.get('market_cap_change_percentage_24h_usd', 0)
                
                result = {
                    'total_market_cap': f"${total_mcap / 1e12:.2f}T",
                    'total_market_cap_raw': total_mcap,
                    'btc_dominance': round(btc_dom, 1),
                    'eth_dominance': round(eth_dom, 1),
                    'total_volume_24h': f"${total_vol / 1e9:.0f}B",
                    'market_cap_change_24h': round(mcap_change, 2),
                    'active_cryptocurrencies': data.get('active_cryptocurrencies', 0),
                    'data_source': 'coingecko_live'
                }
                
                # Fear & Greed Index
                try:
                    fgi_resp = requests.get("https://api.alternative.me/fng/?limit=1", timeout=5)
                    if fgi_resp.status_code == 200:
                        fgi_data = fgi_resp.json().get('data', [{}])[0]
                        result['fear_greed_index'] = int(fgi_data.get('value', 0))
                        result['fear_greed_label'] = fgi_data.get('value_classification', 'Unknown')
                except Exception:
                    result['fear_greed_index'] = None
                
                return result
        except Exception as e:
            logger.warning(f"CoinGecko global data failed: {e}")
        
        return self._polygon_market_fallback()
    
    def _polygon_market_fallback(self) -> Dict:
        """Fallback market overview from Polygon BTC data."""
        data = self._polygon_get("https://api.polygon.io/v2/aggs/ticker/X:BTCUSD/prev")
        results = data.get('results', [])
        if results:
            btc = results[0]
            change_pct = ((btc['c'] - btc['o']) / btc['o']) * 100 if btc['o'] else 0
            return {
                'btc_price': btc['c'],
                'btc_volume_24h': btc['v'],
                'btc_change_24h': round(change_pct, 2),
                'data_source': 'polygon_fallback'
            }
        return {'error': 'Market overview data unavailable'}
    
    async def _get_price_data(self, symbols: List[str], shared=None) -> Dict:
        """Get real price data from Polygon for each crypto symbol."""
        data = {}
        for symbol in symbols[:5]:
            polygon_ticker = CRYPTO_TICKERS.get(symbol)
            if not polygon_ticker:
                continue
            
            prev = (await shared.get_quote(polygon_ticker)) if shared else self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{polygon_ticker}/prev")
            results = prev.get('results', []) if prev else []
            if results:
                r = results[0]
                change = r['c'] - r['o']
                change_pct = (change / r['o']) * 100 if r['o'] else 0
                
                data[symbol] = {
                    'price': r['c'],
                    'open': r['o'],
                    'high': r['h'],
                    'low': r['l'],
                    'volume': r['v'],
                    'vwap': r.get('vw', 0),
                    'change': round(change, 2),
                    'change_pct': round(change_pct, 2),
                    'data_source': 'polygon_live'
                }
        
        return data
    
    async def _get_coin_fundamentals(self, symbols: List[str], shared=None) -> Dict:
        """Get coin-specific fundamentals from CoinGecko /coins/{id}."""
        fundamentals = {}
        
        for symbol in symbols[:3]:
            coin_id = COINGECKO_IDS.get(symbol)
            if not coin_id:
                continue
            
            try:
                coin = None
                if shared:
                    coin = await shared.get_coingecko_coin(coin_id)
                else:
                    resp = requests.get(
                        f"https://api.coingecko.com/api/v3/coins/{coin_id}",
                        params={'localization': 'false', 'tickers': 'false',
                                'community_data': 'false', 'developer_data': 'false'},
                        timeout=10, headers={'Accept': 'application/json'}
                    )
                    if resp.status_code == 200:
                        coin = resp.json()
                    elif resp.status_code == 429:
                        logger.warning("CoinGecko rate limit hit")
                        break
                
                if coin:
                    market = coin.get('market_data', {})
                    
                    fundamentals[symbol] = {
                        'market_cap_rank': coin.get('market_cap_rank'),
                        'market_cap': market.get('market_cap', {}).get('usd', 0),
                        'fully_diluted_valuation': market.get('fully_diluted_valuation', {}).get('usd', 0),
                        'circulating_supply': market.get('circulating_supply', 0),
                        'total_supply': market.get('total_supply', 0),
                        'max_supply': market.get('max_supply'),
                        'ath': market.get('ath', {}).get('usd', 0),
                        'ath_change_pct': market.get('ath_change_percentage', {}).get('usd', 0),
                        'ath_date': market.get('ath_date', {}).get('usd', ''),
                        'atl': market.get('atl', {}).get('usd', 0),
                        'atl_change_pct': market.get('atl_change_percentage', {}).get('usd', 0),
                        'price_change_24h': market.get('price_change_percentage_24h', 0),
                        'price_change_7d': market.get('price_change_percentage_7d', 0),
                        'price_change_30d': market.get('price_change_percentage_30d', 0),
                        'price_change_1y': market.get('price_change_percentage_1y', 0),
                        'total_volume_24h': market.get('total_volume', {}).get('usd', 0),
                        'data_source': 'coingecko_coins'
                    }
            except Exception as e:
                logger.warning(f"CoinGecko coin data failed for {symbol}: {e}")
        
        return fundamentals if fundamentals else {'note': 'CoinGecko coin data unavailable'}
    
    async def _get_crypto_technicals(self, symbols: List[str], shared=None) -> Dict:
        """Get technical indicators using Polygon server-side API + historical."""
        technicals = {}
        
        for symbol in symbols[:3]:
            polygon_ticker = CRYPTO_TICKERS.get(symbol)
            if not polygon_ticker:
                continue
            
            entry = {'data_source': 'polygon_indicators'}
            
            # Server-side RSI from Polygon
            rsi_data = (await shared.get_rsi(polygon_ticker)) if shared else self._polygon_get(
                f"https://api.polygon.io/v1/indicators/rsi/{polygon_ticker}",
                {'timespan': 'day', 'window': 14, 'series_type': 'close', 'order': 'desc', 'limit': 1}
            )
            rsi_vals = (rsi_data or {}).get('results', {}).get('values', [])
            if rsi_vals:
                rsi = rsi_vals[0].get('value')
                entry['rsi'] = round(float(rsi), 2) if rsi else None
                entry['rsi_signal'] = 'overbought' if rsi and rsi > 70 else 'oversold' if rsi and rsi < 30 else 'neutral'
            
            # Server-side SMA from Polygon
            for window in [20, 50]:
                sma_data = (await shared.get_sma(polygon_ticker, window)) if shared else self._polygon_get(
                    f"https://api.polygon.io/v1/indicators/sma/{polygon_ticker}",
                    {'timespan': 'day', 'window': window, 'series_type': 'close', 'order': 'desc', 'limit': 1}
                )
                sma_vals = (sma_data or {}).get('results', {}).get('values', [])
                if sma_vals:
                    entry[f'sma_{window}'] = round(float(sma_vals[0].get('value', 0)), 2)
            
            # Server-side MACD from Polygon
            macd_data = (await shared.get_macd(polygon_ticker)) if shared else self._polygon_get(
                f"https://api.polygon.io/v1/indicators/macd/{polygon_ticker}",
                {'timespan': 'day', 'short_window': 12, 'long_window': 26,
                 'signal_window': 9, 'series_type': 'close', 'order': 'desc', 'limit': 1}
            )
            macd_vals = (macd_data or {}).get('results', {}).get('values', [])
            if macd_vals:
                entry['macd'] = round(float(macd_vals[0].get('value', 0)), 4)
                entry['macd_signal'] = round(float(macd_vals[0].get('signal', 0)), 4)
                entry['macd_histogram'] = round(float(macd_vals[0].get('histogram', 0)), 4)
            
            # Historical data for volume analysis & volatility
            try:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
                
                hist = (await shared.get_historical(polygon_ticker, 60)) if shared else self._polygon_get(
                    f"https://api.polygon.io/v2/aggs/ticker/{polygon_ticker}/range/1/day/{start_date}/{end_date}",
                    {'sort': 'asc'}
                )
                results = (hist or {}).get('results', [])
                if len(results) >= 14:
                    closes = np.array([r['c'] for r in results])
                    volumes = np.array([r['v'] for r in results])
                    returns = np.diff(closes) / closes[:-1]
                    
                    entry['volatility_annualized'] = round(float(np.std(returns[-20:]) * np.sqrt(365)), 3) if len(returns) >= 20 else None
                    
                    avg_vol_recent = float(np.mean(volumes[-7:])) if len(volumes) >= 7 else 0
                    avg_vol_prior = float(np.mean(volumes[-30:-7])) if len(volumes) >= 30 else avg_vol_recent
                    entry['volume_change_7d'] = round(((avg_vol_recent - avg_vol_prior) / avg_vol_prior) * 100, 1) if avg_vol_prior > 0 else 0
                    
                    # Trend from SMAs
                    sma_20 = entry.get('sma_20')
                    sma_50 = entry.get('sma_50')
                    if sma_20 and sma_50:
                        entry['price_vs_sma20'] = round(((closes[-1] - sma_20) / sma_20) * 100, 2)
                        if closes[-1] > sma_20 > sma_50:
                            entry['trend'] = 'BULLISH'
                        elif closes[-1] < sma_20 < sma_50:
                            entry['trend'] = 'BEARISH'
                        else:
                            entry['trend'] = 'MIXED'
            except Exception:
                pass
            
            technicals[symbol] = entry
        
        return technicals
    
    async def _get_crypto_sentiment(self, shared=None) -> Dict:
        """Get real crypto sentiment from Fear & Greed + CryptoNews PRO (ALL premium endpoints)."""
        sentiment = {}
        
        # Fear & Greed Index
        try:
            fng_data_list = None
            if shared:
                fng_resp = await shared.get_fear_greed()
                if fng_resp:
                    fng_data_list = fng_resp.get('data', [])
            else:
                resp = requests.get("https://api.alternative.me/fng/?limit=7", timeout=5)
                if resp.status_code == 200:
                    fng_data_list = resp.json().get('data', [])
            
            if fng_data_list:
                current = fng_data_list[0]
                sentiment['fear_greed'] = {
                    'value': int(current.get('value', 0)),
                    'label': current.get('value_classification', 'Unknown'),
                    'trend_7d': [int(d.get('value', 0)) for d in fng_data_list],
                    'direction': 'IMPROVING' if int(fng_data_list[0].get('value', 0)) > int(fng_data_list[-1].get('value', 0)) else 'DETERIORATING',
                    'data_source': 'alternative_me_live'
                }
        except Exception:
            pass
        
        # CryptoNews PRO — Ticker news with sentiment + rank score
        if self.crypto_news_key:
            try:
                news_data = None
                if shared:
                    cn_resp = await shared.get_cryptonews('BTC,ETH,SOL')
                    if cn_resp:
                        news_data = cn_resp.get('data', [])
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1", params={
                        'tickers': 'BTC,ETH,SOL',
                        'items': 10, 'sortby': 'rank',
                        'extra-fields': 'id,eventid,rankscore',
                        'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        news_data = resp.json().get('data', [])
                
                if news_data:
                    sentiments = []
                    headlines = []
                    for article in news_data[:10]:
                        sent = article.get('sentiment', 'Neutral')
                        sentiments.append(sent)
                        headlines.append({
                            'title': article.get('title', ''),
                            'sentiment': sent,
                            'source': article.get('source_name', ''),
                            'date': article.get('date', ''),
                            'rank_score': article.get('rankscore', ''),
                        })
                    
                    pos = sentiments.count('Positive') + sentiments.count('Bullish')
                    neg = sentiments.count('Negative') + sentiments.count('Bearish')
                    total = len(sentiments) if sentiments else 1
                    
                    overall = 'BULLISH' if pos > neg else 'BEARISH' if neg > pos else 'NEUTRAL'
                    
                    sentiment['news'] = {
                        'overall': overall,
                        'positive_pct': round(pos / total * 100, 1),
                        'negative_pct': round(neg / total * 100, 1),
                        'article_count': total,
                        'recent_headlines': headlines[:5],
                        'data_source': 'cryptonews_pro'
                    }
            except Exception as e:
                logger.warning(f"CryptoNews sentiment failed: {e}")
        
        # CryptoNews PRO — /stat endpoint for quantitative sentiment over 30 days
        if self.crypto_news_key:
            try:
                stat_data = None
                if shared:
                    stat_resp = await shared.get_cryptonews_stat('BTC')
                    if stat_resp:
                        stat_data = stat_resp.get('data', {})
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1/stat", params={
                        'tickers': 'BTC', 'date': 'last30days', 'page': 1,
                        'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        stat_data = resp.json().get('data', {})
                
                if stat_data:
                    sentiment['btc_sentiment_30d'] = {
                        'data': stat_data,
                        'period': 'last30days',
                        'data_source': 'cryptonews_stat'
                    }
            except Exception:
                pass
        
        # CryptoNews PRO — /top-mention for most discussed coins
        if self.crypto_news_key:
            try:
                top_coins = None
                if shared:
                    tm_resp = await shared.get_cryptonews_top_mentioned()
                    if tm_resp:
                        top_coins = tm_resp.get('data', [])
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1/top-mention", params={
                        'date': 'last7days', 'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        top_coins = resp.json().get('data', [])
                
                if top_coins:
                    sentiment['top_mentioned_coins'] = {
                        'coins': top_coins[:15],
                        'period': 'last7days',
                        'data_source': 'cryptonews_top_mention'
                    }
            except Exception:
                pass
        
        # CryptoNews PRO — /trending-headlines for top trending crypto stories
        if self.crypto_news_key:
            try:
                trending = None
                if shared:
                    tr_resp = await shared.get_cryptonews_trending()
                    if tr_resp:
                        trending = tr_resp.get('data', [])
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1/trending-headlines", params={
                        'page': 1, 'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        trending = resp.json().get('data', [])
                
                if trending:
                    sentiment['trending_headlines'] = {
                        'headlines': [{
                            'title': h.get('title', ''),
                            'source': h.get('source_name', ''),
                            'date': h.get('date', ''),
                        } for h in trending[:8]],
                        'data_source': 'cryptonews_trending'
                    }
            except Exception:
                pass
        
        # CryptoNews PRO — /events for breaking crypto events
        if self.crypto_news_key:
            try:
                events = None
                if shared:
                    ev_resp = await shared.get_cryptonews_events()
                    if ev_resp:
                        events = ev_resp.get('data', [])
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1/events", params={
                        'page': 1, 'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        events = resp.json().get('data', [])
                
                if events:
                    sentiment['breaking_events'] = {
                        'events': [{
                            'title': e.get('title', ''),
                            'event_id': e.get('eventid', ''),
                            'source': e.get('source_name', ''),
                            'date': e.get('date', ''),
                        } for e in events[:8]],
                        'data_source': 'cryptonews_events'
                    }
            except Exception:
                pass
        
        # CryptoNews PRO — /category?section=general for regulation, DeFi, NFT, market news
        if self.crypto_news_key:
            try:
                general = None
                if shared:
                    cat_resp = await shared.get_cryptonews_category('general')
                    if cat_resp:
                        general = cat_resp.get('data', [])
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1/category", params={
                        'section': 'general', 'items': 5, 'sortby': 'rank',
                        'extra-fields': 'id,eventid,rankscore',
                        'page': 1, 'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        general = resp.json().get('data', [])
                
                if general:
                    sentiment['general_crypto_news'] = {
                        'articles': [{
                            'title': a.get('title', ''),
                            'sentiment': a.get('sentiment', ''),
                            'source': a.get('source_name', ''),
                            'rank_score': a.get('rankscore', ''),
                        } for a in general[:5]],
                        'data_source': 'cryptonews_general'
                    }
            except Exception:
                pass
        
        # CryptoNews PRO — /sundown-digest for daily crypto summary
        if self.crypto_news_key:
            try:
                digest = None
                if shared:
                    sd_resp = await shared.get_cryptonews_sundown()
                    if sd_resp:
                        digest = sd_resp.get('data', [])
                else:
                    resp = requests.get("https://cryptonews-api.com/api/v1/sundown-digest", params={
                        'page': 1, 'token': self.crypto_news_key
                    }, timeout=10)
                    if resp.status_code == 200:
                        digest = resp.json().get('data', [])
                
                if digest:
                    sentiment['sundown_digest'] = {
                        'digest': [{
                            'title': d.get('title', ''),
                            'text': d.get('text', ''),
                            'date': d.get('date', ''),
                        } for d in digest[:2]],
                        'data_source': 'cryptonews_sundown'
                    }
            except Exception:
                pass
        
        if not sentiment:
            sentiment['error'] = 'Sentiment data unavailable'
        
        return sentiment
    
    async def _get_defi_overview(self, shared=None) -> Dict:
        """Get DeFi overview from CoinGecko."""
        try:
            defi_data = None
            if shared:
                defi_resp = await shared.get_coingecko_defi()
                if defi_resp:
                    defi_data = defi_resp.get('data', {})
            else:
                resp = requests.get("https://api.coingecko.com/api/v3/global/decentralized_finance_defi",
                                    timeout=10, headers={'Accept': 'application/json'})
                if resp.status_code == 200:
                    defi_data = resp.json().get('data', {})
            
            if defi_data:
                return {
                    'defi_market_cap': defi_data.get('defi_market_cap', ''),
                    'eth_market_cap': defi_data.get('eth_market_cap', ''),
                    'defi_to_eth_ratio': defi_data.get('defi_to_eth_ratio', ''),
                    'defi_dominance': defi_data.get('defi_dominance', ''),
                    'top_coin': defi_data.get('top_coin_name', ''),
                    'top_coin_defi_dominance': defi_data.get('top_coin_defi_dominance', ''),
                    'trading_volume_24h': defi_data.get('trading_volume_24h', ''),
                    'data_source': 'coingecko_defi_live'
                }
        except Exception as e:
            logger.warning(f"DeFi overview fetch failed: {e}")
        
        return {'note': 'DeFi data unavailable'}


__all__ = ['CryptoSpecialistAgent']
