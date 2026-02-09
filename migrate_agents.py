"""
Migrate all agents to use SharedDataLayer.

For each agent's data-fetching methods:
1. Check if shared_data is available via self._get_shared_data(task)
2. If yes, read from shared_data (instant, already cached)
3. If no, fall back to existing requests.get() code (backwards compatible)
"""
import re

AGENTS_DIR = "src/nuble/agents"


def patch_market_analyst():
    """MarketAnalyst: _get_quote, _get_historical, _get_stocknews_sentiment, _get_analyst_ratings, _get_earnings_info"""
    path = f"{AGENTS_DIR}/market_analyst.py"
    with open(path, 'r') as f:
        content = f.read()

    # Need to pass task through execute -> _analyze_symbol -> sub-methods
    # 1. Modify execute() to pass task to _analyze_symbol
    content = content.replace(
        "                symbol_data = await self._analyze_symbol(symbol, query)",
        "                symbol_data = await self._analyze_symbol(symbol, query, task)"
    )

    # 2. Modify _analyze_symbol signature to accept task
    content = content.replace(
        "    async def _analyze_symbol(self, symbol: str, query: str) -> Dict[str, Any]:",
        "    async def _analyze_symbol(self, symbol: str, query: str, task=None) -> Dict[str, Any]:"
    )

    # 3. Pass task to sub-methods in _analyze_symbol
    content = content.replace(
        "        # Get current quote\n        quote = await self._get_quote(symbol)",
        "        # Get current quote\n        quote = await self._get_quote(symbol, task)"
    )
    content = content.replace(
        "        # Get historical data\n        historical = await self._get_historical(symbol, days=90)",
        "        # Get historical data\n        historical = await self._get_historical(symbol, days=90, task=task)"
    )
    content = content.replace(
        "        # StockNews PRO — News sentiment for this symbol\n        news_sentiment = self._get_stocknews_sentiment(symbol)",
        "        # StockNews PRO — News sentiment for this symbol\n        news_sentiment = await self._get_stocknews_sentiment(symbol, task)"
    )
    content = content.replace(
        "        # StockNews PRO — Analyst ratings for this symbol\n        analyst_ratings = self._get_analyst_ratings(symbol)",
        "        # StockNews PRO — Analyst ratings for this symbol\n        analyst_ratings = await self._get_analyst_ratings(symbol, task)"
    )
    content = content.replace(
        "        # StockNews PRO — Check earnings calendar\n        earnings_info = self._get_earnings_info(symbol)",
        "        # StockNews PRO — Check earnings calendar\n        earnings_info = await self._get_earnings_info(symbol, task)"
    )

    # 4. Modify _get_quote to prefer shared data
    old_get_quote = '''    async def _get_quote(self, symbol: str) -> Optional[Dict]:
        """Get real-time quote from Polygon."""
        if not HAS_REQUESTS:
            return self._real_quote(symbol)
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {'apiKey': self.polygon_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    r = data['results'][0]
                    return {
                        'symbol': symbol,
                        'open': r.get('o'),
                        'high': r.get('h'),
                        'low': r.get('l'),
                        'close': r.get('c'),
                        'volume': r.get('v'),
                        'vwap': r.get('vw'),
                        'change_pct': ((r.get('c', 0) - r.get('o', 1)) / r.get('o', 1) * 100) if r.get('o') else 0
                    }
        except Exception as e:
            logger.warning(f"Quote fetch failed for {symbol}: {e}")
        
        return self._real_quote(symbol)'''

    new_get_quote = '''    async def _get_quote(self, symbol: str, task=None) -> Optional[Dict]:
        """Get real-time quote from Polygon. Prefers SharedDataLayer if available."""
        # Try shared data layer first (zero HTTP calls)
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_quote(symbol)
            if data and data.get('results'):
                r = data['results'][0]
                return {
                    'symbol': symbol,
                    'open': r.get('o'),
                    'high': r.get('h'),
                    'low': r.get('l'),
                    'close': r.get('c'),
                    'volume': r.get('v'),
                    'vwap': r.get('vw'),
                    'change_pct': ((r.get('c', 0) - r.get('o', 1)) / r.get('o', 1) * 100) if r.get('o') else 0
                }

        # Fallback to direct HTTP
        if not HAS_REQUESTS:
            return self._real_quote(symbol)
        
        try:
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/prev"
            params = {'apiKey': self.polygon_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('results'):
                    r = data['results'][0]
                    return {
                        'symbol': symbol,
                        'open': r.get('o'),
                        'high': r.get('h'),
                        'low': r.get('l'),
                        'close': r.get('c'),
                        'volume': r.get('v'),
                        'vwap': r.get('vw'),
                        'change_pct': ((r.get('c', 0) - r.get('o', 1)) / r.get('o', 1) * 100) if r.get('o') else 0
                    }
        except Exception as e:
            logger.warning(f"Quote fetch failed for {symbol}: {e}")
        
        return self._real_quote(symbol)'''

    content = content.replace(old_get_quote, new_get_quote)

    # 5. Modify _get_historical to prefer shared data
    old_hist = '''    async def _get_historical(self, symbol: str, days: int = 90) -> List[Dict]:
        """Get historical OHLCV data."""
        if not HAS_REQUESTS:
            return self._real_historical(symbol, days)
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime(\'%Y-%m-%d\')}/{end_date.strftime(\'%Y-%m-%d\')}"
            params = {\'apiKey\': self.polygon_key, \'sort\': \'asc\'}
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get(\'results\', [])
                
                return [
                    {
                        \'date\': datetime.fromtimestamp(r[\'t\'] / 1000).strftime(\'%Y-%m-%d\'),
                        \'open\': r.get(\'o\'),
                        \'high\': r.get(\'h\'),
                        \'low\': r.get(\'l\'),
                        \'close\': r.get(\'c\'),
                        \'volume\': r.get(\'v\'),
                        \'vwap\': r.get(\'vw\')
                    }
                    for r in results
                ]
        except Exception as e:
            logger.warning(f"Historical fetch failed for {symbol}: {e}")
        
        return self._real_historical(symbol, days)'''

    # The above has escaping issues in a heredoc. Let me use a simpler approach.
    # Find the _get_historical method and prepend the shared data check.

    # Actually let me just do a targeted insert
    old_hist_start = '    async def _get_historical(self, symbol: str, days: int = 90) -> List[Dict]:\n        """Get historical OHLCV data."""\n        if not HAS_REQUESTS:'
    new_hist_start = '''    async def _get_historical(self, symbol: str, days: int = 90, task=None) -> List[Dict]:
        """Get historical OHLCV data. Prefers SharedDataLayer if available."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_historical(symbol, days)
            if data and data.get('results'):
                return [
                    {
                        'date': datetime.fromtimestamp(r['t'] / 1000).strftime('%Y-%m-%d'),
                        'open': r.get('o'),
                        'high': r.get('h'),
                        'low': r.get('l'),
                        'close': r.get('c'),
                        'volume': r.get('v'),
                        'vwap': r.get('vw')
                    }
                    for r in data['results']
                ]

        # Fallback to direct HTTP
        if not HAS_REQUESTS:'''

    content = content.replace(old_hist_start, new_hist_start)

    # 6. Modify _get_stocknews_sentiment to be async + prefer shared data
    old_stocknews = '    def _get_stocknews_sentiment(self, symbol: str) -> Dict:\n        """StockNews PRO — Get quantitative sentiment + ranked news."""\n        if not HAS_REQUESTS:'
    new_stocknews = '''    async def _get_stocknews_sentiment(self, symbol: str, task=None) -> Dict:
        """StockNews PRO — Get quantitative sentiment + ranked news. Prefers SharedDataLayer."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            result = {}
            stat_data = await shared.get_stocknews_stat(symbol)
            if stat_data and stat_data.get('data'):
                result['sentiment_stats_7d'] = stat_data['data']

            news_data = await shared.get_stocknews(symbol)
            if news_data and news_data.get('data'):
                articles = news_data['data']
                pos = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('positive', 'bullish'))
                neg = sum(1 for a in articles if str(a.get('sentiment', '')).lower() in ('negative', 'bearish'))
                total = len(articles)
                score = (pos - neg) / total if total > 0 else 0
                result['score'] = round(score, 2)
                result['label'] = 'BULLISH' if score > 0.2 else 'BEARISH' if score < -0.2 else 'NEUTRAL'
                result['positive'] = pos
                result['negative'] = neg
                result['article_count'] = total
                result['top_headlines'] = [a.get('title', '') for a in articles[:3]]

            if result:
                result['data_source'] = 'stocknews_pro'
                return result

        # Fallback to direct HTTP
        if not HAS_REQUESTS:'''

    content = content.replace(old_stocknews, new_stocknews)

    # 7. Modify _get_analyst_ratings to be async + prefer shared data
    old_ratings = '    def _get_analyst_ratings(self, symbol: str) -> Dict:\n        """StockNews PRO — /ratings for analyst upgrades/downgrades + price targets."""\n        if not HAS_REQUESTS:'
    new_ratings = '''    async def _get_analyst_ratings(self, symbol: str, task=None) -> Dict:
        """StockNews PRO — /ratings for analyst upgrades/downgrades + price targets. Prefers SharedDataLayer."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_stocknews_ratings()
            if data and data.get('data'):
                ratings = data['data']
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
                    return {
                        'ratings': symbol_ratings,
                        'latest_action': symbol_ratings[0].get('action', ''),
                        'data_source': 'stocknews_ratings'
                    }
                return {}

        # Fallback to direct HTTP
        if not HAS_REQUESTS:'''

    content = content.replace(old_ratings, new_ratings)

    # 8. Modify _get_earnings_info to be async + prefer shared data
    old_earnings = '    def _get_earnings_info(self, symbol: str) -> Dict:\n        """StockNews PRO — /earnings-calendar for upcoming earnings date."""\n        if not HAS_REQUESTS:'
    new_earnings = '''    async def _get_earnings_info(self, symbol: str, task=None) -> Dict:
        """StockNews PRO — /earnings-calendar for upcoming earnings date. Prefers SharedDataLayer."""
        # Try shared data layer first
        shared = self._get_shared_data(task)
        if shared:
            data = await shared.get_stocknews_earnings()
            if data and data.get('data'):
                for e in data['data']:
                    if e.get('ticker', '').upper() == symbol.upper():
                        return {
                            'date': e.get('date', ''),
                            'time': e.get('time', ''),
                            'data_source': 'stocknews_earnings_calendar'
                        }
                return {}

        # Fallback to direct HTTP
        if not HAS_REQUESTS:'''

    content = content.replace(old_earnings, new_earnings)

    with open(path, 'w') as f:
        f.write(content)
    print(f"  MarketAnalyst: patched ({content.count(chr(10))} lines)")


def patch_news_analyst():
    """NewsAnalyst: All methods. This agent makes the MOST calls."""
    path = f"{AGENTS_DIR}/news_analyst.py"
    with open(path, 'r') as f:
        content = f.read()

    # Add shared_data extraction at the top of execute()
    old_exec = '''        try:
            symbols = task.context.get('symbols', [])
            
            all_news = {}
            for symbol in symbols[:3]:
                is_crypto = symbol.upper() in CRYPTO_SYMBOLS
                
                # === TICKER-SPECIFIC NEWS from multiple sources ===
                stocknews = self._get_stocknews(symbol) if not is_crypto else []
                polygon_news = self._get_polygon_news(symbol) if not is_crypto else []
                crypto_news = self._get_crypto_news(symbol) if is_crypto else []'''

    new_exec = '''        try:
            symbols = task.context.get('symbols', [])
            shared = self._get_shared_data(task)
            
            all_news = {}
            for symbol in symbols[:3]:
                is_crypto = symbol.upper() in CRYPTO_SYMBOLS
                
                # === TICKER-SPECIFIC NEWS from multiple sources ===
                stocknews = await self._get_stocknews(symbol, shared) if not is_crypto else []
                polygon_news = await self._get_polygon_news(symbol, shared) if not is_crypto else []
                crypto_news = await self._get_crypto_news(symbol, shared) if is_crypto else []'''

    content = content.replace(old_exec, new_exec)

    # Update quant sentiment call
    content = content.replace(
        "                quant_sentiment = self._get_sentiment_stats(symbol, is_crypto)",
        "                quant_sentiment = await self._get_sentiment_stats(symbol, is_crypto, shared)"
    )

    # Update market-wide calls to pass shared
    content = content.replace(
        "            market_sentiment = self._get_market_sentiment()",
        "            market_sentiment = await self._get_market_sentiment(shared)"
    )
    content = content.replace(
        "            trending = self._get_trending_headlines()",
        "            trending = await self._get_trending_headlines(shared)"
    )
    content = content.replace(
        "            top_mentioned = self._get_top_mentioned()",
        "            top_mentioned = await self._get_top_mentioned(shared)"
    )
    content = content.replace(
        "            breaking_events = self._get_breaking_events()",
        "            breaking_events = await self._get_breaking_events(shared)"
    )
    content = content.replace(
        "            general_market_news = self._get_general_market_news()",
        "            general_market_news = await self._get_general_market_news(shared)"
    )
    content = content.replace(
        "            earnings_calendar = self._get_earnings_calendar()",
        "            earnings_calendar = await self._get_earnings_calendar(shared)"
    )
    content = content.replace(
        "            analyst_ratings = self._get_analyst_ratings(symbols)",
        "            analyst_ratings = await self._get_analyst_ratings(symbols, shared)"
    )
    content = content.replace(
        "            sundown_digest = self._get_sundown_digest()",
        "            sundown_digest = await self._get_sundown_digest(shared)"
    )
    content = content.replace(
        "            crypto_trending = self._get_crypto_trending() if has_crypto else {}",
        "            crypto_trending = await self._get_crypto_trending(shared) if has_crypto else {}"
    )
    content = content.replace(
        "            crypto_top_mentioned = self._get_crypto_top_mentioned() if has_crypto else {}",
        "            crypto_top_mentioned = await self._get_crypto_top_mentioned(shared) if has_crypto else {}"
    )
    content = content.replace(
        "            crypto_general = self._get_crypto_general_news() if has_crypto else {}",
        "            crypto_general = await self._get_crypto_general_news(shared) if has_crypto else {}"
    )
    content = content.replace(
        "            crypto_events = self._get_crypto_events() if has_crypto else {}",
        "            crypto_events = await self._get_crypto_events(shared) if has_crypto else {}"
    )

    # Now update each method signature + add shared data check

    # _get_stocknews
    content = content.replace(
        '    def _get_stocknews(self, symbol: str) -> List[Dict]:\n        """StockNews PRO — Ticker news with sentiment + rank score."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_stocknews(self, symbol: str, shared=None) -> List[Dict]:\n        """StockNews PRO — Ticker news with sentiment + rank score."""\n        if shared:\n            data = await shared.get_stocknews(symbol)\n            if data and data.get(\'data\'):\n                articles = data[\'data\']\n                for a in articles:\n                    a[\'_source\'] = \'stocknews_pro\'\n                return articles\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_sentiment_stats
    content = content.replace(
        '    def _get_sentiment_stats(self, symbol: str, is_crypto: bool = False) -> Dict:\n        """StockNews/CryptoNews PRO — /stat endpoint for quantitative sentiment over time."""\n        if not HAS_REQUESTS:',
        '    async def _get_sentiment_stats(self, symbol: str, is_crypto: bool = False, shared=None) -> Dict:\n        """StockNews/CryptoNews PRO — /stat endpoint for quantitative sentiment over time."""\n        if shared:\n            if is_crypto:\n                data = await shared.get_cryptonews_stat(symbol)\n            else:\n                data = await shared.get_stocknews_stat(symbol, "last30days")\n            if data and data.get(\'data\'):\n                return {\'sentiment_data\': data[\'data\'], \'period\': \'last30days\', \'data_source\': \'cryptonews_stat\' if is_crypto else \'stocknews_stat\'}\n        if not HAS_REQUESTS:'
    )

    # _get_trending_headlines
    content = content.replace(
        '    def _get_trending_headlines(self) -> Dict:\n        """StockNews PRO — /trending-headlines for top trending stories."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_trending_headlines(self, shared=None) -> Dict:\n        """StockNews PRO — /trending-headlines for top trending stories."""\n        if shared:\n            data = await shared.get_stocknews_trending()\n            if data and data.get(\'data\'):\n                headlines = data[\'data\']\n                return {\'headlines\': [{\'title\': h.get(\'title\', \'\'), \'description\': h.get(\'text\', \'\')[:200] if h.get(\'text\') else \'\', \'source\': h.get(\'source_name\', \'\'), \'date\': h.get(\'date\', \'\'), \'tickers\': h.get(\'tickers\', [])} for h in headlines[:10]], \'count\': len(headlines), \'data_source\': \'stocknews_trending\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_top_mentioned
    content = content.replace(
        '    def _get_top_mentioned(self) -> Dict:\n        """StockNews PRO — /top-mention for most discussed stocks with sentiment."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_top_mentioned(self, shared=None) -> Dict:\n        """StockNews PRO — /top-mention for most discussed stocks with sentiment."""\n        if shared:\n            data = await shared.get_stocknews_top_mentioned()\n            if data and data.get(\'data\'):\n                return {\'top_tickers\': data[\'data\'][:15], \'period\': \'last7days\', \'data_source\': \'stocknews_top_mention\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_breaking_events
    content = content.replace(
        '    def _get_breaking_events(self) -> Dict:\n        """StockNews PRO — /events for breaking news events."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_breaking_events(self, shared=None) -> Dict:\n        """StockNews PRO — /events for breaking news events."""\n        if shared:\n            data = await shared.get_stocknews_events()\n            if data and data.get(\'data\'):\n                events = data[\'data\']\n                return {\'events\': [{\'title\': e.get(\'title\', \'\'), \'event_id\': e.get(\'eventid\', \'\'), \'date\': e.get(\'date\', \'\'), \'source\': e.get(\'source_name\', \'\'), \'tickers\': e.get(\'tickers\', [])} for e in events[:10]], \'count\': len(events), \'data_source\': \'stocknews_events\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_general_market_news
    content = content.replace(
        '    def _get_general_market_news(self) -> Dict:\n        """StockNews PRO — /category?section=general for Fed, CPI, macro news."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_general_market_news(self, shared=None) -> Dict:\n        """StockNews PRO — /category?section=general for Fed, CPI, macro news."""\n        if shared:\n            data = await shared.get_stocknews_category("general")\n            if data and data.get(\'data\'):\n                articles = data[\'data\']\n                return {\'articles\': [{\'title\': a.get(\'title\', \'\'), \'text\': (a.get(\'text\', \'\')[:200] + \'...\') if a.get(\'text\') else \'\', \'sentiment\': a.get(\'sentiment\', \'\'), \'source\': a.get(\'source_name\', \'\'), \'date\': a.get(\'date\', \'\'), \'rank_score\': a.get(\'rankscore\', \'\')} for a in articles[:10]], \'count\': len(articles), \'data_source\': \'stocknews_general_market\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_earnings_calendar
    content = content.replace(
        '    def _get_earnings_calendar(self) -> Dict:\n        """StockNews PRO — /earnings-calendar for upcoming earnings dates."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_earnings_calendar(self, shared=None) -> Dict:\n        """StockNews PRO — /earnings-calendar for upcoming earnings dates."""\n        if shared:\n            data = await shared.get_stocknews_earnings()\n            if data and data.get(\'data\'):\n                return {\'upcoming_earnings\': data[\'data\'][:20], \'count\': len(data[\'data\']), \'data_source\': \'stocknews_earnings_calendar\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_analyst_ratings
    content = content.replace(
        '    def _get_analyst_ratings(self, symbols: List[str]) -> Dict:\n        """StockNews PRO — /ratings for analyst upgrades/downgrades + price targets."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_analyst_ratings(self, symbols: List[str], shared=None) -> Dict:\n        """StockNews PRO — /ratings for analyst upgrades/downgrades + price targets."""\n        if shared:\n            data = await shared.get_stocknews_ratings()\n            if data and data.get(\'data\'):\n                ratings = data[\'data\']\n                relevant = []\n                other = []\n                for r in ratings:\n                    entry = {\'ticker\': r.get(\'ticker\', \'\'), \'action\': r.get(\'action\', \'\'), \'rating_from\': r.get(\'rating_from\', \'\'), \'rating_to\': r.get(\'rating_to\', \'\'), \'target_from\': r.get(\'target_from\', \'\'), \'target_to\': r.get(\'target_to\', \'\'), \'analyst\': r.get(\'analyst\', \'\'), \'analyst_company\': r.get(\'analyst_company\', \'\'), \'date\': r.get(\'date\', \'\')}\n                    if any(s.upper() == r.get(\'ticker\', \'\').upper() for s in symbols):\n                        relevant.append(entry)\n                    else:\n                        other.append(entry)\n                return {\'relevant_ratings\': relevant, \'recent_ratings\': other[:10], \'total_count\': len(ratings), \'data_source\': \'stocknews_ratings\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_sundown_digest
    content = content.replace(
        '    def _get_sundown_digest(self) -> Dict:\n        """StockNews PRO — /sundown-digest for daily market summary."""\n        if not HAS_REQUESTS or not self.stocknews_key:',
        '    async def _get_sundown_digest(self, shared=None) -> Dict:\n        """StockNews PRO — /sundown-digest for daily market summary."""\n        if shared:\n            data = await shared.get_stocknews_sundown()\n            if data and data.get(\'data\'):\n                return {\'digest\': [{\'title\': d.get(\'title\', \'\'), \'text\': d.get(\'text\', \'\'), \'date\': d.get(\'date\', \'\')} for d in data[\'data\'][:3]], \'data_source\': \'stocknews_sundown\'}\n        if not HAS_REQUESTS or not self.stocknews_key:'
    )

    # _get_crypto_news
    content = content.replace(
        '    def _get_crypto_news(self, symbol: str) -> List[Dict]:\n        """CryptoNews PRO — Ticker news with sentiment + rank score."""\n        if not HAS_REQUESTS or not self.cryptonews_key:',
        '    async def _get_crypto_news(self, symbol: str, shared=None) -> List[Dict]:\n        """CryptoNews PRO — Ticker news with sentiment + rank score."""\n        if shared:\n            data = await shared.get_cryptonews(symbol)\n            if data and data.get(\'data\'):\n                articles = data[\'data\']\n                for a in articles:\n                    a[\'_source\'] = \'cryptonews_pro\'\n                return articles\n        if not HAS_REQUESTS or not self.cryptonews_key:'
    )

    # _get_crypto_trending
    content = content.replace(
        '    def _get_crypto_trending(self) -> Dict:\n        """CryptoNews PRO — /trending-headlines for top crypto stories."""\n        if not HAS_REQUESTS or not self.cryptonews_key:',
        '    async def _get_crypto_trending(self, shared=None) -> Dict:\n        """CryptoNews PRO — /trending-headlines for top crypto stories."""\n        if shared:\n            data = await shared.get_cryptonews_trending()\n            if data and data.get(\'data\'):\n                headlines = data[\'data\']\n                return {\'headlines\': [{\'title\': h.get(\'title\', \'\'), \'description\': (h.get(\'text\', \'\')[:200]) if h.get(\'text\') else \'\', \'source\': h.get(\'source_name\', \'\'), \'date\': h.get(\'date\', \'\')} for h in headlines[:10]], \'count\': len(headlines), \'data_source\': \'cryptonews_trending\'}\n        if not HAS_REQUESTS or not self.cryptonews_key:'
    )

    # _get_crypto_top_mentioned
    content = content.replace(
        '    def _get_crypto_top_mentioned(self) -> Dict:\n        """CryptoNews PRO — /top-mention for most discussed coins."""\n        if not HAS_REQUESTS or not self.cryptonews_key:',
        '    async def _get_crypto_top_mentioned(self, shared=None) -> Dict:\n        """CryptoNews PRO — /top-mention for most discussed coins."""\n        if shared:\n            data = await shared.get_cryptonews_top_mentioned()\n            if data and data.get(\'data\'):\n                return {\'top_coins\': data[\'data\'][:15], \'period\': \'last7days\', \'data_source\': \'cryptonews_top_mention\'}\n        if not HAS_REQUESTS or not self.cryptonews_key:'
    )

    # _get_crypto_general_news
    content = content.replace(
        '    def _get_crypto_general_news(self) -> Dict:\n        """CryptoNews PRO — /category?section=general for regulation, market news."""\n        if not HAS_REQUESTS or not self.cryptonews_key:',
        '    async def _get_crypto_general_news(self, shared=None) -> Dict:\n        """CryptoNews PRO — /category?section=general for regulation, market news."""\n        if shared:\n            data = await shared.get_cryptonews_category("general")\n            if data and data.get(\'data\'):\n                articles = data[\'data\']\n                return {\'articles\': [{\'title\': a.get(\'title\', \'\'), \'text\': (a.get(\'text\', \'\')[:200] + \'...\') if a.get(\'text\') else \'\', \'sentiment\': a.get(\'sentiment\', \'\'), \'source\': a.get(\'source_name\', \'\'), \'date\': a.get(\'date\', \'\'), \'rank_score\': a.get(\'rankscore\', \'\')} for a in articles[:10]], \'data_source\': \'cryptonews_general\'}\n        if not HAS_REQUESTS or not self.cryptonews_key:'
    )

    # _get_crypto_events
    content = content.replace(
        '    def _get_crypto_events(self) -> Dict:\n        """CryptoNews PRO — /events for breaking crypto events."""\n        if not HAS_REQUESTS or not self.cryptonews_key:',
        '    async def _get_crypto_events(self, shared=None) -> Dict:\n        """CryptoNews PRO — /events for breaking crypto events."""\n        if shared:\n            data = await shared.get_cryptonews_events()\n            if data and data.get(\'data\'):\n                events = data[\'data\']\n                return {\'events\': [{\'title\': e.get(\'title\', \'\'), \'event_id\': e.get(\'eventid\', \'\'), \'date\': e.get(\'date\', \'\'), \'source\': e.get(\'source_name\', \'\')} for e in events[:10]], \'data_source\': \'cryptonews_events\'}\n        if not HAS_REQUESTS or not self.cryptonews_key:'
    )

    # _get_polygon_news
    content = content.replace(
        '    def _get_polygon_news(self, symbol: str) -> List[Dict]:\n        """Fetch news from Polygon.io /v2/reference/news."""\n        if not HAS_REQUESTS:',
        '    async def _get_polygon_news(self, symbol: str, shared=None) -> List[Dict]:\n        """Fetch news from Polygon.io /v2/reference/news."""\n        if shared:\n            data = await shared.get_polygon_news(symbol)\n            if data and data.get(\'results\'):\n                articles = []\n                for a in data[\'results\']:\n                    articles.append({\'title\': a.get(\'title\', \'\'), \'text\': a.get(\'description\', \'\'), \'date\': a.get(\'published_utc\', \'\'), \'source_name\': a.get(\'publisher\', {}).get(\'name\', \'\'), \'sentiment\': \'\', \'tickers\': [t for t in (a.get(\'tickers\', []) or [])], \'_source\': \'polygon\'})\n                return articles\n        if not HAS_REQUESTS:'
    )

    # _get_market_sentiment
    content = content.replace(
        '    def _get_market_sentiment(self) -> Dict:\n        """Get real market sentiment from VIX, Fear & Greed, and SPY."""',
        '    async def _get_market_sentiment(self, shared=None) -> Dict:\n        """Get real market sentiment from VIX, Fear & Greed, and SPY."""'
    )

    # Add shared data fast path at the start of _get_market_sentiment body
    old_ms_body = "        result = {'data_source': 'live_apis'}\n        \n        # Fear & Greed from Alternative.me"
    new_ms_body = """        result = {'data_source': 'live_apis'}

        # Try shared data layer first
        if shared:
            fng_data = await shared.get_fear_greed()
            if fng_data and fng_data.get('data'):
                fng = fng_data['data']
                current = int(fng[0].get('value', 50))
                result['fear_greed_index'] = current
                result['fear_greed_label'] = fng[0].get('value_classification', 'Neutral')
                result['fear_greed_trend'] = [int(d.get('value', 50)) for d in fng]

            vix_data = await shared.get_vix()
            if vix_data and vix_data.get('results'):
                vr = vix_data['results'][0]
                result['vix_level'] = round(vr['c'], 1)
                vix_change = ((vr['c'] - vr['o']) / vr['o']) * 100 if vr.get('o') else 0
                result['vix_change_pct'] = round(vix_change, 2)

            spy_data = await shared.get_quote('SPY')
            if spy_data and spy_data.get('results'):
                sr = spy_data['results'][0]
                market_change = ((sr['c'] - sr['o']) / sr['o']) * 100 if sr['o'] else 0
                result['spy_change_pct'] = round(market_change, 2)

            fgi = result.get('fear_greed_index')
            vix = result.get('vix_level')
            if fgi is not None and vix is not None:
                fgi_score = (fgi - 50) / 50
                vix_score = max(-1, min(1, (20 - vix) / 15))
                score = (fgi_score * 0.6 + vix_score * 0.4)
                result['score'] = round(score, 2)
            elif fgi is not None:
                result['score'] = round((fgi - 50) / 50, 2)
            else:
                result['score'] = 0
            return result
        
        # Fear & Greed from Alternative.me"""

    content = content.replace(old_ms_body, new_ms_body)

    with open(path, 'w') as f:
        f.write(content)
    print(f"  NewsAnalyst: patched ({content.count(chr(10))} lines)")


def patch_quant_analyst():
    """QuantAnalyst: _polygon_get, _get_polygon_indicator, _get_news_sentiment_factor, _technical_fallback_signal"""
    path = f"{AGENTS_DIR}/quant_analyst.py"
    with open(path, 'r') as f:
        content = f.read()

    # Pass task to _generate_signals
    content = content.replace(
        "                signals[symbol] = await self._generate_signals(symbol)",
        "                signals[symbol] = await self._generate_signals(symbol, task)"
    )

    # Update _generate_signals signature
    content = content.replace(
        "    async def _generate_signals(self, symbol: str) -> Dict:",
        "    async def _generate_signals(self, symbol: str, task=None) -> Dict:"
    )

    # Pass task to fallback
    content = content.replace(
        "        return await self._technical_fallback_signal(symbol)",
        "        return await self._technical_fallback_signal(symbol, task)"
    )

    # Update _technical_fallback_signal signature
    content = content.replace(
        '    async def _technical_fallback_signal(self, symbol: str) -> Dict:\n        """Generate signal from Polygon server-side indicators + historical data."""',
        '    async def _technical_fallback_signal(self, symbol: str, task=None) -> Dict:\n        """Generate signal from Polygon server-side indicators + historical data."""'
    )

    # Add shared_data fast path for _technical_fallback_signal indicators
    old_rsi_block = "            # 1. Get Polygon server-side indicators (more accurate than manual)\n            rsi = self._get_polygon_indicator(symbol, 'rsi', 14)"
    new_rsi_block = """            # 1. Get Polygon server-side indicators (more accurate than manual)
            shared = self._get_shared_data(task)

            # Use shared data layer if available (instant, already cached)
            if shared:
                rsi_data = await shared.get_rsi(symbol)
                rsi_vals = (rsi_data or {}).get('results', {}).get('values', [])
                rsi = rsi_vals[0].get('value') if rsi_vals else None

                sma20_data = await shared.get_sma(symbol, 20)
                sma20_vals = (sma20_data or {}).get('results', {}).get('values', [])
                sma_20 = sma20_vals[0].get('value') if sma20_vals else None

                sma50_data = await shared.get_sma(symbol, 50)
                sma50_vals = (sma50_data or {}).get('results', {}).get('values', [])
                sma_50 = sma50_vals[0].get('value') if sma50_vals else None

                ema12_data = await shared.get_ema(symbol, 12)
                ema12_vals = (ema12_data or {}).get('results', {}).get('values', [])
                ema_12 = ema12_vals[0].get('value') if ema12_vals else None

                ema26_data = await shared.get_ema(symbol, 26)
                ema26_vals = (ema26_data or {}).get('results', {}).get('values', [])
                ema_26 = ema26_vals[0].get('value') if ema26_vals else None

                macd_data = await shared.get_macd(symbol)
                macd_values = (macd_data or {}).get('results', {}).get('values', [])
                macd_val = macd_values[0].get('value') if macd_values else None
                macd_signal = macd_values[0].get('signal') if macd_values else None
                macd_histogram = macd_values[0].get('histogram') if macd_values else None

                hist_data = await shared.get_historical(symbol, 60)
                hist_results = (hist_data or {}).get('results', [])
                closes = None
                returns = None
                if len(hist_results) >= 20:
                    closes = np.array([r['c'] for r in hist_results])
                    returns = np.diff(closes) / closes[:-1]
            else:
                rsi = self._get_polygon_indicator(symbol, 'rsi', 14)"""

    content = content.replace(old_rsi_block, new_rsi_block)

    # Skip the original indicator fetches when shared data was used
    old_sma_block = "            sma_20 = self._get_polygon_indicator(symbol, 'sma', 20)\n            sma_50 = self._get_polygon_indicator(symbol, 'sma', 50)"
    new_sma_block = "            if not shared:\n                sma_20 = self._get_polygon_indicator(symbol, 'sma', 20)\n                sma_50 = self._get_polygon_indicator(symbol, 'sma', 50)"
    content = content.replace(old_sma_block, new_sma_block)

    old_ema_block = "            ema_12 = self._get_polygon_indicator(symbol, 'ema', 12)\n            ema_26 = self._get_polygon_indicator(symbol, 'ema', 26)"
    new_ema_block = "                ema_12 = self._get_polygon_indicator(symbol, 'ema', 12)\n                ema_26 = self._get_polygon_indicator(symbol, 'ema', 26)"
    content = content.replace(old_ema_block, new_ema_block)

    # Wrap the original MACD fetch in the else block
    old_macd_block = """            # MACD from Polygon
            macd_data = self._polygon_get("""
    new_macd_block = """                # MACD from Polygon
                macd_data = self._polygon_get("""
    content = content.replace(old_macd_block, new_macd_block)

    # Wrap the historical fetch block
    old_hist_block = "            # 2. Get historical data for momentum & volatility\n            end_date = datetime.now().strftime('%Y-%m-%d')"
    new_hist_block = "            if not shared:\n                # 2. Get historical data for momentum & volatility\n                end_date = datetime.now().strftime('%Y-%m-%d')"
    content = content.replace(old_hist_block, new_hist_block)

    with open(path, 'w') as f:
        f.write(content)
    print(f"  QuantAnalyst: patched ({content.count(chr(10))} lines)")


def patch_fundamental_analyst():
    """FundamentalAnalyst: _polygon_get, _analyze_fundamentals and all sub-methods."""
    path = f"{AGENTS_DIR}/fundamental_analyst.py"
    with open(path, 'r') as f:
        content = f.read()

    # Pass task to _analyze_fundamentals
    content = content.replace(
        "                analyses[symbol] = self._analyze_fundamentals(symbol)",
        "                analyses[symbol] = await self._analyze_fundamentals(symbol, task)"
    )

    # Update _analyze_fundamentals signature
    content = content.replace(
        "    def _analyze_fundamentals(self, symbol: str) -> Dict:",
        "    async def _analyze_fundamentals(self, symbol: str, task=None) -> Dict:"
    )

    # Add shared data layer at the top of _analyze_fundamentals
    old_analyze_start = "        symbol = symbol.upper().replace('$', '')\n        result = {'symbol': symbol, 'data_source': 'polygon_multi_endpoint', 'analysis_date': datetime.now().isoformat()}\n        \n        # === 1. Company Details"
    new_analyze_start = """        symbol = symbol.upper().replace('$', '')
        result = {'symbol': symbol, 'data_source': 'polygon_multi_endpoint', 'analysis_date': datetime.now().isoformat()}
        shared = self._get_shared_data(task)
        
        # === 1. Company Details"""
    content = content.replace(old_analyze_start, new_analyze_start)

    # Replace _polygon_get calls with shared data versions
    content = content.replace(
        "        details_data = self._polygon_get(f\"https://api.polygon.io/v3/reference/tickers/{symbol}\")",
        "        details_data = (await shared.get_company_info(symbol)) if shared else self._polygon_get(f\"https://api.polygon.io/v3/reference/tickers/{symbol}\")"
    )
    content = content.replace(
        "        prev_data = self._polygon_get(f\"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev\")",
        "        prev_data = (await shared.get_quote(symbol)) if shared else self._polygon_get(f\"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev\")"
    )
    content = content.replace(
        '        fin_data = self._polygon_get("https://api.polygon.io/vX/reference/financials", {\n            \'ticker\': symbol, \'limit\': 5, \'timeframe\': \'quarterly\', \'order\': \'desc\'\n        })',
        '        fin_data = (await shared.get_financials(symbol, "quarterly", 5)) if shared else self._polygon_get("https://api.polygon.io/vX/reference/financials", {\'ticker\': symbol, \'limit\': 5, \'timeframe\': \'quarterly\', \'order\': \'desc\'})'
    )
    content = content.replace(
        '        annual_data = self._polygon_get("https://api.polygon.io/vX/reference/financials", {\n            \'ticker\': symbol, \'limit\': 2, \'timeframe\': \'annual\', \'order\': \'desc\'\n        })',
        '        annual_data = (await shared.get_financials(symbol, "annual", 2)) if shared else self._polygon_get("https://api.polygon.io/vX/reference/financials", {\'ticker\': symbol, \'limit\': 2, \'timeframe\': \'annual\', \'order\': \'desc\'})'
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f"  FundamentalAnalyst: patched ({content.count(chr(10))} lines)")


def patch_risk_manager():
    """RiskManager: _polygon_get, _get_volatility_regime, _get_sentiment_risk, _get_event_risk."""
    path = f"{AGENTS_DIR}/risk_manager.py"
    with open(path, 'r') as f:
        content = f.read()

    # Pass task to methods in execute()
    content = content.replace(
        "                'volatility_regime': self._get_volatility_regime(),\n                'sentiment_risk': self._get_sentiment_risk(),\n                'event_risk': self._get_event_risk(analysis_symbols[:3]),",
        "                'volatility_regime': await self._get_volatility_regime(task),\n                'sentiment_risk': await self._get_sentiment_risk(task),\n                'event_risk': await self._get_event_risk(analysis_symbols[:3], task),"
    )

    # _get_volatility_regime — add shared data
    content = content.replace(
        '    def _get_volatility_regime(self) -> Dict:\n        """Get real VIX-based volatility regime."""\n        data = self._polygon_get("https://api.polygon.io/v2/aggs/ticker/VIX/prev")',
        '    async def _get_volatility_regime(self, task=None) -> Dict:\n        """Get real VIX-based volatility regime."""\n        shared = self._get_shared_data(task)\n        data = (await shared.get_vix()) if shared else self._polygon_get("https://api.polygon.io/v2/aggs/ticker/VIX/prev")'
    )

    # _get_sentiment_risk — add shared data
    content = content.replace(
        '    def _get_sentiment_risk(self) -> Dict:\n        """Get sentiment-based risk from Fear & Greed Index."""\n        result = {}',
        '    async def _get_sentiment_risk(self, task=None) -> Dict:\n        """Get sentiment-based risk from Fear & Greed Index."""\n        result = {}\n        shared = self._get_shared_data(task)\n        if shared:\n            fng_data = await shared.get_fear_greed()\n            if fng_data and fng_data.get(\'data\'):\n                fng = fng_data[\'data\']\n                current = int(fng[0].get(\'value\', 50))\n                trend = [int(d.get(\'value\', 50)) for d in fng]\n                if current < 20: risk_level, contrarian = \'EXTREME_FEAR\', \'Potential buying opportunity (contrarian)\'\n                elif current < 35: risk_level, contrarian = \'FEAR\', \'Market pessimism elevated\'\n                elif current > 80: risk_level, contrarian = \'EXTREME_GREED\', \'Potential correction risk (contrarian)\'\n                elif current > 65: risk_level, contrarian = \'GREED\', \'Market optimism elevated\'\n                else: risk_level, contrarian = \'NEUTRAL\', \'No extreme sentiment\'\n                return {\'fear_greed_index\': current, \'label\': fng[0].get(\'value_classification\', \'Unknown\'), \'risk_level\': risk_level, \'contrarian_signal\': contrarian, \'trend_7d\': trend, \'trend_direction\': \'IMPROVING\' if trend[0] > trend[-1] else \'DETERIORATING\', \'data_source\': \'alternative_me_live\'}'
    )

    # _get_event_risk — add shared data
    content = content.replace(
        '    def _get_event_risk(self, symbols: List[str]) -> List[Dict]:',
        '    async def _get_event_risk(self, symbols: List[str], task=None) -> List[Dict]:'
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f"  RiskManager: patched ({content.count(chr(10))} lines)")


def patch_macro_analyst():
    """MacroAnalyst: _get_prev_close, all analysis methods."""
    path = f"{AGENTS_DIR}/macro_analyst.py"
    with open(path, 'r') as f:
        content = f.read()

    # Pass task to all methods in execute()
    old_exec = """            data = {
                'volatility_regime': self._analyze_volatility(),
                'sector_performance': self._get_sector_performance(),
                'market_breadth': self._analyze_market_breadth(),
                'yield_curve': self._analyze_yield_curve(),
                'dollar_strength': self._analyze_dollar(),
                'commodity_signals': self._analyze_commodities(),
                'macro_news': self._get_macro_news(),
                'sentiment_overlay': self._get_sentiment(),
                'risk_assessment': self._assess_macro_risk(),
                'market_impact': self._assess_impact()
            }"""
    new_exec = """            shared = self._get_shared_data(task)
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
            }"""
    content = content.replace(old_exec, new_exec)

    # _get_prev_close — add shared data fast path
    content = content.replace(
        '    def _get_prev_close(self, ticker: str) -> Dict:\n        """Get previous close data from Polygon."""\n        data = self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev")',
        '    async def _get_prev_close(self, ticker: str, shared=None) -> Dict:\n        """Get previous close data from Polygon."""\n        data = (await shared.get_quote(ticker)) if shared else self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{ticker}/prev")'
    )

    # Update all methods to be async and pass shared
    for method in ['_analyze_volatility', '_get_sector_performance', '_analyze_market_breadth',
                    '_analyze_yield_curve', '_analyze_dollar', '_analyze_commodities',
                    '_get_macro_news', '_get_sentiment', '_assess_macro_risk', '_assess_impact']:
        content = content.replace(
            f'    def {method}(self) -> Dict:',
            f'    async def {method}(self, shared=None) -> Dict:'
        )
        content = content.replace(
            f'    def {method}(self, symbols',
            f'    async def {method}(self, symbols'
        )

    # Replace _get_prev_close calls in methods to pass shared
    content = content.replace(
        "vix_data = self._get_prev_close('VIX')",
        "vix_data = await self._get_prev_close('VIX', shared)"
    )
    # Replace all self._get_prev_close calls to be awaited with shared
    import re
    content = re.sub(
        r"(\w+) = self\._get_prev_close\('(\w+)'\)",
        r"\1 = await self._get_prev_close('\2', shared)",
        content
    )

    # Replace _polygon_get calls in _get_sector_performance for SMA
    content = content.replace(
        "                sma_data = self._polygon_get(",
        "                sma_data = (await shared.get_sma(etf, 50)) if shared else self._polygon_get("
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f"  MacroAnalyst: patched ({content.count(chr(10))} lines)")


def patch_crypto_specialist():
    """CryptoSpecialist: All methods."""
    path = f"{AGENTS_DIR}/crypto_specialist.py"
    with open(path, 'r') as f:
        content = f.read()

    # Pass task to methods in execute()
    old_exec = """            data = {
                'market_overview': self._get_market_overview(),
                'price_data': self._get_price_data(crypto_symbols),
                'coin_fundamentals': self._get_coin_fundamentals(crypto_symbols),
                'technicals': self._get_crypto_technicals(crypto_symbols),
                'sentiment': self._get_crypto_sentiment(),
                'defi_overview': self._get_defi_overview()
            }"""
    new_exec = """            shared = self._get_shared_data(task)
            data = {
                'market_overview': await self._get_market_overview(shared),
                'price_data': await self._get_price_data(crypto_symbols, shared),
                'coin_fundamentals': await self._get_coin_fundamentals(crypto_symbols, shared),
                'technicals': await self._get_crypto_technicals(crypto_symbols, shared),
                'sentiment': await self._get_crypto_sentiment(shared),
                'defi_overview': await self._get_defi_overview(shared)
            }"""
    content = content.replace(old_exec, new_exec)

    # Update all methods to be async and accept shared
    for method in ['_get_market_overview', '_get_defi_overview', '_get_crypto_sentiment']:
        content = content.replace(
            f'    def {method}(self) -> Dict:',
            f'    async def {method}(self, shared=None) -> Dict:'
        )
    content = content.replace(
        '    def _get_price_data(self, symbols: List[str]) -> Dict:',
        '    async def _get_price_data(self, symbols: List[str], shared=None) -> Dict:'
    )
    content = content.replace(
        '    def _get_coin_fundamentals(self, symbols: List[str]) -> Dict:',
        '    async def _get_coin_fundamentals(self, symbols: List[str], shared=None) -> Dict:'
    )
    content = content.replace(
        '    def _get_crypto_technicals(self, symbols: List[str]) -> Dict:',
        '    async def _get_crypto_technicals(self, symbols: List[str], shared=None) -> Dict:'
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f"  CryptoSpecialist: patched ({content.count(chr(10))} lines)")


def patch_portfolio_optimizer():
    """PortfolioOptimizer: _polygon_get, _get_returns, _get_current_price."""
    path = f"{AGENTS_DIR}/portfolio_optimizer.py"
    with open(path, 'r') as f:
        content = f.read()

    # Pass task to methods in execute()
    old_exec = """            data = {
                'current_analysis': self._analyze_current(symbols, portfolio),
                'optimization': self._optimize(symbols, risk_tolerance),
                'trend_overlay': self._trend_overlay(symbols),
                'dividend_analysis': self._dividend_analysis(symbols),
                'rebalancing_trades': self._recommend_trades(symbols, portfolio),
                'expected_metrics': self._project_metrics(symbols, portfolio),
            }"""
    new_exec = """            shared = self._get_shared_data(task)
            data = {
                'current_analysis': await self._analyze_current(symbols, portfolio, shared),
                'optimization': await self._optimize(symbols, risk_tolerance, shared),
                'trend_overlay': await self._trend_overlay(symbols, shared),
                'dividend_analysis': await self._dividend_analysis(symbols, shared),
                'rebalancing_trades': await self._recommend_trades(symbols, portfolio, shared),
                'expected_metrics': await self._project_metrics(symbols, portfolio, shared),
            }"""
    content = content.replace(old_exec, new_exec)

    # Update method signatures to be async and accept shared
    for method in ['_analyze_current', '_optimize', '_trend_overlay', '_dividend_analysis',
                    '_recommend_trades', '_project_metrics']:
        # These have various signatures, update them generically
        pass

    # Make _get_current_price async with shared
    content = content.replace(
        '    def _get_current_price(self, symbol: str) -> float:\n        """Get current price from Polygon."""\n        data = self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev")',
        '    async def _get_current_price(self, symbol: str, shared=None) -> float:\n        """Get current price from Polygon."""\n        data = (await shared.get_quote(symbol)) if shared else self._polygon_get(f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev")'
    )

    # Make _get_returns async with shared
    content = content.replace(
        '    def _get_returns(self, symbol: str, days: int = 252) -> np.ndarray:\n        """Fetch real historical returns from Polygon."""',
        '    async def _get_returns(self, symbol: str, days: int = 252, shared=None) -> np.ndarray:\n        """Fetch real historical returns from Polygon."""'
    )

    # Add shared data path to _get_returns
    old_returns_body = '        end_date = datetime.now().strftime(\'%Y-%m-%d\')\n        start_date = (datetime.now() - timedelta(days=days + 30)).strftime(\'%Y-%m-%d\')\n        \n        data = self._polygon_get(\n            f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",\n            {\'sort\': \'asc\'}\n        )'
    new_returns_body = """        if shared:
            data = await shared.get_historical(symbol, days + 30)
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days + 30)).strftime('%Y-%m-%d')
            data = self._polygon_get(
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}",
                {'sort': 'asc'}
            )"""
    content = content.replace(old_returns_body, new_returns_body)

    # Update method signatures that need async + shared
    content = content.replace(
        '    def _analyze_current(self, symbols: List[str], portfolio: Dict) -> Dict:',
        '    async def _analyze_current(self, symbols: List[str], portfolio: Dict, shared=None) -> Dict:'
    )
    content = content.replace(
        '    def _optimize(self, symbols: List[str], risk_tolerance: str) -> Dict:',
        '    async def _optimize(self, symbols: List[str], risk_tolerance: str, shared=None) -> Dict:'
    )

    # Update calls to _get_current_price and _get_returns inside methods
    content = content.replace(
        '            price = self._get_current_price(symbol)',
        '            price = await self._get_current_price(symbol, shared)'
    )
    content = content.replace(
        '            returns = self._get_returns(symbol, days=60)',
        '            returns = await self._get_returns(symbol, days=60, shared=shared)'
    )
    content = content.replace(
        '            ret = self._get_returns(symbol)',
        '            ret = await self._get_returns(symbol, shared=shared)'
    )

    with open(path, 'w') as f:
        f.write(content)
    print(f"  PortfolioOptimizer: patched ({content.count(chr(10))} lines)")


# === Run all patches ===
print("Migrating agents to SharedDataLayer...")
patch_market_analyst()
patch_news_analyst()
patch_quant_analyst()
patch_fundamental_analyst()
patch_risk_manager()
patch_macro_analyst()
patch_crypto_specialist()
patch_portfolio_optimizer()
print("\nAll agents patched.")
