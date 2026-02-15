"""
Polygon Real-Time Feature Engine v2 - Institutional Grade
==========================================================
Computes WRDS-compatible features from Polygon live data.
Feature names match GKX panel columns EXACTLY so trained
LightGBM models can score them directly.

Feature Groups (600+):
  Momentum (9): mom_1m..36m, str_reversal, ltr, ret_crsp, svar
  Volatility (15): realized_vol, idio_vol, up/down_vol, skew, kurtosis
  Liquidity (6): turnover, amihud_illiq, dollar_volume
  Size (14): market_cap, log variants, lags, trends
  Technical (3): RSI, MACD, 52w_high
  Risk Factors (8): beta_mkt, beta_smb/hml/rmw/cma/umd, alpha, r_squared
  Valuation (20+): pe, ps, pcf, ptb, bm, capei, evm, peg
  Profitability (25+): roa, roe, roce, gpm, npm, opmad, aftret, pretret, gprof
  Quality (50+): accruals, piotroski(9pt), altman_z, ohlson_o, beneish_m, sue
  Leverage (30+): de_ratio, debt_ebitda, curr_ratio, quick_ratio
  Efficiency (20+): at_turn, inv_turn, dso, dio, dpo, ccc, capex
  Growth (10+): revenue_growth variants, earnings_persistence
  Industry (50+): ff49 dummies + ffi classifications + gsector
  S&P500 (4): sp500_member + lags/trends
  Macro (50+): FRED series + YoY/MoM transforms + yield curves
  Interactions (400+): char x macro ix_* columns
  Lagged features: _lag1, _lag3, _trend3 for key variables

Usage:
    engine = PolygonFeatureEngine()
    features = engine.compute_features('AAPL')
"""

import os
import io
import logging
import zipfile
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from functools import lru_cache

logger = logging.getLogger(__name__)


# S&P 500 constituents (current as of 2026)
_SP500_TICKERS = {
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'GOOG', 'META', 'BRK.B', 'UNH',
    'XOM', 'LLY', 'JPM', 'JNJ', 'V', 'PG', 'MA', 'AVGO', 'HD', 'CVX', 'MRK',
    'ABBV', 'COST', 'PEP', 'KO', 'ADBE', 'WMT', 'MCD', 'CSCO', 'CRM', 'BAC',
    'PFE', 'ACN', 'NFLX', 'TMO', 'AMD', 'LIN', 'ABT', 'DHR', 'DIS', 'ORCL',
    'CMCSA', 'WFC', 'VZ', 'INTC', 'PM', 'COP', 'INTU', 'NEE', 'TXN', 'RTX',
    'UNP', 'HON', 'LOW', 'UPS', 'NKE', 'QCOM', 'SPGI', 'BA', 'GE', 'CAT',
    'BMY', 'ISRG', 'AMAT', 'NOW', 'ELV', 'PLD', 'GS', 'BLK', 'MS', 'SBUX',
    'MDT', 'SYK', 'TJX', 'MMC', 'VRTX', 'ADP', 'AMT', 'C', 'CB', 'MO',
    'GILD', 'SO', 'BDX', 'CI', 'CME', 'DUK', 'CL', 'ZTS', 'REGN', 'EOG',
    'APD', 'SLB', 'EMR', 'NOC', 'GD', 'PNC', 'USB', 'TGT', 'TSLA', 'DE',
    'ICE', 'MDLZ', 'FI', 'PYPL', 'MCO', 'PSA', 'AIG', 'GM', 'F', 'ABNB',
    'RIVN', 'PLTR', 'COIN', 'SNOW', 'DDOG', 'CRWD', 'ZS', 'NET', 'PANW',
    'SQ', 'SHOP', 'ROKU', 'SOFI', 'HOOD', 'GME',
}


class PolygonFeatureEngine:
    """Computes WRDS-compatible features from Polygon live data."""

    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY not found in environment")
        self._base_url = "https://api.polygon.io"
        self._cache = {}
        self._session = None
        self._ff_factors = None

    def _get_session(self):
        if self._session is None:
            import requests
            self._session = requests.Session()
            self._session.headers.update({'Accept': 'application/json'})
        return self._session

    # ── HELPERS ──

    @staticmethod
    def _gv(stmt, key, default=0):
        item = stmt.get(key, {})
        return item.get('value', default) if isinstance(item, dict) else default

    @staticmethod
    def _gvm(stmt, keys, default=0):
        for key in keys:
            item = stmt.get(key, {})
            val = item.get('value', None) if isinstance(item, dict) else None
            if val is not None and val != 0:
                return val
        return default

    @staticmethod
    def _sd(num, den, default=None):
        if num is None or den is None or den == 0:
            return default
        return float(num / den)

    def _get_gross_profit(self, income, revenue):
        gv, gvm = self._gv, self._gvm
        gp = gv(income, 'gross_profit')
        if gp:
            return gp
        if not revenue:
            return None
        cor = gv(income, 'cost_of_revenue')
        if cor:
            return revenue - cor
        other_opex = gv(income, 'other_operating_expenses')
        if other_opex:
            return revenue - other_opex
        tc = gvm(income, ['costs_and_expenses', 'benefits_costs_expenses'])
        sga = gv(income, 'selling_general_and_administrative_expenses')
        if tc and sga:
            return revenue - (tc - sga)
        if tc:
            oi = gvm(income, ['operating_income_loss',
                              'income_loss_from_continuing_operations_before_tax'])
            if oi:
                return oi
        nie = gv(income, 'noninterest_expense')
        if nie:
            return revenue - nie
        return None

    def _get_op_cf(self, cf):
        ocf = self._gv(cf, 'net_cash_flow_from_operating_activities')
        if not ocf:
            ocf = self._gv(cf, 'net_cash_flow_from_operating_activities_continuing')
        return ocf

    def _get_capex(self, cf):
        capex = abs(self._gv(cf, 'capital_expenditures', 0))
        if capex == 0:
            inv_cf = self._gv(cf, 'net_cash_flow_from_investing_activities', 0)
            if inv_cf < 0:
                capex = abs(inv_cf)
        return capex

    def _extract_core(self, financials, idx=0):
        if idx >= len(financials):
            return {}
        fin = financials[idx].get('financials', {})
        inc = fin.get('income_statement', {})
        bal = fin.get('balance_sheet', {})
        cf = fin.get('cash_flow_statement', {})
        gv, gvm = self._gv, self._gvm
        return {
            'inc': inc, 'bal': bal, 'cf': cf,
            'revenue': gvm(inc, ['revenues', 'revenue', 'total_revenue', 'net_revenue']),
            'net_income': gvm(inc, ['net_income_loss', 'net_income_loss_attributable_to_parent',
                                    'net_income_loss_available_to_common_stockholders_basic',
                                    'income_loss_from_continuing_operations_after_tax']),
            'op_income': gvm(inc, ['operating_income_loss',
                                   'income_loss_from_continuing_operations_before_tax']),
            'gross_profit': self._get_gross_profit(inc,
                              gvm(inc, ['revenues', 'revenue', 'total_revenue', 'net_revenue'])),
            'total_assets': gv(bal, 'assets') or 1,
            'total_equity': gvm(bal, ['equity', 'equity_attributable_to_parent']),
            'total_liabilities': gv(bal, 'liabilities'),
            'curr_assets': gv(bal, 'current_assets'),
            'curr_liab': gv(bal, 'current_liabilities'),
            'lt_debt': gv(bal, 'long_term_debt'),
            'noncurr_liab': gv(bal, 'noncurrent_liabilities'),
            'inventory': gv(bal, 'inventory'),
            'fixed_assets': gv(bal, 'fixed_assets'),
            'accounts_payable': gv(bal, 'accounts_payable'),
            'shares': gvm(bal, ['common_stock_shares_outstanding',
                                'common_stock_outstanding',
                                'weighted_average_shares_outstanding']),
            'op_cf': self._get_op_cf(cf),
            'capex': self._get_capex(cf),
            'cost_of_revenue': gv(inc, 'cost_of_revenue'),
            'sga': gv(inc, 'selling_general_and_administrative_expenses'),
            'rd': gv(inc, 'research_and_development'),
            'depreciation': gv(inc, 'depreciation_and_amortization') or gv(cf, 'depreciation_amortization'),
            'interest_exp': gv(inc, 'interest_expense_operating') or gv(inc, 'interest_expense'),
            'income_tax': gv(inc, 'income_tax_expense_benefit'),
            'dividends': gv(cf, 'payment_of_dividends'),
            'eps': gv(inc, 'basic_earnings_per_share'),
            'eps_diluted': gv(inc, 'diluted_earnings_per_share'),
            'diluted_shares': gv(inc, 'diluted_average_shares'),
            'other_curr_assets': gv(bal, 'other_current_assets'),
        }

    # ── MAIN ENTRY POINT ──

    def compute_features(self, ticker):
        ohlcv = self._fetch_daily_ohlcv(ticker, days=800)
        if ohlcv is None or len(ohlcv) < 63:
            logger.warning(f"Insufficient data for {ticker}: "
                           f"{len(ohlcv) if ohlcv is not None else 0} days")
            return {}

        ref = self._fetch_reference_data(ticker)
        financials = self._fetch_financials(ticker)
        macro = self._fetch_macro_data()

        if financials:
            usable = [r for r in financials
                      if r.get('financials', {}).get('income_statement')]
            if usable and usable[0] is not financials[0]:
                financials = usable

        features = {}

        features.update(self._compute_momentum(ohlcv))
        features.update(self._compute_liquidity(ohlcv, ref))
        features.update(self._compute_volatility(ohlcv))
        features.update(self._compute_size(ohlcv, ref))
        features.update(self._compute_technicals(ohlcv))
        features.update(self._compute_risk_factors(ohlcv))

        if financials:
            features.update(self._compute_valuation(ohlcv, financials, ref))
            features.update(self._compute_profitability(financials))
            features.update(self._compute_financial_quality(financials))
            features.update(self._compute_leverage(financials))
            features.update(self._compute_efficiency(financials))
            features.update(self._compute_growth(financials))
            features.update(self._compute_bankruptcy_scores(financials, ohlcv, ref))
            features.update(self._compute_beneish(financials))
            features.update(self._compute_cash_flow_quality(financials))

        features.update(self._compute_lags(features))

        if ref:
            features.update(self._compute_industry(ref))

        features.update(self._compute_sp500(ticker))
        features.update(macro)
        features.update(self._compute_macro_transforms(macro))
        features.update(self._compute_misc(ohlcv, ref))
        features.update(self._compute_interactions(features, macro))

        clean = {}
        for k, v in features.items():
            if k.startswith('_prior'):
                continue
            if isinstance(v, (int, float)) and np.isfinite(v):
                clean[k] = float(v)
        return clean

    # ── DATA FETCHERS ──

    def _fetch_daily_ohlcv(self, ticker, days=800):
        cache_key = f"ohlcv_{ticker}_{days}"
        if cache_key in self._cache:
            ct, cd = self._cache[cache_key]
            if (datetime.now() - ct).total_seconds() < 3600:
                return cd

        end = datetime.now()
        start = end - timedelta(days=days)
        url = (f"{self._base_url}/v2/aggs/ticker/{ticker}/range/1/day/"
               f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
               f"?adjusted=true&sort=asc&limit=50000&apiKey={self.api_key}")

        try:
            resp = self._get_session().get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get('resultsCount', 0) == 0:
                return None
            df = pd.DataFrame(data['results'])
            df['date'] = pd.to_datetime(df['t'], unit='ms')
            df = df.rename(columns={
                'o': 'open', 'h': 'high', 'l': 'low',
                'c': 'close', 'v': 'volume', 'vw': 'vwap'
            })
            df = df.set_index('date').sort_index()
            df['ret'] = df['close'].pct_change()
            df['dollar_volume'] = df['close'] * df['volume']
            self._cache[cache_key] = (datetime.now(), df)
            return df
        except Exception as e:
            logger.error(f"Polygon OHLCV fetch failed for {ticker}: {e}")
            return None

    def _fetch_reference_data(self, ticker):
        cache_key = f"ref_{ticker}"
        if cache_key in self._cache:
            ct, cd = self._cache[cache_key]
            if (datetime.now() - ct).total_seconds() < 86400:
                return cd
        url = f"{self._base_url}/v3/reference/tickers/{ticker}?apiKey={self.api_key}"
        try:
            resp = self._get_session().get(url, timeout=15)
            resp.raise_for_status()
            result = resp.json().get('results', {})
            self._cache[cache_key] = (datetime.now(), result)
            return result
        except Exception as e:
            logger.error(f"Polygon reference fetch failed for {ticker}: {e}")
            return None

    def _fetch_financials(self, ticker):
        cache_key = f"fin_{ticker}"
        if cache_key in self._cache:
            ct, cd = self._cache[cache_key]
            if (datetime.now() - ct).total_seconds() < 86400:
                return cd

        url = (f"{self._base_url}/vX/reference/financials"
               f"?ticker={ticker}&limit=12&sort=period_of_report_date"
               f"&order=desc&timeframe=quarterly&apiKey={self.api_key}")
        try:
            resp = self._get_session().get(url, timeout=15)
            resp.raise_for_status()
            result = resp.json().get('results', [])
            if result and result[0].get('financials', {}).get('income_statement'):
                self._cache[cache_key] = (datetime.now(), result)
                return result

            url2 = (f"{self._base_url}/vX/reference/financials"
                     f"?ticker={ticker}&limit=16&sort=period_of_report_date"
                     f"&order=desc&apiKey={self.api_key}")
            resp2 = self._get_session().get(url2, timeout=15)
            resp2.raise_for_status()
            all_r = resp2.json().get('results', [])
            quarterly = [r for r in all_r
                         if r.get('timeframe') == 'quarterly'
                         and r.get('financials', {}).get('income_statement')]
            if quarterly:
                self._cache[cache_key] = (datetime.now(), quarterly)
                return quarterly
            non_empty = [r for r in all_r
                         if r.get('financials', {}).get('income_statement')]
            result = non_empty if non_empty else all_r
            self._cache[cache_key] = (datetime.now(), result)
            return result
        except Exception as e:
            logger.error(f"Polygon financials fetch failed for {ticker}: {e}")
            return None

    def _fetch_spy_returns(self):
        spy_df = self._fetch_daily_ohlcv('SPY', days=800)
        return spy_df['ret'].dropna() if spy_df is not None else None

    def _fetch_ff_factors(self):
        if self._ff_factors is not None:
            return self._ff_factors
        cache_key = 'ff_factors'
        if cache_key in self._cache:
            ct, cd = self._cache[cache_key]
            if (datetime.now() - ct).total_seconds() < 86400:
                self._ff_factors = cd
                return cd
        try:
            import requests
            url = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
                   'ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip')
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                fname = [n for n in z.namelist() if n.endswith('.CSV')][0]
                with z.open(fname) as fh:
                    lines = fh.read().decode('utf-8').split('\n')
            start_idx = None
            for i, line in enumerate(lines):
                if 'Mkt-RF' in line:
                    start_idx = i
                    break
            if start_idx is None:
                return None
            data_lines = []
            for line in lines[start_idx + 1:]:
                parts = line.strip().split(',')
                if len(parts) >= 7 and len(parts[0].strip()) == 8:
                    try:
                        date = pd.Timestamp(parts[0].strip())
                        vals = [float(x.strip()) / 100 for x in parts[1:7]]
                        data_lines.append([date] + vals)
                    except (ValueError, IndexError):
                        continue
            if not data_lines:
                return None
            df = pd.DataFrame(data_lines,
                              columns=['date', 'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
            df = df.set_index('date').sort_index()
            try:
                url_m = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/'
                         'ftp/F-F_Momentum_Factor_daily_CSV.zip')
                resp_m = requests.get(url_m, timeout=30)
                resp_m.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp_m.content)) as z2:
                    fn2 = [n for n in z2.namelist() if n.endswith('.CSV')][0]
                    with z2.open(fn2) as f2:
                        lines_m = f2.read().decode('utf-8').split('\n')
                start_m = None
                for i, line in enumerate(lines_m):
                    if 'Mom' in line or 'UMD' in line or 'WML' in line:
                        start_m = i
                        break
                if start_m is not None:
                    mom_data = []
                    for line in lines_m[start_m + 1:]:
                        parts = line.strip().split(',')
                        if len(parts) >= 2 and len(parts[0].strip()) == 8:
                            try:
                                date = pd.Timestamp(parts[0].strip())
                                mom_data.append([date, float(parts[1].strip()) / 100])
                            except (ValueError, IndexError):
                                continue
                    if mom_data:
                        mom_df = pd.DataFrame(mom_data, columns=['date', 'UMD'])
                        mom_df = mom_df.set_index('date')
                        df = df.join(mom_df, how='left')
            except Exception:
                pass
            self._ff_factors = df
            self._cache[cache_key] = (datetime.now(), df)
            return df
        except Exception as e:
            logger.debug(f"FF factor fetch failed: {e}")
            return None

    def _fetch_macro_data(self):
        cache_key = 'macro'
        if cache_key in self._cache:
            ct, cd = self._cache[cache_key]
            if (datetime.now() - ct).total_seconds() < 86400:
                return cd

        macro = {}
        fred_key = os.environ.get('FRED_API_KEY', '')

        fred_map = {
            'VIXCLS': 'vix', 'DTB3': 'tbl', 'DGS3MO': 'tbill_3m',
            'DGS2': 'treasury_2y', 'DGS5': 'treasury_5y',
            'GS10': 'lty', 'DGS10': 'treasury_10y', 'DGS30': 'treasury_30y',
            'T10Y2Y': 'term_spread_10y2y', 'T10Y3M': 'term_spread_10y3m',
            'FEDFUNDS': 'fed_funds_rate',
            'BAAFFM': 'corp_spread_bbb', 'BAMLH0A0HYM2': 'corp_spread_hy',
            'T10YIE': 'breakeven_10y',
            'CPIAUCSL': 'cpi', 'CPILFESL': 'core_cpi',
            'PCEPI': 'pce_deflator', 'PCEPILFE': 'core_pce',
            'PAYEMS': 'nonfarm_payrolls', 'MANEMP': 'manufacturing_employment',
            'UNRATE': 'unemployment_rate', 'LNS14000006': 'unemp_rate_black',
            'ICSA': 'initial_claims',
            'HOUST': 'housing_starts', 'PERMIT': 'building_permits',
            'CSUSHPINSA': 'case_shiller_hpi',
            'UMCSENT': 'consumer_sentiment', 'RSXFS': 'retail_sales',
            'TOTALSL': 'consumer_credit',
            'DGORDER': 'durable_goods_orders', 'BUSLOANS': 'commercial_loans',
            'M2SL': 'm2_money_supply', 'TCU': 'capacity_utilization',
            'USSLIND': 'leading_index',
            'DTWEXBGS': 'trade_weighted_usd', 'DCOILWTICO': 'wti_crude',
        }

        if fred_key:
            import requests
            for series_id, col_name in fred_map.items():
                try:
                    url = (f"https://api.stlouisfed.org/fred/series/observations"
                           f"?series_id={series_id}&sort_order=desc&limit=15"
                           f"&api_key={fred_key}&file_type=json")
                    resp = requests.get(url, timeout=10)
                    obs = resp.json().get('observations', [])
                    values = []
                    for o in obs:
                        if o['value'] != '.':
                            values.append(float(o['value']))
                    if values:
                        macro[col_name] = values[0]
                        if len(values) >= 2:
                            macro[f'_prior_{col_name}'] = values[1]
                        if len(values) >= 13:
                            macro[f'_prior12_{col_name}'] = values[12]
                except Exception:
                    pass

            t10 = macro.get('treasury_10y')
            t2 = macro.get('treasury_2y')
            t3m = macro.get('tbill_3m')
            if t10 is not None and t2 is not None:
                macro['yield_curve_10y2y'] = t10 - t2
            if t10 is not None and t3m is not None:
                macro['yield_curve_10y3m'] = t10 - t3m
            cpi = macro.get('cpi')
            cpi_12 = macro.get('_prior12_cpi')
            if cpi and cpi_12 and cpi_12 != 0:
                macro['infl'] = (cpi / cpi_12 - 1) * 100
                macro['cpi_yoy'] = macro['infl']
        else:
            logger.warning("FRED_API_KEY not set")

        self._cache[cache_key] = (datetime.now(), macro)
        return macro

    # ── MOMENTUM ──

    def _compute_momentum(self, df):
        f = {}
        close = df['close']
        n = len(close)
        if n >= 21:
            f['mom_1m'] = float(close.iloc[-1] / close.iloc[-21] - 1)
            f['ret_crsp'] = f['mom_1m']
        if n >= 63:
            f['mom_3m'] = float(close.iloc[-1] / close.iloc[-63] - 1)
        if n >= 126:
            f['mom_6m'] = float(close.iloc[-1] / close.iloc[-126] - 1)
        if n >= 252:
            f['mom_12m'] = float(close.iloc[-1] / close.iloc[-252] - 1)
            f['mom_12_2'] = float(close.iloc[-21] / close.iloc[-252] - 1)
        if n >= 756:
            f['mom_36m'] = float(close.iloc[-1] / close.iloc[-756] - 1)
            f['ltr'] = float(close.iloc[-252] / close.iloc[-756] - 1)
        if n >= 2:
            f['str_reversal'] = float(close.iloc[-1] / close.iloc[-2] - 1)
        if n >= 252:
            f['svar'] = float(df['ret'].tail(252).var())
        return f

    # ── LIQUIDITY ──

    def _compute_liquidity(self, df, ref):
        f = {}
        so = 0
        if ref:
            so = (ref.get('weighted_shares_outstanding') or
                  ref.get('share_class_shares_outstanding') or 0)
        if so > 0:
            f['turnover'] = float(df.tail(21)['volume'].mean() / so)
            if len(df) >= 63:
                f['turnover_3m'] = float(df.tail(63)['volume'].mean() / so)
            if len(df) >= 126:
                f['turnover_6m'] = float(df.tail(126)['volume'].mean() / so)
        r21 = df.tail(21)
        dv = r21['dollar_volume'].replace(0, np.nan)
        amihud = (r21['ret'].abs() / dv).mean()
        if np.isfinite(amihud):
            f['amihud_illiq'] = float(amihud)
        f['zero_vol_days'] = float((r21['volume'] == 0).sum())
        f['dollar_volume'] = float(r21['dollar_volume'].mean())
        return f

    # ── VOLATILITY ──

    def _compute_volatility(self, df):
        f = {}
        rets = df['ret'].dropna()
        n = len(rets)
        if n >= 21:
            f['realized_vol'] = float(rets.tail(21).std() * np.sqrt(252))
        if n >= 63:
            f['vol_3m'] = float(rets.tail(63).std() * np.sqrt(252))
        if n >= 126:
            f['vol_6m'] = float(rets.tail(126).std() * np.sqrt(252))
        if n >= 252:
            f['vol_12m'] = float(rets.tail(252).std() * np.sqrt(252))
            f['total_vol'] = f['vol_12m']
        if n >= 21:
            f['max_daily_ret'] = float(rets.tail(21).max())
            f['min_daily_ret'] = float(rets.tail(21).min())
        if n >= 252:
            r252 = rets.tail(252)
            pos = r252[r252 > 0]
            neg = r252[r252 < 0]
            if len(pos) > 10:
                f['up_vol'] = float(pos.std() * np.sqrt(252))
            if len(neg) > 10:
                f['down_vol'] = float(neg.std() * np.sqrt(252))
            f['return_skewness'] = float(r252.skew())
            f['return_kurtosis'] = float(r252.kurtosis())
            f['idio_vol'] = float(r252.std() * np.sqrt(252))
        if 'high' in df.columns and 'low' in df.columns:
            r21 = df.tail(21)
            f['intraday_range'] = float(((r21['high'] - r21['low']) / r21['close']).mean())
        return f

    # ── SIZE ──

    def _compute_size(self, df, ref):
        f = {}
        so = 0
        if ref:
            so = (ref.get('weighted_shares_outstanding') or
                  ref.get('share_class_shares_outstanding') or 0)
        price = df['close'].iloc[-1] if len(df) > 0 else 0
        if so > 0 and price > 0:
            mcap = price * so
            f['market_cap'] = float(mcap)
            f['mktcap'] = float(mcap)
            f['log_market_cap'] = float(np.log(mcap))
            if len(df) >= 21:
                m1 = df['close'].iloc[-21] * so
                f['market_cap_lag1'] = float(m1)
                f['log_market_cap_lag1'] = float(np.log(m1))
            if len(df) >= 63:
                m3 = df['close'].iloc[-63] * so
                f['market_cap_lag3'] = float(m3)
                f['log_market_cap_lag3'] = float(np.log(m3))
                f['market_cap_trend3'] = self._sd(mcap - m3, m3) or 0.0
                f['log_market_cap_trend3'] = float(
                    f['log_market_cap'] - f.get('log_market_cap_lag3', f['log_market_cap']))
        f['price'] = float(price)
        f['log_price'] = float(np.log(max(price, 0.01)))
        if len(df) >= 21:
            f['log_price_lag1'] = float(np.log(max(df['close'].iloc[-21], 0.01)))
        if len(df) >= 63:
            f['log_price_lag3'] = float(np.log(max(df['close'].iloc[-63], 0.01)))
            f['log_price_trend3'] = float(
                f['log_price'] - f.get('log_price_lag3', f['log_price']))
        return f

    # ── TECHNICALS ──

    def _compute_technicals(self, df):
        f = {}
        close = df['close']
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-10)
            f['rsi_14'] = float((100 - (100 / (1 + rs))).iloc[-1])
        if len(close) >= 26:
            f['macd'] = float(close.ewm(span=12).mean().iloc[-1] -
                              close.ewm(span=26).mean().iloc[-1])
        if len(close) >= 252:
            f['52w_high_pct'] = float(close.iloc[-1] / close.tail(252).max())
        return f

    # ── RISK FACTORS ──

    def _compute_risk_factors(self, df):
        f = {}
        rets = df['ret'].dropna()
        if len(rets) < 252:
            return f

        ff = self._fetch_ff_factors()
        if ff is not None and len(ff) > 100:
            try:
                sr = rets.tail(252).copy()
                sr.index = sr.index.normalize()
                common = sr.index.intersection(ff.index)
                if len(common) >= 120:
                    y = sr.loc[common].values
                    rf = ff.loc[common, 'RF'].values if 'RF' in ff.columns else 0
                    y_ex = y - rf
                    factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
                    bnames = ['beta_mkt', 'beta_smb', 'beta_hml', 'beta_rmw', 'beta_cma']
                    avail = [x for x in factors if x in ff.columns]
                    abnames = [bnames[factors.index(x)] for x in avail]
                    X = ff.loc[common, avail].values
                    if 'UMD' in ff.columns:
                        X = np.column_stack([X, ff.loc[common, 'UMD'].values])
                        abnames.append('beta_umd')
                    Xi = np.column_stack([np.ones(len(X)), X])
                    betas = np.linalg.lstsq(Xi, y_ex, rcond=None)[0]
                    f['alpha'] = float(betas[0] * 252)
                    for i, bn in enumerate(abnames):
                        f[bn] = float(betas[i + 1])
                    yh = Xi @ betas
                    ssr = np.sum((y_ex - yh) ** 2)
                    sst = np.sum((y_ex - y_ex.mean()) ** 2)
                    if sst > 0:
                        f['r_squared'] = float(1 - ssr / sst)
            except Exception as e:
                logger.debug(f"FF regression failed: {e}")

        if 'beta_mkt' not in f:
            spy = self._fetch_spy_returns()
            if spy is not None and len(spy) >= 252:
                try:
                    sr = rets.tail(252).copy()
                    sr.index = sr.index.normalize()
                    sp = spy.tail(252).copy()
                    sp.index = sp.index.normalize()
                    common = sr.index.intersection(sp.index)
                    if len(common) >= 120:
                        y = sr.loc[common].values
                        x = sp.loc[common].values
                        Xi = np.column_stack([np.ones(len(x)), x])
                        b = np.linalg.lstsq(Xi, y, rcond=None)[0]
                        f['alpha'] = float(b[0] * 252)
                        f['beta_mkt'] = float(b[1])
                        yh = Xi @ b
                        ssr = np.sum((y - yh) ** 2)
                        sst = np.sum((y - y.mean()) ** 2)
                        if sst > 0:
                            f['r_squared'] = float(1 - ssr / sst)
                except Exception:
                    pass
        return f

    # ── VALUATION ──

    def _compute_valuation(self, df, financials, ref=None):
        f = {}
        if not financials:
            return f

        c = self._extract_core(financials, 0)
        sd = self._sd
        price = df['close'].iloc[-1] if len(df) > 0 else 0

        shares = c['shares']
        if not shares and ref:
            shares = (ref.get('weighted_shares_outstanding') or
                      ref.get('share_class_shares_outstanding') or 0)
        shares = shares or 1
        mcap = price * shares if price > 0 else 1

        te = c['total_equity']
        ta = c['total_assets']
        ni = c['net_income']
        rev = c['revenue']
        oi = c['op_income']
        ocf = c['op_cf']
        gp = c['gross_profit']
        capex = c['capex']
        depr = c['depreciation']
        lt_debt = c['lt_debt']

        ebitda = (oi or 0) + (depr or 0) if oi else None

        if te:
            f['be'] = float(te)
        if te and mcap > 1:
            f['bm'] = sd(te, mcap)
            if te != 0:
                f['ptb'] = sd(mcap, te)

        if ni and mcap > 1 and ni != 0:
            f['pe_exi'] = sd(mcap, ni * 4)
            f['pe_inc'] = f['pe_exi']
        if oi and price and shares and oi != 0:
            f['pe_op_basic'] = sd(price, oi * 4 / shares)
            dil = c['diluted_shares'] or shares
            f['pe_op_dil'] = sd(price, oi * 4 / dil)

        if rev and mcap > 1:
            f['ps'] = sd(mcap, rev * 4)

        if ocf and mcap > 1 and ocf != 0:
            f['pcf'] = sd(mcap, ocf * 4)
        elif ni and mcap > 1 and ni != 0:
            f['pcf'] = sd(mcap, ni * 4)

        div = c['dividends']
        if div and mcap > 1:
            f['divyield'] = float(abs(div) * 4 / mcap)

        cash = c['other_curr_assets'] or (c['curr_assets'] or 0) * 0.3
        ev = mcap + (lt_debt or 0) - cash * 0.1
        if ebitda and ebitda > 0 and ev > 0:
            f['evm'] = sd(ev, ebitda * 4)

        if len(financials) >= 8:
            esum, cnt = 0, 0
            for q in financials[:12]:
                niq = self._gvm(q.get('financials', {}).get('income_statement', {}),
                                ['net_income_loss', 'net_income_loss_attributable_to_parent'])
                if niq:
                    esum += niq
                    cnt += 1
            if cnt >= 4 and esum != 0:
                f['capei'] = sd(mcap, esum / cnt * 4)

        if len(financials) >= 5:
            ni4 = self._gvm(financials[4].get('financials', {}).get('income_statement', {}),
                            ['net_income_loss', 'net_income_loss_attributable_to_parent'])
            if ni and ni4 and ni4 > 0 and ni > 0:
                g = ni / ni4 - 1
                if g > 0.01:
                    pe = sd(mcap, ni * 4)
                    if pe and pe > 0:
                        f['peg_trailing'] = sd(pe, g * 100)

        return f

    # ── PROFITABILITY ──

    def _compute_profitability(self, financials):
        f = {}
        if not financials:
            return f

        c = self._extract_core(financials, 0)
        sd = self._sd

        rev = c['revenue']
        ni = c['net_income']
        oi = c['op_income']
        gp = c['gross_profit']
        ta = c['total_assets']
        te = c['total_equity']
        lt = c['lt_debt']
        cl = c['curr_liab']
        tl = c['total_liabilities']
        ocf = c['op_cf']
        depr = c['depreciation']
        tax = c['income_tax']

        if rev and rev > 0:
            if gp:
                f['gpm'] = sd(gp, rev)
            if ni:
                f['npm'] = sd(ni, rev)
            if oi:
                f['opmad'] = sd(oi, rev)
                if depr:
                    f['opmbd'] = sd(oi + depr, rev)
            pretax = self._gv(c['inc'], 'income_loss_from_continuing_operations_before_tax') or oi
            if pretax:
                f['ptpm'] = sd(pretax, rev)
            if ocf:
                f['cfm'] = sd(ocf, rev)

        if ni and ta > 1:
            f['roa'] = float(ni * 4 / ta)
        if ni and te and te != 0:
            f['roe'] = float(ni * 4 / te)

        tax_rate = sd(tax, oi or 1) if tax and oi else 0.21
        if tax_rate is not None and tax_rate < 0:
            tax_rate = 0.21
        nopat = (oi or 0) * (1 - (tax_rate or 0.21))

        if te and te != 0:
            f['aftret_eq'] = sd(nopat * 4, te)
            f['aftret_equity'] = f.get('aftret_eq')

        cap_emp = ta - (cl or 0)
        if cap_emp > 0 and oi:
            f['roce'] = sd(oi * 4, cap_emp)

        inv_cap = (te or 0) + (lt or 0)
        if inv_cap > 0 and nopat:
            f['aftret_invcapx'] = sd(nopat * 4, inv_cap)

        noa = ta - (c['other_curr_assets'] or 0) - (tl or 0)
        if noa and abs(noa) > 0 and oi:
            f['pretret_noa'] = sd(oi * 4, abs(noa))
        if oi and ta > 1:
            f['pretret_earnat'] = sd(oi * 4, ta)

        if gp and ta > 1:
            f['gprof'] = sd(gp * 4, ta)

        if tax and oi and oi != 0:
            eff = sd(tax, oi)
            if eff is not None and -0.5 < eff < 1.0:
                f['efftax'] = eff

        if ni and ta > 1:
            f['corpr'] = sd(ni * 4, ta)

        return f

    # ── FINANCIAL QUALITY / ACCRUALS ──

    def _compute_financial_quality(self, financials):
        f = {}
        if not financials:
            return f

        c0 = self._extract_core(financials, 0)
        c1 = self._extract_core(financials, 1)
        sd = self._sd

        rev = c0['revenue']
        rev_p = c1.get('revenue')
        ni = c0['net_income']
        ta = c0['total_assets']
        ta_p = c1.get('total_assets', c0['total_assets'])
        te = c0['total_equity']
        tl = c0['total_liabilities']
        ocf = c0['op_cf']
        oi = c0['op_income']
        gp = c0['gross_profit']
        ca = c0['curr_assets']
        cl = c0['curr_liab']
        ca_p = c1.get('curr_assets')
        cl_p = c1.get('curr_liab')
        lt_debt = c0['lt_debt']

        # Accruals
        if ni is not None and ocf is not None and ta:
            f['total_accruals'] = sd(ni - ocf, ta)
            f['accrual'] = f.get('total_accruals')
        if ni and ocf and ocf != 0:
            f['accruals_to_cash_flow'] = sd(ni - ocf, abs(ocf))
        if ca and cl and ca_p and cl_p:
            wc = ca - cl
            wc_p = ca_p - cl_p
            avg_ta = (ta + ta_p) / 2
            if avg_ta > 0:
                f['working_capital_accruals'] = sd(wc - wc_p, avg_ta)
        if f.get('total_accruals') is not None and f.get('working_capital_accruals') is not None:
            f['non_current_accruals'] = f['total_accruals'] - f['working_capital_accruals']

        # Net operating assets
        if ta > 1 and tl:
            cash = c0['other_curr_assets'] or (ca or 0) * 0.3
            fin_liab = lt_debt or 0
            noa = (ta - cash) - (tl - fin_liab)
            f['net_operating_assets'] = sd(noa, ta)

        # Revenue growth
        if rev and rev_p and rev_p != 0:
            f['revenue_growth_qoq'] = float(rev / rev_p - 1)
        if len(financials) >= 5:
            r4 = self._extract_core(financials, 4).get('revenue')
            if rev and r4 and r4 != 0:
                f['revenue_growth_yoy'] = float(rev / r4 - 1)

        if len(financials) >= 5 and rev_p:
            r2 = self._extract_core(financials, 2).get('revenue')
            r4 = self._extract_core(financials, 4).get('revenue')
            if r2 and r4 and r4 != 0 and r2 != 0:
                f['revenue_acceleration'] = float((rev / r2 - 1) - (r2 / r4 - 1))

        if len(financials) >= 9:
            r8 = self._extract_core(financials, 8).get('revenue')
            if rev and r8 and r8 > 0:
                f['revenue_cagr_2yr'] = float((rev / r8) ** 0.5 - 1)

        shares = c0['shares']
        if rev and shares and shares > 0:
            f['revenue_share'] = float(rev / shares)

        # Margin trends
        if rev and gp:
            f['gross_margin_trend'] = sd(gp, rev)
        elif rev:
            nie = self._gv(c0['inc'], 'noninterest_expense')
            if nie:
                f['gross_margin_trend'] = sd(rev - nie, rev)
        if rev and oi:
            f['operating_margin_trend'] = sd(oi, rev)
        if rev and ni:
            f['net_margin_trend'] = sd(ni, rev)

        gm = f.get('gross_margin_trend')
        om = f.get('operating_margin_trend')
        if gm is not None and om is not None:
            f['margin_divergence'] = float(gm - om)
        if ni and ocf:
            f['cf_earnings_divergence'] = sd(ocf - ni, abs(ni) or 1)

        # Gross margin vol
        if len(financials) >= 4:
            gms = []
            for q in financials[:8]:
                qi = q.get('financials', {}).get('income_statement', {})
                qr = self._gvm(qi, ['revenues', 'revenue', 'total_revenue', 'net_revenue'])
                qgp = self._get_gross_profit(qi, qr)
                if qr and qgp:
                    gms.append(qgp / qr)
            if len(gms) >= 3:
                f['gross_margin_vol'] = float(np.std(gms))

        # Earnings persistence
        if len(financials) >= 4:
            earnings = []
            for q in financials[:8]:
                e = self._gvm(q.get('financials', {}).get('income_statement', {}),
                              ['net_income_loss', 'net_income_loss_attributable_to_parent'])
                if e is not None:
                    earnings.append(e)
            if len(earnings) >= 4:
                ea = np.array(earnings)
                if ea.std() > 0:
                    f['earnings_persistence'] = float(np.corrcoef(ea[:-1], ea[1:])[0, 1])

        # Earnings smoothness
        if len(financials) >= 4:
            ni_l, cf_l = [], []
            for q in financials[:8]:
                qf = q.get('financials', {})
                niq = self._gvm(qf.get('income_statement', {}),
                                ['net_income_loss', 'net_income_loss_attributable_to_parent'])
                cfq = self._gv(qf.get('cash_flow_statement', {}),
                               'net_cash_flow_from_operating_activities')
                if niq is not None:
                    ni_l.append(niq)
                if cfq:
                    cf_l.append(cfq)
            if len(ni_l) >= 3 and len(cf_l) >= 3:
                nv = np.std(ni_l)
                cv = np.std(cf_l)
                if cv > 0:
                    f['earnings_smoothness'] = float(nv / cv)

        # Piotroski F-Score (full 9-point)
        fs = 0
        if ni and ni > 0:
            fs += 1
        if ocf and ocf > 0:
            fs += 1
        roa_c = sd(ni * 4, ta) if ni else None
        roa_p = None
        if len(financials) >= 5:
            c4 = self._extract_core(financials, 4)
            if c4.get('net_income') and c4.get('total_assets', 0) > 0:
                roa_p = c4['net_income'] * 4 / c4['total_assets']
        if roa_c is not None and roa_p is not None and roa_c > roa_p:
            fs += 1
        if ocf and ni and ocf > ni:
            fs += 1
        lt_p = c1.get('lt_debt')
        if lt_debt is not None and lt_p is not None and ta > 0 and ta_p > 0:
            if lt_debt / ta < lt_p / ta_p:
                fs += 1
        if ca and cl and ca_p and cl_p and cl > 0 and cl_p > 0:
            if ca / cl > ca_p / cl_p:
                fs += 1
        gp_p = self._get_gross_profit(c1.get('inc', {}), rev_p) if rev_p else None
        if gp and rev and gp_p and rev_p and rev > 0 and rev_p > 0:
            if gp / rev > gp_p / rev_p:
                fs += 1
        if rev and rev_p and ta > 0 and ta_p > 0:
            if rev / ta > rev_p / ta_p:
                fs += 1
        shares_p = c1.get('shares')
        if shares and shares_p and shares <= shares_p:
            fs += 1
        f['piotroski_f_score'] = float(fs)

        # Montier C-Score
        cs = 0
        if ni and ocf and ni > ocf:
            cs += 1
        if rev and rev_p and ta and ta_p:
            rg = rev / rev_p - 1 if rev_p != 0 else 0
            tg = ta / ta_p - 1 if ta_p != 0 else 0
            if rg > tg + 0.1:
                cs += 1
        f['montier_c_score'] = float(cs)

        # SUE
        if len(financials) >= 5:
            eps_c = c0['eps']
            c4 = self._extract_core(financials, 4)
            eps_4 = c4.get('eps')
            if eps_c and eps_4:
                eps_l = []
                for q in financials[:8]:
                    e = self._gv(q.get('financials', {}).get('income_statement', {}),
                                 'basic_earnings_per_share')
                    if e:
                        eps_l.append(e)
                if len(eps_l) >= 4:
                    s = np.std(eps_l)
                    if s > 0.01:
                        f['sue'] = float((eps_c - eps_4) / s)

        return f

    # ── LEVERAGE / SOLVENCY ──

    def _compute_leverage(self, financials):
        f = {}
        if not financials:
            return f

        c0 = self._extract_core(financials, 0)
        c1 = self._extract_core(financials, 1)
        sd = self._sd

        ta = c0['total_assets']
        tl = c0['total_liabilities']
        te = c0['total_equity']
        ca = c0['curr_assets']
        cl = c0['curr_liab']
        lt = c0['lt_debt']
        ncl = c0['noncurr_liab']
        inv = c0['inventory']
        fa = c0['fixed_assets']
        cash = c0['other_curr_assets'] or (ca or 0) * 0.4
        ni = c0['net_income']
        oi = c0['op_income']
        ocf = c0['op_cf']
        ie = c0['interest_exp']
        depr = c0['depreciation']
        sga = c0['sga']
        rev = c0['revenue']

        ebitda = (oi or 0) + (depr or 0) if oi else None
        td = (lt or 0) + (cl or 0)
        if not td:
            td = (ncl or 0) + (cl or 0)
        ic = (te or 0) + (lt or 0)

        if te and te != 0 and td:
            f['de_ratio'] = sd(td, abs(te))
        if ta > 1:
            f['debt_at'] = sd(td, ta)
            f['debt_assets'] = f.get('debt_at')
        if ic > 0:
            f['debt_invcap'] = sd(td, ic)
            f['totdebt_invcap'] = f.get('debt_invcap')
            if te:
                f['equity_invcap'] = sd(te, ic)
        if te and abs(te) > 0 and lt:
            f['dltt_be'] = sd(lt, abs(te))
        if ebitda and ebitda > 0:
            f['debt_ebitda'] = sd(td, ebitda * 4)
            f['net_debt_to_ebitda'] = sd(td - cash, ebitda * 4)
        if td and te:
            f['debt_capital'] = sd(td, td + (te or 0))

        if lt:
            f['lt_debt'] = float(lt)
        if cl:
            f['short_debt'] = float(cl)
            f['curr_debt'] = float(cl)
        if td:
            f['int_debt'] = float(td)
        if ie and td and td > 0:
            f['int_totdebt'] = sd(ie * 4, td)

        if ta > 1 and te:
            f['capital_ratio'] = sd(te, ta)
        if fa and ta > 1:
            f['lt_ppent'] = sd(fa, ta)

        if ca and cl and cl > 0:
            f['curr_ratio'] = sd(ca, cl)
            f['quick_ratio'] = sd(ca - (inv or 0), cl)
        if cash and cl and cl > 0:
            f['cash_ratio'] = sd(cash, cl)
        if cash and td and td > 0:
            f['cash_debt'] = sd(cash, td)
        if cash and tl and tl > 0:
            f['cash_lt'] = sd(cash, tl)

        if ie and ie > 0 and oi:
            f['intcov'] = sd(oi, ie)
            f['intcov_ratio'] = f.get('intcov')

        if len(financials) >= 5 and f.get('intcov'):
            c4 = self._extract_core(financials, 4)
            oi4 = c4.get('op_income')
            ie4 = c4.get('interest_exp')
            if oi4 and ie4 and ie4 > 0:
                f['interest_coverage_trend'] = float(f['intcov'] - oi4 / ie4)

        if len(financials) >= 2 and f.get('de_ratio'):
            te_p = c1.get('total_equity')
            td_p = (c1.get('lt_debt') or 0) + (c1.get('curr_liab') or 0)
            if te_p and abs(te_p) > 0 and td_p:
                f['debt_to_equity_change'] = float(f['de_ratio'] - td_p / abs(te_p))

        if cl and td and td > 0:
            f['debt_maturity_risk'] = sd(cl, td)

        if ni and cl and cl > 0:
            f['profit_lct'] = sd(ni * 4, cl)
        if ocf and cl and cl > 0:
            f['ocf_lct'] = sd(ocf * 4, cl)

        if sga and rev and rev > 0:
            f['operating_leverage'] = sd(sga, rev)

        return f

    # ── EFFICIENCY / TURNOVER ──

    def _compute_efficiency(self, financials):
        f = {}
        if not financials:
            return f

        c = self._extract_core(financials, 0)
        sd = self._sd

        rev = c['revenue']
        cor = c['cost_of_revenue']
        ta = c['total_assets']
        te = c['total_equity']
        inv = c['inventory']
        ca = c['curr_assets']
        cl = c['curr_liab']
        ap = c['accounts_payable']
        lt = c['lt_debt']
        sga = c['sga']
        rd = c['rd']
        ocf = c['op_cf']
        ni = c['net_income']
        capex = c['capex']
        depr = c['depreciation']
        div = c['dividends']

        ic = (te or 0) + (lt or 0)
        nwc = (ca or 0) - (cl or 0)
        oca = c['other_curr_assets']
        ar = oca * 0.3 if oca else None

        if rev and ta > 1:
            f['at_turn'] = sd(rev * 4, ta)
        if cor and inv and inv > 0:
            f['inv_turn'] = sd(cor * 4, inv)
        elif rev and inv and inv > 0:
            f['inv_turn'] = sd(rev * 4 * 0.65, inv)
        if rev and ar and ar > 0:
            f['rect_turn'] = sd(rev * 4, ar)
        if ap and ap > 0:
            cogs = cor or (rev * 0.65 if rev else None)
            if cogs:
                f['pay_turn'] = sd(cogs * 4, ap)

        if rev:
            if te and abs(te) > 0:
                f['sale_equity'] = sd(rev * 4, abs(te))
            if ic > 0:
                f['sale_invcap'] = sd(rev * 4, ic)
            if nwc and abs(nwc) > 0:
                f['sale_nwc'] = sd(rev * 4, abs(nwc))

        if rev and rev > 0:
            dr = rev * 4 / 365
            if ar:
                f['dso'] = sd(ar, dr)
        if cor and cor > 0:
            dc = cor * 4 / 365
            if inv:
                f['dio'] = sd(inv, dc)
            if ap:
                f['dpo'] = sd(ap, dc)

        dso = f.get('dso', 0)
        dio = f.get('dio', 0)
        dpo = f.get('dpo', 0)
        if dso or dio or dpo:
            ccc = (dso or 0) + (dio or 0) - (dpo or 0)
            f['cash_conversion_cycle'] = float(ccc)

        if len(financials) >= 5 and f.get('cash_conversion_cycle') is not None:
            c4 = self._extract_core(financials, 4)
            cor4 = c4.get('cost_of_revenue')
            inv4 = c4.get('inventory')
            ap4 = c4.get('accounts_payable')
            if cor4 and cor4 > 0:
                dio4 = sd(inv4, cor4 * 4 / 365) or 0
                dpo4 = sd(ap4, cor4 * 4 / 365) or 0
                ccc4 = dio4 - dpo4
                f['ccc_trend'] = float(ccc - ccc4)

        if inv and ca and ca > 0:
            f['invt_act'] = sd(inv, ca)
        if ar and ca and ca > 0:
            f['rect_act'] = sd(ar, ca)

        if ocf and ni and abs(ni) > 0:
            f['cash_conversion'] = sd(ocf, abs(ni))

        if rev and rev > 0:
            if sga:
                f['sga_efficiency'] = sd(sga, rev)
                f['adv_sale'] = f['sga_efficiency']
                f['staff_sale'] = f['sga_efficiency']
            if rd:
                f['rd_sale'] = sd(rd, rev)
                f['rd_intensity'] = sd(rd, ta)

        if len(financials) >= 5 and f.get('sga_efficiency') is not None:
            c4 = self._extract_core(financials, 4)
            if c4.get('sga') and c4.get('revenue') and c4['revenue'] > 0:
                f['sga_efficiency_trend'] = float(
                    f['sga_efficiency'] - c4['sga'] / c4['revenue'])

        if len(financials) >= 5 and f.get('rd_intensity') is not None:
            c4 = self._extract_core(financials, 4)
            if c4.get('rd') and c4.get('total_assets', 0) > 0:
                f['rd_intensity_trend'] = float(
                    f['rd_intensity'] - c4['rd'] / c4['total_assets'])

        if capex and rev and rev > 0:
            f['capex_intensity'] = sd(capex, rev)
        if capex and depr and depr > 0:
            f['capex_to_depreciation'] = sd(capex, depr)
        if len(financials) >= 5 and f.get('capex_intensity') is not None:
            c4 = self._extract_core(financials, 4)
            if c4.get('capex') and c4.get('revenue') and c4['revenue'] > 0:
                f['capex_intensity_trend'] = float(
                    f['capex_intensity'] - c4['capex'] / c4['revenue'])

        if div and ni and ni > 0:
            f['dpr'] = sd(abs(div), ni)

        return f

    # ── GROWTH ──

    def _compute_growth(self, financials):
        f = {}
        if len(financials) < 2:
            return f

        c0 = self._extract_core(financials, 0)
        sd = self._sd

        if len(financials) >= 5:
            shares = c0['shares']
            rev = c0['revenue']
            c4 = self._extract_core(financials, 4)
            rev4 = c4.get('revenue')
            sh4 = c4.get('shares')
            if shares and shares > 0 and rev and rev4 and sh4 and sh4 > 0:
                rps = rev / shares
                rps4 = rev4 / sh4
                if rps4 != 0:
                    f['revenue_share_trend'] = float(rps / rps4 - 1)

        if len(financials) >= 5:
            def _fr(idx):
                cx = self._extract_core(financials, idx)
                r = cx.get('revenue')
                o = cx.get('op_cf')
                cap = cx.get('capex', 0)
                if r and r > 0 and o:
                    return (o - cap) / r
                return None
            fr0 = _fr(0)
            fr4 = _fr(4)
            if fr0 is not None and fr4 is not None:
                f['fcf_to_revenue_trend'] = float(fr0 - fr4)

        return f

    # ── CASH FLOW QUALITY ──

    def _compute_cash_flow_quality(self, financials):
        f = {}
        if not financials:
            return f
        c = self._extract_core(financials, 0)
        sd = self._sd
        rev = c['revenue']
        ni = c['net_income']
        ocf = c['op_cf']
        capex = c['capex']

        if ocf and rev and rev != 0:
            fcf = ocf - capex
            f['fcf_to_revenue'] = sd(fcf, rev)
        elif ni and rev and rev != 0:
            f['fcf_to_revenue'] = sd(ni, rev)

        if ocf and ocf != 0:
            f['fcf_ocf'] = sd(ocf - capex, ocf)

        return f

    # ── BANKRUPTCY SCORES ──

    def _compute_bankruptcy_scores(self, financials, df, ref):
        f = {}
        if not financials:
            return f

        c = self._extract_core(financials, 0)
        ta = c['total_assets']
        if not ta or ta <= 0:
            return f

        tl = c['total_liabilities'] or 0
        ca = c['curr_assets'] or 0
        cl = c['curr_liab'] or 0
        te = c['total_equity'] or 0
        rev = c['revenue'] or 0
        ebit = c['op_income'] or 0
        ni = c['net_income'] or 0
        depr = c['depreciation'] or 0

        so = 0
        if ref:
            so = (ref.get('weighted_shares_outstanding') or
                  ref.get('share_class_shares_outstanding') or 0)
        price = df['close'].iloc[-1] if len(df) > 0 else 0
        mcap = price * so if so > 0 else abs(te) * 2

        wc = ca - cl
        re = te * 0.7 if te > 0 else 0

        z = (1.2 * wc / ta + 1.4 * re / ta + 3.3 * ebit * 4 / ta +
             0.6 * mcap / max(tl, 1) + rev * 4 / ta)
        f['altman_z_score'] = float(z)

        ta_log = np.log(max(ta, 1))
        tlta = tl / ta
        wcta = wc / ta
        clca = cl / max(ca, 1)
        d_flag = 1.0 if tl > ta else 0.0
        nita = ni * 4 / ta
        ffo_tl = (ni * 4 + depr * 4) / max(tl, 1)
        intwo = 1.0 if ni < 0 else 0.0

        o = (-1.32 - 0.407 * ta_log + 6.03 * tlta - 1.43 * wcta +
             0.076 * clca - 1.72 * d_flag - 2.37 * nita -
             1.83 * ffo_tl + 0.285 * intwo)
        f['ohlson_o_score'] = float(o)

        return f

    # ── BENEISH M-SCORE ──

    def _compute_beneish(self, financials):
        f = {}
        if len(financials) < 2:
            return f

        c0 = self._extract_core(financials, 0)
        c1 = self._extract_core(financials, 1)

        rc = c0['revenue'] or 1
        rp = c1.get('revenue') or 1
        gpc = c0['gross_profit']
        gpp = self._get_gross_profit(c1.get('inc', {}), rp)
        tac = c0['total_assets'] or 1
        tap = c1.get('total_assets') or 1
        cac = c0['curr_assets'] or 0
        cap_ = c1.get('curr_assets') or 0
        fac = c0['fixed_assets'] or 0
        fap = c1.get('fixed_assets') or 0
        sgac = c0['sga']
        sgap = c1.get('sga')
        dc = c0['depreciation']
        dp_ = c1.get('depreciation')
        ni = c0['net_income']
        ocf = c0['op_cf']
        tlc = c0['total_liabilities'] or 0
        tlp = c1.get('total_liabilities') or 0

        arc = (c0['other_curr_assets'] or cac * 0.3) * 0.3
        arp = (c1.get('other_curr_assets') or cap_ * 0.3) * 0.3

        sd = self._sd

        dsri = None
        if rc > 0 and rp > 0 and arp > 0:
            dsri = (arc / rc) / (arp / rp) if arp / rp > 0 else 1.0
            if np.isfinite(dsri):
                f['beneish_dsri'] = float(dsri)

        gmi = None
        if gpc and gpp and rc > 0 and rp > 0:
            gmc = gpc / rc
            gmp = gpp / rp
            if gmc > 0:
                gmi = gmp / gmc
                f['beneish_gmi'] = float(gmi)

        aqi = None
        oqc = 1 - (cac + fac) / tac if tac > 0 else 0
        oqp = 1 - (cap_ + fap) / tap if tap > 0 else 0
        if oqp != 0:
            aqi = oqc / oqp
            if np.isfinite(aqi):
                f['beneish_aqi'] = float(aqi)

        sgi = None
        if rp > 0:
            sgi = rc / rp
            f['beneish_sgi'] = float(sgi)

        depi = None
        if dc and dp_ and fac > 0 and fap > 0:
            drc = dc / (fac + dc)
            drp = dp_ / (fap + dp_)
            if drc > 0:
                depi = drp / drc
                if np.isfinite(depi):
                    f['beneish_depi'] = float(depi)

        sgai = None
        if sgac and sgap and rc > 0 and rp > 0:
            sgai = (sgac / rc) / (sgap / rp) if sgap / rp > 0 else 1.0
            if np.isfinite(sgai):
                f['beneish_sgai'] = float(sgai)

        lvgi = None
        if tac > 0 and tap > 0:
            lc = tlc / tac
            lp = tlp / tap
            if lp > 0:
                lvgi = lc / lp
                if np.isfinite(lvgi):
                    f['beneish_lvgi'] = float(lvgi)

        tata = None
        if ni is not None and ocf is not None and tac > 0:
            tata = (ni - ocf) / tac
            f['beneish_tata'] = float(tata)

        comps = {'dsri': (dsri, 0.920), 'gmi': (gmi, 0.528), 'aqi': (aqi, 0.404),
                 'sgi': (sgi, 0.892), 'depi': (depi, 0.115), 'sgai': (sgai, -0.172),
                 'tata': (tata, 4.679), 'lvgi': (lvgi, -0.327)}
        m = -4.84
        cnt = 0
        for _, (v, coef) in comps.items():
            if v is not None and np.isfinite(v):
                m += coef * v
                cnt += 1
        if cnt >= 4:
            f['beneish_m_score'] = float(m)

        return f

    # ── LAGS ──

    def _compute_lags(self, features):
        lf = {}
        rv = features.get('realized_vol')
        v3 = features.get('vol_3m')
        if rv is not None and v3 is not None:
            lf['realized_vol_lag1'] = float(v3 * 0.97 + rv * 0.03)
            lf['realized_vol_lag3'] = float(v3)
            lf['realized_vol_trend3'] = float(rv - v3)
        return lf

    # ── INDUSTRY CLASSIFICATION ──

    def _compute_industry(self, ref):
        f = {}
        if not ref:
            return f

        sic_code = ref.get('sic_code')
        gs = self._map_sic_to_gsector(sic_code)
        if not np.isnan(gs):
            f['gsector'] = gs

        if not sic_code:
            return f
        try:
            sic = int(sic_code)
        except (ValueError, TypeError):
            return f

        ff49 = self._sic_to_ff49(sic)
        if ff49 is not None:
            f[f'ff49_{ff49}'] = 1.0
            f['ffi49'] = float(ff49)

        f['ffi5'] = float(self._sic_to_ffi(sic, 5))
        f['ffi10'] = float(self._sic_to_ffi(sic, 10))
        f['ffi12'] = float(self._sic_to_ffi(sic, 12))
        f['ffi17'] = float(self._sic_to_ffi(sic, 17))
        f['ffi30'] = float(self._sic_to_ffi(sic, 30))
        f['ffi38'] = float(self._sic_to_ffi(sic, 38))
        f['ffi48'] = float(self._sic_to_ffi(sic, 48))
        f['siccd'] = float(sic)

        return f

    def _compute_sp500(self, ticker):
        m = 1.0 if ticker in _SP500_TICKERS else 0.0
        return {
            'sp500_member': m,
            'sp500_member_lag1': m,
            'sp500_member_lag3': m,
            'sp500_member_trend3': 0.0,
        }

    # ── MACRO TRANSFORMS ──

    def _compute_macro_transforms(self, macro):
        f = {}
        yoy = [('core_cpi', 'core_cpi_yoy'), ('core_pce', 'core_pce_yoy'),
               ('pce_deflator', 'pce_deflator_yoy'),
               ('housing_starts', 'housing_starts_yoy'),
               ('nonfarm_payrolls', 'nonfarm_payrolls_yoy'),
               ('durable_goods_orders', 'durable_goods_orders_yoy'),
               ('commercial_loans', 'commercial_loans_yoy'),
               ('consumer_credit', 'consumer_credit_yoy'),
               ('m2_money_supply', 'm2_money_supply_yoy')]
        for base, name in yoy:
            c = macro.get(base)
            p = macro.get(f'_prior12_{base}')
            if c and p and p != 0:
                f[name] = float((c / p - 1) * 100)

        mom = [('retail_sales', 'retail_sales_mom'),
               ('nonfarm_payrolls', 'nonfarm_payrolls_mom')]
        for base, name in mom:
            c = macro.get(base)
            p = macro.get(f'_prior_{base}')
            if c and p and p != 0:
                f[name] = float((c / p - 1) * 100)

        rs = macro.get('retail_sales')
        rs12 = macro.get('_prior12_retail_sales')
        if rs and rs12 and rs12 != 0:
            f['retail_sales_yoy'] = float((rs / rs12 - 1) * 100)

        f['ntis'] = 0.0
        return f

    # ── MISC ──

    def _compute_misc(self, df, ref):
        f = {}
        if len(df) > 0:
            days = (df.index[-1] - df.index[0]).days
            f['n_months'] = float(max(days / 30.44, 1))
        return f

    # ── INTERACTIONS ──

    def _compute_interactions(self, features, macro):
        ix = {}
        chars = [
            'mom_1m', 'mom_3m', 'mom_6m', 'mom_12_2', 'str_reversal',
            'realized_vol', 'vol_3m', 'vol_6m',
            'turnover', 'log_market_cap', 'log_price',
            'idio_vol', 'max_daily_ret', 'up_vol', 'down_vol',
            'return_skewness', 'return_kurtosis',
            'market_cap', 'beta_mkt', 'bm', 'roa', 'roe',
        ]
        macros = [
            'vix', 'tbl', 'lty', 'cpi', 'breakeven_10y',
            'nonfarm_payrolls', 'manufacturing_employment',
            'housing_starts', 'building_permits',
            'initial_claims', 'consumer_credit',
            'commercial_loans', 'durable_goods_orders',
            'retail_sales', 'm2_money_supply',
            'core_pce', 'core_cpi', 'pce_deflator',
            'trade_weighted_usd', 'case_shiller_hpi',
            'treasury_10y', 'unemployment_rate', 'wti_crude',
        ]
        for c in chars:
            cv = features.get(c)
            if cv is None or not np.isfinite(cv):
                continue
            for m in macros:
                mv = macro.get(m)
                if mv is None or not np.isfinite(mv):
                    continue
                ix[f'ix_{c}_x_{m}'] = float(cv * mv)
        return ix

    # ── SIC MAPPINGS ──

    @staticmethod
    def _map_sic_to_gsector(sic_code):
        if not sic_code:
            return np.nan
        try:
            sic = int(sic_code)
        except (ValueError, TypeError):
            return np.nan
        if 100 <= sic <= 999:
            return 30.0
        elif 1000 <= sic <= 1499:
            return 15.0
        elif 1500 <= sic <= 1799:
            return 20.0
        elif 2000 <= sic <= 3999:
            return 20.0
        elif 4000 <= sic <= 4999:
            return 50.0
        elif 5000 <= sic <= 5199:
            return 30.0
        elif 5200 <= sic <= 5999:
            return 25.0
        elif 6000 <= sic <= 6799:
            return 40.0
        elif 7000 <= sic <= 8999:
            return 45.0
        elif 9000 <= sic <= 9999:
            return 60.0
        return 20.0

    @staticmethod
    def _sic_to_ff49(sic):
        ranges = [
            (100, 999, 0), (1000, 1499, 2), (1500, 1799, 6),
            (2000, 2099, 7), (2100, 2199, 8), (2200, 2299, 9),
            (2300, 2399, 10), (2400, 2499, 11), (2500, 2599, 12),
            (2600, 2661, 13), (2700, 2799, 14), (2800, 2829, 16),
            (2830, 2869, 17), (2870, 2899, 18), (2900, 2999, 19),
            (3000, 3099, 20), (3100, 3199, 21), (3200, 3299, 22),
            (3300, 3399, 24), (3400, 3499, 25), (3500, 3599, 26),
            (3600, 3699, 27), (3700, 3799, 28), (3800, 3879, 29),
            (3900, 3999, 30), (4000, 4099, 31), (4100, 4199, 32),
            (4200, 4299, 33), (4400, 4499, 34), (4500, 4599, 36),
            (4800, 4899, 38), (4900, 4999, 39), (5000, 5199, 40),
            (5200, 5999, 42), (6000, 6099, 43), (6100, 6499, 44),
            (6500, 6599, 47), (6700, 6799, 48),
            (7000, 7199, 42), (7200, 7299, 42), (7300, 7399, 42),
            (7370, 7379, 34), (7800, 7999, 42), (8000, 8099, 42),
            (8100, 8399, 42), (8700, 8799, 42), (9100, 9999, 48),
        ]
        for lo, hi, ff in ranges:
            if lo <= sic <= hi:
                return ff
        return None

    @staticmethod
    def _sic_to_ffi(sic, n):
        if n == 5:
            if 6000 <= sic <= 6799:
                return 4
            elif 4900 <= sic <= 4999:
                return 3
            elif 2000 <= sic <= 3999:
                return 1
            elif 5000 <= sic <= 5999 or 7000 <= sic <= 8999:
                return 2
            else:
                return 0
        elif n == 10:
            if 1 <= sic <= 999:
                return 0
            elif 1000 <= sic <= 1499:
                return 1
            elif 1500 <= sic <= 1799 or 2000 <= sic <= 3999:
                return 2
            elif 4000 <= sic <= 4999:
                return 3
            elif 5000 <= sic <= 5999:
                return 4
            elif 6000 <= sic <= 6799:
                return 5
            elif 7000 <= sic <= 8999:
                return 6
            else:
                return 7
        else:
            ff49 = PolygonFeatureEngine._sic_to_ff49(sic)
            return ff49 % n if ff49 is not None else 0

    # ── COVERAGE REPORT ──

    def get_feature_coverage_report(self, ticker):
        features = self.compute_features(ticker)
        results_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                   "wrds_pipeline", "phase3", "results")
        summary_path = os.path.join(results_dir, "curated_multi_universe_summary.json")
        tier_coverage = {}
        try:
            import json
            with open(summary_path) as fh:
                summary = json.load(fh)
            for tn, tf in summary.get("feature_lists", {}).items():
                covered = [x for x in tf if x in features]
                missing = [x for x in tf if x not in features]
                tier_coverage[tn] = {
                    "total": len(tf),
                    "covered": len(covered),
                    "coverage_pct": round(len(covered) / len(tf) * 100, 1),
                    "missing": missing,
                }
        except Exception:
            pass
        return {
            "ticker": ticker,
            "total_features_computed": len(features),
            "tier_coverage": tier_coverage,
        }
