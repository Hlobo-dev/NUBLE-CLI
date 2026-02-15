"""
SEC EDGAR XBRL Parser — Institutional Fundamental Data
========================================================

Parses SEC EDGAR XBRL data to extract 24 standardized accounting items
for ANY publicly traded US company. Computes the 40 fundamental ratios
in the Gu-Kelly-Xiu framework.

DATA SOURCE: SEC EDGAR Company Facts API (FREE, no API key needed)
Rate limit: 10 requests/second with User-Agent header.

The HARD PROBLEM: XBRL concept names vary across companies.
"Revenue" might be filed as RevenueFromContractWithCustomerExcludingAssessedTax,
Revenues, SalesRevenueNet, etc. This module handles ALL variants with
an ordered fallback chain.

Author: NUBLE ML Pipeline — Phase 1 Institutional Upgrade
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# XBRL Concept Fallback Chains
# ══════════════════════════════════════════════════════════════

CONCEPT_MAP: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "cost_of_revenue": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "total_assets": [
        "Assets",
    ],
    "current_assets": [
        "AssetsCurrent",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ],
    "total_liabilities": [
        "Liabilities",
    ],
    "current_liabilities": [
        "LiabilitiesCurrent",
    ],
    "stockholders_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent",
    ],
    "ppe_net": [
        "PropertyPlantAndEquipmentNet",
    ],
    "inventories": [
        "InventoryNet",
        "Inventories",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "rd_expense": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestIncomeExpenseNet",
    ],
    "dividends_per_share": [
        "CommonStockDividendsPerShareDeclared",
        "CommonStockDividendsPerShareCashPaid",
    ],
    "operating_cash_flow": [
        "NetCashProvidedByOperatingActivities",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "eps_basic": [
        "EarningsPerShareBasic",
    ],
    "income_tax": [
        "IncomeTaxExpenseBenefit",
    ],
    "accounts_receivable": [
        "AccountsReceivableNetCurrent",
        "AccountsReceivableNet",
    ],
}


# ══════════════════════════════════════════════════════════════
# SECEdgarXBRL
# ══════════════════════════════════════════════════════════════

class SECEdgarXBRL:
    """
    Parses SEC EDGAR XBRL data to extract standardized accounting items
    and compute the 40 Gu-Kelly-Xiu fundamental ratios.
    """

    TICKER_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
    FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
    USER_AGENT = "ROKET Trading humberto@rokettrading.com"

    def __init__(self, cache_dir: str = "~/.nuble/sec_edgar/"):
        self.cache_dir = Path(os.path.expanduser(cache_dir))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": self.USER_AGENT,
            "Accept": "application/json",
        })

        # Ticker → CIK mapping
        self._ticker_cik: Dict[str, str] | None = None
        self._last_request_time: float = 0.0

    # ── Rate limiting ─────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Enforce 100ms minimum between requests (10 req/s max)."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.1:
            time.sleep(0.1 - elapsed)
        self._last_request_time = time.time()

    # ── CIK mapping ──────────────────────────────────────────

    def _load_ticker_cik_map(self) -> Dict[str, str]:
        """Load or download ticker → CIK mapping."""
        if self._ticker_cik is not None:
            return self._ticker_cik

        cache_file = self.cache_dir / "ticker_cik_map.json"

        # Refresh weekly
        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 7 * 86400:  # 7 days
                with open(cache_file) as f:
                    self._ticker_cik = json.load(f)
                return self._ticker_cik

        try:
            self._rate_limit()
            resp = self._session.get(self.TICKER_CIK_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            mapping = {}
            for _, entry in data.items():
                ticker = entry.get("ticker", "").upper()
                cik = str(entry.get("cik_str", "")).zfill(10)
                if ticker:
                    mapping[ticker] = cik

            with open(cache_file, "w") as f:
                json.dump(mapping, f)

            self._ticker_cik = mapping
            logger.info("Loaded %d ticker→CIK mappings from SEC", len(mapping))
            return mapping

        except Exception as e:
            logger.error("Failed to download ticker-CIK mapping: %s", e)
            self._ticker_cik = {}
            return {}

    def _get_cik(self, ticker: str) -> str | None:
        """Map a ticker to 10-digit zero-padded CIK."""
        mapping = self._load_ticker_cik_map()
        ticker = ticker.upper().replace(".", "-")  # BRK.B → BRK-B
        cik = mapping.get(ticker)
        if not cik:
            # Try without hyphen
            cik = mapping.get(ticker.replace("-", ""))
        return cik

    # ── Company Facts fetch ───────────────────────────────────

    def _fetch_company_facts(self, cik: str) -> dict | None:
        """
        Fetch full XBRL company facts from SEC EDGAR.
        Cache raw JSON locally. Cache expiry: 24 hours.
        """
        cache_file = self.cache_dir / f"facts_{cik}.json"

        if cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < 86400:  # 24 hours
                with open(cache_file) as f:
                    return json.load(f)

        try:
            self._rate_limit()
            url = self.FACTS_URL.format(cik=cik)
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            with open(cache_file, "w") as f:
                json.dump(data, f)

            return data

        except Exception as e:
            logger.warning("Failed to fetch company facts for CIK %s: %s", cik, e)
            return None

    # ── Concept extraction ────────────────────────────────────

    def _extract_concept(
        self,
        facts: dict,
        concept_names: List[str],
        form_types: List[str] | None = None,
        units: str = "USD",
    ) -> pd.DataFrame:
        """
        Extract a specific accounting concept from company facts.
        Tries each concept name in order until one has data.
        """
        if form_types is None:
            form_types = ["10-K", "10-Q"]

        us_gaap = facts.get("facts", {}).get("us-gaap", {})

        for concept_name in concept_names:
            concept_data = us_gaap.get(concept_name, {})
            unit_data = concept_data.get("units", {})

            # Try requested units first, then USD/shares for per-share items
            data_list = unit_data.get(units, [])
            if not data_list and units == "USD":
                data_list = unit_data.get("USD/shares", [])
            if not data_list:
                # Try any available unit
                for unit_key, unit_vals in unit_data.items():
                    if unit_vals:
                        data_list = unit_vals
                        break

            if not data_list:
                continue

            records = []
            for item in data_list:
                form = item.get("form", "")
                if form not in form_types:
                    continue

                records.append({
                    "value": item.get("val"),
                    "period_start": item.get("start"),
                    "period_end": item.get("end"),
                    "filed_date": item.get("filed"),
                    "form_type": form,
                    "fiscal_year": item.get("fy"),
                    "fiscal_quarter": item.get("fp", ""),
                })

            if records:
                df = pd.DataFrame(records)
                df["filed_date"] = pd.to_datetime(df["filed_date"], errors="coerce")
                df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
                df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
                return df.dropna(subset=["value", "filed_date"])

        return pd.DataFrame()

    # ── Public API ────────────────────────────────────────────

    def get_fundamentals(self, ticker: str) -> pd.DataFrame | None:
        """
        Get all 24 accounting items for a company, organized by quarter.

        Returns DataFrame with index (fiscal_year, fiscal_quarter),
        columns = all 24 CONCEPT_MAP items + filing_date, period_end, form_type.
        """
        cik = self._get_cik(ticker)
        if not cik:
            logger.warning("No CIK found for %s", ticker)
            return None

        facts = self._fetch_company_facts(cik)
        if not facts:
            return None

        all_data = {}
        for item_name, concept_names in CONCEPT_MAP.items():
            # Determine units
            if item_name in ("eps_basic", "dividends_per_share"):
                units = "USD/shares"
            else:
                units = "USD"

            df = self._extract_concept(facts, concept_names, units=units)
            if df.empty:
                continue

            # Deduplicate: keep latest filing per (fiscal_year, fiscal_quarter)
            if "fiscal_year" in df.columns and "fiscal_quarter" in df.columns:
                df = df.sort_values("filed_date").drop_duplicates(
                    subset=["fiscal_year", "fiscal_quarter"], keep="last"
                )

            all_data[item_name] = df

        if not all_data:
            logger.warning("No XBRL data found for %s", ticker)
            return None

        # Merge all items into a single DataFrame
        # Use the item with the most records as the base
        base_item = max(all_data, key=lambda k: len(all_data[k]))
        base_df = all_data[base_item][
            ["fiscal_year", "fiscal_quarter", "filed_date", "period_end", "form_type"]
        ].copy()
        base_df[base_item] = all_data[base_item]["value"].values

        for item_name, df in all_data.items():
            if item_name == base_item:
                continue
            if df.empty:
                continue

            merge_df = df[["fiscal_year", "fiscal_quarter", "value"]].rename(
                columns={"value": item_name}
            )
            base_df = base_df.merge(
                merge_df, on=["fiscal_year", "fiscal_quarter"],
                how="left", suffixes=("", f"_{item_name}_dup"),
            )

        # Compute derived items
        if "gross_profit" not in base_df.columns or base_df["gross_profit"].isna().all():
            if "revenue" in base_df.columns and "cost_of_revenue" in base_df.columns:
                base_df["gross_profit"] = base_df["revenue"] - base_df["cost_of_revenue"]

        if "revenue" in base_df.columns and "cost_of_revenue" in base_df.columns:
            if "gross_profit" not in base_df.columns:
                base_df["gross_profit"] = base_df["revenue"] - base_df["cost_of_revenue"]

        if "operating_cash_flow" in base_df.columns and "capex" in base_df.columns:
            base_df["free_cash_flow"] = (
                base_df["operating_cash_flow"].fillna(0) - base_df["capex"].fillna(0).abs()
            )

        # Sort by filing date
        base_df = base_df.sort_values("filed_date", ascending=True)
        return base_df.reset_index(drop=True)

    def get_fundamental_ratios(
        self,
        ticker: str,
        market_cap: float | None = None,
        price: float | None = None,
    ) -> dict | None:
        """
        Compute the 40 Gu-Kelly-Xiu fundamental ratios.
        Returns dict of {ratio_name: value}, None for missing data.
        """
        fundamentals = self.get_fundamentals(ticker)
        if fundamentals is None or len(fundamentals) < 4:
            return None

        ratios: Dict[str, float | None] = {}

        # Get recent quarters (sorted by filing date)
        recent = fundamentals.tail(8)
        latest = recent.iloc[-1]

        # TTM = sum of most recent 4 quarters for income/cash flow items
        last_4 = recent.tail(4)

        def _ttm(col: str) -> float | None:
            if col not in last_4.columns:
                return None
            vals = last_4[col].dropna()
            return float(vals.sum()) if len(vals) >= 3 else None

        def _latest(col: str) -> float | None:
            if col not in latest.index:
                return None
            v = latest[col]
            return float(v) if pd.notna(v) else None

        def _lag(col: str, periods: int = 4) -> float | None:
            """Get value from N quarters ago."""
            if col not in recent.columns or len(recent) < periods + 1:
                return None
            v = recent.iloc[-(periods + 1)][col]
            return float(v) if pd.notna(v) else None

        def _safe_div(a: float | None, b: float | None) -> float | None:
            if a is None or b is None or b == 0:
                return None
            return a / b

        # ── Balance sheet items (instant — use latest) ────────
        total_assets = _latest("total_assets")
        current_assets = _latest("current_assets")
        cash = _latest("cash")
        total_liabs = _latest("total_liabilities")
        current_liabs = _latest("current_liabilities")
        equity = _latest("stockholders_equity")
        lt_debt = _latest("long_term_debt") or 0
        st_debt = _latest("short_term_debt") or 0
        total_debt = lt_debt + st_debt
        ppe = _latest("ppe_net")
        inventories = _latest("inventories")
        ar = _latest("accounts_receivable")

        # ── TTM items ──────────────────────────────────────────
        revenue_ttm = _ttm("revenue")
        net_income_ttm = _ttm("net_income")
        operating_income_ttm = _ttm("operating_income")
        gross_profit_ttm = _ttm("gross_profit")
        ocf_ttm = _ttm("operating_cash_flow")
        capex_ttm = _ttm("capex")
        rd_ttm = _ttm("rd_expense")
        interest_ttm = _ttm("interest_expense")
        dep_ttm = _ttm("depreciation")
        dividends_ps_ttm = _ttm("dividends_per_share")

        fcf_ttm = None
        if ocf_ttm is not None and capex_ttm is not None:
            fcf_ttm = ocf_ttm - abs(capex_ttm)

        ebitda_ttm = None
        if operating_income_ttm is not None and dep_ttm is not None:
            ebitda_ttm = operating_income_ttm + abs(dep_ttm)

        # ── Lag values ─────────────────────────────────────────
        lag_assets = _lag("total_assets")
        lag_equity = _lag("stockholders_equity")
        lag_revenue = _lag("revenue", periods=4)
        lag_ppe = _lag("ppe_net")
        lag_inv = _lag("inventories")
        lag_ca = _lag("current_assets")
        lag_cash = _lag("cash")
        lag_cl = _lag("current_liabilities")
        lag_std = _lag("short_term_debt")

        avg_assets = None
        if total_assets and lag_assets:
            avg_assets = (total_assets + lag_assets) / 2

        avg_equity = None
        if equity and lag_equity:
            avg_equity = (equity + lag_equity) / 2

        # ═══════════════════════════════════════════════════════
        # VALUE RATIOS (14)
        # ═══════════════════════════════════════════════════════

        ratios["book_to_market"] = _safe_div(equity, market_cap)
        ratios["earnings_to_price"] = _safe_div(net_income_ttm, market_cap)
        ratios["cashflow_to_price"] = _safe_div(ocf_ttm, market_cap)
        ratios["dividend_yield"] = _safe_div(dividends_ps_ttm, price)
        ratios["sales_to_price"] = _safe_div(revenue_ttm, market_cap)

        ev = None
        if market_cap is not None and cash is not None:
            ev = market_cap + total_debt - cash
        ratios["ev_ebitda"] = _safe_div(ev, ebitda_ttm)

        ratios["asset_growth"] = _safe_div(
            (total_assets - lag_assets) if total_assets and lag_assets else None,
            lag_assets,
        )
        ratios["free_cf_yield"] = _safe_div(fcf_ttm, market_cap)
        ratios["tangibility"] = _safe_div(ppe, total_assets)

        noa = None
        if total_assets and cash and total_liabs:
            noa = (total_assets - cash - total_liabs)
        ratios["net_operating_assets"] = _safe_div(noa, total_assets)

        delta_ppe = (ppe - lag_ppe) if ppe is not None and lag_ppe is not None else None
        delta_inv = (inventories - lag_inv) if inventories is not None and lag_inv is not None else None
        investment_num = None
        if delta_ppe is not None and delta_inv is not None:
            investment_num = delta_ppe + delta_inv
        ratios["investment"] = _safe_div(investment_num, lag_assets)

        ratios["cf_to_debt"] = _safe_div(ocf_ttm, total_debt if total_debt else None)
        ratios["pe_ratio"] = _safe_div(market_cap, net_income_ttm)

        # ═══════════════════════════════════════════════════════
        # QUALITY RATIOS (20)
        # ═══════════════════════════════════════════════════════

        ratios["roe"] = _safe_div(net_income_ttm, avg_equity)
        ratios["roa"] = _safe_div(net_income_ttm, avg_assets)
        ratios["gross_profitability"] = _safe_div(gross_profit_ttm, total_assets)
        ratios["operating_profitability"] = _safe_div(operating_income_ttm, equity)
        ratios["cash_profitability"] = _safe_div(ocf_ttm, total_assets)

        # Accruals (Sloan 1996)
        accruals = None
        if all(v is not None for v in [current_assets, lag_ca, cash, lag_cash,
                                        current_liabs, lag_cl, st_debt, lag_std, dep_ttm]):
            accruals = (
                (current_assets - lag_ca)
                - (cash - lag_cash)
                - (current_liabs - lag_cl)
                + ((st_debt or 0) - (lag_std or 0))
                - abs(dep_ttm)
            )
        ratios["accruals"] = _safe_div(accruals, avg_assets)
        ratios["accruals_pct"] = _safe_div(
            abs(accruals) if accruals is not None else None,
            abs(net_income_ttm) if net_income_ttm else None,
        )

        ratios["asset_turnover"] = _safe_div(revenue_ttm, avg_assets)

        # Gross margin change YoY
        gp_ratio_now = _safe_div(gross_profit_ttm, revenue_ttm)
        lag_gp = _ttm("gross_profit") if len(recent) >= 8 else None
        lag_rev_ttm = lag_revenue  # Already TTM 4Q ago approximation
        gp_ratio_lag = _safe_div(lag_gp, lag_rev_ttm)
        if gp_ratio_now is not None and gp_ratio_lag is not None:
            ratios["gross_margin_change"] = gp_ratio_now - gp_ratio_lag
        else:
            ratios["gross_margin_change"] = None

        # Earnings consistency (std of quarterly EPS growth)
        if "eps_basic" in recent.columns and len(recent) >= 5:
            eps_vals = recent["eps_basic"].dropna()
            if len(eps_vals) >= 5:
                eps_growth = eps_vals.pct_change().dropna()
                ratios["earnings_consistency"] = float(eps_growth.std()) if len(eps_growth) >= 3 else None
            else:
                ratios["earnings_consistency"] = None
        else:
            ratios["earnings_consistency"] = None

        ratios["rd_to_market"] = _safe_div(rd_ttm, market_cap)
        ratios["rd_to_sales"] = _safe_div(rd_ttm, revenue_ttm)
        ratios["capex_to_assets"] = _safe_div(abs(capex_ttm) if capex_ttm else None, total_assets)
        ratios["payout_ratio"] = _safe_div(
            dividends_ps_ttm,
            _latest("eps_basic") if _latest("eps_basic") else None,
        )
        ratios["debt_to_equity"] = _safe_div(total_debt, equity)
        ratios["current_ratio"] = _safe_div(current_assets, current_liabs)
        ratios["interest_coverage"] = _safe_div(operating_income_ttm, abs(interest_ttm) if interest_ttm else None)
        ratios["sales_growth"] = _safe_div(
            (revenue_ttm - lag_revenue) if revenue_ttm and lag_revenue else None,
            abs(lag_revenue) if lag_revenue else None,
        )

        # Earnings surprise (latest quarter vs same quarter last year)
        if "eps_basic" in recent.columns and len(recent) >= 5:
            eps_now = _latest("eps_basic")
            eps_4ago = _lag("eps_basic", periods=4)
            ratios["earnings_surprise"] = _safe_div(
                (eps_now - eps_4ago) if eps_now is not None and eps_4ago is not None else None,
                abs(eps_4ago) if eps_4ago else None,
            )
        else:
            ratios["earnings_surprise"] = None

        ratios["cash_to_assets"] = _safe_div(cash, total_assets)

        return ratios

    def get_point_in_time_fundamentals(
        self, ticker: str, as_of_date: str
    ) -> dict | None:
        """
        Get fundamentals AS THEY WERE KNOWN on a specific date.
        Only uses filings with filing_date <= as_of_date.
        Critical for avoiding lookahead bias in ML training.
        """
        fundamentals = self.get_fundamentals(ticker)
        if fundamentals is None:
            return None

        cutoff = pd.Timestamp(as_of_date)
        available = fundamentals[fundamentals["filed_date"] <= cutoff]
        if available.empty:
            return None

        # Return most recent values as a flat dict
        latest = available.iloc[-1]
        result = {}
        for col in CONCEPT_MAP.keys():
            if col in latest.index and pd.notna(latest[col]):
                result[col] = float(latest[col])

        # Add derived
        if "revenue" in result and "cost_of_revenue" in result:
            result["gross_profit"] = result.get("gross_profit", result["revenue"] - result["cost_of_revenue"])
        if "operating_cash_flow" in result and "capex" in result:
            result["free_cash_flow"] = result["operating_cash_flow"] - abs(result["capex"])

        result["filing_date"] = latest["filed_date"].isoformat() if pd.notna(latest["filed_date"]) else None
        result["period_end"] = latest["period_end"].isoformat() if pd.notna(latest.get("period_end")) else None

        return result

    def get_quality_score(self, ticker: str, market_cap: float | None = None, price: float | None = None) -> dict:
        """
        Compute a composite fundamental quality score.
        Returns {'score': 0-100, 'grade': A/B/C/D/F, 'components': {...}}.
        """
        ratios = self.get_fundamental_ratios(ticker, market_cap=market_cap, price=price)
        if not ratios:
            return {"score": None, "grade": "N/A", "components": {}}

        score = 50.0  # Start at neutral
        components = {}

        # Profitability (higher is better)
        roe = ratios.get("roe")
        if roe is not None:
            if roe > 0.15:
                score += 8
            elif roe > 0.10:
                score += 4
            elif roe < 0:
                score -= 8
            components["roe"] = roe

        roa = ratios.get("roa")
        if roa is not None:
            if roa > 0.08:
                score += 6
            elif roa > 0.04:
                score += 3
            elif roa < 0:
                score -= 6
            components["roa"] = roa

        gp = ratios.get("gross_profitability")
        if gp is not None:
            if gp > 0.30:
                score += 5
            elif gp > 0.15:
                score += 2
            components["gross_profitability"] = gp

        # Balance sheet health
        cr = ratios.get("current_ratio")
        if cr is not None:
            if cr > 2.0:
                score += 4
            elif cr > 1.5:
                score += 2
            elif cr < 1.0:
                score -= 5
            components["current_ratio"] = cr

        de = ratios.get("debt_to_equity")
        if de is not None:
            if de < 0.3:
                score += 4
            elif de < 1.0:
                score += 2
            elif de > 3.0:
                score -= 6
            components["debt_to_equity"] = de

        ic = ratios.get("interest_coverage")
        if ic is not None:
            if ic > 10:
                score += 4
            elif ic > 5:
                score += 2
            elif ic is not None and ic < 1.5:
                score -= 8
            components["interest_coverage"] = ic

        # Growth
        sg = ratios.get("sales_growth")
        if sg is not None:
            if sg > 0.20:
                score += 5
            elif sg > 0.05:
                score += 2
            elif sg < -0.10:
                score -= 5
            components["sales_growth"] = sg

        # Earnings quality (lower accruals = higher quality)
        acc = ratios.get("accruals")
        if acc is not None:
            if abs(acc) < 0.03:
                score += 4  # High quality
            elif abs(acc) > 0.10:
                score -= 4  # Low quality
            components["accruals"] = acc

        # Clamp score
        score = max(0, min(100, score))

        # Grade
        if score >= 80:
            grade = "A"
        elif score >= 65:
            grade = "B"
        elif score >= 50:
            grade = "C"
        elif score >= 35:
            grade = "D"
        else:
            grade = "F"

        return {"score": round(score, 1), "grade": grade, "components": components}
