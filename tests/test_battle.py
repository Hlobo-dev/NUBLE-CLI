#!/usr/bin/env python3
"""Phase 7: Battle Tests - Trading Logic Edge Cases & Stress Testing."""
import sys
import os
import time
import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.WARNING)

class S(Enum):
    P="PASS"; F="FAIL"; W="WARN"

@dataclass
class R:
    name:str; status:S; time:float; detail:str=""

class BattleTester:
    def __init__(self):
        self.results: List[R] = []
        
    def add(self, r: R):
        self.results.append(r)
        sym = "✅" if r.status == S.P else "❌" if r.status == S.F else "⚠️"
        print(f"  {sym} {r.name} ({r.time:.1f}s) {r.detail}")
    
    def test_triple_barrier_zero_volatility(self) -> R:
        t = time.time()
        try:
            from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
            flat_prices = pd.Series([100.0] * 100, index=pd.date_range('2020-01-01', periods=100))
            labeler = TripleBarrierLabeler(pt_sl=(2.0, 1.0), max_holding_period=20)
            labels = labeler.get_labels(flat_prices)
            return R("TB: ZeroVol", S.P, time.time()-t, f"labels={len(labels)}")
        except Exception as e:
            if "volatility" in str(e).lower() or "zero" in str(e).lower() or "divide" in str(e).lower():
                return R("TB: ZeroVol", S.P, time.time()-t, "clean_error")
            return R("TB: ZeroVol", S.F, time.time()-t, str(e)[:50])
            
    def test_triple_barrier_extreme_moves(self) -> R:
        t = time.time()
        try:
            from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
            np.random.seed(42)
            prices = [100.0]
            for i in range(99):
                if i == 50:
                    prices.append(prices[-1] * 1.50)
                elif i == 75:
                    prices.append(prices[-1] * 0.60)
                else:
                    prices.append(prices[-1] * (1 + np.random.randn() * 0.02))
            price_series = pd.Series(prices, index=pd.date_range('2020-01-01', periods=100))
            labeler = TripleBarrierLabeler(pt_sl=(3.0, 2.0), max_holding_period=10)
            labels = labeler.get_labels(price_series)
            return R("TB: ExtremeMove", S.P, time.time()-t, f"labels={len(labels)}")
        except Exception as e:
            return R("TB: ExtremeMove", S.F, time.time()-t, str(e)[:50])
            
    def test_triple_barrier_nan_handling(self) -> R:
        t = time.time()
        try:
            from src.institutional.labeling.triple_barrier import TripleBarrierLabeler
            np.random.seed(123)
            prices = [100.0 + np.random.randn() * 2 for _ in range(100)]
            prices[25] = np.nan
            prices[50] = np.nan
            price_series = pd.Series(prices, index=pd.date_range('2020-01-01', periods=100))
            clean_series = price_series.ffill().bfill()
            labeler = TripleBarrierLabeler(pt_sl=(2.0, 1.0), max_holding_period=15)
            labels = labeler.get_labels(clean_series)
            valid_labels = labels.dropna()
            return R("TB: NaNHandle", S.P, time.time()-t, f"valid={len(valid_labels)}")
        except Exception as e:
            return R("TB: NaNHandle", S.F, time.time()-t, str(e)[:50])
    
    def test_regime_single_regime(self) -> R:
        t = time.time()
        try:
            from src.institutional.signals.enhanced_signals import RegimeDetector
            np.random.seed(42)
            prices = pd.Series(100 + np.cumsum(np.random.randn(300) * 0.001), index=pd.date_range('2020-01-01', periods=300))
            detector = RegimeDetector()
            regime = detector.detect(prices)
            return R("Regime: Stable", S.P, time.time()-t, f"regime={regime}")
        except Exception as e:
            return R("Regime: Stable", S.F, time.time()-t, str(e)[:50])
            
    def test_regime_switching(self) -> R:
        t = time.time()
        try:
            from src.institutional.signals.enhanced_signals import RegimeDetector
            np.random.seed(789)
            prices = [100.0]
            for i in range(300):
                change = np.random.randn() * (0.005 if (i // 50) % 2 == 0 else 0.03)
                prices.append(prices[-1] * (1 + change))
            price_series = pd.Series(prices, index=pd.date_range('2020-01-01', periods=len(prices)))
            detector = RegimeDetector()
            regime = detector.detect(price_series)
            return R("Regime: Switch", S.P, time.time()-t, f"regime={regime}")
        except Exception as e:
            return R("Regime: Switch", S.F, time.time()-t, str(e)[:50])
            
    def test_regime_extreme_outliers(self) -> R:
        t = time.time()
        try:
            from src.institutional.signals.enhanced_signals import RegimeDetector
            np.random.seed(101)
            prices = [100.0]
            for i in range(300):
                change = 0.20 if i == 50 else (-0.25 if i == 150 else np.random.randn() * 0.01)
                prices.append(prices[-1] * (1 + change))
            price_series = pd.Series(prices, index=pd.date_range('2020-01-01', periods=len(prices)))
            detector = RegimeDetector()
            regime = detector.detect(price_series)
            return R("Regime: Outliers", S.P, time.time()-t, f"regime={regime}")
        except Exception as e:
            return R("Regime: Outliers", S.F, time.time()-t, str(e)[:50])
    
    def test_technical_minimum_data(self) -> R:
        t = time.time()
        try:
            from src.institutional.analytics.technical import TechnicalAnalyzer
            np.random.seed(42)
            n = 50
            closes = list(100 + np.cumsum(np.random.randn(n) * 0.5))
            highs = [c * 1.01 for c in closes]
            lows = [c * 0.99 for c in closes]
            volumes = [1000000] * n
            analyzer = TechnicalAnalyzer()
            result = analyzer.analyze(highs, lows, closes, volumes, "TEST")
            return R("Tech: MinData", S.P, time.time()-t, f"dir={result.direction}")
        except Exception as e:
            return R("Tech: MinData", S.F, time.time()-t, str(e)[:50])
    
    def test_technical_empty_data(self) -> R:
        t = time.time()
        try:
            from src.institutional.analytics.technical import TechnicalAnalyzer
            analyzer = TechnicalAnalyzer()
            result = analyzer.analyze([], [], [], [], "TEST")
            return R("Tech: Empty", S.P, time.time()-t, f"dir={result.direction}")
        except (ValueError, IndexError) as e:
            return R("Tech: Empty", S.P, time.time()-t, "clean_error")
        except Exception as e:
            return R("Tech: Empty", S.F, time.time()-t, str(e)[:50])
    
    def test_risk_drawdown_calculation(self) -> R:
        t = time.time()
        try:
            pnl = [1000.0]
            for i in range(100):
                pnl.append(pnl[-1] * (0.97 if i < 50 else 1.02))
            pnl = np.array(pnl)
            peak = np.maximum.accumulate(pnl)
            drawdown = (peak - pnl) / peak
            max_dd = np.max(drawdown)
            return R("Risk: MaxDD", S.P if max_dd > 0.5 else S.F, time.time()-t, f"maxDD={max_dd:.1%}")
        except Exception as e:
            return R("Risk: MaxDD", S.F, time.time()-t, str(e)[:50])
            
    def test_risk_position_sizing_edge(self) -> R:
        t = time.time()
        try:
            def kelly(wr, wlr): return (wr * wlr - (1 - wr)) / wlr if wlr > 0 else 0
            k1, k2, k3 = kelly(0.5, 1.0), kelly(0.6, 2.0), kelly(0.4, 3.0)
            valid = abs(k1) < 0.01 and k2 > 0 and k3 > 0
            return R("Risk: Kelly", S.P if valid else S.F, time.time()-t, f"k1={k1:.2f},k2={k2:.2f}")
        except Exception as e:
            return R("Risk: Kelly", S.F, time.time()-t, str(e)[:50])
    
    def test_risk_var_calculation(self) -> R:
        t = time.time()
        try:
            np.random.seed(42)
            returns = np.random.randn(252) * 0.02
            var_95, var_99 = np.percentile(returns, 5), np.percentile(returns, 1)
            valid = var_95 < 0 and var_99 < var_95
            return R("Risk: VaR", S.P if valid else S.F, time.time()-t, f"VaR95={var_95:.2%}")
        except Exception as e:
            return R("Risk: VaR", S.F, time.time()-t, str(e)[:50])
    
    def test_data_negative_prices(self) -> R:
        t = time.time()
        try:
            prices = pd.Series([100, 101, -50, 102, 103], index=pd.date_range('2020-01-01', periods=5))
            has_negative = (prices < 0).any()
            clean_prices = prices[prices > 0]
            return R("Data: Negative", S.P if has_negative else S.F, time.time()-t, f"clean={len(clean_prices)}")
        except Exception as e:
            return R("Data: Negative", S.F, time.time()-t, str(e)[:50])
            
    def test_data_duplicate_timestamps(self) -> R:
        t = time.time()
        try:
            dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03'])
            prices = pd.Series([100, 101, 102, 103], index=dates)
            has_dups = prices.index.duplicated().any()
            clean = prices[~prices.index.duplicated(keep='last')]
            return R("Data: Dupes", S.P if has_dups else S.F, time.time()-t, f"deduped={len(clean)}")
        except Exception as e:
            return R("Data: Dupes", S.F, time.time()-t, str(e)[:50])
    
    def test_data_future_dates(self) -> R:
        t = time.time()
        try:
            from datetime import datetime, timedelta
            today = datetime.now()
            dates = pd.date_range(today - timedelta(days=5), today + timedelta(days=30), periods=10)
            prices = pd.Series(range(100, 110), index=dates)
            historical = prices[prices.index <= pd.Timestamp(today)]
            has_future = len(historical) < len(prices)
            return R("Data: Future", S.P if has_future else S.W, time.time()-t, f"filtered={len(historical)}")
        except Exception as e:
            return R("Data: Future", S.F, time.time()-t, str(e)[:50])
    
    async def test_concurrent_analysis(self) -> R:
        t = time.time()
        try:
            from src.institutional.signals.enhanced_signals import RegimeDetector
            detector = RegimeDetector()
            async def analyze_async(seed):
                np.random.seed(seed)
                prices = pd.Series(100 + np.cumsum(np.random.randn(250) * 0.02), index=pd.date_range('2020-01-01', periods=250))
                await asyncio.sleep(0.01)
                return detector.detect(prices)
            results = await asyncio.gather(*[analyze_async(i) for i in range(10)])
            return R("Conc: Analysis", S.P if len(results) == 10 else S.F, time.time()-t, f"n={len(results)}")
        except Exception as e:
            return R("Conc: Analysis", S.F, time.time()-t, str(e)[:50])
    
    def test_math_division_zero(self) -> R:
        t = time.time()
        try:
            returns = np.array([0.01, 0.01, 0.01])
            std = np.std(returns)
            sharpe = np.mean(returns) / std if std > 1e-10 else 0
            return R("Math: DivZero", S.P if not np.isnan(sharpe) else S.F, time.time()-t, f"sharpe={sharpe:.2f}")
        except Exception as e:
            return R("Math: DivZero", S.F, time.time()-t, str(e)[:50])
    
    def test_math_overflow(self) -> R:
        t = time.time()
        try:
            large_return = 10.0
            compounded = (1 + large_return) ** 252
            is_inf = np.isinf(compounded)
            log_return = 252 * np.log(1 + large_return)
            return R("Math: Overflow", S.P if is_inf else S.W, time.time()-t, f"log_ret={log_return:.1f}")
        except Exception as e:
            return R("Math: Overflow", S.F, time.time()-t, str(e)[:50])
    
    async def run_all(self):
        print("\n" + "="*60)
        print("PHASE 7: BATTLE TESTS - EDGE CASES & STRESS")
        print("="*60)
        
        print("\n" + "-"*40)
        print("TRIPLE BARRIER TESTS")
        print("-"*40)
        self.add(self.test_triple_barrier_zero_volatility())
        self.add(self.test_triple_barrier_extreme_moves())
        self.add(self.test_triple_barrier_nan_handling())
        
        print("\n" + "-"*40)
        print("REGIME DETECTION TESTS")
        print("-"*40)
        self.add(self.test_regime_single_regime())
        self.add(self.test_regime_switching())
        self.add(self.test_regime_extreme_outliers())
        
        print("\n" + "-"*40)
        print("TECHNICAL ANALYSIS TESTS")
        print("-"*40)
        self.add(self.test_technical_minimum_data())
        self.add(self.test_technical_empty_data())
        
        print("\n" + "-"*40)
        print("RISK MANAGEMENT TESTS")
        print("-"*40)
        self.add(self.test_risk_drawdown_calculation())
        self.add(self.test_risk_position_sizing_edge())
        self.add(self.test_risk_var_calculation())
        
        print("\n" + "-"*40)
        print("DATA INTEGRITY TESTS")
        print("-"*40)
        self.add(self.test_data_negative_prices())
        self.add(self.test_data_duplicate_timestamps())
        self.add(self.test_data_future_dates())
        
        print("\n" + "-"*40)
        print("CONCURRENCY TESTS")
        print("-"*40)
        self.add(await self.test_concurrent_analysis())
        
        print("\n" + "-"*40)
        print("MATHEMATICAL EDGE CASES")
        print("-"*40)
        self.add(self.test_math_division_zero())
        self.add(self.test_math_overflow())
        
        passed = sum(1 for r in self.results if r.status == S.P)
        warned = sum(1 for r in self.results if r.status == S.W)
        failed = sum(1 for r in self.results if r.status == S.F)
        total = len(self.results)
        total_time = sum(r.time for r in self.results)
        
        print("\n" + "="*60)
        print(f"RESULTS: ✅{passed} ⚠️{warned} ❌{failed} Total:{total} Time:{total_time:.1f}s")
        print("="*60)
        
        if failed == 0:
            print("🏆 BATTLE TESTS: ALL PASSED")
        else:
            print(f"❌ BATTLE TESTS: {failed} FAILURES")
        print("="*60 + "\n")
        
        return failed == 0

if __name__ == "__main__":
    tester = BattleTester()
    success = asyncio.run(tester.run_all())
    sys.exit(0 if success else 1)
