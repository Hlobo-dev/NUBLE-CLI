"""
Alpha Attribution Analysis

Critical Question: Is our return ALPHA or just BETA exposure?

Key Analyses:
1. Regression-based alpha (vs SPY, QQQ)
2. Factor exposure (Fama-French: MKT, SMB, HML, MOM)
3. Contribution analysis (is one symbol driving everything?)
4. Conditional alpha (alpha in different regimes)

If alpha is just beta, the strategy provides no value.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


@dataclass
class AlphaResult:
    """Alpha attribution result."""
    # Regression results
    alpha_annualized: float
    alpha_t_stat: float
    beta: float
    r_squared: float
    
    # Factor exposures
    factor_betas: Dict[str, float] = None
    factor_t_stats: Dict[str, float] = None
    
    # Contribution
    top_contributor: str = None
    top_contributor_pct: float = 0.0
    concentration_risk: bool = False
    
    # Regime alpha
    alpha_bull: float = np.nan
    alpha_bear: float = np.nan
    alpha_sideways: float = np.nan
    
    # Verdict
    has_alpha: bool = False
    confidence: str = "Low"
    summary: str = ""


class AlphaAttribution:
    """
    Comprehensive alpha attribution analysis.
    
    Answers: Is our outperformance skill or just risk exposure?
    """
    
    def __init__(self):
        self.benchmark_data = {}
        self.strategy_returns = None
        self.results: Dict[str, AlphaResult] = {}
    
    def calculate_alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        benchmark_name: str = "SPY"
    ) -> AlphaResult:
        """
        Calculate regression-based alpha and beta.
        
        Alpha = E[R_strategy] - beta * E[R_benchmark]
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily strategy returns
        benchmark_returns : pd.Series  
            Daily benchmark returns (SPY, QQQ)
        benchmark_name : str
            Name of benchmark for labeling
            
        Returns:
        --------
        AlphaResult with alpha, beta, t-stats
        """
        # Align dates
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        aligned.columns = ['strategy', 'benchmark']
        aligned = aligned.dropna()
        
        if len(aligned) < 30:
            return AlphaResult(
                alpha_annualized=np.nan,
                alpha_t_stat=np.nan,
                beta=np.nan,
                r_squared=np.nan,
                summary="Insufficient data for regression"
            )
        
        y = aligned['strategy'].values
        X = aligned['benchmark'].values
        
        # OLS regression: strategy = alpha + beta * benchmark + epsilon
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            # Solve OLS
            beta_hat = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha_daily = beta_hat[0]
            beta = beta_hat[1]
            
            # Residuals and R-squared
            y_pred = X_with_const @ beta_hat
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            # Standard errors (HAC would be better but this is simpler)
            n = len(y)
            k = 2  # intercept + beta
            mse = ss_res / (n - k)
            XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
            se = np.sqrt(mse * np.diag(XtX_inv))
            
            alpha_t_stat = alpha_daily / se[0] if se[0] > 0 else 0
            
            # Annualize alpha
            alpha_annualized = alpha_daily * 252
            
        except Exception as e:
            return AlphaResult(
                alpha_annualized=np.nan,
                alpha_t_stat=np.nan,
                beta=np.nan,
                r_squared=np.nan,
                summary=f"Regression failed: {str(e)}"
            )
        
        # Verdict
        has_alpha = alpha_annualized > 0.01 and alpha_t_stat > 2.0  # 1%+ alpha, t > 2
        
        if has_alpha:
            confidence = "High" if alpha_t_stat > 3.0 else "Medium"
            summary = f"✅ Significant alpha of {alpha_annualized:.1%} (t={alpha_t_stat:.2f}) vs {benchmark_name}"
        else:
            confidence = "Low"
            if alpha_annualized <= 0:
                summary = f"❌ Negative alpha of {alpha_annualized:.1%} vs {benchmark_name}"
            elif alpha_t_stat <= 2.0:
                summary = f"⚠️ Alpha of {alpha_annualized:.1%} not statistically significant (t={alpha_t_stat:.2f})"
            else:
                summary = f"⚠️ Alpha present but below threshold"
        
        return AlphaResult(
            alpha_annualized=alpha_annualized,
            alpha_t_stat=alpha_t_stat,
            beta=beta,
            r_squared=r_squared,
            has_alpha=has_alpha,
            confidence=confidence,
            summary=summary
        )
    
    def multi_factor_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: Dict[str, pd.Series]
    ) -> AlphaResult:
        """
        Fama-French style multi-factor attribution.
        
        strategy = alpha + beta_mkt*MKT + beta_smb*SMB + beta_hml*HML + beta_mom*MOM + epsilon
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily strategy returns
        factor_returns : Dict[str, pd.Series]
            Dictionary of factor returns (MKT, SMB, HML, MOM, etc.)
            
        Returns:
        --------
        AlphaResult with factor betas and residual alpha
        """
        # Align all series
        all_series = [strategy_returns] + list(factor_returns.values())
        aligned = pd.concat(all_series, axis=1, join='inner').dropna()
        
        if len(aligned) < 60:  # Need more data for multi-factor
            return AlphaResult(
                alpha_annualized=np.nan,
                alpha_t_stat=np.nan,
                beta=np.nan,
                r_squared=np.nan,
                summary="Insufficient data for multi-factor analysis"
            )
        
        y = aligned.iloc[:, 0].values  # Strategy
        X = aligned.iloc[:, 1:].values  # Factors
        factor_names = list(factor_returns.keys())
        
        # Add constant for alpha
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        try:
            # OLS
            beta_hat = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            alpha_daily = beta_hat[0]
            factor_betas = dict(zip(factor_names, beta_hat[1:]))
            
            # R-squared
            y_pred = X_with_const @ beta_hat
            residuals = y - y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - y.mean())**2)
            r_squared = 1 - ss_res/ss_tot if ss_tot > 0 else 0
            
            # Standard errors
            n = len(y)
            k = len(beta_hat)
            mse = ss_res / (n - k)
            XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
            se = np.sqrt(mse * np.diag(XtX_inv))
            
            alpha_t_stat = alpha_daily / se[0] if se[0] > 0 else 0
            factor_t_stats = dict(zip(factor_names, beta_hat[1:] / se[1:]))
            
            # Annualize
            alpha_annualized = alpha_daily * 252
            
        except Exception as e:
            return AlphaResult(
                alpha_annualized=np.nan,
                alpha_t_stat=np.nan,
                beta=np.nan,
                r_squared=np.nan,
                summary=f"Multi-factor regression failed: {str(e)}"
            )
        
        # Determine if alpha is real
        has_alpha = alpha_annualized > 0.01 and alpha_t_stat > 2.0
        
        # Find dominant factor
        abs_t_stats = {k: abs(v) for k, v in factor_t_stats.items()}
        dominant_factor = max(abs_t_stats, key=abs_t_stats.get)
        
        if has_alpha:
            confidence = "High" if alpha_t_stat > 3.0 else "Medium"
            summary = f"✅ Multi-factor alpha of {alpha_annualized:.1%} (t={alpha_t_stat:.2f})"
        else:
            confidence = "Low"
            summary = f"⚠️ Returns explained by {dominant_factor} (beta={factor_betas[dominant_factor]:.2f}, t={factor_t_stats[dominant_factor]:.2f})"
        
        return AlphaResult(
            alpha_annualized=alpha_annualized,
            alpha_t_stat=alpha_t_stat,
            beta=factor_betas.get('MKT', np.nan),
            r_squared=r_squared,
            factor_betas=factor_betas,
            factor_t_stats=factor_t_stats,
            has_alpha=has_alpha,
            confidence=confidence,
            summary=summary
        )
    
    def contribution_analysis(
        self,
        portfolio_returns: pd.Series,
        position_returns: Dict[str, pd.Series],
        weights: Dict[str, float] = None
    ) -> Dict:
        """
        Analyze return contribution by position.
        
        Key question: Is one symbol (e.g., NVDA) driving all returns?
        
        Parameters:
        -----------
        portfolio_returns : pd.Series
            Total portfolio returns
        position_returns : Dict[str, pd.Series]
            Returns for each position
        weights : Dict[str, float]
            Average weights (optional, assume equal if not provided)
            
        Returns:
        --------
        Dict with contribution breakdown
        """
        if weights is None:
            weights = {symbol: 1.0/len(position_returns) for symbol in position_returns}
        
        contributions = {}
        
        for symbol, rets in position_returns.items():
            if symbol not in weights:
                continue
            
            weight = weights[symbol]
            
            # Align with portfolio
            aligned = pd.concat([rets, portfolio_returns], axis=1, join='inner').dropna()
            if len(aligned) < 10:
                continue
            
            pos_ret = aligned.iloc[:, 0]
            
            # Contribution = weight * return
            contribution = (pos_ret * weight).sum()
            
            # Share of total (normalized)
            contributions[symbol] = {
                'return': pos_ret.sum(),
                'weighted_return': contribution,
                'weight': weight,
                'volatility': pos_ret.std() * np.sqrt(252),
                'sharpe': pos_ret.mean() / pos_ret.std() * np.sqrt(252) if pos_ret.std() > 0 else 0
            }
        
        # Calculate total and normalize
        total_contribution = sum(c['weighted_return'] for c in contributions.values())
        
        for symbol in contributions:
            contributions[symbol]['pct_of_total'] = (
                contributions[symbol]['weighted_return'] / total_contribution 
                if total_contribution != 0 else 0
            )
        
        # Sort by contribution
        sorted_contributions = dict(sorted(
            contributions.items(), 
            key=lambda x: abs(x[1]['weighted_return']), 
            reverse=True
        ))
        
        # Check concentration risk
        top_contributor = list(sorted_contributions.keys())[0] if sorted_contributions else None
        top_pct = abs(sorted_contributions[top_contributor]['pct_of_total']) if top_contributor else 0
        
        concentration_risk = top_pct > 0.5  # One position > 50% of returns
        
        return {
            'contributions': sorted_contributions,
            'top_contributor': top_contributor,
            'top_contributor_pct': top_pct,
            'concentration_risk': concentration_risk,
            'summary': (
                f"⚠️ CONCENTRATION RISK: {top_contributor} contributes {top_pct:.0%} of returns"
                if concentration_risk
                else f"Diversified: top contributor ({top_contributor}) is {top_pct:.0%}"
            )
        }
    
    def regime_alpha(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        regime_threshold: float = 0.02
    ) -> Dict:
        """
        Calculate alpha in different market regimes.
        
        Regimes:
        - Bull: benchmark return > +threshold (annualized ~50%+)
        - Bear: benchmark return < -threshold (annualized ~50%-) 
        - Sideways: in between
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily strategy returns
        benchmark_returns : pd.Series
            Daily benchmark returns
        regime_threshold : float
            Daily threshold for regime classification (0.02 = ~2%)
            
        Returns:
        --------
        Dict with regime-specific alphas
        """
        aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner').dropna()
        aligned.columns = ['strategy', 'benchmark']
        
        # Classify regimes
        aligned['regime'] = 'sideways'
        aligned.loc[aligned['benchmark'] > regime_threshold/np.sqrt(252), 'regime'] = 'bull'
        aligned.loc[aligned['benchmark'] < -regime_threshold/np.sqrt(252), 'regime'] = 'bear'
        
        results = {}
        
        for regime in ['bull', 'bear', 'sideways']:
            regime_data = aligned[aligned['regime'] == regime]
            
            if len(regime_data) < 20:
                results[regime] = {
                    'alpha': np.nan,
                    'n_days': len(regime_data),
                    'summary': 'Insufficient data'
                }
                continue
            
            # Calculate alpha in this regime
            strat_mean = regime_data['strategy'].mean() * 252
            bench_mean = regime_data['benchmark'].mean() * 252
            alpha = strat_mean - bench_mean
            
            # Outperformance frequency
            outperform_pct = (regime_data['strategy'] > regime_data['benchmark']).mean()
            
            results[regime] = {
                'alpha': alpha,
                'strategy_return': strat_mean,
                'benchmark_return': bench_mean,
                'n_days': len(regime_data),
                'pct_of_total': len(regime_data) / len(aligned),
                'outperform_frequency': outperform_pct,
                'summary': f"Alpha={alpha:.1%}, outperform {outperform_pct:.0%} of days"
            }
        
        # Check for regime-specific skill
        bull_alpha = results['bull']['alpha'] if not np.isnan(results.get('bull', {}).get('alpha', np.nan)) else 0
        bear_alpha = results['bear']['alpha'] if not np.isnan(results.get('bear', {}).get('alpha', np.nan)) else 0
        
        if bear_alpha > bull_alpha and bear_alpha > 0.05:
            skill_summary = "💡 Strategy excels in BEAR markets (defensive)"
        elif bull_alpha > bear_alpha and bull_alpha > 0.05:
            skill_summary = "💡 Strategy excels in BULL markets (momentum)"
        elif bull_alpha > 0 and bear_alpha > 0:
            skill_summary = "✅ Strategy generates alpha in both regimes (all-weather)"
        else:
            skill_summary = "⚠️ No consistent alpha across regimes"
        
        results['skill_summary'] = skill_summary
        
        return results
    
    def full_attribution_report(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        position_returns: Dict[str, pd.Series] = None,
        factor_returns: Dict[str, pd.Series] = None
    ) -> Dict:
        """
        Generate comprehensive alpha attribution report.
        
        Parameters:
        -----------
        strategy_returns : pd.Series
            Daily portfolio returns
        benchmark_returns : pd.Series
            Daily benchmark returns (SPY or similar)
        position_returns : Dict[str, pd.Series], optional
            Returns by position for contribution analysis
        factor_returns : Dict[str, pd.Series], optional
            Factor returns for multi-factor analysis
            
        Returns:
        --------
        Complete attribution report
        """
        report = {}
        
        print("="*70)
        print("ALPHA ATTRIBUTION REPORT")
        print("="*70)
        
        # 1. Simple regression vs benchmark
        print("\n📊 1. CAPM ALPHA (vs Benchmark)")
        print("-"*50)
        capm_result = self.calculate_alpha_beta(strategy_returns, benchmark_returns)
        report['capm'] = capm_result
        
        print(f"   Alpha (annualized): {capm_result.alpha_annualized:.2%}")
        print(f"   Alpha t-stat: {capm_result.alpha_t_stat:.2f}")
        print(f"   Beta: {capm_result.beta:.2f}")
        print(f"   R-squared: {capm_result.r_squared:.2%}")
        print(f"   Verdict: {capm_result.summary}")
        
        # 2. Multi-factor if available
        if factor_returns:
            print("\n📊 2. MULTI-FACTOR ALPHA")
            print("-"*50)
            factor_result = self.multi_factor_attribution(strategy_returns, factor_returns)
            report['factor'] = factor_result
            
            print(f"   Alpha (annualized): {factor_result.alpha_annualized:.2%}")
            print(f"   Alpha t-stat: {factor_result.alpha_t_stat:.2f}")
            print(f"   R-squared: {factor_result.r_squared:.2%}")
            print("   Factor exposures:")
            if factor_result.factor_betas:
                for factor, beta in factor_result.factor_betas.items():
                    t_stat = factor_result.factor_t_stats.get(factor, np.nan)
                    sig = "***" if abs(t_stat) > 3 else "**" if abs(t_stat) > 2 else "*" if abs(t_stat) > 1.5 else ""
                    print(f"     {factor}: {beta:.3f} (t={t_stat:.2f}){sig}")
            print(f"   Verdict: {factor_result.summary}")
        
        # 3. Contribution analysis if available
        if position_returns:
            print("\n📊 3. RETURN CONTRIBUTION")
            print("-"*50)
            contribution = self.contribution_analysis(strategy_returns, position_returns)
            report['contribution'] = contribution
            
            print(f"   Top contributor: {contribution['top_contributor']} ({contribution['top_contributor_pct']:.0%})")
            print(f"   Concentration risk: {'YES ⚠️' if contribution['concentration_risk'] else 'No ✅'}")
            print("   Breakdown:")
            for symbol, data in list(contribution['contributions'].items())[:5]:
                print(f"     {symbol}: {data['pct_of_total']:.0%} of returns (Sharpe={data['sharpe']:.2f})")
        
        # 4. Regime analysis
        print("\n📊 4. REGIME ANALYSIS")
        print("-"*50)
        regime = self.regime_alpha(strategy_returns, benchmark_returns)
        report['regime'] = regime
        
        for r in ['bull', 'bear', 'sideways']:
            if r in regime and not np.isnan(regime[r].get('alpha', np.nan)):
                alpha = regime[r]['alpha']
                n_days = regime[r]['n_days']
                pct = regime[r]['pct_of_total']
                emoji = "📈" if r == "bull" else "📉" if r == "bear" else "➡️"
                print(f"   {emoji} {r.upper()}: Alpha={alpha:.1%} ({n_days} days, {pct:.0%} of period)")
        
        print(f"\n   {regime.get('skill_summary', '')}")
        
        # 5. Final verdict
        print("\n" + "="*70)
        print("FINAL VERDICT")
        print("="*70)
        
        has_alpha = capm_result.has_alpha
        is_concentrated = position_returns and report.get('contribution', {}).get('concentration_risk', False)
        
        if has_alpha and not is_concentrated:
            print("✅ ALPHA IS REAL")
            print(f"   Annualized alpha: {capm_result.alpha_annualized:.1%}")
            print(f"   Statistical significance: t={capm_result.alpha_t_stat:.2f} (>2 is significant)")
            print(f"   Confidence: {capm_result.confidence}")
            report['verdict'] = 'ALPHA'
        elif has_alpha and is_concentrated:
            print("⚠️ ALPHA EXISTS BUT CONCENTRATION RISK")
            print(f"   Alpha is real ({capm_result.alpha_annualized:.1%})")
            print(f"   But {report['contribution']['top_contributor']} drives {report['contribution']['top_contributor_pct']:.0%} of returns")
            print("   Risk: If this one position fails, so does the strategy")
            report['verdict'] = 'CONCENTRATED_ALPHA'
        else:
            print("❌ RETURNS ARE LIKELY BETA, NOT ALPHA")
            print(f"   Beta to benchmark: {capm_result.beta:.2f}")
            print(f"   R-squared: {capm_result.r_squared:.1%}")
            print("   This strategy may just be leveraged market exposure")
            report['verdict'] = 'BETA'
        
        print("="*70 + "\n")
        
        return report


def demo():
    """Demo the alpha attribution system."""
    print("Alpha Attribution Demo")
    print("="*60)
    
    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-12-31', freq='B')
    n = len(dates)
    
    # Benchmark (SPY-like)
    benchmark = pd.Series(
        np.random.normal(0.0003, 0.01, n),  # ~8% annual return, 16% vol
        index=dates
    )
    
    # Strategy with some alpha
    alpha_daily = 0.0002  # ~5% annual alpha
    beta = 0.8
    strategy = alpha_daily + beta * benchmark + pd.Series(
        np.random.normal(0, 0.005, n),  # Idiosyncratic risk
        index=dates
    )
    
    # Position returns (simulate 5 positions)
    position_returns = {}
    for symbol in ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META']:
        specific_beta = beta + np.random.uniform(-0.2, 0.2)
        specific_alpha = alpha_daily * np.random.uniform(0.5, 1.5)
        position_returns[symbol] = specific_alpha + specific_beta * benchmark + pd.Series(
            np.random.normal(0, 0.015, n),
            index=dates
        )
    
    # Create mock factors
    factor_returns = {
        'MKT': benchmark,
        'SMB': pd.Series(np.random.normal(0.0001, 0.005, n), index=dates),
        'HML': pd.Series(np.random.normal(0.0001, 0.004, n), index=dates),
        'MOM': pd.Series(np.random.normal(0.0002, 0.006, n), index=dates),
    }
    
    # Run attribution
    attribution = AlphaAttribution()
    report = attribution.full_attribution_report(
        strategy_returns=strategy,
        benchmark_returns=benchmark,
        position_returns=position_returns,
        factor_returns=factor_returns
    )
    
    return report


if __name__ == "__main__":
    demo()
