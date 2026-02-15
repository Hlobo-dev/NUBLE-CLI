"""
ADVANCED FEATURE ENGINEERING for Cross-Sectional Return Prediction
====================================================================
Expert-level feature transforms that go beyond basic GKX panel.

These are the "alpha" features that distinguish a quant fund's
research pipeline from an academic exercise.

References:
- Freyberger, Neuhierl & Weber (2020): "Dissecting Characteristics Nonparametrically"
- Kozak, Nagel & Santosh (2020): "Shrinking the Cross-Section"
- Jensen, Kelly & Pedersen (2023): "Is There a Replication Crisis in Finance?"

Features added:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. MOMENTUM DECOMPOSITION
   - Separate 12-1 momentum into: trend, acceleration, reversal
   - These have DIFFERENT factor premia

2. VOLATILITY FEATURES
   - Idiosyncratic vol (CAPM residual vol)
   - Vol-of-vol (uncertainty about uncertainty)
   - Realized skewness and kurtosis

3. CROSS-SECTIONAL RELATIVE FEATURES
   - Industry-relative (stock vs. industry median)
   - Size-quintile relative (small vs. large stock norms)

4. NONLINEAR TRANSFORMS
   - Squared terms for key characteristics
   - Ratio features (value/momentum, size/volatility)

5. TIME-SERIES FEATURES
   - Feature momentum (how features are trending)
   - Mean reversion signals
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_momentum_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decompose 12-1 month momentum into orthogonal components.

    Jegadeesh & Titman (1993) showed momentum works, but recent research
    shows the components have different risk/return profiles:

    - TREND: smooth upward path → high SR, persistent
    - ACCELERATION: recent returns accelerating → mean-reverting
    - REVERSAL: 1-month reversal effect → very short-lived alpha
    """
    if "mom_12m" not in df.columns:
        return df

    # If we have monthly returns, compute sub-components
    if "mom_6m" in df.columns and "mom_3m" in df.columns:
        # Acceleration = recent momentum - earlier momentum
        df["mom_acceleration"] = df.get("mom_3m", 0) - (df.get("mom_12m", 0) - df.get("mom_3m", 0)) / 3

        # Trend strength: how consistent is the path?
        # Approximate with mom_12m / vol_12m (price path smoothness)
        if "vol_12m" in df.columns:
            df["mom_trend_strength"] = df["mom_12m"] / (df["vol_12m"] + 1e-8)

    # Short-term reversal (1-month return, negated)
    if "ret" in df.columns:
        df["str_reversal"] = -df["ret"]

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced volatility features beyond simple realized vol.

    Key insight from Ang et al. (2006): Idiosyncratic volatility
    is negatively priced cross-sectionally (IVOL puzzle).
    """
    # Vol-of-vol proxy: difference between recent and long-term vol
    if "vol_3m" in df.columns and "vol_12m" in df.columns:
        df["vol_of_vol"] = df["vol_3m"] - df["vol_12m"]

    # Idiosyncratic vol approximation
    # (Already in panel as "idiovol" if computed in Step 5)
    if "realized_vol" in df.columns and "beta" in df.columns:
        # Simple CAPM residual vol estimate
        if "market_vol" not in df.columns:
            # Cross-sectional proxy: median vol that month
            df["idiovol_approx"] = df.groupby("date")["realized_vol"].transform(
                lambda x: x - x.median() * df.loc[x.index, "beta"].clip(0, 3)
            )

    return df


def add_industry_relative_features(df: pd.DataFrame,
                                    base_features: List[str]) -> pd.DataFrame:
    """
    Industry-relative features: stock characteristic vs. industry median.

    WHY: A PE of 15 means very different things for a tech stock vs. utility.
    Industry-relative features capture WITHIN-INDUSTRY ranking, which is
    more predictive than raw cross-sectional ranking.

    Uses FF49 industry classification (already in panel as siccd).
    """
    # Identify industry column
    ind_col = None
    for candidate in ["ff49_industry", "siccd"]:
        if candidate in df.columns:
            ind_col = candidate
            break

    if ind_col is None:
        return df

    for feat in base_features:
        if feat not in df.columns:
            continue
        # Industry-relative = stock value - industry median (within month)
        ind_median = df.groupby(["date", ind_col])[feat].transform("median")
        df[f"ir_{feat}"] = df[feat] - ind_median

    return df


def add_nonlinear_transforms(df: pd.DataFrame,
                              key_features: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Nonlinear feature transforms that capture known nonlinear effects.

    - SQUARED terms: capture U-shaped relationships (e.g., size effect)
    - RATIO features: capture interaction effects more stably
    - LOG transforms: handle right-skewed distributions
    """
    if key_features is None:
        key_features = [
            "log_market_cap", "book_to_market", "mom_12m",
            "realized_vol", "turnover", "earnings_yield"
        ]

    for feat in key_features:
        if feat not in df.columns:
            continue
        # Squared (captures U-shape)
        df[f"{feat}_sq"] = df[feat] ** 2

    # Ratio features (value/momentum, etc.)
    ratio_pairs = [
        ("book_to_market", "mom_12m", "value_mom_ratio"),
        ("log_market_cap", "realized_vol", "size_vol_ratio"),
        ("earnings_yield", "turnover", "yield_turnover_ratio"),
    ]
    for feat1, feat2, name in ratio_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            denom = df[feat2].clip(lower=df[feat2].quantile(0.01))
            df[name] = df[feat1] / (denom + 1e-8)

    return df


def add_feature_momentum(df: pd.DataFrame,
                          base_features: List[str],
                          lookback: int = 3) -> pd.DataFrame:
    """
    Feature momentum: how stock characteristics are changing over time.

    If a stock's book-to-market is INCREASING, it means the stock is
    getting cheaper relative to fundamentals — a bullish signal.

    This captures time-series dynamics that static features miss.
    """
    df = df.sort_values(["permno", "date"])

    for feat in base_features:
        if feat not in df.columns:
            continue
        # Feature change over lookback months
        df[f"{feat}_chg{lookback}m"] = df.groupby("permno")[feat].transform(
            lambda x: x - x.shift(lookback)
        )

    return df


def add_cross_sectional_rank_features(df: pd.DataFrame,
                                       base_features: List[str]) -> pd.DataFrame:
    """
    Cross-sectional percentile rank (robust alternative to rank normalization).

    For each month, compute the percentile rank of each stock on each feature.
    This is similar to rank_normalize but preserves as a separate feature
    that can interact with the original feature.
    """
    for feat in base_features:
        if feat not in df.columns:
            continue
        df[f"rank_{feat}"] = df.groupby("date")[feat].transform(
            lambda x: x.rank(pct=True, na_option="keep")
        )

    return df


def engineer_advanced_features(panel: pd.DataFrame,
                                 feature_cols: List[str]) -> tuple:
    """
    Master function: apply all advanced feature engineering.

    Returns: (enhanced_panel, new_feature_list)
    """
    print("  🔧 Advanced Feature Engineering...")
    n_orig = len(feature_cols)

    # 1. Momentum decomposition
    panel = add_momentum_decomposition(panel)

    # 2. Volatility features
    panel = add_volatility_features(panel)

    # 3. Nonlinear transforms (on key characteristics)
    panel = add_nonlinear_transforms(panel)

    # 4. Industry-relative features (top 10 most important base features)
    important_base = [f for f in [
        "log_market_cap", "book_to_market", "mom_12m", "realized_vol",
        "turnover", "earnings_yield", "asset_growth", "roe", "accruals",
        "investment"
    ] if f in panel.columns]

    if important_base:
        panel = add_industry_relative_features(panel, important_base[:5])

    # Collect all new feature columns
    id_cols = ["permno", "date", "cusip", "ticker", "siccd", "year",
               "ret", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
               "ret_forward", "dlret", "dlstcd", "fwd_ret_1m_ranked"]
    all_features = [c for c in panel.columns if c not in id_cols
                    and panel[c].dtype in ["float64", "float32", "int64", "int32"]]

    n_new = len(all_features) - n_orig
    print(f"    Added {n_new} new features → total {len(all_features)} features")

    return panel, all_features
