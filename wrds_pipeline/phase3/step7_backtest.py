"""
PHASE 3 — STEP 7: Backtesting Engine
======================================
Academic-standard long-short portfolio backtesting.
Strategy: Long D10, Short D1, equal-weight, monthly rebalance.
Transaction costs: 10bps round-trip.
Metrics: Sharpe, max drawdown, turnover, subperiod analysis.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import subprocess
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
S3_BUCKET = "nuble-data-warehouse"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"

TRANSACTION_COST = 0.0010  # 10bps round-trip


def compute_drawdown(returns):
    """Compute maximum drawdown from return series."""
    cum = (1 + returns).cumprod()
    peak = cum.expanding().max()
    dd = (cum - peak) / peak
    return dd.min()


def compute_sharpe(returns, annual_factor=12):
    """Annualized Sharpe ratio."""
    if returns.std() == 0:
        return 0
    return returns.mean() / returns.std() * np.sqrt(annual_factor)


def compute_sortino(returns, annual_factor=12):
    """Annualized Sortino ratio (downside deviation)."""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0
    return returns.mean() / downside.std() * np.sqrt(annual_factor)


def compute_calmar(returns, annual_factor=12):
    """Calmar ratio = annualized return / |max drawdown|."""
    ann_ret = returns.mean() * annual_factor
    dd = compute_drawdown(returns)
    if dd == 0:
        return 0
    return ann_ret / abs(dd)


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 7: BACKTESTING ENGINE")
    print("=" * 70)
    start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Load predictions ─────────────────────────────────────────────────
    pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    if not os.path.exists(pred_path):
        print("  ❌ lgb_predictions.parquet not found — run Step 6 first!")
        return

    df = pd.read_parquet(pred_path)
    df["date"] = pd.to_datetime(df["date"])
    # Auto-detect target column
    target_col = None
    for candidate in ["ret_forward", "fwd_ret_1m"]:
        if candidate in df.columns:
            target_col = candidate
            break
    if target_col is None:
        print("  ❌ No forward return column in predictions!")
        return
    print(f"  Predictions: {len(df):,} rows")
    print(f"  Target column: {target_col}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # ── Monthly portfolio construction ───────────────────────────────────
    print("\n📊 CONSTRUCTING LONG-SHORT PORTFOLIOS...")

    monthly_results = []
    prev_long = set()
    prev_short = set()

    for dt, month_data in df.groupby("date"):
        if len(month_data) < 100:
            continue

        # Assign deciles
        month_data = month_data.copy()
        month_data["decile"] = pd.qcut(
            month_data["prediction"], 10, labels=False, duplicates="drop"
        )

        # Long = top decile, Short = bottom decile
        long_stocks = set(month_data[month_data["decile"] == month_data["decile"].max()]["permno"])
        short_stocks = set(month_data[month_data["decile"] == month_data["decile"].min()]["permno"])

        # Returns
        long_ret = month_data[month_data["permno"].isin(long_stocks)][target_col].mean()
        short_ret = month_data[month_data["permno"].isin(short_stocks)][target_col].mean()
        ls_ret = long_ret - short_ret

        # Turnover
        if prev_long:
            long_turnover = 1 - len(long_stocks & prev_long) / max(len(long_stocks), 1)
            short_turnover = 1 - len(short_stocks & prev_short) / max(len(short_stocks), 1)
            turnover = (long_turnover + short_turnover) / 2
        else:
            turnover = 1.0

        # Net of transaction costs
        tc = turnover * TRANSACTION_COST * 2  # Both legs
        ls_ret_net = ls_ret - tc

        monthly_results.append({
            "date": dt,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_gross": ls_ret,
            "ls_net": ls_ret_net,
            "turnover": turnover,
            "tc_cost": tc,
            "n_long": len(long_stocks),
            "n_short": len(short_stocks),
            "n_total": len(month_data),
        })

        prev_long = long_stocks
        prev_short = short_stocks

    results = pd.DataFrame(monthly_results)
    results = results.sort_values("date").reset_index(drop=True)

    if len(results) == 0:
        print("  ❌ No monthly results!")
        return

    # ── Performance Metrics ──────────────────────────────────────────────
    print("\n📊 PERFORMANCE METRICS")

    # Full period
    sharpe_gross = compute_sharpe(results["ls_gross"])
    sharpe_net = compute_sharpe(results["ls_net"])
    sortino_net = compute_sortino(results["ls_net"])
    calmar_net = compute_calmar(results["ls_net"])
    max_dd = compute_drawdown(results["ls_net"])
    ann_ret_gross = results["ls_gross"].mean() * 12 * 100
    ann_ret_net = results["ls_net"].mean() * 12 * 100
    ann_vol = results["ls_net"].std() * np.sqrt(12) * 100
    avg_turnover = results["turnover"].mean()
    hit_rate = (results["ls_gross"] > 0).mean()

    print(f"  Ann. Return (gross):  {ann_ret_gross:+.1f}%")
    print(f"  Ann. Return (net):    {ann_ret_net:+.1f}%")
    print(f"  Ann. Volatility:      {ann_vol:.1f}%")
    print(f"  Sharpe (gross):       {sharpe_gross:.2f}")
    print(f"  Sharpe (net):         {sharpe_net:.2f}")
    print(f"  Sortino (net):        {sortino_net:.2f}")
    print(f"  Calmar (net):         {calmar_net:.2f}")
    print(f"  Max Drawdown:         {max_dd*100:.1f}%")
    print(f"  Avg Monthly Turnover: {avg_turnover*100:.0f}%")
    print(f"  Hit Rate:             {hit_rate*100:.0f}%")

    # ── Subperiod Analysis ───────────────────────────────────────────────
    print("\n📊 SUBPERIOD ANALYSIS")
    results["year"] = results["date"].dt.year

    subperiods = [
        ("2000-2004", 2000, 2004),
        ("2005-2007", 2005, 2007),
        ("2008-2009 (GFC)", 2008, 2009),
        ("2010-2015", 2010, 2015),
        ("2016-2019", 2016, 2019),
        ("2020-2021 (COVID)", 2020, 2021),
        ("2022-2024", 2022, 2024),
    ]

    subperiod_data = []
    for name, y1, y2 in subperiods:
        mask = (results["year"] >= y1) & (results["year"] <= y2)
        sub = results[mask]
        if len(sub) > 3:
            sp_sharpe = compute_sharpe(sub["ls_net"])
            sp_ret = sub["ls_net"].mean() * 12 * 100
            sp_dd = compute_drawdown(sub["ls_net"])
            print(f"  {name:<25} Sharpe={sp_sharpe:>5.2f}  Ret={sp_ret:>+6.1f}%  DD={sp_dd*100:>5.1f}%  N={len(sub)}")
            subperiod_data.append({
                "period": name, "sharpe": sp_sharpe,
                "ann_return_pct": sp_ret, "max_dd_pct": sp_dd * 100,
                "months": len(sub),
            })

    # ── Decile Analysis ──────────────────────────────────────────────────
    print("\n📊 DECILE MONOTONICITY")
    decile_rets = []
    for dt, month_data in df.groupby("date"):
        if len(month_data) < 100:
            continue
        month_data = month_data.copy()
        month_data["decile"] = pd.qcut(
            month_data["prediction"], 10, labels=range(1, 11), duplicates="drop"
        )
        for d in range(1, 11):
            d_data = month_data[month_data["decile"] == d]
            if len(d_data) > 0:
                decile_rets.append({"decile": d, "ret": d_data[target_col].mean()})

    if decile_rets:
        dec_df = pd.DataFrame(decile_rets)
        dec_avg = dec_df.groupby("decile")["ret"].mean() * 100
        print("  Decile → Avg Monthly Return (%):")
        for d in range(1, 11):
            if d in dec_avg.index:
                bar = "█" * int(max(0, (dec_avg[d] + 1) * 10))
                print(f"    D{d:>2}: {dec_avg[d]:>+.2f}%  {bar}")

    # ── Save Results ─────────────────────────────────────────────────────
    print("\n💾 SAVING BACKTEST RESULTS...")

    results.to_csv(os.path.join(RESULTS_DIR, "backtest_monthly.csv"), index=False)
    pd.DataFrame(subperiod_data).to_csv(os.path.join(RESULTS_DIR, "backtest_subperiods.csv"), index=False)

    backtest_summary = {
        "strategy": "Long D10 / Short D1, equal-weight",
        "rebalance": "monthly",
        "transaction_cost": f"{TRANSACTION_COST*10000:.0f} bps",
        "period": f"{results['date'].min().date()} to {results['date'].max().date()}",
        "months": len(results),
        "ann_return_gross_pct": round(ann_ret_gross, 1),
        "ann_return_net_pct": round(ann_ret_net, 1),
        "ann_volatility_pct": round(ann_vol, 1),
        "sharpe_gross": round(sharpe_gross, 2),
        "sharpe_net": round(sharpe_net, 2),
        "sortino_net": round(sortino_net, 2),
        "calmar_net": round(calmar_net, 2),
        "max_drawdown_pct": round(max_dd * 100, 1),
        "avg_turnover_pct": round(avg_turnover * 100, 0),
        "hit_rate_pct": round(hit_rate * 100, 0),
        "subperiods": subperiod_data,
    }
    with open(os.path.join(RESULTS_DIR, "backtest_summary.json"), "w") as f:
        json.dump(backtest_summary, f, indent=2)

    # S3 upload
    for fname in ["backtest_monthly.csv", "backtest_subperiods.csv", "backtest_summary.json"]:
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            subprocess.run(
                ["aws", "s3", "cp", fpath,
                 f"s3://{S3_BUCKET}/results/{fname}"],
                capture_output=True,
            )

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"BACKTESTING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Sharpe (net): {sharpe_net:.2f}")
    print(f"  Ann Return:   {ann_ret_net:+.1f}%")
    print(f"  Max DD:       {max_dd*100:.1f}%")
    print(f"  Time:         {elapsed:.0f}s")
    print(f"  ✅ All saved and uploaded to S3")


if __name__ == "__main__":
    main()
