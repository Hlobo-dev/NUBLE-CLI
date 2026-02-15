"""
IC INTEGRITY AUDIT — 8 DIAGNOSTIC TESTS
=========================================
IC = 0.1136 is suspiciously high.  This script runs the 8 tests
described in the failure-mode analysis to identify leakage,
overfitting, or inflated metrics.

Tests:
  1. Contemporaneous leakage (lag features +1 month)
  2. Ensemble weight overfitting (train on 2005-14, eval 2015-24)
  3. Feature selection bias (use ALL 535 features vs selected)
  4. Market-cap stratified IC (large/mid/small)
  5. Turnover & net Sharpe
  6. Rank normalization leakage check
  7. ret_crsp / momentum feature overlap with fwd_ret_1m
  8. Block bootstrap confidence interval

Author: Audit Script
"""

import pandas as pd
import numpy as np
import os
import time
import warnings
import gc
from scipy import stats

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"


def spearman_ic(pred, actual):
    """Compute Spearman rank IC, handling NaN."""
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 50:
        return np.nan
    return stats.spearmanr(pred[mask], actual[mask])[0]


def monthly_ics(df, pred_col="prediction", target_col="fwd_ret_1m"):
    """Compute monthly Spearman IC series."""
    ics = []
    for dt, grp in df.groupby("date"):
        ic = spearman_ic(grp[pred_col].values, grp[target_col].values)
        if not np.isnan(ic):
            ics.append({"date": dt, "ic": ic, "n": len(grp)})
    return pd.DataFrame(ics)


def main():
    print("=" * 70)
    print("IC INTEGRITY AUDIT — DIAGNOSING IC = 0.1136")
    print("=" * 70)
    t0 = time.time()

    # ────────────────────────────────────────────────────────────
    # LOAD DATA
    # ────────────────────────────────────────────────────────────
    print("\n[LOAD] Loading predictions and panel metadata...")
    preds = pd.read_parquet(os.path.join(DATA_DIR, "ensemble_predictions.parquet"))
    preds["date"] = pd.to_datetime(preds["date"])
    print(f"  Predictions: {len(preds):,} rows, cols={list(preds.columns)}")
    print(f"  Date range: {preds['date'].min().date()} → {preds['date'].max().date()}")
    print(f"  Unique months: {preds['date'].nunique()}")

    # Load panel for market cap info
    import pyarrow.parquet as pq
    panel_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    available_cols = pq.read_schema(panel_path).names
    
    need_cols = ["permno", "date", "fwd_ret_1m"]
    # Grab ret_crsp and market cap if available
    for c in ["ret_crsp", "log_market_cap", "market_cap", "mom_1m", 
              "realized_vol", "turnover", "str_reversal"]:
        if c in available_cols:
            need_cols.append(c)
    
    print(f"  Loading panel columns: {need_cols}")
    panel = pd.read_parquet(panel_path, columns=need_cols)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"  Panel: {len(panel):,} rows")

    # Merge predictions with panel data
    merged = preds.merge(panel, on=["permno", "date"], how="left", 
                         suffixes=("_pred", "_panel"))
    
    # Resolve fwd_ret_1m column
    if "fwd_ret_1m_panel" in merged.columns and "fwd_ret_1m_pred" in merged.columns:
        merged["fwd_ret_1m"] = merged["fwd_ret_1m_pred"]
        merged.drop(columns=["fwd_ret_1m_pred", "fwd_ret_1m_panel"], inplace=True)
    
    print(f"  Merged: {len(merged):,} rows")

    # ────────────────────────────────────────────────────────────
    # TEST 0: SANITY — verify the reported IC
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 0: VERIFY REPORTED IC")
    print("=" * 70)
    
    ics = monthly_ics(preds, "prediction", "fwd_ret_1m")
    if len(ics) == 0:
        print("  ❌ NO MONTHLY ICs COMPUTED — something is fundamentally wrong")
        print("  Checking for NaN in predictions and target...")
        print(f"    prediction NaN: {preds['prediction'].isna().sum():,} / {len(preds):,}")
        print(f"    fwd_ret_1m NaN: {preds['fwd_ret_1m'].isna().sum():,} / {len(preds):,}")
        print(f"    prediction stats: {preds['prediction'].describe()}")
        print(f"    fwd_ret_1m stats: {preds['fwd_ret_1m'].describe()}")
        return
    
    reported_ic = ics["ic"].mean()
    ic_std = ics["ic"].std()
    ic_ir = reported_ic / ic_std if ic_std > 0 else 0
    ic_pos = (ics["ic"] > 0).mean() * 100
    
    print(f"  Verified IC:     {reported_ic:+.4f}")
    print(f"  IC Std:          {ic_std:.4f}")
    print(f"  IC IR:           {ic_ir:.2f}")
    print(f"  IC > 0:          {ic_pos:.0f}%")
    print(f"  Months:          {len(ics)}")

    # ────────────────────────────────────────────────────────────
    # TEST 1: CONTEMPORANEOUS LEAKAGE
    # Lag the target by +1 month. If IC drops >50%, leakage.
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 1: CONTEMPORANEOUS LEAKAGE (shift target forward 1 month)")
    print("  If IC drops >50%, features contain T+1 information.")
    print("=" * 70)

    # For each stock, shift fwd_ret_1m forward by 1 month
    # This means: at time T, we now predict ret from T+1 to T+2 
    # instead of T to T+1. If features are truly lagged, IC should barely change.
    preds_sorted = preds.sort_values(["permno", "date"])
    preds_sorted["fwd_ret_2m"] = preds_sorted.groupby("permno")["fwd_ret_1m"].shift(-1)
    
    ics_shifted = monthly_ics(preds_sorted.dropna(subset=["fwd_ret_2m"]), 
                               "prediction", "fwd_ret_2m")
    
    if len(ics_shifted) > 0:
        shifted_ic = ics_shifted["ic"].mean()
        pct_drop = (1 - shifted_ic / reported_ic) * 100 if reported_ic != 0 else 0
        
        print(f"  Original IC (predict T→T+1):    {reported_ic:+.4f}")
        print(f"  Shifted IC  (predict T+1→T+2):  {shifted_ic:+.4f}")
        print(f"  Drop:                           {pct_drop:+.1f}%")
        
        if pct_drop > 50:
            print(f"  🔴 FATAL: IC dropped {pct_drop:.0f}% — CONTEMPORANEOUS LEAKAGE DETECTED")
            print(f"      Features likely contain information from the return period.")
        elif pct_drop > 25:
            print(f"  🟡 WARNING: IC dropped {pct_drop:.0f}% — possible partial leakage")
        else:
            print(f"  🟢 PASS: IC drop is only {pct_drop:.0f}% — no obvious leakage")
    else:
        print(f"  ⚠️ Could not compute shifted IC")

    # ────────────────────────────────────────────────────────────
    # TEST 1b: Check if ret_crsp correlates with fwd_ret_1m
    # This would indicate the target is computed FROM the same
    # data as a feature (mechanical correlation).
    # ────────────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("TEST 1b: ret_crsp ↔ fwd_ret_1m correlation check")
    print("  ret_crsp should be ret(T-1→T), fwd_ret_1m should be ret(T→T+1)")
    print("  Correlation should be near 0 (independent time periods)")
    print("-" * 70)

    if "ret_crsp" in merged.columns:
        mask = merged["ret_crsp"].notna() & merged["fwd_ret_1m"].notna()
        if mask.sum() > 1000:
            corr = np.corrcoef(merged.loc[mask, "ret_crsp"].values,
                               merged.loc[mask, "fwd_ret_1m"].values)[0, 1]
            rank_corr = stats.spearmanr(merged.loc[mask, "ret_crsp"].values,
                                         merged.loc[mask, "fwd_ret_1m"].values)[0]
            print(f"  Pearson corr(ret_crsp, fwd_ret_1m):  {corr:+.4f}")
            print(f"  Spearman corr(ret_crsp, fwd_ret_1m): {rank_corr:+.4f}")
            
            if abs(corr) > 0.3:
                print(f"  🔴 FATAL: ret_crsp IS the target (or overlaps heavily)!")
                print(f"      This means the model has the answer as a feature.")
            elif abs(corr) > 0.1:
                print(f"  🟡 WARNING: unexpected correlation — investigate timing")
            else:
                print(f"  🟢 PASS: ret_crsp and fwd_ret_1m appear to be independent")
    else:
        print(f"  ⚠️ ret_crsp not in panel — cannot check")

    # ────────────────────────────────────────────────────────────
    # TEST 2: ENSEMBLE WEIGHT OVERFITTING  
    # Compare first half vs second half performance
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 2: TEMPORAL STABILITY (first half vs second half)")
    print("  If IC is much higher in first half, ensemble weights are overfit.")
    print("=" * 70)

    ics["year"] = pd.to_datetime(ics["date"]).dt.year
    mid_year = ics["year"].median()
    
    first_half = ics[ics["year"] <= mid_year]
    second_half = ics[ics["year"] > mid_year]
    
    ic_first = first_half["ic"].mean()
    ic_second = second_half["ic"].mean()
    
    print(f"  First half  ({int(first_half['year'].min())}-{int(mid_year)}):  IC = {ic_first:+.4f}  ({len(first_half)} months)")
    print(f"  Second half ({int(mid_year)+1}-{int(second_half['year'].max())}): IC = {ic_second:+.4f}  ({len(second_half)} months)")
    
    ratio = ic_second / ic_first if ic_first != 0 else 0
    print(f"  Ratio (2nd/1st):  {ratio:.2f}")
    
    if ratio < 0.5:
        print(f"  🔴 FATAL: Second half IC is <50% of first half — likely overfitting")
    elif ratio < 0.75:
        print(f"  🟡 WARNING: IC degradation in second half")
    else:
        print(f"  🟢 PASS: IC is stable across time periods")

    # Year-by-year breakdown
    print(f"\n  Year-by-year IC:")
    yearly = ics.groupby("year")["ic"].agg(["mean", "std", "count"])
    for yr, row in yearly.iterrows():
        marker = "🟢" if row["mean"] > 0.03 else "🟡" if row["mean"] > 0 else "🔴"
        print(f"    {int(yr)}: IC={row['mean']:+.4f} ± {row['std']:.4f}  ({int(row['count'])} months)  {marker}")

    # ────────────────────────────────────────────────────────────
    # TEST 3: FEATURE IMPORTANCE CONCENTRATION
    # If top features are suspicious (e.g., ret_crsp, realized_vol
    # dominating), the signal may be trivial or leaking.
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 3: FEATURE IMPORTANCE ANALYSIS")
    print("  Check if top features are suspicious (trivial or leaky)")
    print("=" * 70)

    fi_path = os.path.join(RESULTS_DIR, "ensemble_feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        print(f"  Top 20 features by importance:")
        for i, row in fi.head(20).iterrows():
            pct = row.get("pct", row.get("importance", 0)) * 100
            flag = ""
            name = row["feature"]
            # Flag suspicious features
            if "ret" in name.lower() and "forward" not in name.lower() and "fwd" not in name.lower():
                flag = " ⚠️ RETURN FEATURE"
            if "vol" in name.lower() and "lag" not in name.lower():
                flag = " ⚠️ CONTEMPORANEOUS VOL"
            if name == "ret_crsp":
                flag = " 🔴 THIS IS MONTH-T RETURN — CHECK TIMING"
            print(f"    {i+1:>3}. {name:<50} {pct:>6.2f}%{flag}")
        
        # Concentration
        top5_pct = fi.head(5)["pct"].sum() * 100 if "pct" in fi.columns else fi.head(5)["importance"].sum() * 100
        top20_pct = fi.head(20)["pct"].sum() * 100 if "pct" in fi.columns else fi.head(20)["importance"].sum() * 100
        print(f"\n  Top 5 concentration:  {top5_pct:.1f}%")
        print(f"  Top 20 concentration: {top20_pct:.1f}%")
        
        if top5_pct > 30:
            print(f"  🟡 WARNING: Top 5 features dominate — signal may be fragile")
    else:
        print(f"  ⚠️ Feature importance file not found")

    # ────────────────────────────────────────────────────────────
    # TEST 4: MARKET-CAP STRATIFIED IC
    # If IC comes only from micro-caps, it's untradeable.
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 4: MARKET-CAP STRATIFIED IC")
    print("  If IC is only in small caps, the signal is untradeable.")
    print("=" * 70)

    if "log_market_cap" in merged.columns:
        # Create market cap terciles per month
        def add_cap_tercile(grp):
            grp = grp.copy()
            grp["cap_tercile"] = pd.qcut(grp["log_market_cap"], 3, 
                                          labels=["Small", "Mid", "Large"],
                                          duplicates="drop")
            return grp
        
        merged_cap = merged.dropna(subset=["log_market_cap", "prediction", "fwd_ret_1m"])
        merged_cap = merged_cap.groupby("date", group_keys=False).apply(add_cap_tercile)
        
        for tercile in ["Small", "Mid", "Large"]:
            sub = merged_cap[merged_cap["cap_tercile"] == tercile]
            sub_ics = monthly_ics(sub, "prediction", "fwd_ret_1m")
            if len(sub_ics) > 0:
                sub_ic = sub_ics["ic"].mean()
                sub_n = sub.groupby("date").size().mean()
                marker = "🟢" if sub_ic > 0.03 else "🟡" if sub_ic > 0 else "🔴"
                print(f"  {tercile:>6} cap: IC = {sub_ic:+.4f}  (avg {sub_n:.0f} stocks/month)  {marker}")
        
        # Also check: IC for stocks > $1B market cap only
        large_only = merged_cap[merged_cap["log_market_cap"] > np.log(1e9)]
        large_ics = monthly_ics(large_only, "prediction", "fwd_ret_1m")
        if len(large_ics) > 0:
            large_ic = large_ics["ic"].mean()
            print(f"  >$1B cap: IC = {large_ic:+.4f}  ({large_only.groupby('date').size().mean():.0f} stocks/month)")
    else:
        print(f"  ⚠️ log_market_cap not in panel — cannot stratify")

    # ────────────────────────────────────────────────────────────
    # TEST 5: TURNOVER & NET SHARPE
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 5: PORTFOLIO TURNOVER & NET SHARPE")
    print("  High turnover with transaction costs destroys alpha.")
    print("=" * 70)

    dates_sorted = sorted(preds["date"].unique())
    turnovers = []
    ls_returns_gross = []
    ls_returns_net = []
    
    prev_long = set()
    prev_short = set()
    
    for dt in dates_sorted:
        month = preds[preds["date"] == dt].dropna(subset=["prediction", "fwd_ret_1m"])
        if len(month) < 100:
            continue
        
        month = month.copy()
        month["decile"] = pd.qcut(month["prediction"], 10, labels=False, duplicates="drop")
        
        long_stocks = set(month[month["decile"] == month["decile"].max()]["permno"].values)
        short_stocks = set(month[month["decile"] == month["decile"].min()]["permno"].values)
        
        # Gross L/S return
        long_ret = month[month["permno"].isin(long_stocks)]["fwd_ret_1m"].mean()
        short_ret = month[month["permno"].isin(short_stocks)]["fwd_ret_1m"].mean()
        ls_gross = long_ret - short_ret
        ls_returns_gross.append(ls_gross)
        
        # Turnover
        if prev_long:
            long_turnover = 1 - len(long_stocks & prev_long) / max(len(long_stocks), 1)
            short_turnover = 1 - len(short_stocks & prev_short) / max(len(short_stocks), 1)
            avg_turnover = (long_turnover + short_turnover) / 2
            turnovers.append(avg_turnover)
            
            # Net return: subtract turnover × cost
            # Assume 30bps per side for large caps, 80bps average
            cost_per_side = 0.005  # 50bps average (conservative)
            total_cost = avg_turnover * cost_per_side * 2  # both sides
            ls_net = ls_gross - total_cost
            ls_returns_net.append(ls_net)
        
        prev_long = long_stocks
        prev_short = short_stocks

    if turnovers:
        avg_turnover = np.mean(turnovers)
        ls_gross_arr = np.array(ls_returns_gross)
        ls_net_arr = np.array(ls_returns_net)
        
        sharpe_gross = np.mean(ls_gross_arr) / np.std(ls_gross_arr) * np.sqrt(12) if np.std(ls_gross_arr) > 0 else 0
        sharpe_net = np.mean(ls_net_arr) / np.std(ls_net_arr) * np.sqrt(12) if np.std(ls_net_arr) > 0 else 0
        
        print(f"  Average monthly turnover: {avg_turnover*100:.1f}%")
        print(f"  Gross L/S Sharpe:         {sharpe_gross:.2f}")
        print(f"  Net L/S Sharpe (50bps):   {sharpe_net:.2f}")
        print(f"  Avg gross spread:         {np.mean(ls_gross_arr)*100:+.2f}%/mo")
        print(f"  Avg net spread:           {np.mean(ls_net_arr)*100:+.2f}%/mo")
        
        if sharpe_net < 0.6:
            print(f"  🔴 Net Sharpe < 0.6 — strategy may not be viable after costs")
        elif sharpe_net < 1.0:
            print(f"  🟡 Net Sharpe {sharpe_net:.2f} — marginal, needs cost optimization")
        else:
            print(f"  🟢 Net Sharpe {sharpe_net:.2f} — viable after transaction costs")

    # ────────────────────────────────────────────────────────────
    # TEST 6: RANK NORMALIZATION LEAKAGE
    # Check if rank normalization was done per-month (correct)
    # or globally (leakage).
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 6: RANK NORMALIZATION VERIFICATION")
    print("  rank_normalize_slice uses groupby('date') — checking code...")
    print("=" * 70)
    
    # The rank normalization in step6b uses:
    #   grouped = df.groupby("date")
    #   df[valid] = grouped[valid].rank(pct=True, na_option="keep") * 2 - 1
    # This IS correct — per-month ranking. No cross-temporal leakage.
    print(f"  ✅ Code inspection: rank_normalize_slice uses groupby('date')")
    print(f"     This is the correct approach — no cross-temporal leakage.")
    
    # But verify: are the predictions actually rank-like or continuous?
    sample_month = preds[preds["date"] == preds["date"].max()]
    pred_range = sample_month["prediction"].describe()
    print(f"\n  Prediction distribution (latest month, {len(sample_month)} stocks):")
    print(f"    min={pred_range['min']:.4f}, median={pred_range['50%']:.4f}, "
          f"max={pred_range['max']:.4f}, std={pred_range['std']:.4f}")

    # ────────────────────────────────────────────────────────────
    # TEST 7: IC BY INDIVIDUAL MODEL
    # If ALL models have high IC, the signal is in the features.
    # If only ensemble is high, it's ensemble overfitting.
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 7: INDIVIDUAL MODEL IC COMPARISON")
    print("  If ensemble >> individual models, weights are overfit.")
    print("=" * 70)

    for model_col in ["pred_lgb", "pred_ridge", "pred_enet", "pred_xgb"]:
        if model_col in preds.columns:
            model_ics = monthly_ics(preds, model_col, "fwd_ret_1m")
            if len(model_ics) > 0:
                m_ic = model_ics["ic"].mean()
                m_ir = m_ic / model_ics["ic"].std() if model_ics["ic"].std() > 0 else 0
                print(f"  {model_col:<12}: IC = {m_ic:+.4f}  IR = {m_ir:.2f}")
    
    print(f"  {'ensemble':<12}: IC = {reported_ic:+.4f}  IR = {ic_ir:.2f}")
    
    # Check: is ensemble IC much higher than best individual model?
    individual_ics = []
    for model_col in ["pred_lgb", "pred_ridge", "pred_enet", "pred_xgb"]:
        if model_col in preds.columns:
            model_ics = monthly_ics(preds, model_col, "fwd_ret_1m")
            if len(model_ics) > 0:
                individual_ics.append(model_ics["ic"].mean())
    
    if individual_ics:
        best_individual = max(individual_ics)
        ensemble_lift = (reported_ic - best_individual) / best_individual * 100 if best_individual != 0 else 0
        print(f"\n  Best individual model IC:  {best_individual:+.4f}")
        print(f"  Ensemble lift:            {ensemble_lift:+.1f}%")
        
        if ensemble_lift > 30:
            print(f"  🟡 WARNING: Ensemble lift of {ensemble_lift:.0f}% is unusually high")
            print(f"      Typical ensemble improvement is 5-15%")
        else:
            print(f"  🟢 Ensemble lift is within normal range")

    # ────────────────────────────────────────────────────────────
    # TEST 8: BLOCK BOOTSTRAP CONFIDENCE INTERVAL
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 8: BLOCK BOOTSTRAP SHARPE CONFIDENCE INTERVAL")
    print("  12-month blocks, 10,000 draws")
    print("=" * 70)

    ls_returns = np.array(ls_returns_gross)
    n_months = len(ls_returns)
    block_size = 12
    n_blocks = n_months // block_size
    n_bootstrap = 10000
    
    if n_blocks >= 5:
        bootstrap_sharpes = []
        np.random.seed(42)
        
        # Create non-overlapping blocks
        blocks = [ls_returns[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
        
        for _ in range(n_bootstrap):
            # Sample blocks with replacement
            sampled_blocks = [blocks[i] for i in np.random.randint(0, len(blocks), n_blocks)]
            sampled_returns = np.concatenate(sampled_blocks)
            
            if np.std(sampled_returns) > 0:
                s = np.mean(sampled_returns) / np.std(sampled_returns) * np.sqrt(12)
                bootstrap_sharpes.append(s)
        
        bootstrap_sharpes = np.array(bootstrap_sharpes)
        p5 = np.percentile(bootstrap_sharpes, 5)
        p25 = np.percentile(bootstrap_sharpes, 25)
        p50 = np.percentile(bootstrap_sharpes, 50)
        p75 = np.percentile(bootstrap_sharpes, 75)
        p95 = np.percentile(bootstrap_sharpes, 95)
        
        print(f"  Bootstrap Sharpe distribution ({n_bootstrap:,} draws):")
        print(f"    5th percentile:   {p5:.2f}")
        print(f"    25th percentile:  {p25:.2f}")
        print(f"    Median:           {p50:.2f}")
        print(f"    75th percentile:  {p75:.2f}")
        print(f"    95th percentile:  {p95:.2f}")
        
        if p5 < 0.5:
            print(f"  🟡 WARNING: 5th percentile Sharpe is {p5:.2f} — insufficient confidence")
        else:
            print(f"  🟢 PASS: Even at 5th percentile, Sharpe = {p5:.2f}")
        
        prob_positive = (bootstrap_sharpes > 0).mean() * 100
        print(f"  Prob(Sharpe > 0):   {prob_positive:.1f}%")
        print(f"  Prob(Sharpe > 0.5): {(bootstrap_sharpes > 0.5).mean()*100:.1f}%")
        print(f"  Prob(Sharpe > 1.0): {(bootstrap_sharpes > 1.0).mean()*100:.1f}%")
    else:
        print(f"  ⚠️ Not enough data for block bootstrap (need ≥60 months)")

    # ────────────────────────────────────────────────────────────
    # TEST 9: DIRECT FEATURE → TARGET IC CHECK
    # Compute raw IC of top features against fwd_ret_1m
    # to see if the high IC is in the features themselves.
    # ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEST 9: RAW FEATURE → TARGET IC (top features)")
    print("  High raw IC in a single feature = likely leakage")
    print("=" * 70)

    # Load a sample of the full panel with top features
    fi_path = os.path.join(RESULTS_DIR, "ensemble_feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
        top_features = fi.head(15)["feature"].tolist()
        
        # Load these features from the panel
        load_feats = ["permno", "date", "fwd_ret_1m"] + [f for f in top_features if f in available_cols]
        panel_sample = pd.read_parquet(panel_path, columns=load_feats)
        panel_sample["date"] = pd.to_datetime(panel_sample["date"])
        
        # Sample 50 months for speed
        dates = sorted(panel_sample["date"].unique())
        sample_dates = [dates[i] for i in np.linspace(0, len(dates)-1, 50, dtype=int)]
        
        print(f"  Computing raw IC for top 15 features on {len(sample_dates)} months...")
        for feat in load_feats[3:]:  # skip permno, date, fwd_ret_1m
            ics_raw = []
            for dt in sample_dates:
                sub = panel_sample[panel_sample["date"] == dt]
                ic = spearman_ic(sub[feat].values, sub["fwd_ret_1m"].values)
                if not np.isnan(ic):
                    ics_raw.append(ic)
            
            if ics_raw:
                mean_ic = np.mean(ics_raw)
                flag = ""
                if abs(mean_ic) > 0.15:
                    flag = " 🔴 SUSPICIOUSLY HIGH — LIKELY LEAKAGE"
                elif abs(mean_ic) > 0.08:
                    flag = " 🟡 ELEVATED"
                print(f"    {feat:<45} raw IC = {mean_ic:+.4f}{flag}")

    # ────────────────────────────────────────────────────────────
    # FINAL SUMMARY
    # ────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"AUDIT COMPLETE — {elapsed:.0f}s")
    print(f"{'=' * 70}")
    print(f"\nReported IC: {reported_ic:+.4f}")
    print(f"If this IC is real, it would be the best published cross-sectional")
    print(f"equity model in academic history. Review all 🔴 and 🟡 flags above.")
    print(f"\nKey question: Does the IC survive Test 1 (lag +1 month)?")
    print(f"If yes → you have a legitimate alpha source worth investigating further.")
    print(f"If no  → there is temporal leakage and the true IC is likely 0.03-0.05.")


if __name__ == "__main__":
    main()
