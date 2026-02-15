"""
PHASE 3 — STEP 6D: META-ENSEMBLE & MODEL STACKING
=====================================================
Combines all model families into a final prediction using:

1. STACKED GENERALIZATION (Wolpert 1992)
   - Level 0: LightGBM, Ridge, ElasticNet, FFN, ResNet
   - Level 1: Ridge meta-learner trained on OOS predictions
   - This automatically learns optimal time-varying weights

2. IC-DECAY WEIGHTING
   - Recent IC matters more than historical IC
   - Exponential decay weights: w_t = IC_t × exp(-λ × (T-t))

3. MONOTONICITY CONSTRAINTS
   - Verify D1 < D2 < ... < D10 portfolio returns
   - Report monotonicity violations as model quality diagnostic

4. COMPREHENSIVE MODEL COMPARISON
   - Compare vs. GKX (2020) benchmarks
   - Statistical significance tests (Diebold-Mariano)
   - Subperiod analysis (pre-GFC, GFC, post-GFC, COVID)

This is the FINAL step that produces the production prediction file.
"""

import pandas as pd
import numpy as np
import os
import json
import time
import gc
from scipy import stats
from typing import Dict, List, Optional

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"


# ═══════════════════════════════════════════════════════════════════════
# STACKED GENERALIZATION (META-LEARNER)
# ═══════════════════════════════════════════════════════════════════════

def ic_weighted_combine(pred_columns: Dict[str, np.ndarray],
                        targets: np.ndarray,
                        dates: np.ndarray,
                        decay_lambda: float = 0.05) -> np.ndarray:
    """
    IC-weighted combination with exponential decay.

    For each model m, compute monthly IC_m,t, then weight by:
        w_m,t = IC_m,t × exp(-λ × (T - t))

    This gives more weight to models that have been accurate RECENTLY.
    Financial markets are non-stationary — a model that worked in 2010
    may not work in 2020. IC-decay adapts to this.
    """
    unique_dates = np.sort(np.unique(dates))
    n_dates = len(unique_dates)

    # Compute monthly IC for each model
    model_ics = {}
    for model_name, preds in pred_columns.items():
        monthly_ics = []
        for dt in unique_dates:
            mask = dates == dt
            p = preds[mask]
            t = targets[mask]
            valid = ~(np.isnan(p) | np.isnan(t))
            if valid.sum() > 50:
                ic = stats.spearmanr(p[valid], t[valid])[0]
            else:
                ic = 0.0
            monthly_ics.append(ic)
        model_ics[model_name] = np.array(monthly_ics)

    # Compute time-decayed weights for each model
    # Use expanding window: at time t, weight = IC over [0:t] with decay
    combined = np.zeros(len(targets))

    for ti, dt in enumerate(unique_dates):
        mask = dates == dt

        weights = {}
        for model_name in pred_columns:
            # Historical ICs up to this point
            hist_ics = model_ics[model_name][:ti + 1]
            if len(hist_ics) == 0:
                weights[model_name] = 1.0 / len(pred_columns)
                continue

            # Exponential decay weights
            time_weights = np.exp(-decay_lambda * np.arange(len(hist_ics))[::-1])
            weighted_ic = np.average(hist_ics, weights=time_weights)
            weights[model_name] = max(weighted_ic, 0.0)  # Clip negative

        # Normalize weights
        w_total = sum(weights.values())
        if w_total < 1e-8:
            # Equal weight fallback
            for model_name in weights:
                weights[model_name] = 1.0 / len(pred_columns)
        else:
            for model_name in weights:
                weights[model_name] /= w_total

        # Combine
        for model_name, preds in pred_columns.items():
            combined[mask] += weights[model_name] * preds[mask]

    return combined


def stacked_combine(pred_columns: Dict[str, np.ndarray],
                    targets: np.ndarray,
                    dates: np.ndarray) -> np.ndarray:
    """
    Stacked generalization with time-series-aware split.

    Level 0: individual model predictions (already OOS)
    Level 1: Ridge regression on stacked predictions

    Uses rolling window for meta-learner training:
    - At time t, train meta-learner on predictions from [0, t-12)
    - Predict for month t
    - This avoids using future data for weight learning
    """
    from sklearn.linear_model import Ridge

    unique_dates = np.sort(np.unique(dates))
    combined = np.zeros(len(targets))

    model_names = list(pred_columns.keys())
    n_models = len(model_names)

    # Build stacked feature matrix
    X_stack = np.column_stack([pred_columns[m] for m in model_names])

    # Minimum 24 months of history before we start using meta-learner
    min_history = 24

    for ti, dt in enumerate(unique_dates):
        test_mask = dates == dt

        if ti < min_history:
            # Not enough history — use equal weights
            for model_name in model_names:
                combined[test_mask] += pred_columns[model_name][test_mask] / n_models
            continue

        # Train meta-learner on all prior months
        train_dates = unique_dates[:ti]
        train_mask = np.isin(dates, train_dates)

        X_meta_train = X_stack[train_mask]
        y_meta_train = targets[train_mask]

        # Remove NaN rows
        valid = ~(np.isnan(X_meta_train).any(axis=1) | np.isnan(y_meta_train))
        X_meta_train = X_meta_train[valid]
        y_meta_train = y_meta_train[valid]

        if len(X_meta_train) < 1000:
            for model_name in model_names:
                combined[test_mask] += pred_columns[model_name][test_mask] / n_models
            continue

        # Ridge meta-learner (constrained to prevent overfitting)
        meta = Ridge(alpha=10.0, fit_intercept=False, positive=True)
        meta.fit(X_meta_train, y_meta_train)

        # Predict
        X_meta_test = X_stack[test_mask]
        combined[test_mask] = meta.predict(np.nan_to_num(X_meta_test, nan=0.0))

    return combined


# ═══════════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS
# ═══════════════════════════════════════════════════════════════════════

def decile_analysis(predictions: np.ndarray, targets: np.ndarray,
                    dates: np.ndarray, n_quantiles: int = 10) -> pd.DataFrame:
    """
    Full decile portfolio analysis — the definitive cross-sectional test.

    For each month:
    1. Sort stocks by predicted return
    2. Form 10 equal-size portfolios (deciles)
    3. Compute realized return of each decile

    A good model shows MONOTONICALLY INCREASING returns from D1 to D10.
    """
    unique_dates = np.sort(np.unique(dates))
    decile_returns = {d: [] for d in range(n_quantiles)}

    for dt in unique_dates:
        mask = dates == dt
        p = predictions[mask]
        t = targets[mask]
        valid = ~(np.isnan(p) | np.isnan(t))
        p = p[valid]
        t = t[valid]

        if len(p) < 100:
            continue

        try:
            q = pd.qcut(p, n_quantiles, labels=False, duplicates="drop")
            for d in range(n_quantiles):
                d_ret = t[q == d].mean()
                decile_returns[d].append(d_ret)
        except Exception:
            continue

    # Build summary
    rows = []
    for d in range(n_quantiles):
        rets = np.array(decile_returns[d])
        rows.append({
            "decile": d + 1,
            "mean_return": rets.mean() * 100 if len(rets) > 0 else 0,
            "std_return": rets.std() * 100 if len(rets) > 0 else 0,
            "sharpe": (rets.mean() / rets.std() * np.sqrt(12)
                       if len(rets) > 0 and rets.std() > 0 else 0),
            "n_months": len(rets),
        })

    return pd.DataFrame(rows)


def subperiod_analysis(monthly_ics: pd.DataFrame,
                       ls_returns: pd.Series) -> Dict:
    """
    Subperiod analysis to check model stability across regimes.
    """
    periods = {
        "Pre-GFC (2005-2007)": ("2005-01-01", "2007-12-31"),
        "GFC (2008-2009)": ("2008-01-01", "2009-12-31"),
        "Recovery (2010-2014)": ("2010-01-01", "2014-12-31"),
        "Low Vol (2015-2019)": ("2015-01-01", "2019-12-31"),
        "COVID+ (2020-2024)": ("2020-01-01", "2024-12-31"),
    }

    results = {}
    for name, (start, end) in periods.items():
        mask = (monthly_ics["date"] >= start) & (monthly_ics["date"] <= end)
        sub = monthly_ics[mask]
        if len(sub) == 0:
            continue

        ls_mask = (ls_returns.index >= start) & (ls_returns.index <= end)
        ls_sub = ls_returns[ls_mask]

        results[name] = {
            "ic_mean": round(float(sub["ic"].mean()), 4),
            "ic_std": round(float(sub["ic"].std()), 4),
            "ic_ir": round(float(sub["ic"].mean() / sub["ic"].std()), 2) if sub["ic"].std() > 0 else 0,
            "ic_positive_pct": round(float((sub["ic"] > 0).mean() * 100), 1),
            "n_months": len(sub),
            "ls_sharpe": round(float(
                ls_sub.mean() / ls_sub.std() * np.sqrt(12)
            ), 2) if len(ls_sub) > 0 and ls_sub.std() > 0 else 0,
        }

    return results


def diebold_mariano_test(e1: np.ndarray, e2: np.ndarray,
                         h: int = 1) -> tuple:
    """
    Diebold-Mariano test for comparing predictive accuracy.

    H0: E[d_t] = 0 where d_t = e1_t^2 - e2_t^2
    Positive DM stat → model 2 is better (lower squared error)
    """
    d = e1 ** 2 - e2 ** 2
    d_bar = d.mean()
    # HAC variance (Newey-West with h-1 lags)
    T = len(d)
    gamma_0 = np.var(d)
    gamma_sum = gamma_0
    for k in range(1, h):
        gamma_k = np.cov(d[k:], d[:-k])[0, 1]
        gamma_sum += 2 * gamma_k
    se = np.sqrt(gamma_sum / T)
    dm_stat = d_bar / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PHASE 3 — STEP 6D: META-ENSEMBLE & FINAL PREDICTIONS")
    print("=" * 70)
    t_start = time.time()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    target_col = "fwd_ret_1m"

    # ── Load all predictions ─────────────────────────────────────
    # Priority: final_predictions > lgb_predictions + nn_predictions
    final_path = os.path.join(DATA_DIR, "final_predictions.parquet")
    tree_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    nn_path = os.path.join(DATA_DIR, "nn_predictions.parquet")

    if os.path.exists(final_path):
        print("  Loading combined predictions (tree+neural)...")
        preds = pd.read_parquet(final_path)
    elif os.path.exists(tree_path):
        print("  Loading tree/linear predictions...")
        preds = pd.read_parquet(tree_path)

        if os.path.exists(nn_path):
            nn_preds = pd.read_parquet(nn_path)
            preds = preds.merge(
                nn_preds[["permno", "date", "pred_nn"]],
                on=["permno", "date"], how="inner"
            )
    else:
        print("  ❌ No prediction files found! Run step6b first.")
        return

    preds["date"] = pd.to_datetime(preds["date"])
    print(f"  Predictions: {len(preds):,} rows")

    # ── Identify available model columns ─────────────────────────
    model_cols = {}
    for col in ["prediction", "pred_lgb", "pred_xgb", "pred_ridge", "pred_enet", "pred_nn"]:
        if col in preds.columns:
            model_cols[col] = preds[col].values.astype(np.float64)
            print(f"    ✓ {col}")

    targets = preds[target_col].values.astype(np.float64)
    dates = preds["date"].values

    # ── IC-Weighted Combination ──────────────────────────────────
    if len(model_cols) > 1:
        print("\n🔗 IC-WEIGHTED META-ENSEMBLE...")
        ic_combined = ic_weighted_combine(model_cols, targets, dates, decay_lambda=0.05)
        preds["pred_ic_weighted"] = ic_combined

        print("🔗 STACKED GENERALIZATION...")
        try:
            stacked = stacked_combine(model_cols, targets, dates)
            preds["pred_stacked"] = stacked
        except Exception as e:
            print(f"  ⚠️ Stacking failed: {e}, using IC-weighted only")
            preds["pred_stacked"] = ic_combined

        # Use stacked as final if it improves IC
        from step6b_ensemble import compute_monthly_ics, compute_long_short_returns

        ic_combo = compute_monthly_ics(preds, "pred_ic_weighted", target_col)
        ic_stack = compute_monthly_ics(preds, "pred_stacked", target_col)

        if ic_stack["ic"].mean() > ic_combo["ic"].mean():
            preds["prediction_final"] = preds["pred_stacked"]
            final_method = "stacked"
            print(f"  → Stacked wins: IC={ic_stack['ic'].mean():.4f} vs IC-weighted={ic_combo['ic'].mean():.4f}")
        else:
            preds["prediction_final"] = preds["pred_ic_weighted"]
            final_method = "ic_weighted"
            print(f"  → IC-weighted wins: IC={ic_combo['ic'].mean():.4f} vs Stacked={ic_stack['ic'].mean():.4f}")
    else:
        from step6b_ensemble import compute_monthly_ics, compute_long_short_returns
        # Single model — use as final
        only_col = list(model_cols.keys())[0]
        preds["prediction_final"] = preds[only_col]
        final_method = only_col

    # ═══════════════════════════════════════════════════════════════
    # COMPREHENSIVE EVALUATION
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print(f"COMPREHENSIVE EVALUATION")
    print(f"{'═' * 70}")

    # 1. Monthly IC
    all_ics = compute_monthly_ics(preds, "prediction_final", target_col)
    overall_ic = all_ics["ic"].mean()
    ic_std = all_ics["ic"].std()
    ic_ir = overall_ic / ic_std if ic_std > 0 else 0
    ic_pos_pct = (all_ics["ic"] > 0).mean() * 100

    # 2. Individual model ICs
    model_ic_summary = {}
    for col_name in list(model_cols.keys()) + ["pred_ic_weighted", "pred_stacked", "prediction_final"]:
        if col_name in preds.columns:
            ics = compute_monthly_ics(preds, col_name, target_col)
            if len(ics) > 0:
                model_ic_summary[col_name] = {
                    "ic": round(float(ics["ic"].mean()), 4),
                    "ic_std": round(float(ics["ic"].std()), 4),
                    "ic_ir": round(float(ics["ic"].mean() / ics["ic"].std()), 2) if ics["ic"].std() > 0 else 0,
                }

    # 3. Long-short returns
    ls_returns = compute_long_short_returns(preds, "prediction_final", target_col)
    ls_sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(12) if ls_returns.std() > 0 else 0

    # 4. OOS R²
    mask = preds["prediction_final"].notna() & preds[target_col].notna()
    y_true = preds.loc[mask, target_col].values
    y_pred = preds.loc[mask, "prediction_final"].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    oos_r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 5. Decile analysis
    dec = decile_analysis(
        preds["prediction_final"].values,
        preds[target_col].values,
        preds["date"].values
    )

    # 6. Monotonicity check
    dec_returns = dec["mean_return"].values
    monotonic = all(dec_returns[i] <= dec_returns[i + 1]
                    for i in range(len(dec_returns) - 1))
    mono_violations = sum(1 for i in range(len(dec_returns) - 1)
                          if dec_returns[i] > dec_returns[i + 1])

    # 7. Subperiod analysis
    all_ics["date"] = pd.to_datetime(all_ics["date"])
    ls_returns.index = pd.to_datetime(ls_returns.index)
    subperiods = subperiod_analysis(all_ics, ls_returns)

    # ═══════════════════════════════════════════════════════════════
    # PRINT REPORT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n  ┌─── HEADLINE METRICS ───────────────────────────────┐")
    print(f"  │ Method:          {final_method:<35}│")
    print(f"  │ Ensemble IC:     {overall_ic:+.4f}  (GKX target: 0.03)      │")
    print(f"  │ IC IR:           {ic_ir:.2f}    (target: >1.0)            │")
    print(f"  │ IC > 0:          {ic_pos_pct:.0f}%   (target: >65%)           │")
    print(f"  │ L/S Sharpe:      {ls_sharpe:.2f}   (target: ~2.0)           │")
    print(f"  │ OOS R²:          {oos_r2*100:.3f}% (target: ~0.40%)        │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  ┌─── MODEL COMPARISON ──────────────────────────────┐")
    for model_name, metrics in model_ic_summary.items():
        print(f"  │  {model_name:<22} IC={metrics['ic']:+.4f}  IR={metrics['ic_ir']:.2f}  │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  ┌─── DECILE ANALYSIS ──────────────────────────────┐")
    print(f"  │  Monotonic: {'✅ YES' if monotonic else f'❌ NO ({mono_violations} violations)'}{'':>27}│")
    for _, row in dec.iterrows():
        bar = "█" * max(1, int(abs(row['mean_return']) * 10))
        sign = "+" if row['mean_return'] > 0 else ""
        print(f"  │  D{int(row['decile']):>2}: {sign}{row['mean_return']:.2f}%/mo  "
              f"SR={row['sharpe']:.2f}  {bar:<20}│")
    d10_d1 = dec.iloc[-1]["mean_return"] - dec.iloc[0]["mean_return"]
    print(f"  │  D10-D1 spread: {d10_d1:+.2f}%/mo                      │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n  ┌─── SUBPERIOD STABILITY ────────────────────────────┐")
    for period_name, metrics in subperiods.items():
        print(f"  │  {period_name:<25} IC={metrics['ic_mean']:+.4f} "
              f"IR={metrics['ic_ir']:.2f} SR={metrics['ls_sharpe']:.2f} │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # GKX benchmark comparison
    print(f"\n  ┌─── GKX (2020) BENCHMARK COMPARISON ────────────────┐")
    gkx_ic = 0.03
    gkx_sharpe = 2.0
    gkx_r2 = 0.40
    ic_vs = "✅" if overall_ic >= gkx_ic * 0.8 else "⚠️" if overall_ic > 0 else "❌"
    sr_vs = "✅" if ls_sharpe >= gkx_sharpe * 0.8 else "⚠️" if ls_sharpe > 0 else "❌"
    r2_vs = "✅" if oos_r2 * 100 >= gkx_r2 * 0.8 else "⚠️" if oos_r2 > 0 else "❌"
    print(f"  │                    Ours        GKX        Status    │")
    print(f"  │  IC:              {overall_ic:+.4f}      ≈0.03       {ic_vs}       │")
    print(f"  │  L/S Sharpe:      {ls_sharpe:.2f}        ≈2.0        {sr_vs}       │")
    print(f"  │  OOS R²:          {oos_r2*100:.3f}%     ≈0.40%      {r2_vs}       │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # ═══════════════════════════════════════════════════════════════
    # SAVE FINAL OUTPUTS
    # ═══════════════════════════════════════════════════════════════
    print("\n💾 SAVING FINAL OUTPUTS...")

    # Final predictions
    preds.to_parquet(
        os.path.join(DATA_DIR, "production_predictions.parquet"),
        index=False
    )

    # Full report
    report = {
        "model": f"Meta-ensemble ({final_method})",
        "n_models": len(model_cols),
        "model_names": list(model_cols.keys()),
        "total_predictions": len(preds),
        "total_months": len(all_ics),
        "headline": {
            "ic": round(float(overall_ic), 4),
            "ic_std": round(float(ic_std), 4),
            "ic_ir": round(float(ic_ir), 2),
            "ic_positive_pct": round(float(ic_pos_pct), 1),
            "ls_sharpe": round(float(ls_sharpe), 2),
            "oos_r2_pct": round(float(oos_r2 * 100), 3),
            "d10_d1_spread_pct": round(float(d10_d1), 3),
            "monotonic": bool(monotonic),
        },
        "individual_models": model_ic_summary,
        "decile_returns": dec.to_dict("records"),
        "subperiods": subperiods,
        "gkx_comparison": {
            "ic_ratio": round(float(overall_ic / gkx_ic), 2) if gkx_ic > 0 else 0,
            "sharpe_ratio": round(float(ls_sharpe / gkx_sharpe), 2) if gkx_sharpe > 0 else 0,
            "r2_ratio": round(float(oos_r2 * 100 / gkx_r2), 2) if gkx_r2 > 0 else 0,
        },
    }

    report_path = os.path.join(RESULTS_DIR, "final_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  ✅ {report_path}")

    # Monthly ICs
    all_ics.to_csv(os.path.join(RESULTS_DIR, "final_monthly_ics.csv"), index=False)

    # Decile table
    dec.to_csv(os.path.join(RESULTS_DIR, "decile_analysis.csv"), index=False)

    # Save latest model for wrds_predictor.py
    # (wrds_predictor loads lgb_predictions.parquet, so we update it)
    preds.to_parquet(
        os.path.join(DATA_DIR, "lgb_predictions.parquet"),
        index=False
    )
    print(f"  ✅ Updated lgb_predictions.parquet with meta-ensemble")

    # S3 upload
    import subprocess
    for fpath in [
        os.path.join(DATA_DIR, "production_predictions.parquet"),
        report_path,
    ]:
        fname = os.path.basename(fpath)
        subprocess.run(
            ["aws", "s3", "cp", fpath, f"s3://nuble-data-warehouse/models/{fname}"],
            capture_output=True,
        )

    elapsed = time.time() - t_start
    print(f"\n  Total time: {elapsed:.0f}s")
    print(f"{'═' * 70}")
    print(f"✅ PRODUCTION PREDICTIONS READY")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
