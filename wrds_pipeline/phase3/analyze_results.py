"""
Phase 3 Results Analysis — GKX Benchmark Comparison Report.

Reads the outputs from Step 6 (walk-forward) and Step 7 (backtest)
and produces a comprehensive performance report comparing to GKX (2020)
benchmark targets:

GKX Targets:
  - Monthly IC: ≈ 0.03
  - OOS R²: ≈ 0.40%
  - Long-Short Sharpe: ≈ 2.0
  - D10-D1 monthly spread: monotonically increasing

Also produces:
  - Feature importance analysis (interaction vs base vs macro)
  - Subperiod stability analysis
  - IC time-series with regime overlay
  - Decile return monotonicity
"""

import pandas as pd
import numpy as np
import os
import json
from scipy import stats

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "wrds")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def analyze():
    print("=" * 70)
    print("PHASE 3 — GKX BENCHMARK COMPARISON REPORT")
    print("=" * 70)

    # ── Load walk-forward metrics ─────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, "walk_forward_metrics.csv")
    if not os.path.exists(metrics_path):
        print("❌ walk_forward_metrics.csv not found — run Step 6 first!")
        return
    metrics = pd.read_csv(metrics_path)

    # ── Load predictions ─────────────────────────────────────────────
    pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    if os.path.exists(pred_path):
        predictions = pd.read_parquet(pred_path)
        predictions["date"] = pd.to_datetime(predictions["date"])
    else:
        predictions = None

    # ── Load model summary ───────────────────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, "model_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            model_summary = json.load(f)
    else:
        model_summary = {}

    # ── Load feature importance ──────────────────────────────────────
    fi_path = os.path.join(RESULTS_DIR, "feature_importance.csv")
    if os.path.exists(fi_path):
        fi = pd.read_csv(fi_path)
    else:
        fi = None

    # ══════════════════════════════════════════════════════════════════
    # 1. GKX BENCHMARK COMPARISON
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("1. GKX (2020) BENCHMARK COMPARISON")
    print(f"{'─' * 70}")

    overall_ic = metrics["avg_ic"].mean()
    ic_median = metrics["avg_ic"].median()
    ic_std = metrics["avg_ic"].std()
    ic_ir = overall_ic / ic_std if ic_std > 0 else 0
    ic_positive = (metrics["avg_ic"] > 0).mean() * 100
    ic_above_002 = (metrics["avg_ic"] > 0.02).mean() * 100
    overall_spread = metrics["avg_spread"].mean()

    # Compute Sharpe from predictions
    sharpe = np.nan
    oos_r2 = np.nan
    if predictions is not None:
        target_col = "fwd_ret_1m" if "fwd_ret_1m" in predictions.columns else "ret_forward"
        # Monthly long-short returns
        ls_returns = []
        for dt, grp in predictions.groupby("date"):
            if len(grp) < 100:
                continue
            grp = grp.copy()
            grp["decile"] = pd.qcut(grp["prediction"], 10, labels=False, duplicates="drop")
            d10 = grp[grp["decile"] == grp["decile"].max()][target_col].mean()
            d1 = grp[grp["decile"] == grp["decile"].min()][target_col].mean()
            ls_returns.append(d10 - d1)

        ls_returns = pd.Series(ls_returns)
        if len(ls_returns) > 0 and ls_returns.std() > 0:
            sharpe = ls_returns.mean() / ls_returns.std() * np.sqrt(12)

        # OOS R²
        mask = predictions["prediction"].notna() & predictions[target_col].notna()
        if mask.sum() > 1000:
            ss_res = ((predictions.loc[mask, target_col] - predictions.loc[mask, "prediction"]) ** 2).sum()
            ss_tot = ((predictions.loc[mask, target_col] - predictions.loc[mask, target_col].mean()) ** 2).sum()
            if ss_tot > 0:
                oos_r2 = 1 - ss_res / ss_tot

    print(f"\n{'Metric':<30} {'Ours':>10} {'GKX Target':>12} {'Status':>10}")
    print(f"{'─' * 70}")

    def status(val, target, higher_better=True):
        if np.isnan(val):
            return "❓"
        if higher_better:
            return "✅" if val >= target else "⚠️" if val >= target * 0.7 else "❌"
        else:
            return "✅" if val <= target else "⚠️" if val <= target * 1.3 else "❌"

    print(f"{'Monthly IC (mean):':<30} {overall_ic:>+10.4f} {'≈ 0.03':>12} {status(overall_ic, 0.03):>10}")
    print(f"{'Monthly IC (median):':<30} {ic_median:>+10.4f} {'≈ 0.03':>12} {status(ic_median, 0.03):>10}")
    print(f"{'IC Information Ratio:':<30} {ic_ir:>10.2f} {'> 1.0':>12} {status(ic_ir, 1.0):>10}")
    print(f"{'IC > 0 (% years):':<30} {ic_positive:>9.0f}% {'> 80%':>12} {status(ic_positive, 80):>10}")
    print(f"{'IC > 0.02 (% years):':<30} {ic_above_002:>9.0f}% {'> 60%':>12} {status(ic_above_002, 60):>10}")
    print(f"{'D10-D1 Spread (monthly):':<30} {overall_spread:>+10.4f} {'> 0.005':>12} {status(overall_spread, 0.005):>10}")
    print(f"{'Long-Short Sharpe:':<30} {sharpe:>10.2f} {'≈ 2.0':>12} {status(sharpe, 1.5):>10}")
    if not np.isnan(oos_r2):
        print(f"{'OOS R² (monthly):':<30} {oos_r2*100:>9.2f}% {'≈ 0.40%':>12} {status(oos_r2*100, 0.2):>10}")

    # ══════════════════════════════════════════════════════════════════
    # 2. ANNUAL IC BREAKDOWN
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("2. ANNUAL IC & SPREAD BREAKDOWN")
    print(f"{'─' * 70}")
    print(f"{'Year':>6} {'IC':>8} {'Spread':>10} {'Train':>10} {'Test':>8} {'Trees':>6}")
    for _, row in metrics.iterrows():
        ic_bar = "█" * int(max(0, row["avg_ic"]) * 200)
        stat = "✅" if row["avg_ic"] > 0.02 else "⚠️" if row["avg_ic"] > 0 else "❌"
        print(f"  {int(row['year']):>4} {row['avg_ic']:>+8.4f} {row['avg_spread']:>+10.4f} "
              f"{int(row['n_train']):>10,} {int(row['n_test']):>8,} {int(row['n_trees']):>6} {stat} {ic_bar}")

    # ══════════════════════════════════════════════════════════════════
    # 3. SUBPERIOD ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'─' * 70}")
    print("3. SUBPERIOD IC ANALYSIS")
    print(f"{'─' * 70}")

    subperiods = [
        ("Pre-GFC (2005-2007)", 2005, 2007),
        ("GFC (2008-2009)", 2008, 2009),
        ("Recovery (2010-2012)", 2010, 2012),
        ("Bull Run (2013-2019)", 2013, 2019),
        ("COVID Era (2020-2021)", 2020, 2021),
        ("Rate Hike (2022-2024)", 2022, 2024),
    ]

    for name, y1, y2 in subperiods:
        sub = metrics[(metrics["year"] >= y1) & (metrics["year"] <= y2)]
        if len(sub) > 0:
            avg_ic = sub["avg_ic"].mean()
            avg_sp = sub["avg_spread"].mean()
            stat = "✅" if avg_ic > 0.02 else "⚠️" if avg_ic > 0 else "❌"
            print(f"  {name:<30} IC={avg_ic:>+.4f}  Spread={avg_sp:>+.4f}  N={len(sub)} {stat}")

    # ══════════════════════════════════════════════════════════════════
    # 4. FEATURE IMPORTANCE ANALYSIS
    # ══════════════════════════════════════════════════════════════════
    if fi is not None and len(fi) > 0:
        print(f"\n{'─' * 70}")
        print("4. FEATURE IMPORTANCE ANALYSIS")
        print(f"{'─' * 70}")

        # Categorize features
        fi["category"] = "BASE"
        fi.loc[fi["feature"].str.startswith("ix_"), "category"] = "INTERACTION"
        fi.loc[fi["feature"].str.startswith("ff49_"), "category"] = "INDUSTRY"
        fi.loc[fi["feature"].str.contains("_lag|_chg|_trend", na=False), "category"] = "LAG/TREND"

        # Check if it's a macro column
        macro_indicators = ["unemployment", "cpi", "pce", "fed_funds", "vix", "treasury",
                            "spread", "yield", "housing", "nonfarm", "retail", "durable",
                            "consumer", "commercial", "leading", "manufacturing", "trade",
                            "tbl", "svar", "ntis", "corpr", "ltr"]
        for idx, row in fi.iterrows():
            if row["category"] == "BASE":
                feat_lower = row["feature"].lower()
                if any(m in feat_lower for m in macro_indicators):
                    fi.loc[idx, "category"] = "MACRO"

        # Category breakdown
        cat_importance = fi.groupby("category")["importance"].sum()
        cat_total = cat_importance.sum()
        print(f"\n  Category Breakdown:")
        for cat in ["INTERACTION", "BASE", "MACRO", "INDUSTRY", "LAG/TREND"]:
            if cat in cat_importance.index:
                pct = cat_importance[cat] / cat_total * 100
                n = (fi["category"] == cat).sum()
                bar = "█" * int(pct)
                print(f"    {cat:<15} {pct:>5.1f}%  ({n:>3} features)  {bar}")

        # Top 20 features
        print(f"\n  Top 20 Features:")
        for i, row in fi.head(20).iterrows():
            pct = row.get("importance_pct", row["importance"] / fi["importance"].sum()) * 100
            print(f"    {i+1:>3}. {row['feature']:<50} {pct:>5.2f}%  [{row['category']}]")

        # Interaction contribution
        n_inter_top30 = (fi.head(30)["category"] == "INTERACTION").sum()
        print(f"\n  Interaction features in Top 30: {n_inter_top30}/30")
        if n_inter_top30 >= 10:
            print(f"  ✅ Interactions are adding SIGNIFICANT value (GKX's 'secret sauce' works!)")
        elif n_inter_top30 >= 5:
            print(f"  ⚠️ Interactions moderately important")
        else:
            print(f"  ❌ Interactions underrepresented — may need more macro conditioning")

    # ══════════════════════════════════════════════════════════════════
    # 5. DECILE MONOTONICITY
    # ══════════════════════════════════════════════════════════════════
    if predictions is not None:
        print(f"\n{'─' * 70}")
        print("5. DECILE RETURN MONOTONICITY")
        print(f"{'─' * 70}")

        target_col = "fwd_ret_1m" if "fwd_ret_1m" in predictions.columns else "ret_forward"
        decile_rets = []
        for dt, grp in predictions.groupby("date"):
            if len(grp) < 100:
                continue
            grp = grp.copy()
            try:
                grp["decile"] = pd.qcut(grp["prediction"], 10, labels=range(1, 11), duplicates="drop")
                for d in range(1, 11):
                    d_data = grp[grp["decile"] == d]
                    if len(d_data) > 0:
                        decile_rets.append({"decile": d, "ret": d_data[target_col].mean()})
            except Exception:
                continue

        if decile_rets:
            dec_df = pd.DataFrame(decile_rets)
            dec_avg = dec_df.groupby("decile")["ret"].mean() * 100

            print(f"\n  {'Decile':>8} {'Avg Ret %':>10} {'Bar'}")
            for d in range(1, 11):
                if d in dec_avg.index:
                    r = dec_avg[d]
                    bar_len = int((r + 2) * 15)  # Normalize for display
                    bar = "█" * max(0, bar_len)
                    marker = " ← SHORT" if d == 1 else " ← LONG" if d == 10 else ""
                    print(f"    D{d:>2}:   {r:>+7.3f}%  {bar}{marker}")

            if len(dec_avg) >= 10:
                monotonic = all(dec_avg.iloc[i] <= dec_avg.iloc[i+1] for i in range(len(dec_avg)-1))
                spread = dec_avg.iloc[-1] - dec_avg.iloc[0]
                print(f"\n  D10-D1 spread: {spread:+.3f}%/month ({spread*12:+.1f}%/year)")
                print(f"  Monotonic: {'✅ YES' if monotonic else '⚠️ NO (some inversions)'}")

    # ══════════════════════════════════════════════════════════════════
    # 6. OVERALL ASSESSMENT
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'═' * 70}")
    print("OVERALL ASSESSMENT")
    print(f"{'═' * 70}")

    scores = 0
    total = 0

    checks = [
        ("IC > 0.02", overall_ic > 0.02),
        ("IC > 0.03 (GKX target)", overall_ic > 0.03),
        ("IC positive > 80% of years", ic_positive > 80),
        ("D10-D1 spread > 0", overall_spread > 0),
        ("Sharpe > 1.0", sharpe > 1.0 if not np.isnan(sharpe) else False),
        ("Sharpe > 1.5 (GKX target)", sharpe > 1.5 if not np.isnan(sharpe) else False),
    ]

    for name, passed in checks:
        total += 1
        if passed:
            scores += 1
        status_str = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status_str}  {name}")

    print(f"\n  Score: {scores}/{total}")
    if scores >= 5:
        print(f"  🏆 INSTITUTIONAL GRADE — Ready for production")
    elif scores >= 3:
        print(f"  📊 STRONG ALPHA — Needs refinement for production")
    else:
        print(f"  ⚠️ NEEDS WORK — Feature engineering or data issues")

    print(f"\n{'═' * 70}")


if __name__ == "__main__":
    analyze()
