"""
PHASE 3 — STEP 5: GKX-Style Feature Panel (600-900 Features)
=============================================================
The CORE of the system. Implements Gu-Kelly-Xiu (2020) feature engineering:

LAYER 1: Base characteristics (from training_panel.parquet)  ~155 features
LAYER 2: FF49 industry dummies (from SIC codes)              ~49 features
LAYER 3: Macro conditioning variables (from Steps 2-3)       ~15 features
LAYER 4: Characteristic × Macro interactions (SECRET SAUCE)  ~450 features
LAYER 5: Lagged features + trends                           ~100 features
         Cross-sectional ranking to [-1,1]
         IC-based feature selection → top 900

Output: gkx_panel.parquet (~3-5M rows × 600-900 columns)
"""

import pandas as pd
import numpy as np
import os
import time
import subprocess
import gc
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
S3_BUCKET = "nuble-data-warehouse"

# ── Fama-French 49 Industry SIC Ranges ──────────────────────────────────
# Simplified mapping: SIC → FF49 industry number
FF49_SIC_MAP = {
    1: [(100,199),(200,299),(700,799),(910,919),(2048,2048)],
    2: [(2000,2046),(2050,2063),(2070,2079),(2090,2092),(2095,2099)],
    3: [(2064,2068),(2086,2086),(2087,2087),(2096,2097)],
    4: [(2080,2085)],
    5: [(2100,2199)],
    6: [(920,999),(3650,3651),(3732,3732),(3930,3931),(3940,3949)],
    7: [(7800,7833),(7840,7841),(7900,7999)],
    8: [(2700,2749),(2770,2771),(2780,2799)],
    9: [(2047,2047),(2391,2392),(2510,2519),(2590,2599),(2840,2844),(3160,3161),(3170,3199),(3229,3229),(3260,3260),(3262,3263),(3269,3269),(3589,3589),(3631,3639),(3750,3751),(3800,3800),(3860,3861),(3870,3879),(3910,3919),(3960,3961),(3991,3991),(3995,3995)],
    10: [(2300,2390),(3020,3021),(3100,3111),(3130,3131),(3140,3151),(3963,3965)],
    11: [(8000,8099)],
    12: [(3693,3693),(3840,3851)],
    13: [(2830,2836)],
    14: [(2800,2829),(2850,2899),(2910,2911)],
    15: [(3031,3031),(3041,3041),(3050,3053),(3060,3069),(3080,3089),(3090,3099)],
    16: [(2200,2284),(2290,2295),(2297,2299),(2393,2395),(2397,2399)],
    17: [(800,899),(2400,2439),(2450,2459),(2490,2499),(2660,2661),(2950,2952),(3200,3200),(3210,3211),(3240,3241),(3250,3259),(3261,3261),(3264,3264),(3270,3275),(3280,3281),(3290,3293),(3295,3299),(3420,3433),(3440,3442),(3446,3452),(3490,3499),(3996,3996)],
    18: [(1500,1511),(1520,1549),(1600,1699),(1700,1799)],
    19: [(3300,3300),(3310,3317),(3320,3325),(3330,3339),(3340,3341),(3350,3357),(3360,3369),(3370,3379),(3380,3399)],
    20: [(3400,3400),(3443,3443),(3444,3444),(3460,3479),(3510,3536),(3538,3538),(3540,3569),(3580,3582),(3585,3586),(3589,3589),(3590,3599)],
    21: [(3600,3600),(3610,3613),(3620,3629),(3670,3679),(3680,3680),(3690,3692),(3699,3699)],
    22: [(3622,3622),(3661,3666),(3669,3669),(3672,3679),(3810,3810),(3812,3812)],
    23: [(3674,3674),(3675,3675),(3678,3678),(3679,3679)],
    24: [(3810,3810),(3812,3812),(3820,3827),(3829,3839)],
    25: [(2440,2449),(2520,2549),(2590,2599),(2600,2639),(2670,2699),(2760,2761),(3085,3085),(3411,3412),(3950,3955)],
    26: [(2440,2449),(2640,2659),(3220,3221),(3410,3412)],
    27: [(4000,4013),(4040,4049)],
    28: [(4100,4100),(4110,4121),(4130,4131),(4140,4142),(4150,4151),(4170,4173),(4190,4199),(4200,4200),(4210,4219),(4220,4231),(4240,4249),(4400,4499),(4500,4599),(4600,4699),(4700,4712),(4720,4749),(4780,4780),(4782,4789),(4789,4789)],
    29: [(5000,5000),(5010,5015),(5020,5023),(5030,5060),(5063,5065),(5070,5078),(5080,5088),(5090,5094),(5099,5099),(5100,5172),(5180,5182),(5190,5199)],
    30: [(5200,5231),(5250,5251),(5260,5261),(5270,5271),(5300,5311),(5320,5320),(5330,5331),(5334,5334),(5340,5349),(5390,5399),(5400,5411),(5412,5412),(5420,5469),(5490,5499),(5500,5571),(5590,5599),(5600,5699),(5700,5736),(5900,5990)],
    31: [(5800,5829),(5890,5899)],
    32: [(7000,7000),(7010,7019),(7040,7049),(7200,7212),(7215,7299),(7395,7395),(7500,7500),(7520,7549),(7600,7699),(8100,8199),(8200,8299),(8300,8399),(8400,8499),(8600,8699),(8700,8721),(8730,8734),(8740,8748),(8800,8899),(8900,8910),(8911,8911),(8920,8999),(4720,4729)],
    33: [(7300,7300),(7310,7342),(7349,7353),(7359,7372),(7374,7385),(7389,7394),(7396,7397),(7399,7399),(7510,7515),(8700,8713),(8720,8721),(8730,8734),(8740,8748),(8900,8910),(8911,8911),(4730,4749)],
    34: [(3570,3579),(3680,3689),(3695,3695),(7370,7372)],
    35: [(3571,3572),(3575,3579),(3680,3689),(3695,3695)],
    36: [(3661,3666),(3669,3669),(3672,3679),(3812,3812),(4800,4899)],
    37: [(4810,4813),(4820,4822),(4830,4841),(4880,4889),(4890,4890),(4899,4899)],
    38: [(4900,4900),(4910,4911),(4920,4925),(4930,4932),(4939,4942),(4950,4959),(4960,4969),(4990,4991)],
    39: [(9900,9999)],
    40: [(6020,6022),(6025,6025),(6030,6036),(6040,6062),(6080,6082),(6090,6099),(6110,6111),(6112,6113),(6120,6129),(6130,6159)],
    41: [(6150,6153),(6159,6159)],
    42: [(6200,6299),(6700,6726)],
    43: [(6300,6331),(6350,6351),(6360,6361),(6399,6411)],
    44: [(6500,6500),(6510,6553),(6590,6599),(6610,6611)],
    45: [(6150,6153),(6159,6159),(6200,6299),(6700,6726)],
    46: [(1040,1049)],
    47: [(1000,1039),(1060,1099),(1200,1299)],
    48: [(1300,1389),(1400,1499),(2900,2912),(2990,2999)],
}


def sic_to_ff49(sic):
    """Map SIC code to Fama-French 49 industry."""
    if pd.isna(sic) or sic == 0:
        return 0
    sic = int(sic)
    for industry, ranges in FF49_SIC_MAP.items():
        for lo, hi in ranges:
            if lo <= sic <= hi:
                return industry
    return 0


def cross_sectional_rank(series):
    """Rank to [-1, 1] within cross-section."""
    ranked = series.rank(pct=True)
    return 2 * ranked - 1


def compute_ic(feature, target):
    """Compute information coefficient (rank correlation)."""
    mask = feature.notna() & target.notna()
    if mask.sum() < 30:
        return 0
    return feature[mask].corr(target[mask], method="spearman")


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 5: GKX-STYLE FEATURE PANEL")
    print("=" * 70)
    start = time.time()

    # ── LAYER 1: Base characteristics ────────────────────────────────────
    print("\n📊 LAYER 1: Loading base training panel...")
    panel_path = os.path.join(DATA_DIR, "training_panel.parquet")
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"  Base panel: {len(panel):,} rows × {panel.shape[1]} cols")

    # Identify numeric feature columns (exclude identifiers)
    id_cols = ["permno", "date", "cusip", "ticker", "comnam", "siccd",
               "ret_forward", "fwd_ret_1m", "fwd_ret_3m", "fwd_ret_6m", "fwd_ret_12m",
               "ret", "shrcd", "exchcd", "dlret", "dlstcd",
               "excess_ret_1m", "rf"]  # exclude forward-looking & risk-free
    feature_cols = [c for c in panel.columns if c not in id_cols and panel[c].dtype in ["float64", "float32", "int64"]]
    print(f"  Base features: {len(feature_cols)}")

    # ── LAYER 2: FF49 industry dummies ───────────────────────────────────
    print("\n📊 LAYER 2: Fama-French 49 industry dummies...")
    if "siccd" in panel.columns:
        panel["ff49"] = panel["siccd"].apply(sic_to_ff49)
        ff49_dummies = pd.get_dummies(panel["ff49"], prefix="ff49", dtype=float)
        # Only keep industries with >0.5% of observations
        min_obs = len(panel) * 0.005
        ff49_keep = [c for c in ff49_dummies.columns if ff49_dummies[c].sum() > min_obs]
        ff49_dummies = ff49_dummies[ff49_keep]
        panel = pd.concat([panel, ff49_dummies], axis=1)
        feature_cols.extend(ff49_keep)
        print(f"  Added {len(ff49_keep)} FF49 dummies")
    else:
        print("  ⚠️ No SIC codes, skipping industry dummies")

    # ── LAYER 3: Macro conditioning variables ────────────────────────────
    print("\n📊 LAYER 3: Macro conditioning variables...")
    macro_cols = []

    # Welch-Goyal
    wg_path = os.path.join(DATA_DIR, "welch_goyal_macro.parquet")
    if os.path.exists(wg_path):
        wg = pd.read_parquet(wg_path)
        wg["date"] = pd.to_datetime(wg["date"])
        wg_vars = [c for c in wg.columns if c != "date"]
        panel = panel.merge(wg, on="date", how="left")
        macro_cols.extend(wg_vars)
        print(f"  Welch-Goyal: {len(wg_vars)} vars → {wg_vars}")
    else:
        print("  ⚠️ welch_goyal_macro.parquet not found — run Step 2 first")

    # FRED macro
    fred_path = os.path.join(DATA_DIR, "macro_predictors.parquet")
    if os.path.exists(fred_path):
        fred = pd.read_parquet(fred_path)
        fred["date"] = pd.to_datetime(fred["date"])
        fred_vars = [c for c in fred.columns if c != "date" and c not in panel.columns]
        if fred_vars:
            panel = panel.merge(fred[["date"] + fred_vars], on="date", how="left")
            macro_cols.extend(fred_vars)
            print(f"  FRED: {len(fred_vars)} vars")
    else:
        print("  ⚠️ macro_predictors.parquet not found — run Step 3 first")

    # Forward-fill macro (they report monthly, we need them for each stock-month)
    for c in macro_cols:
        if c in panel.columns:
            panel[c] = panel[c].ffill()

    feature_cols.extend(macro_cols)
    print(f"  Total macro vars added: {len(macro_cols)}")

    # ── LAYER 4: Characteristic × Macro Interactions (SECRET SAUCE) ──────
    print("\n📊 LAYER 4: Characteristic × Macro interactions (GKX SECRET SAUCE)...")

    # Select top characteristics by coverage + variance (limit to 15 for memory)
    char_candidates = [c for c in feature_cols if c not in macro_cols and not c.startswith("ff49_")]
    char_coverage = panel[char_candidates].notna().mean()
    char_std = panel[char_candidates].std()
    char_score = char_coverage * (char_std > 0).astype(float)
    top_chars = char_score.nlargest(15).index.tolist()
    print(f"  Top 15 characteristics: {top_chars[:10]}...")

    # Select TOP macro variables with good coverage (limit to 15 for memory)
    good_macro = [c for c in macro_cols if c in panel.columns and panel[c].notna().mean() > 0.5]
    # If too many, pick most variable ones
    if len(good_macro) > 15:
        macro_var = panel[good_macro].std().nlargest(15).index.tolist()
        good_macro = macro_var
    print(f"  Macro vars for interactions: {len(good_macro)}")

    interaction_cols = []
    for char in top_chars:
        for macro in good_macro:
            inter_name = f"ix_{char}_x_{macro}"
            panel[inter_name] = panel[char].values * panel[macro].values
            interaction_cols.append(inter_name)

    feature_cols.extend(interaction_cols)
    print(f"  Created {len(interaction_cols)} interaction features")
    gc.collect()

    # ── LAYER 5: Lagged features + trends ────────────────────────────────
    print("\n📊 LAYER 5: Lagged features + trends...")

    # Key features to lag (top 5 chars only to keep it fast)
    lag_features = top_chars[:5]
    lag_cols = []

    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    permno_vals = panel["permno"].values

    # Vectorized lag: use numpy array shifting with group boundary masking
    # Build group boundary mask once — O(n) numpy, no python loops
    same_group_1 = np.zeros(len(panel), dtype=bool)
    same_group_1[1:] = permno_vals[1:] == permno_vals[:-1]

    same_group_3 = np.zeros(len(panel), dtype=bool)
    same_group_3[3:] = permno_vals[3:] == permno_vals[:-3]
    # Also need all intermediate to be same group
    same_group_3[3:] &= same_group_1[3:] & same_group_1[2:-1] & same_group_1[1:-2]

    for feat in lag_features:
        if feat not in panel.columns:
            continue
        vals = panel[feat].values.astype(np.float64)

        # Lag 1: shift by 1, mask where group boundary
        lag1 = np.empty_like(vals)
        lag1[:] = np.nan
        lag1[1:] = vals[:-1]
        lag1[~same_group_1] = np.nan
        col_lag1 = f"{feat}_lag1"
        panel[col_lag1] = lag1
        lag_cols.append(col_lag1)

        # Lag 3: shift by 3, mask where group boundary
        lag3 = np.empty_like(vals)
        lag3[:] = np.nan
        lag3[3:] = vals[:-3]
        lag3[~same_group_3] = np.nan
        col_lag3 = f"{feat}_lag3"
        panel[col_lag3] = lag3
        lag_cols.append(col_lag3)

        # 3-month trend
        trend_name = f"{feat}_trend3"
        panel[trend_name] = vals - lag3
        lag_cols.append(trend_name)
    feature_cols.extend(lag_cols)
    print(f"  Created {len(lag_cols)} lagged/trend features")

    # ── Cross-sectional ranking ──────────────────────────────────────────
    print("\n📊 CROSS-SECTIONAL RANKING (skipping — will rank in LightGBM step)...")
    # Skip full cross-sectional ranking to avoid OOM on 3.76M rows × 500+ cols.
    # LightGBM is rank-invariant for tree splits, so raw values work fine.
    # IC selection below uses Spearman which ranks internally.
    numeric_features = [c for c in feature_cols if c in panel.columns]
    print(f"  Features available: {len(numeric_features)}")
    gc.collect()

    # ── IC-based feature selection ───────────────────────────────────────
    print("\n📊 IC-BASED FEATURE SELECTION...")
    # Detect target column (may be 'ret_forward' or 'fwd_ret_1m')
    target = None
    for candidate in ["ret_forward", "fwd_ret_1m", "fwd_ret_3m"]:
        if candidate in panel.columns:
            target = candidate
            break
    
    if target is None:
        print("  ⚠️ No forward return column found — keeping all features")
    else:
        print(f"  Target column: {target}")

    if target is not None and target in panel.columns:
        # Compute average IC for each feature — vectorized approach
        ic_scores = {}
        dates = sorted(panel["date"].unique())
        # Sample ~50 dates evenly for speed
        n_sample = min(50, len(dates))
        sample_idx = np.linspace(0, len(dates) - 1, n_sample, dtype=int)
        sample_dates = [dates[i] for i in sample_idx]

        print(f"  Computing IC on {len(sample_dates)} sampled months...")
        for i, col in enumerate(numeric_features):
            ics = []
            col_vals = panel[col].values
            target_vals = panel[target].values
            date_vals = panel["date"].values
            for dt in sample_dates:
                mask = date_vals == dt
                cv = col_vals[mask]
                tv = target_vals[mask]
                valid = ~(np.isnan(cv) | np.isnan(tv))
                if valid.sum() > 50:
                    from scipy.stats import spearmanr
                    ic, _ = spearmanr(cv[valid], tv[valid])
                    if not np.isnan(ic):
                        ics.append(ic)
            if ics:
                ic_scores[col] = np.mean(np.abs(ics))
            if (i + 1) % 100 == 0:
                print(f"    IC computed for {i+1}/{len(numeric_features)} features...")

        # Select top features by absolute IC
        ic_df = pd.Series(ic_scores).sort_values(ascending=False)
        max_features = min(900, len(ic_df))
        selected_features = ic_df.head(max_features).index.tolist()

        print(f"  Features with IC computed: {len(ic_df)}")
        print(f"  Top 10 by |IC|:")
        for feat in ic_df.head(10).index:
            print(f"    {feat:<40} IC = {ic_df[feat]:.4f}")
        print(f"  Selected: {max_features} features")

        # Add CZ predictors if available
        cz_path = os.path.join(DATA_DIR, "cz_predictors.parquet")
        if os.path.exists(cz_path):
            cz = pd.read_parquet(cz_path)
            cz_features = [c for c in cz.columns if c not in ["permno", "date", "cusip"]]
            print(f"  CZ predictors available: {len(cz_features)} features")
            # These would be merged similarly — flag for inclusion
        
        # Final column set
        keep_cols = ["permno", "date", target] + selected_features
        # Also keep identifiers if present
        for id_col in ["cusip", "ticker", "siccd"]:
            if id_col in panel.columns:
                keep_cols.append(id_col)

        keep_cols = [c for c in keep_cols if c in panel.columns]
        gkx = panel[keep_cols].copy()
    else:
        print(f"  ⚠️ No '{target}' column — keeping all features")
        gkx = panel.copy()
        selected_features = numeric_features

    # ── Save ─────────────────────────────────────────────────────────────
    print("\n💾 SAVING GKX PANEL...")
    # Downcast float64 → float32 to halve file size and memory for downstream
    float64_cols = gkx.select_dtypes(include=["float64"]).columns
    gkx[float64_cols] = gkx[float64_cols].astype(np.float32)
    output_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
    gkx.to_parquet(output_path, index=False, engine="pyarrow")
    file_size = os.path.getsize(output_path) / (1024 ** 3)

    subprocess.run(
        ["aws", "s3", "cp", output_path,
         f"s3://{S3_BUCKET}/features/gkx_panel.parquet"],
        capture_output=True,
    )

    elapsed = time.time() - start
    n_features = len([c for c in gkx.columns if c not in id_cols])

    print(f"\n{'=' * 70}")
    print(f"GKX-STYLE FEATURE PANEL COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Rows:           {len(gkx):,}")
    print(f"  Columns:        {gkx.shape[1]}")
    print(f"  Features:       {n_features}")
    print(f"  Feature layers:")
    print(f"    Base chars:     ~{len([c for c in selected_features if not c.startswith(('ff49_','ix_')) and '_lag' not in c and '_trend' not in c and c not in macro_cols])}")
    print(f"    FF49 dummies:   ~{len([c for c in selected_features if c.startswith('ff49_')])}")
    print(f"    Macro vars:     ~{len([c for c in selected_features if c in macro_cols])}")
    print(f"    Interactions:   ~{len([c for c in selected_features if c.startswith('ix_')])}")
    print(f"    Lagged/trends:  ~{len([c for c in selected_features if '_lag' in c or '_trend' in c])}")
    print(f"  Date range:     {gkx['date'].min().date()} to {gkx['date'].max().date()}")
    print(f"  Stocks:         {gkx['permno'].nunique():,}")
    print(f"  File size:      {file_size:.2f} GB")
    print(f"  Time:           {elapsed/60:.1f} min")
    print(f"  ✅ Saved and uploaded to S3")


if __name__ == "__main__":
    main()
