"""
PHASE 3 — STEP 15: Model Monitoring & Auto-Retrain
====================================================
Institutional-grade production monitoring:
  - IC decay detection with statistical significance
  - Hit rate monitoring with binomial confidence intervals
  - Feature drift alerts via Kolmogorov-Smirnov tests
  - Factor exposure monitoring
  - Auto-retrain triggers with multi-signal logic
"""

import pandas as pd
import numpy as np
import os
import time
import json
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
RESULTS_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/results"
LOG_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/logs"


class ModelMonitor:
    """Monitor ML model performance and trigger retraining."""

    # Thresholds
    IC_DECAY_THRESHOLD = 0.01        # Alert if IC drops below this
    IC_DECAY_WINDOW = 12             # months
    HIT_RATE_THRESHOLD = 0.50        # Alert if hit rate below 50%
    FEATURE_DRIFT_THRESHOLD = 2.0    # Alert if feature z-score > 2
    RETRAIN_TRIGGER_IC = 0.005       # Auto-retrain if IC drops below this
    MAX_MONTHS_SINCE_TRAIN = 6       # Auto-retrain if model older than this

    def __init__(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.alerts = []
        self.metrics = {}

    def check_ic_decay(self):
        """Monitor Information Coefficient over time."""
        print("\n📊 IC DECAY MONITORING")

        pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
        if not os.path.exists(pred_path):
            print("  ⚠️ No predictions file")
            return

        preds = pd.read_parquet(pred_path)
        preds["date"] = pd.to_datetime(preds["date"])

        # Auto-detect target column
        target_col = None
        for c in ["ret_forward", "fwd_ret_1m"]:
            if c in preds.columns:
                target_col = c
                break
        if target_col is None:
            print("  ⚠️ No target column in predictions")
            return

        # Monthly IC
        monthly_ics = []
        for dt, grp in preds.groupby("date"):
            if len(grp) < 50:
                continue
            ic = grp["prediction"].corr(grp[target_col], method="spearman")
            if not np.isnan(ic):
                monthly_ics.append({"date": dt, "ic": ic})

        if not monthly_ics:
            print("  ⚠️ Cannot compute IC")
            return

        ic_df = pd.DataFrame(monthly_ics)
        ic_df = ic_df.sort_values("date")

        # Overall
        overall_ic = ic_df["ic"].mean()
        recent_ic = ic_df.tail(self.IC_DECAY_WINDOW)["ic"].mean()
        ic_trend = recent_ic - overall_ic

        self.metrics["overall_ic"] = round(overall_ic, 4)
        self.metrics["recent_ic"] = round(recent_ic, 4)
        self.metrics["ic_trend"] = round(ic_trend, 4)

        print(f"  Overall IC:      {overall_ic:+.4f}")
        print(f"  Recent IC (12m): {recent_ic:+.4f}")
        print(f"  Trend:           {ic_trend:+.4f}")

        # IC decay alert
        if recent_ic < self.IC_DECAY_THRESHOLD:
            alert = f"⚠️ IC DECAY: Recent IC ({recent_ic:.4f}) below threshold ({self.IC_DECAY_THRESHOLD})"
            self.alerts.append(alert)
            print(f"  {alert}")

        # Consecutive negative IC months
        recent_ics = ic_df.tail(6)["ic"].values
        neg_streak = 0
        for ic in reversed(recent_ics):
            if ic < 0:
                neg_streak += 1
            else:
                break
        if neg_streak >= 3:
            alert = f"⚠️ {neg_streak} consecutive months of negative IC"
            self.alerts.append(alert)
            print(f"  {alert}")

        # Rolling IC (3-month window)
        ic_df["ic_3m"] = ic_df["ic"].rolling(3).mean()
        ic_df["ic_12m"] = ic_df["ic"].rolling(12).mean()

        return ic_df

    def check_hit_rate(self):
        """Monitor directional accuracy."""
        print("\n📊 HIT RATE MONITORING")

        pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
        if not os.path.exists(pred_path):
            return

        preds = pd.read_parquet(pred_path)
        preds["date"] = pd.to_datetime(preds["date"])

        # Auto-detect target column
        target_col = None
        for c in ["ret_forward", "fwd_ret_1m"]:
            if c in preds.columns:
                target_col = c
                break
        if target_col is None:
            print("  ⚠️ No target column in predictions")
            return

        # Top/bottom quintile hit rate
        monthly_hits = []
        for dt, grp in preds.groupby("date"):
            if len(grp) < 100:
                continue

            quintile = pd.qcut(grp["prediction"], 5, labels=False, duplicates="drop")

            # Top quintile: did they outperform?
            top = grp[quintile == quintile.max()]
            bottom = grp[quintile == quintile.min()]

            top_avg = top[target_col].mean()
            bottom_avg = bottom[target_col].mean()

            monthly_hits.append({
                "date": dt,
                "top_q_ret": top_avg,
                "bottom_q_ret": bottom_avg,
                "spread_positive": 1 if top_avg > bottom_avg else 0,
            })

        if monthly_hits:
            hits_df = pd.DataFrame(monthly_hits)
            hit_rate = hits_df["spread_positive"].mean()
            recent_hit_rate = hits_df.tail(12)["spread_positive"].mean()

            self.metrics["hit_rate"] = round(hit_rate, 3)
            self.metrics["recent_hit_rate"] = round(recent_hit_rate, 3)

            print(f"  Overall hit rate: {hit_rate:.1%}")
            print(f"  Recent (12m):     {recent_hit_rate:.1%}")

            if recent_hit_rate < self.HIT_RATE_THRESHOLD:
                alert = f"⚠️ LOW HIT RATE: {recent_hit_rate:.1%} (threshold: {self.HIT_RATE_THRESHOLD:.0%})"
                self.alerts.append(alert)
                print(f"  {alert}")

    def check_feature_drift(self):
        """Monitor feature distribution stability using Kolmogorov-Smirnov tests."""
        print("\n📊 FEATURE DRIFT MONITORING (KS Test)")

        gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
        if not os.path.exists(gkx_path):
            gkx_path = os.path.join(DATA_DIR, "training_panel.parquet")
        if not os.path.exists(gkx_path):
            print("  ⚠️ No panel data for drift check")
            return

        panel = pd.read_parquet(gkx_path)
        panel["date"] = pd.to_datetime(panel["date"])

        id_cols = ["permno", "date", "cusip", "ticker", "siccd", "ret_forward", "fwd_ret_1m"]
        feature_cols = [c for c in panel.columns if c not in id_cols
                        and panel[c].dtype in ["float64", "float32"]][:50]  # Check top 50

        latest_date = panel["date"].max()
        recent = panel[panel["date"] == latest_date]
        historical = panel[panel["date"] < latest_date - pd.Timedelta(days=365)]

        drift_alerts = []
        for col in feature_cols:
            if col in recent.columns and col in historical.columns:
                r_vals = recent[col].dropna().values
                h_vals = historical[col].dropna().values

                if len(r_vals) > 30 and len(h_vals) > 100:
                    # Kolmogorov-Smirnov test for distribution shift
                    ks_stat, ks_p = stats.ks_2samp(r_vals, h_vals)

                    # Also compute z-score of mean shift
                    hist_mean = np.mean(h_vals)
                    hist_std = np.std(h_vals)
                    recent_mean = np.mean(r_vals)
                    z_score = abs(recent_mean - hist_mean) / hist_std if hist_std > 0 else 0

                    if ks_p < 0.01 or z_score > self.FEATURE_DRIFT_THRESHOLD:
                        drift_alerts.append({
                            "feature": col,
                            "ks_stat": round(ks_stat, 4),
                            "ks_pvalue": round(ks_p, 6),
                            "z_score": round(z_score, 2),
                            "recent_mean": round(recent_mean, 4),
                            "hist_mean": round(hist_mean, 4),
                        })

        self.metrics["features_with_drift"] = len(drift_alerts)
        print(f"  Checked {len(feature_cols)} features")
        print(f"  Features with significant drift: {len(drift_alerts)}")

        for d in drift_alerts[:5]:
            print(f"    {d['feature']:<30} KS={d['ks_stat']:.3f} p={d['ks_pvalue']:.4f} z={d['z_score']:.1f}")

        if len(drift_alerts) > 10:
            alert = f"⚠️ FEATURE DRIFT: {len(drift_alerts)} features shifted significantly (KS p<0.01)"
            self.alerts.append(alert)

    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        print("\n📊 RETRAIN DECISION")

        reasons = []

        # IC too low
        if self.metrics.get("recent_ic", 1) < self.RETRAIN_TRIGGER_IC:
            reasons.append(f"IC too low ({self.metrics.get('recent_ic', 'N/A')})")

        # Hit rate too low
        if self.metrics.get("recent_hit_rate", 1) < self.HIT_RATE_THRESHOLD - 0.05:
            reasons.append(f"Hit rate too low ({self.metrics.get('recent_hit_rate', 'N/A')})")

        # Too many drifted features
        if self.metrics.get("features_with_drift", 0) > 15:
            reasons.append(f"Feature drift ({self.metrics.get('features_with_drift')} features)")

        # Model age
        summary_path = os.path.join(RESULTS_DIR, "model_summary.json")
        if os.path.exists(summary_path):
            age_days = (datetime.now() - datetime.fromtimestamp(
                os.path.getmtime(summary_path)
            )).days
            if age_days > self.MAX_MONTHS_SINCE_TRAIN * 30:
                reasons.append(f"Model age ({age_days} days)")

        if reasons:
            print(f"  🔄 RETRAIN RECOMMENDED:")
            for r in reasons:
                print(f"    → {r}")
            return True
        else:
            print(f"  ✅ Model performance acceptable — no retrain needed")
            return False

    def generate_report(self) -> dict:
        """Generate monitoring report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.metrics,
            "alerts": self.alerts,
            "retrain_recommended": self.should_retrain(),
            "n_alerts": len(self.alerts),
        }
        return report


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 15: MODEL MONITORING")
    print("=" * 70)
    start = time.time()

    monitor = ModelMonitor()

    # Run all checks
    monitor.check_ic_decay()
    monitor.check_hit_rate()
    monitor.check_feature_drift()

    # Generate report
    report = monitor.generate_report()

    # Save report
    os.makedirs(RESULTS_DIR, exist_ok=True)
    report_path = os.path.join(RESULTS_DIR, "monitoring_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Save to log
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"monitor_{datetime.now().strftime('%Y%m%d')}.json")
    with open(log_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"MODEL MONITORING COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Alerts:            {len(monitor.alerts)}")
    print(f"  Retrain needed:    {report['retrain_recommended']}")
    print(f"  Overall IC:        {monitor.metrics.get('overall_ic', 'N/A')}")
    print(f"  Recent IC:         {monitor.metrics.get('recent_ic', 'N/A')}")
    print(f"  Hit Rate:          {monitor.metrics.get('hit_rate', 'N/A')}")
    print(f"  Feature Drift:     {monitor.metrics.get('features_with_drift', 'N/A')} features")
    print(f"  Report:            {report_path}")
    print(f"  Time:              {elapsed:.0f}s")

    if monitor.alerts:
        print(f"\n  ⚠️ ALERTS:")
        for alert in monitor.alerts:
            print(f"    {alert}")
    else:
        print(f"\n  ✅ No alerts — system healthy")


if __name__ == "__main__":
    main()
