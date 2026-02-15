"""
PHASE 3 — STEP 10: Monthly Refresh Pipeline
=============================================
Automated monthly data refresh: download latest WRDS data,
rebuild features, retrain model, update predictions.
Run on 5th of each month (data available with ~1 week lag).
Can be triggered via cron, Lambda, or manual execution.
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import json
import subprocess
from datetime import datetime, timedelta

sys.path.insert(0, "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3")

DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"
S3_BUCKET = "nuble-data-warehouse"
LOG_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/wrds_pipeline/phase3/logs"

WRDS_PARAMS = {
    "host": os.environ.get("WRDS_PGHOST", "wrds-pgdata.wharton.upenn.edu"),
    "port": int(os.environ.get("WRDS_PGPORT", "9737")),
    "dbname": os.environ.get("WRDS_PGDATABASE", "wrds"),
    "user": os.environ.get("WRDS_USERNAME", "hlobo"),
    "password": os.environ.get("WRDS_PASSWORD", ""),
}


def log_step(step_name, status, details=""):
    """Log pipeline step result."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "status": status,
        "details": details,
    }
    log_file = os.path.join(LOG_DIR, f"refresh_{datetime.now().strftime('%Y%m')}.jsonl")
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")
    symbol = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
    print(f"  {symbol} [{step_name}] {status} {details}")


def step1_download_latest_crsp():
    """Download latest month of CRSP data."""
    import psycopg2

    try:
        conn = psycopg2.connect(**WRDS_PARAMS)
        # Get the latest available month
        latest = pd.read_sql(
            "SELECT MAX(date) as max_date FROM crsp.msf", conn
        )
        max_date = latest["max_date"].iloc[0]

        # Download new month
        query = f"""
        SELECT a.permno, a.date, a.ret, a.retx, a.prc, a.vol, a.shrout,
               b.shrcd, b.exchcd, b.siccd, b.ticker, b.comnam
        FROM crsp.msf a
        LEFT JOIN crsp.msenames b ON a.permno = b.permno
          AND a.date BETWEEN b.namedt AND b.nameendt
        WHERE a.date >= '{(max_date - timedelta(days=60)).strftime('%Y-%m-%d')}'
          AND b.shrcd IN (10, 11)
        """
        df = pd.read_sql(query, conn)
        conn.close()
        log_step("download_crsp", "SUCCESS", f"{len(df):,} rows, latest={max_date}")
        return df
    except Exception as e:
        log_step("download_crsp", "FAILED", str(e)[:100])
        return None


def step2_download_latest_compustat():
    """Download latest Compustat quarter."""
    import psycopg2

    try:
        conn = psycopg2.connect(**WRDS_PARAMS)
        query = """
        SELECT gvkey, datadate, fyearq, fqtr, atq, ltq, seqq, ceqq,
               revtq, cogsq, xsgaq, niq, oibdpq, cheq, dlttq, dlcq,
               saleq, ibq, dpq, cshoq
        FROM comp.fundq
        WHERE datadate >= (SELECT MAX(datadate) - INTERVAL '6 months' FROM comp.fundq)
          AND indfmt = 'INDL' AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
        """
        df = pd.read_sql(query, conn)
        conn.close()
        log_step("download_compustat", "SUCCESS", f"{len(df):,} rows")
        return df
    except Exception as e:
        log_step("download_compustat", "FAILED", str(e)[:100])
        return None


def step3_update_fred():
    """Refresh FRED macro data."""
    try:
        exec_script = os.path.join(
            os.path.dirname(__file__), "step3_fred_macro.py"
        )
        result = subprocess.run(
            [sys.executable, exec_script],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            log_step("update_fred", "SUCCESS")
        else:
            log_step("update_fred", "FAILED", result.stderr[:100])
    except Exception as e:
        log_step("update_fred", "FAILED", str(e)[:100])


def step4_rebuild_features():
    """Rebuild GKX feature panel with latest data."""
    try:
        exec_script = os.path.join(
            os.path.dirname(__file__), "step5_gkx_panel.py"
        )
        result = subprocess.run(
            [sys.executable, exec_script],
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode == 0:
            log_step("rebuild_features", "SUCCESS")
        else:
            log_step("rebuild_features", "FAILED", result.stderr[:100])
    except Exception as e:
        log_step("rebuild_features", "FAILED", str(e)[:100])


def step5_retrain_model():
    """Retrain LightGBM with latest data."""
    try:
        exec_script = os.path.join(
            os.path.dirname(__file__), "step6_lightgbm.py"
        )
        result = subprocess.run(
            [sys.executable, exec_script],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode == 0:
            log_step("retrain_model", "SUCCESS")
        else:
            log_step("retrain_model", "FAILED", result.stderr[:100])
    except Exception as e:
        log_step("retrain_model", "FAILED", str(e)[:100])


def step6_generate_predictions():
    """Generate fresh predictions for current month."""
    try:
        import lightgbm as lgb

        gkx_path = os.path.join(DATA_DIR, "gkx_panel.parquet")
        if not os.path.exists(gkx_path):
            log_step("generate_predictions", "FAILED", "No GKX panel")
            return

        gkx = pd.read_parquet(gkx_path)
        gkx["date"] = pd.to_datetime(gkx["date"])
        latest_date = gkx["date"].max()
        latest_data = gkx[gkx["date"] == latest_date]

        # Load latest model (would be saved by step6)
        model_path = os.path.join(DATA_DIR, "lgb_latest_model.txt")
        if os.path.exists(model_path):
            model = lgb.Booster(model_file=model_path)
            id_cols = ["permno", "date", "cusip", "ticker", "siccd", "ret_forward"]
            feature_cols = [c for c in latest_data.columns if c not in id_cols
                           and latest_data[c].dtype in ["float64", "float32", "int64"]]
            X = np.nan_to_num(latest_data[feature_cols].values, nan=0.0)
            preds = model.predict(X)

            result = latest_data[["permno", "date"]].copy()
            result["prediction"] = preds
            result["prediction_date"] = datetime.now().strftime("%Y-%m-%d")

            pred_path = os.path.join(DATA_DIR, "latest_predictions.parquet")
            result.to_parquet(pred_path, index=False, engine="pyarrow")
            log_step("generate_predictions", "SUCCESS", f"{len(result):,} predictions")
        else:
            log_step("generate_predictions", "SKIPPED", "No model file yet")

    except Exception as e:
        log_step("generate_predictions", "FAILED", str(e)[:100])


def step7_upload_to_s3():
    """Upload latest data to S3."""
    try:
        files_to_upload = [
            "macro_predictors.parquet", "fred_daily.parquet", "fred_monthly.parquet",
            "gkx_panel.parquet", "lgb_predictions.parquet", "latest_predictions.parquet",
        ]
        uploaded = 0
        for fname in files_to_upload:
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.exists(fpath):
                subprocess.run(
                    ["aws", "s3", "cp", fpath,
                     f"s3://{S3_BUCKET}/features/{fname}"],
                    capture_output=True, timeout=300,
                )
                uploaded += 1

        log_step("s3_upload", "SUCCESS", f"{uploaded} files uploaded")
    except Exception as e:
        log_step("s3_upload", "FAILED", str(e)[:100])


def step8_validate():
    """Run validation checks on refreshed data."""
    checks_passed = 0
    checks_total = 0

    # Check 1: Panel exists and is recent
    checks_total += 1
    panel_path = os.path.join(DATA_DIR, "training_panel.parquet")
    if os.path.exists(panel_path):
        panel = pd.read_parquet(panel_path, columns=["date"])
        max_date = pd.to_datetime(panel["date"]).max()
        days_old = (datetime.now() - max_date).days
        if days_old < 90:
            checks_passed += 1
        else:
            print(f"    ⚠️ Panel is {days_old} days old")
    del panel

    # Check 2: Macro data updated
    checks_total += 1
    macro_path = os.path.join(DATA_DIR, "macro_predictors.parquet")
    if os.path.exists(macro_path):
        macro = pd.read_parquet(macro_path, columns=["date"])
        macro_max = pd.to_datetime(macro["date"]).max()
        if (datetime.now() - macro_max).days < 60:
            checks_passed += 1
    del macro

    # Check 3: Predictions exist
    checks_total += 1
    pred_path = os.path.join(DATA_DIR, "lgb_predictions.parquet")
    if os.path.exists(pred_path):
        preds = pd.read_parquet(pred_path)
        if len(preds) > 0:
            checks_passed += 1

    log_step("validation", "SUCCESS" if checks_passed == checks_total else "WARNING",
             f"{checks_passed}/{checks_total} checks passed")


def main():
    print("=" * 70)
    print(f"PHASE 3 — STEP 10: MONTHLY REFRESH PIPELINE")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    start = time.time()

    # 10-step monthly refresh
    steps = [
        ("1. Download latest CRSP", step1_download_latest_crsp),
        ("2. Download latest Compustat", step2_download_latest_compustat),
        ("3. Update FRED macro", step3_update_fred),
        ("4. Rebuild features", step4_rebuild_features),
        ("5. Retrain model", step5_retrain_model),
        ("6. Generate predictions", step6_generate_predictions),
        ("7. Upload to S3", step7_upload_to_s3),
        ("8. Validate", step8_validate),
    ]

    print("\n🔄 EXECUTING REFRESH PIPELINE...")
    for step_name, step_func in steps:
        print(f"\n  ── {step_name} ──")
        try:
            step_func()
        except Exception as e:
            log_step(step_name, "FAILED", str(e)[:100])

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"MONTHLY REFRESH COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Log: {LOG_DIR}/refresh_{datetime.now().strftime('%Y%m')}.jsonl")
    print(f"  ✅ Pipeline complete — check logs for details")


if __name__ == "__main__":
    main()
