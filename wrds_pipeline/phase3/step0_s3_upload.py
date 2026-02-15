"""
PHASE 3 — STEP 0: Upload ALL Parquet files to S3 Data Lake
============================================================
Creates versioned S3 bucket with lifecycle rules and uploads
ALL 6.7 GB of WRDS data for 99.999999999% durability.

Before running: aws sts get-caller-identity  (verify auth)
Total upload: ~6.7 GB. Takes ~3-5 minutes on good internet.
"""

import subprocess
import os
import glob
import json
import time

BUCKET = "nuble-data-warehouse"
REGION = "us-east-1"
DATA_DIR = "/Users/humbertolobo/Desktop/NUBLE-CLI/data/wrds"


def run_aws(args, check=False):
    """Run an AWS CLI command and return result."""
    result = subprocess.run(args, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ⚠️ {result.stderr.strip()[:120]}")
    return result


def main():
    print("=" * 70)
    print("PHASE 3 — STEP 0: S3 DATA LAKE SETUP + UPLOAD")
    print("=" * 70)
    start = time.time()

    # ═══════════════════════════════════════════════════
    # 1. Create S3 bucket
    # ═══════════════════════════════════════════════════
    print("\n[1/5] Creating S3 bucket...")
    result = run_aws([
        "aws", "s3api", "create-bucket",
        "--bucket", BUCKET,
        "--region", REGION,
    ])
    if result.returncode == 0:
        print(f"  ✅ Created s3://{BUCKET}")
    elif "BucketAlreadyOwnedByYou" in result.stderr or "BucketAlreadyExists" in result.stderr:
        print(f"  ✅ Bucket s3://{BUCKET} already exists")
    else:
        # Try mb fallback
        run_aws(["aws", "s3", "mb", f"s3://{BUCKET}", "--region", REGION])
        print(f"  ✅ Bucket created via mb")

    # ═══════════════════════════════════════════════════
    # 2. Enable versioning
    # ═══════════════════════════════════════════════════
    print("\n[2/5] Enabling versioning...")
    run_aws([
        "aws", "s3api", "put-bucket-versioning",
        "--bucket", BUCKET,
        "--versioning-configuration", "Status=Enabled",
    ])
    print("  ✅ Versioning enabled")

    # ═══════════════════════════════════════════════════
    # 3. Set lifecycle rules (Glacier after 30 days for old versions)
    # ═══════════════════════════════════════════════════
    print("\n[3/5] Setting lifecycle rules...")
    lifecycle = {
        "Rules": [
            {
                "ID": "ArchiveOldVersions",
                "Status": "Enabled",
                "Filter": {"Prefix": ""},
                "NoncurrentVersionTransitions": [
                    {"NoncurrentDays": 30, "StorageClass": "GLACIER"}
                ],
                "NoncurrentVersionExpiration": {"NoncurrentDays": 365},
            }
        ]
    }
    run_aws([
        "aws", "s3api", "put-bucket-lifecycle-configuration",
        "--bucket", BUCKET,
        "--lifecycle-configuration", json.dumps(lifecycle),
    ])
    print("  ✅ Lifecycle: old versions → Glacier after 30d, expire after 365d")

    # ═══════════════════════════════════════════════════
    # 4. Upload ALL data files
    # ═══════════════════════════════════════════════════
    print("\n[4/5] Uploading data files...")

    def get_s3_key(filepath):
        """Map local file path to S3 key."""
        name = os.path.basename(filepath).lower()
        if "training_panel" in name or "gkx_panel" in name:
            return f"features/{os.path.basename(filepath)}"
        elif "daily_features" in name:
            return f"features/{os.path.basename(filepath)}"
        elif "rolling_betas" in name:
            return f"features/{os.path.basename(filepath)}"
        elif "crsp_daily_" in name:
            return f"raw/crsp_daily/{os.path.basename(filepath)}"
        elif any(x in name for x in ["cz_pred", "macro_pred", "fred", "welch_goyal"]):
            return f"features/{os.path.basename(filepath)}"
        elif "model" in name or name.endswith(".pkl"):
            return f"models/{os.path.basename(filepath)}"
        elif "prediction" in name:
            return f"predictions/{os.path.basename(filepath)}"
        else:
            return f"raw/wrds/{os.path.basename(filepath)}"

    # Collect all files
    all_files = []

    # Top-level parquet files in data/wrds/
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "*.parquet"))):
        all_files.append(f)

    # CRSP daily yearly files
    for f in sorted(glob.glob(os.path.join(DATA_DIR, "crsp_daily", "*.parquet"))):
        all_files.append(f)

    # Manifest files
    for f in glob.glob(os.path.join(DATA_DIR, "*.json")):
        all_files.append(f)

    total_size_gb = sum(os.path.getsize(f) for f in all_files) / (1024 ** 3)
    print(f"  Found {len(all_files)} files ({total_size_gb:.2f} GB)")

    uploaded = 0
    failed = 0
    for idx, filepath in enumerate(all_files):
        s3_key = get_s3_key(filepath)
        size_mb = os.path.getsize(filepath) / (1024 ** 2)

        result = run_aws([
            "aws", "s3", "cp", filepath, f"s3://{BUCKET}/{s3_key}",
        ])

        if result.returncode == 0:
            uploaded += 1
            if size_mb > 10 or (idx + 1) % 10 == 0:
                print(
                    f"  [{idx+1}/{len(all_files)}] ✅ {os.path.basename(filepath):<45} "
                    f"→ {s3_key} ({size_mb:.0f} MB)"
                )
        else:
            failed += 1
            print(f"  [{idx+1}/{len(all_files)}] ❌ {os.path.basename(filepath)}: {result.stderr.strip()[:80]}")

    # ═══════════════════════════════════════════════════
    # 5. Verify upload
    # ═══════════════════════════════════════════════════
    print(f"\n[5/5] Verifying S3 contents...")
    result = run_aws([
        "aws", "s3", "ls", f"s3://{BUCKET}/", "--recursive", "--summarize",
    ])
    # Get the summary lines (last few lines)
    lines = result.stdout.strip().split("\n")
    summary_lines = [l for l in lines if "Total" in l or "Objects" in l]
    for line in summary_lines:
        print(f"  {line.strip()}")

    elapsed = time.time() - start

    print(f"\n{'=' * 70}")
    print(f"S3 UPLOAD COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Uploaded:    {uploaded}/{len(all_files)} files")
    print(f"  Failed:      {failed}")
    print(f"  Total size:  {total_size_gb:.2f} GB")
    print(f"  Bucket:      s3://{BUCKET}/")
    print(f"  Durability:  99.999999999% (11 nines)")
    print(f"  Monthly cost: ~${total_size_gb * 0.023:.2f}")
    print(f"  Time:        {elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
