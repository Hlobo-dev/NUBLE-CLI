#!/bin/bash
# ══════════════════════════════════════════════════════════════════
# DEPLOY WRDS PIPELINE TO EC2 AND RUN
# ══════════════════════════════════════════════════════════════════
#
# This script:
#   1. Packages the wrds_pipeline/ module
#   2. SCPs it to your EC2 instance
#   3. SSHs in, installs dependencies, and runs the full pipeline
#
# USAGE:
#   chmod +x wrds_pipeline/deploy_and_run.sh
#   ./wrds_pipeline/deploy_and_run.sh
#
# PREREQUISITES:
#   - EC2 instance running with SSH access
#   - EC2 security group allows outbound 9737 (WRDS) and inbound from your IP
#   - RDS security group allows EC2 private IP
#
# CONFIGURATION: Set these to your EC2 details
# ══════════════════════════════════════════════════════════════════

# ── EC2 Connection ──
# Replace with your EC2 details:
EC2_USER="ec2-user"                               # Amazon Linux default
EC2_HOST=""                                         # Your EC2 public IP or hostname
EC2_KEY="~/.ssh/your-key.pem"                       # Path to your SSH key
EC2_DIR="/home/${EC2_USER}/wrds_pipeline_deploy"    # Remote working directory

# ══════════════════════════════════════════════════════════════════
# OPTION 1: If you have an EC2 instance, deploy there
# ══════════════════════════════════════════════════════════════════

deploy_to_ec2() {
    echo "═══════════════════════════════════════════════════════"
    echo "  DEPLOYING WRDS PIPELINE TO EC2"
    echo "═══════════════════════════════════════════════════════"

    if [ -z "$EC2_HOST" ]; then
        echo "❌ EC2_HOST not set. Edit this script and set your EC2 hostname/IP."
        echo ""
        echo "   Or run locally if you can reach WRDS port 9737:"
        echo "   python wrds_pipeline/run_complete_pipeline.py"
        exit 1
    fi

    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

    echo "  Project: $PROJECT_DIR"
    echo "  Target:  ${EC2_USER}@${EC2_HOST}:${EC2_DIR}"
    echo ""

    # Create tarball of the pipeline
    echo "  Packaging wrds_pipeline/..."
    cd "$PROJECT_DIR"
    tar -czf /tmp/wrds_pipeline.tar.gz \
        wrds_pipeline/__init__.py \
        wrds_pipeline/config.py \
        wrds_pipeline/download_all.py \
        wrds_pipeline/validate.py \
        wrds_pipeline/characteristics.py \
        wrds_pipeline/data_access.py \
        wrds_pipeline/run_complete_pipeline.py

    echo "  ✅ Package created: $(du -h /tmp/wrds_pipeline.tar.gz | cut -f1)"

    # Upload to EC2
    echo "  Uploading to EC2..."
    ssh -i "$EC2_KEY" "${EC2_USER}@${EC2_HOST}" "mkdir -p ${EC2_DIR}"
    scp -i "$EC2_KEY" /tmp/wrds_pipeline.tar.gz "${EC2_USER}@${EC2_HOST}:${EC2_DIR}/"

    # Install dependencies and run
    echo "  Installing dependencies and running pipeline..."
    ssh -i "$EC2_KEY" "${EC2_USER}@${EC2_HOST}" << 'REMOTE_SCRIPT'
        cd ~/wrds_pipeline_deploy
        tar -xzf wrds_pipeline.tar.gz

        # Install Python if needed
        if ! command -v python3 &> /dev/null; then
            sudo yum install -y python3 python3-pip 2>/dev/null || \
            sudo apt-get install -y python3 python3-pip 2>/dev/null
        fi

        # Create venv
        python3 -m venv .venv 2>/dev/null || true
        source .venv/bin/activate

        # Install packages
        pip install --upgrade pip
        pip install wrds psycopg2-binary pandas numpy scipy sqlalchemy

        # Run the full pipeline
        echo ""
        echo "═══════════════════════════════════════════════════════"
        echo "  STARTING WRDS PIPELINE (this takes 1-4 hours)"
        echo "═══════════════════════════════════════════════════════"
        echo ""

        # Use nohup so it survives SSH disconnection
        nohup python -m wrds_pipeline.run_complete_pipeline > wrds_pipeline_output.log 2>&1 &
        PID=$!
        echo "  Pipeline started with PID $PID"
        echo "  Monitor with: tail -f ~/wrds_pipeline_deploy/wrds_pipeline_output.log"
        echo "  Or reconnect and check: tail -100 ~/wrds_pipeline_deploy/wrds_pipeline_output.log"
        echo ""
        echo "  The pipeline is RESUMABLE — if it stops, just re-run."

        # Wait a bit to show initial output
        sleep 10
        tail -20 wrds_pipeline_output.log
REMOTE_SCRIPT

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  DEPLOYMENT COMPLETE"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "  Monitor remotely:"
    echo "    ssh -i $EC2_KEY ${EC2_USER}@${EC2_HOST} 'tail -f ~/wrds_pipeline_deploy/wrds_pipeline_output.log'"
    echo ""
    echo "  Check if done:"
    echo "    ssh -i $EC2_KEY ${EC2_USER}@${EC2_HOST} 'tail -5 ~/wrds_pipeline_deploy/wrds_pipeline_output.log'"
    echo ""
}

# ══════════════════════════════════════════════════════════════════
# OPTION 2: Run locally (if WRDS port 9737 is reachable)
# ══════════════════════════════════════════════════════════════════

run_locally() {
    echo "═══════════════════════════════════════════════════════"
    echo "  RUNNING WRDS PIPELINE LOCALLY"
    echo "═══════════════════════════════════════════════════════"
    echo ""

    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    cd "$PROJECT_DIR"

    # Activate venv if exists
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi

    python -m wrds_pipeline.run_complete_pipeline
}

# ══════════════════════════════════════════════════════════════════
# OPTION 3: Run via WRDS Cloud SSH
# ══════════════════════════════════════════════════════════════════

run_via_wrds_cloud() {
    echo "═══════════════════════════════════════════════════════"
    echo "  WRDS CLOUD APPROACH"
    echo "═══════════════════════════════════════════════════════"
    echo ""
    echo "  WRDS provides a cloud compute environment where"
    echo "  port 9737 is guaranteed accessible."
    echo ""
    echo "  Step 1: SSH into WRDS cloud"
    echo "    ssh hlobo@wrds-cloud.wharton.upenn.edu"
    echo "    (password: \$WRDS_PASSWORD from .env)"
    echo ""
    echo "  Step 2: Once logged in, install dependencies"
    echo "    pip install --user wrds psycopg2-binary pandas numpy scipy"
    echo ""
    echo "  Step 3: Upload this pipeline"
    echo "    From your Mac:"
    echo "    scp -r wrds_pipeline/ hlobo@wrds-cloud.wharton.upenn.edu:~/"
    echo ""
    echo "  Step 4: Run the pipeline from WRDS cloud"
    echo "    cd ~/wrds_pipeline"
    echo "    python -m wrds_pipeline.run_complete_pipeline"
    echo ""
    echo "  NOTE: WRDS cloud CAN reach port 9737 (same network)"
    echo "  but you need the RDS security group to allow WRDS cloud's IP."
    echo "  WRDS cloud IP range: 165.123.0.0/16 (UPenn)"
    echo ""
}

# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

case "${1:-}" in
    ec2)
        deploy_to_ec2
        ;;
    local)
        run_locally
        ;;
    wrds)
        run_via_wrds_cloud
        ;;
    test)
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
        cd "$PROJECT_DIR"
        if [ -f ".venv/bin/activate" ]; then source .venv/bin/activate; fi
        python -m wrds_pipeline.run_complete_pipeline --test-only
        ;;
    *)
        echo ""
        echo "WRDS Pipeline Deployment"
        echo "========================"
        echo ""
        echo "Usage: $0 {ec2|local|wrds|test}"
        echo ""
        echo "  ec2    Deploy to EC2 instance and run (RECOMMENDED)"
        echo "  local  Run locally (only if port 9737 is reachable)"
        echo "  wrds   Instructions for running via WRDS cloud SSH"
        echo "  test   Test connectivity only"
        echo ""
        echo "Quick connectivity test:"
        echo "  nc -zv wrds-pgdata.wharton.upenn.edu 9737"
        echo ""
        echo "If the above times out, use 'ec2' or 'wrds' option."
        echo ""
        ;;
esac
