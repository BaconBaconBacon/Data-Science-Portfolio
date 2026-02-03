#!/bin/bash
# EC2 Bootstrap Script for Wildfire Risk ML Training
# This script runs automatically when the EC2 instance launches (via user-data)
#
# Usage: Pass this script to --user-data when launching the instance
# ============================================================================

set -e  # Exit on any error

LOG_FILE="/home/ec2-user/training.log"
exec > >(tee -a "$LOG_FILE") 2>&1  # Log everything

echo "=========================================="
echo "Wildfire Risk ML - EC2 Bootstrap"
echo "Started: $(date)"
echo "=========================================="

# Update system
echo "[1/6] Updating system packages..."
sudo yum update -y

# Install Python 3.11
echo "[2/6] Installing Python 3.11..."
sudo yum install -y python3.11 python3.11-pip

# Create working directory
echo "[3/6] Setting up working directory..."
cd /home/ec2-user
mkdir -p wildfire-ml
cd wildfire-ml

# Install Python dependencies
echo "[4/6] Installing Python packages..."
pip3.11 install --user \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    boto3 \
    pyarrow

# Download train.py from S3 (simpler than git clone for a demo)
echo "[5/6] Downloading training script from S3..."
aws s3 cp s3://wildfire-risk-ml/scripts/train.py .

# Run training
echo "[6/6] Starting model training..."
echo "=========================================="

python3.11 train.py \
    --input s3://wildfire-risk-ml/data/model_joined.parquet \
    --output s3://wildfire-risk-ml/models/best_model.pkl \
    --n-iter 50 \
    --model both

echo "=========================================="
echo "Training complete: $(date)"
echo "=========================================="

# Upload log to S3
aws s3 cp "$LOG_FILE" s3://wildfire-risk-ml/logs/training_$(date +%Y%m%d_%H%M%S).log

# Auto-shutdown to save costs
echo "Shutting down instance..."
sudo shutdown -h now