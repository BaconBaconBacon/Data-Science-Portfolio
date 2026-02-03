# Deploy script for Wildfire Risk ML
# Syncs local code and data to S3 bucket
#
# Usage: .\aws\deploy.ps1 [-Code] [-Data] [-All] [-Launch]
# ============================================================================

param(
    [switch]$Code,      # Upload train.py and scripts
    [switch]$Data,      # Upload training data
    [switch]$All,       # Upload everything
    [switch]$Launch     # Launch EC2 instance after upload
)

$BUCKET = "wildfire-risk-ml"
$PROJECT_DIR = Split-Path -Parent $PSScriptRoot  # Parent of aws/

function Upload-Code {
    Write-Host "`n[Code] Uploading scripts to S3..." -ForegroundColor Cyan

    # Main training script
    python -m awscli s3 cp "$PROJECT_DIR\train.py" "s3://$BUCKET/scripts/train.py"

    # Bootstrap script
    python -m awscli s3 cp "$PROJECT_DIR\aws\setup_ec2.sh" "s3://$BUCKET/scripts/setup_ec2.sh"

    Write-Host "[Code] Done!" -ForegroundColor Green
}

function Upload-Data {
    Write-Host "`n[Data] Uploading training data to S3..." -ForegroundColor Cyan

    # Training parquet
    if (Test-Path "$PROJECT_DIR\data\model_joined.parquet") {
        python -m awscli s3 cp "$PROJECT_DIR\data\model_joined.parquet" "s3://$BUCKET/data/model_joined.parquet"
    } else {
        Write-Host "[Data] Warning: model_joined.parquet not found, skipping" -ForegroundColor Yellow
    }

    # Test data (if exists)
    if (Test-Path "$PROJECT_DIR\data\test_joined.parquet") {
        python -m awscli s3 cp "$PROJECT_DIR\data\test_joined.parquet" "s3://$BUCKET/data/test_joined.parquet"
    }

    Write-Host "[Data] Done!" -ForegroundColor Green
}

function Launch-Training {
    Write-Host "`n[Launch] Starting EC2 spot instance..." -ForegroundColor Cyan

    # Use file:// for JSON to avoid PowerShell escaping hell
    $awsDir = "$PROJECT_DIR\aws"

    python -m awscli ec2 run-instances `
        --image-id ami-026992d753d5622bc `
        --instance-type c6i.4xlarge `
        --key-name wildfire-ml-key `
        --security-groups wildfire-ml-sg `
        --instance-market-options "file://$awsDir/spot-options.json" `
        --iam-instance-profile Name=EC2-S3-Access `
        --user-data "file://$awsDir/setup_ec2.sh" `
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=wildfire-ml-training}]"

    Write-Host "[Launch] Instance starting! Check AWS Console or run:" -ForegroundColor Green
    Write-Host "  python -m awscli ec2 describe-instances --filters Name=tag:Name,Values=wildfire-ml-training" -ForegroundColor White
}

# ============================================================================
# Main
# ============================================================================

Write-Host "=========================================="
Write-Host "Wildfire Risk ML - Deploy to AWS"
Write-Host "=========================================="

# Show current S3 contents
Write-Host "`nCurrent S3 bucket contents:" -ForegroundColor Gray
python -m awscli s3 ls "s3://$BUCKET/" --recursive --human-readable 2>$null

# Default to -Code if no flags specified
if (-not ($Code -or $Data -or $All -or $Launch)) {
    Write-Host "`nNo flags specified. Use:" -ForegroundColor Yellow
    Write-Host "  .\deploy.ps1 -Code    # Upload train.py and scripts"
    Write-Host "  .\deploy.ps1 -Data    # Upload training data"
    Write-Host "  .\deploy.ps1 -All     # Upload everything"
    Write-Host "  .\deploy.ps1 -Launch  # Launch EC2 training instance"
    Write-Host "  .\deploy.ps1 -Code -Launch  # Upload code and launch"
    exit
}

if ($All -or $Code) { Upload-Code }
if ($All -or $Data) { Upload-Data }
if ($Launch) { Launch-Training }

Write-Host "`n=========================================="
Write-Host "Deploy complete!"
Write-Host "=========================================="