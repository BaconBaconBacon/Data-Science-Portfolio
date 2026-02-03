# AWS Infrastructure Check for Wildfire Risk ML
# Validates all AWS resources are set up correctly before training
#
# Usage: .\aws\check_aws.ps1
# ============================================================================

$BUCKET = "wildfire-risk-ml"
$KEY_NAME = "wildfire-ml-key"
$SECURITY_GROUP = "wildfire-ml-sg"
$INSTANCE_PROFILE = "EC2-S3-Access"

$errors = 0
$warnings = 0

function Check-Pass { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Check-Fail { param($msg) Write-Host "[FAIL] $msg" -ForegroundColor Red; $script:errors++ }
function Check-Warn { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow; $script:warnings++ }
function Check-Info { param($msg) Write-Host "[INFO] $msg" -ForegroundColor Cyan }

Write-Host "=========================================="
Write-Host "AWS Infrastructure Check"
Write-Host "=========================================="

# ----------------------------------------------------------------------------
# 1. AWS CLI
# ----------------------------------------------------------------------------
Write-Host "`n[1/7] AWS CLI..." -ForegroundColor White
try {
    $identity = python -m awscli sts get-caller-identity 2>&1 | ConvertFrom-Json
    Check-Pass "AWS CLI configured - Account: $($identity.Account)"
} catch {
    Check-Fail "AWS CLI not configured. Run: aws configure"
}

# ----------------------------------------------------------------------------
# 2. S3 Bucket
# ----------------------------------------------------------------------------
Write-Host "`n[2/7] S3 Bucket..." -ForegroundColor White
$bucketCheck = python -m awscli s3 ls "s3://$BUCKET" 2>&1
if ($LASTEXITCODE -eq 0) {
    Check-Pass "Bucket s3://$BUCKET exists"

    # Check for required files
    $files = python -m awscli s3 ls "s3://$BUCKET/" --recursive 2>&1

    if ($files -match "scripts/train.py") {
        Check-Pass "  train.py uploaded"
    } else {
        Check-Warn "  train.py not in S3. Run: .\aws\deploy.ps1 -Code"
    }

    if ($files -match "data/model_joined.parquet") {
        Check-Pass "  model_joined.parquet uploaded"
    } elseif ($files -match "data/test_joined.parquet") {
        Check-Info "  test_joined.parquet found (test mode)"
    } else {
        Check-Warn "  No training data in S3. Run: .\aws\deploy.ps1 -Data"
    }
} else {
    Check-Fail "Bucket s3://$BUCKET not found. Create with:"
    Write-Host "         python -m awscli s3 mb s3://$BUCKET --region us-east-1" -ForegroundColor Gray
}

# ----------------------------------------------------------------------------
# 3. EC2 Key Pair
# ----------------------------------------------------------------------------
Write-Host "`n[3/7] EC2 Key Pair..." -ForegroundColor White
$keyCheck = python -m awscli ec2 describe-key-pairs --key-names $KEY_NAME 2>&1
if ($LASTEXITCODE -eq 0) {
    Check-Pass "Key pair '$KEY_NAME' exists"

    # Check local key file
    $keyPath = "$env:USERPROFILE\.ssh\$KEY_NAME.pem"
    if (Test-Path $keyPath) {
        Check-Pass "  Local key file exists: $keyPath"
    } else {
        Check-Warn "  Local key file not found at $keyPath"
    }
} else {
    Check-Fail "Key pair '$KEY_NAME' not found. Create with:"
    Write-Host "         python -m awscli ec2 create-key-pair --key-name $KEY_NAME --query 'KeyMaterial' --output text > ~\.ssh\$KEY_NAME.pem" -ForegroundColor Gray
}

# ----------------------------------------------------------------------------
# 4. Security Group
# ----------------------------------------------------------------------------
Write-Host "`n[4/7] Security Group..." -ForegroundColor White
$sgCheck = python -m awscli ec2 describe-security-groups --group-names $SECURITY_GROUP 2>&1 | ConvertFrom-Json
if ($LASTEXITCODE -eq 0) {
    Check-Pass "Security group '$SECURITY_GROUP' exists"

    # Check for SSH ingress rule
    $hasSSH = $false
    foreach ($perm in $sgCheck.SecurityGroups[0].IpPermissions) {
        if ($perm.FromPort -eq 22) { $hasSSH = $true }
    }
    if ($hasSSH) {
        Check-Pass "  SSH (port 22) ingress rule configured"
    } else {
        Check-Warn "  No SSH ingress rule. Add with:"
        Write-Host "         python -m awscli ec2 authorize-security-group-ingress --group-name $SECURITY_GROUP --protocol tcp --port 22 --cidr 0.0.0.0/0" -ForegroundColor Gray
    }
} else {
    Check-Fail "Security group '$SECURITY_GROUP' not found. Create with:"
    Write-Host "         python -m awscli ec2 create-security-group --group-name $SECURITY_GROUP --description 'SSH access for ML training'" -ForegroundColor Gray
}

# ----------------------------------------------------------------------------
# 5. IAM Instance Profile
# ----------------------------------------------------------------------------
Write-Host "`n[5/7] IAM Instance Profile..." -ForegroundColor White
$profileCheck = python -m awscli iam get-instance-profile --instance-profile-name $INSTANCE_PROFILE 2>&1
if ($LASTEXITCODE -eq 0) {
    Check-Pass "Instance profile '$INSTANCE_PROFILE' exists"
} else {
    Check-Fail "Instance profile '$INSTANCE_PROFILE' not found."
    Write-Host "         Create IAM role with S3 access and attach to instance profile." -ForegroundColor Gray
    Write-Host "         See AWS Console > IAM > Roles" -ForegroundColor Gray
}

# ----------------------------------------------------------------------------
# 6. Check for running instances
# ----------------------------------------------------------------------------
Write-Host "`n[6/7] Running Instances..." -ForegroundColor White
$instances = python -m awscli ec2 describe-instances --filters "Name=tag:Name,Values=wildfire-ml-training" "Name=instance-state-name,Values=running,pending" 2>&1 | ConvertFrom-Json
$runningCount = ($instances.Reservations | ForEach-Object { $_.Instances }).Count
if ($runningCount -gt 0) {
    Check-Info "$runningCount training instance(s) currently running"
} else {
    Check-Pass "No training instances running (ready to launch)"
}

# ----------------------------------------------------------------------------
# 7. Spot price check
# ----------------------------------------------------------------------------
Write-Host "`n[7/7] Spot Pricing (us-east-1)..." -ForegroundColor White
try {
    $spotPrice = python -m awscli ec2 describe-spot-price-history --instance-types c6i.4xlarge --product-descriptions "Linux/UNIX" --max-items 1 2>&1 | ConvertFrom-Json
    $price = $spotPrice.SpotPriceHistory[0].SpotPrice
    Check-Info "c6i.4xlarge spot price: `$$price/hr"
} catch {
    Check-Warn "Could not fetch spot pricing"
}

# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------
Write-Host "`n=========================================="
if ($errors -eq 0 -and $warnings -eq 0) {
    Write-Host "All checks passed! Ready to launch." -ForegroundColor Green
} elseif ($errors -eq 0) {
    Write-Host "$warnings warning(s), but ready to launch." -ForegroundColor Yellow
} else {
    Write-Host "$errors error(s), $warnings warning(s). Fix errors before launching." -ForegroundColor Red
}
Write-Host "=========================================="