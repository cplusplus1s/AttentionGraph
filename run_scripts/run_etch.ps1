# ================= Configuration =================
# Get current script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Get project root directory
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")

# Define absolute paths
$CodeDir    = Join-Path $ProjectRoot "third_party\iTransformer"
$DataDir    = Join-Path $ProjectRoot "data\processed"
$ResultsDir = Join-Path $ProjectRoot "results"

# ================= Environment Setup =================
$env:CUDA_VISIBLE_DEVICES="0"

Write-Host "========================================="
Write-Host "Starting Training Task..."
Write-Host "Code Dir   : $CodeDir"
Write-Host "Data Dir   : $DataDir"
Write-Host "Results Dir: $ResultsDir"
Write-Host "========================================="

# 1. Change directory to iTransformer source code
# This is crucial for imports to work correctly
Push-Location $CodeDir

# Check if run.py exists (Debug step)
if (-not (Test-Path "run.py")) {
    Write-Error "Error: run.py not found in $CodeDir"
    Pop-Location
    exit
}

# 2. Execute Python command
# Note: --root_path must end with / for some OS, but consistent path handling is key
python -u run.py `
  --is_training 1 `
  --root_path "$DataDir/" `
  --data_path wdl_filtered_aligned.csv `
  --model_id sensor_analysis_short `
  --model iTransformer `
  --data custom `
  --features M `
  --seq_len 50 `
  --pred_len 10 `
  --e_layers 2 `
  --enc_in 35 `
  --dec_in 35 `
  --c_out 35 `
  --des 'Exp' `
  --d_model 64 `
  --d_ff 128 `
  --batch_size 8 `
  --learning_rate 0.001 `
  --train_epochs 10 `
  --output_attention `
  --do_predict `
  --target ChamberManometer_RawAI_Value `
  --checkpoints "$ResultsDir/"

# 3. Restore original directory
Pop-Location

Write-Host "Task Finished! Check results in $ResultsDir"