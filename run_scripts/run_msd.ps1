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
Write-Host "Starting MSD Training Task..."
Write-Host "Code Dir   : $CodeDir"
Write-Host "Data Dir   : $DataDir"
Write-Host "Results Dir: $ResultsDir"
Write-Host "========================================="

# 1. Change directory to iTransformer source code
Push-Location $CodeDir

# Check if run.py exists
if (-not (Test-Path "run.py")) {
    Write-Error "Error: run.py not found in $CodeDir"
    Pop-Location
    exit
}

# 2. Execute Python command
# data_path is for matlab file, enc_in/dec_in/c_out is 3 × Number of mass blocks, target is any mass pos, vel or acc
python -u run.py `
  --is_training 1 `
  --root_path "$DataDir/" `
  --data_path matlab_filtered_aligned.csv `
  --model_id MSD_v1 `
  --model iTransformer `
  --data custom `
  --features M `
  --seq_len 96 `
  --pred_len 96 `
  --e_layers 2 `
  --enc_in 36 `
  --dec_in 36 `
  --c_out 36 `
  --des 'MSD_Exp' `
  --d_model 64 `
  --d_ff 128 `
  --batch_size 32 `
  --learning_rate 0.0001 `
  --train_epochs 10 `
  --output_attention `
  --do_predict `
  --target mass_12_pos `
  --checkpoints "$ResultsDir/"

# 3. Restore original directory
Pop-Location

Write-Host "Task Finished! Check results in $ResultsDir"