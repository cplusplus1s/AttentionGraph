# ================= Configuration =================
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$CodeDir    = Join-Path $ProjectRoot "third_party\iTransformer"
$DataDir    = Join-Path $ProjectRoot "data\processed"
$ResultsDir = Join-Path $ProjectRoot "results"

$env:CUDA_VISIBLE_DEVICES="0"
Push-Location $CodeDir

# Helper Function: extract inference results in .npy
function Move-InferenceResult {
    param([string]$TargetSubDir)

    $SourceDir = Join-Path $ResultsDir "checkpoints"

    $ModelFolder = Get-ChildItem -Path $SourceDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "Brian2_Master_*" } |
        Select-Object -First 1

    if ($ModelFolder) {
        $TargetFullPath = Join-Path $ResultsDir $TargetSubDir
        if (-not (Test-Path $TargetFullPath)) {
            New-Item -ItemType Directory -Force -Path $TargetFullPath | Out-Null
        }

        # Move .npy file (Don't move .pth file!)
        $NpyFiles = Get-ChildItem -Path $ModelFolder.FullName -Filter "*.npy"

        if ($NpyFiles.Count -gt 0) {
            foreach ($file in $NpyFiles) {
                Move-Item -Path $file.FullName -Destination $TargetFullPath -Force
            }
            Write-Host "   ✅ Move inference results to -> $TargetFullPath"
        } else {
            Write-Warning "⚠️ Found target folder, but no .npy file!"
        }
    } else {
        Write-Warning "⚠️ Cannot find folder named Brian2_Master_...!"
    }
}

Write-Host "========================================="
Write-Host "🧠 Phase 1: Training Master Model..."
Write-Host "========================================="
python -u run.py --is_training 1 --root_path "$DataDir/" --data_path combined_healthy_train.csv --model_id Brian2_Master --model iTransformer --data custom --features M --seq_len 50 --label_len 48 --pred_len 10 --e_layers 2 --enc_in 25 --dec_in 25 --c_out 25 --des 'Brian2_Exp' --d_model 64 --d_ff 128 --batch_size 8 --learning_rate 0.001 --train_epochs 10 --output_attention --do_predict --target brian2_sensor_4_4 --patience 3 --checkpoints "$ResultsDir/checkpoints/"

Write-Host "========================================="
Write-Host "📊 Phase 2: Inferring 10 healthy baseline..."
Write-Host "========================================="
for ($i = 1; $i -le 10; $i++) {
    Write-Host "   -> Inferring Healthy Baseline $i"
    python -u run.py --is_training 0 --root_path "$DataDir/" --data_path "brian2_healthy_$i.csv" --model_id Brian2_Master --model iTransformer --data custom --features M --seq_len 50 --label_len 48 --pred_len 10 --e_layers 2 --enc_in 25 --dec_in 25 --c_out 25 --des 'Brian2_Exp' --d_model 64 --d_ff 128 --batch_size 8 --output_attention --do_predict --target brian2_sensor_4_4 --checkpoints "$ResultsDir/checkpoints/"

    Move-InferenceResult "healthy_baseline\run_$i"
}

Write-Host "========================================="
Write-Host "🚨 Phase 3: Inferring Unhealthy dataset..."
Write-Host "========================================="
python -u run.py --is_training 0 --root_path "$DataDir/" --data_path "brian2_unhealthy_1.csv" --model_id Brian2_Master --model iTransformer --data custom --features M --seq_len 50 --label_len 48 --pred_len 10 --e_layers 2 --enc_in 25 --dec_in 25 --c_out 25 --des 'Brian2_Exp' --d_model 64 --d_ff 128 --batch_size 8 --output_attention --do_predict --target brian2_sensor_4_4 --checkpoints "$ResultsDir/checkpoints/"

Move-InferenceResult "unhealthy_test\run_1"

Pop-Location
Write-Host "🎉 Training and Inference Complete!"