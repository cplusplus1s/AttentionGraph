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
        Where-Object { $_.Name -like "MSD_Master_*" } |
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
        Write-Warning "⚠️ Cannot find folder named MSD_Master_...!"
    }
}

Write-Host "========================================="
Write-Host "🧠 Phase 1: Training Master Model..."
Write-Host "========================================="
# 0-padding
#python -u run.py --is_training 1 --root_path "$DataDir/" --data_path combined_healthy_train.csv --model_id MSD_Master --model iTransformer --data custom --features M --seq_len 384 --label_len 192 --pred_len 192 --e_layers 3 --enc_in 12 --dec_in 12 --c_out 12 --des 'MSD_Exp' --d_model 128 --d_ff 128 --batch_size 32 --learning_rate 0.0005 --train_epochs 20 --output_attention --do_predict --target mass_12_acc --patience 5 --lradj "type2" --checkpoints "$ResultsDir/checkpoints/"

Write-Host "========================================="
Write-Host "📊 Phase 2: Inferring 30 healthy baseline..."
Write-Host "========================================="
# for ($i = 1; $i -le 30; $i++) {
#     Write-Host "   -> Inferring Healthy Baseline $i"
#     python -u run.py --is_training 0 --root_path "$DataDir/" --data_path "matlab_healthy_$i.csv" --model_id MSD_Master --model iTransformer --data custom --features M --seq_len 384 --label_len 192 --pred_len 192 --e_layers 3 --enc_in 12 --dec_in 12 --c_out 12 --des 'MSD_Exp' --d_model 128 --d_ff 128 --batch_size 32 --output_attention --do_predict --target mass_12_acc --checkpoints "$ResultsDir/checkpoints/"

#     Move-InferenceResult "healthy_baseline\run_$i"
# }

Write-Host "========================================="
Write-Host "🚨 Phase 3: Inferring Unhealthy dataset..."
Write-Host "========================================="
python -u run.py --is_training 0 --root_path "$DataDir/" --data_path "matlab_unhealthy_1.csv" --model_id MSD_Master --model iTransformer --data custom --features M --seq_len 384 --label_len 192 --pred_len 192 --e_layers 3 --enc_in 12 --dec_in 12 --c_out 12 --des 'MSD_Exp' --d_model 128 --d_ff 128 --batch_size 32 --output_attention --do_predict --target mass_12_acc --checkpoints "$ResultsDir/checkpoints/"

Move-InferenceResult "unhealthy_test\run_1"

Pop-Location
Write-Host "🎉 Training and Inference Complete!"