$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
$CodeDir = Join-Path $ProjectRoot "third_party\iTransformer"
$DataDir = Join-Path $ProjectRoot "data\processed"
$ResultsDir = Join-Path $ProjectRoot "results"

$env:CUDA_VISIBLE_DEVICES="0"
Push-Location $CodeDir

function Move-InferenceResult {
    param([string]$TargetSubDir, [string]$ModelIdPrefix)
    $SourceDir = Join-Path $ResultsDir "checkpoints"
    $ModelFolder = Get-ChildItem -Path $SourceDir -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "$ModelIdPrefix*" } | Select-Object -First 1
    if ($ModelFolder) {
        $TargetFullPath = Join-Path $ResultsDir $TargetSubDir
        if (-not (Test-Path $TargetFullPath)) { New-Item -ItemType Directory -Force -Path $TargetFullPath | Out-Null }
        $NpyFiles = Get-ChildItem -Path $ModelFolder.FullName -Filter "*.npy"
        if ($NpyFiles.Count -gt 0) {
            foreach ($file in $NpyFiles) { Move-Item -Path $file.FullName -Destination $TargetFullPath -Force }
        }
    }
}

Write-Host "Phase 1: Training MSD_SpatialPE..."
python -u run.py --is_training 1 --root_path "$DataDir/" --data_path combined_healthy_train.csv --model_id MSD_SpatialPE --model iTransformer --data custom --features M --seq_len 384 --label_len 192 --pred_len 192 --e_layers 3 --enc_in 24 --dec_in 24 --c_out 24 --des 'MSD_Exp' --d_model 128 --d_ff 128 --batch_size 128 --learning_rate 0.0005 --train_epochs 20 --output_attention --do_predict --target mass_12_acc_y --patience 5 --lradj "type2" --checkpoints "$ResultsDir/checkpoints/" --use_spatial_pe

Write-Host "Phase 2: Inferring Healthy..."
for ($i = 1; $i -le 30; $i++) {
    python -u run.py --is_training 0 --root_path "$DataDir/" --data_path "matlab_healthy_$i.csv" --model_id MSD_SpatialPE --model iTransformer --data custom --features M --seq_len 384 --label_len 192 --pred_len 192 --e_layers 3 --enc_in 24 --dec_in 24 --c_out 24 --des "MSD_Exp" --d_model 128 --d_ff 128 --batch_size 128 --output_attention --do_predict --target mass_12_acc_y --checkpoints "$ResultsDir/checkpoints/" --use_spatial_pe
    Move-InferenceResult "healthy_baseline\run_$i" "MSD_SpatialPE"
}

Write-Host "Phase 3: Inferring Unhealthy..."
# python -u run.py --is_training 0 --root_path "$DataDir/" --data_path "matlab_unhealthy_1.csv" --model_id MSD_SpatialPE --model iTransformer --data custom --features M --seq_len 384 --label_len 192 --pred_len 192 --e_layers 3 --enc_in 24 --dec_in 24 --c_out 24 --des "MSD_Exp" --d_model 128 --d_ff 128 --batch_size 128 --output_attention --do_predict --target mass_12_acc_y --checkpoints "$ResultsDir/checkpoints/" --use_spatial_pe
# Move-InferenceResult "unhealthy_test\run_1" "MSD_SpatialPE"