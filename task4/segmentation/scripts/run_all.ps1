# 一键跑通整个流程：训练 -> 推理 -> 生成提交 CSV -> 校验
#
#   用法:  powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
#   可选:  -Epochs 300 -SkipTrain -Threshold 0.45 -MinSize 12
#   说明:  本文件必须保存为「UTF-8 with BOM」，否则 Windows PowerShell 5.1
#          会按 ANSI 解析中文注释并报 "The string is missing the terminator"。
param(
    [string]$Python   = "python",
    [int]$Epochs      = 250,
    [switch]$SkipTrain,
    [switch]$SkipInfer,
    [double]$Threshold = 0.45,
    [int]$MinSize      = 10
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONIOENCODING = "utf-8"

try {
    Write-Host "==> 工作目录: $root" -ForegroundColor Cyan

    if (-not $SkipTrain) {
        Write-Host "`n==> [1/5] 训练 baseline U-Net ($Epochs epochs)" -ForegroundColor Cyan
        & $Python -m src.train --epochs $Epochs
        if ($LASTEXITCODE -ne 0) { throw "训练失败" }
    } else {
        Write-Host "`n==> [1/5] 跳过训练" -ForegroundColor DarkGray
    }

    if (-not $SkipInfer) {
        # 优先用 33 模型集成（6 个 run × 5 折 + 3 个满数据模型；第六轮加入 scale=2 的 outputs_cv_s2）；
        # 其次 17 模型（第四轮）；再其次 5 折；都没有才退回单模型。
        $fullEns = @()
        foreach ($run in @("outputs_cv", "outputs_cv_att", "outputs_cv_full",
                           "outputs_cv_sa", "outputs_cv_sa_big", "outputs_cv_s2")) {
            for ($k = 0; $k -lt 5; $k++) { $fullEns += "$run\fold$k\best.pt" }
        }
        $fullEns += "outputs_all\best.pt"
        $fullEns += "outputs_all_att\best.pt"
        $fullEns += "outputs_all_sa\best.pt"
        $ens17 = @()
        foreach ($run in @("outputs_cv", "outputs_cv_att", "outputs_cv_full")) {
            for ($k = 0; $k -lt 5; $k++) { $ens17 += "$run\fold$k\best.pt" }
        }
        $ens17 += "outputs_all\best.pt"
        $ens17 += "outputs_all_att\best.pt"
        $cvEns = @()
        for ($k = 0; $k -lt 5; $k++) { $cvEns += "outputs_cv\fold$k\best.pt" }
        $ens = @("outputs\baseline_scratch.pt", "outputs_joint\best.pt", "outputs_all\best.pt")
        $haveEns = $true
        foreach ($p in $fullEns) { if (-not (Test-Path $p)) { $haveEns = $false } }
        $haveEns17 = $true
        foreach ($p in $ens17) { if (-not (Test-Path $p)) { $haveEns17 = $false } }
        $haveCv = $true
        foreach ($p in $cvEns) { if (-not (Test-Path $p)) { $haveCv = $false } }
        $haveAll = $true
        foreach ($p in $ens) { if (-not (Test-Path $p)) { $haveAll = $false } }

        if ($haveEns) {
            Write-Host "`n==> [2/5] 33 模型集成推理（6 run × 5 折 + 3 满数据）-> ./image" -ForegroundColor Cyan
            & $Python scripts/ensemble.py --checkpoints ($fullEns -join ",") --data_dir test --pred_dir image `
                --threshold $Threshold --min_size $MinSize
        } elseif ($haveEns17) {
            Write-Host "`n==> [2/5] 17 模型集成推理（3 架构 × 5 折 + 2 满数据）-> ./image" -ForegroundColor Cyan
            & $Python scripts/ensemble.py --checkpoints ($ens17 -join ",") --data_dir test --pred_dir image `
                --threshold $Threshold --min_size $MinSize
        } elseif ($haveCv) {
            Write-Host "`n==> [2/5] 5 折集成推理（5 模型概率平均）-> ./image" -ForegroundColor Cyan
            & $Python scripts/ensemble.py --checkpoints ($cvEns -join ",") --data_dir test --pred_dir image `
                --threshold $Threshold --min_size $MinSize
        } elseif ($haveAll) {
            Write-Host "`n==> [2/5] 3 模型集成推理 -> ./image" -ForegroundColor Cyan
            & $Python scripts/ensemble.py --checkpoints ($ens -join ",") --data_dir test --pred_dir image `
                --threshold $Threshold --min_size $MinSize
        } else {
            Write-Host "`n==> [2/5] 单模型推理 -> ./image" -ForegroundColor Cyan
            & $Python -m src.infer --threshold $Threshold --min_size $MinSize
        }
        if ($LASTEXITCODE -ne 0) { throw "推理失败" }
    } else {
        Write-Host "`n==> [2/5] 跳过推理" -ForegroundColor DarkGray
    }

    Write-Host "`n==> [3/5] 官方脚本生成 submission_official.csv" -ForegroundColor Cyan
    & $Python segmentation_to_csv.py
    if ($LASTEXITCODE -ne 0) { throw "官方脚本执行失败" }
    Move-Item submission.csv submission_official.csv -Force

    Write-Host "`n==> [4/5] 自研编码器生成 submission.csv（按 Id 数值排序）并校验" -ForegroundColor Cyan
    & $Python -m src.rle --image_dir image --out submission.csv
    & $Python -m src.rle --image_dir image --out submission.csv --verify submission_official.csv

    Write-Host "`n==> [5/5] 完成" -ForegroundColor Green
    Get-ChildItem submission.csv | Select-Object Name, Length, LastWriteTime | Format-Table -AutoSize
}
finally {
    Pop-Location
}
