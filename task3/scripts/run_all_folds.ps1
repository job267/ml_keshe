# ============================================================================
# 一键跑完整流程：阶段一(5 折) -> 阶段二(5 折) -> 交叉验证报告 -> 集成预测 -> 格式校验
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_folds.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_folds.ps1 -Folds 1     # 只跑 1 折
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_folds.ps1 -SkipStage1  # 复用已有阶段一权重
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_folds.ps1 -SkipStage2  # 只跑阶段一（约 40 min）
#
# 参考：RTX 4060 上阶段一 5 折约 45 min，阶段二 5 折约 20 min；两阶段 CV 16.7px。
#      -SkipStage2 时 CV 为 19.6px（阶段一单阶段）。
# ============================================================================
param(
    [int]$Folds = 5,
    [switch]$SkipStage1,
    [switch]$SkipStage2
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
$env:PYTHONIOENCODING = 'utf-8'

try {
    if (-not $SkipStage1) {
        for ($f = 0; $f -lt $Folds; $f++) {
            Write-Host "===== 阶段一 fold $f =====" -ForegroundColor Cyan
            python -m src.train --config configs/stage1.yaml --stage 1 --fold $f
        }
    }
    if (-not $SkipStage2) {
        for ($f = 0; $f -lt $Folds; $f++) {
            Write-Host "===== 阶段二 fold $f =====" -ForegroundColor Cyan
            python -m src.train --config configs/stage2.yaml --stage 2 --fold $f `
                --s1-ckpt "outputs/runs/s1_fold$f/best.pt"
        }
    }

    $runs = @(0..($Folds - 1) | ForEach-Object { "outputs/runs/s2_fold$_" })

    Write-Host "===== 交叉验证报告 =====" -ForegroundColor Cyan
    python -m src.predict --config configs/stage2.yaml --runs @runs --split cv

    Write-Host "===== 生成提交文件 =====" -ForegroundColor Cyan
    python -m src.predict --config configs/stage2.yaml --runs @runs --split test --vis

    Write-Host "===== 提交格式校验 =====" -ForegroundColor Cyan
    python tools\validate_submission.py outputs\submission.csv
}
finally {
    Pop-Location
}
