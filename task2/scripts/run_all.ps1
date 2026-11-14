# 复现入口：按报告轮次依次训练、融合、导出提交。
# 用法示例：
#   powershell -File scripts\run_all.ps1 -Round 1          # 第一轮基线
#   powershell -File scripts\run_all.ps1 -Round 2          # 第二轮三成员 + 融合
#   powershell -File scripts\run_all.ps1 -Round 12 -Refit  # 第十二轮遮挡 + 部署权重
#   powershell -File scripts\run_all.ps1 -Round all        # 全部（耗时长）
param(
    [string]$Round = "1",
    [switch]$Refit,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$refitArg = @()
if ($Refit) { $refitArg = @("--refit") }

function Run-Train([string]$cfg, [string]$out) {
    Write-Host "==> train $cfg -> $out" -ForegroundColor Cyan
    & $Python -m src.train --config $cfg --out-dir $out @refitArg
    if ($LASTEXITCODE -ne 0) { throw "训练失败：$cfg" }
}

function Run-Predict([string[]]$runs, [double[]]$weights, [string]$out, [string]$split = "test", [string]$ckpt = "best.pt") {
    Write-Host "==> predict $out" -ForegroundColor Cyan
    $w = $weights | ForEach-Object { $_.ToString([System.Globalization.CultureInfo]::InvariantCulture) }
    & $Python -m src.predict --runs @runs --weights @w --out $out --split $split --ckpt $ckpt
    if ($LASTEXITCODE -ne 0) { throw "推理失败：$out" }
}

switch ($Round) {
    "1" {
        Run-Train "configs/round1_baseline.yaml" "outputs/runs/r1_baseline"
        Run-Train "configs/round1_attn.yaml"     "outputs/runs/r1_attn"
        Run-Train "configs/round1_resnet18.yaml" "outputs/runs/r1_r18"
        Run-Predict @("outputs/runs/r1_baseline", "outputs/runs/r1_r18") @(0.5, 0.5) "submission_round1.csv"
    }
    "2" {
        Run-Train "configs/round1_resnet18.yaml"  "outputs/runs/r2_r18"
        Run-Train "configs/round2_convnext.yaml"  "outputs/runs/r2_convnext"
        Run-Train "configs/round2_effnet.yaml"    "outputs/runs/r2_b0"
        Run-Predict @("outputs/runs/r2_r18", "outputs/runs/r2_convnext", "outputs/runs/r2_b0") @(0.25, 0.50, 0.25) "submission_round2.csv"
    }
    "7" {
        Run-Train "configs/round7_gce.yaml" "outputs/runs/r7_gce"
        & $Python -m src.train --config configs/round7_gce.yaml --oof 3 --out-dir outputs/runs/oof7
    }
    "12" {
        Run-Train "configs/round12_erasing.yaml" "outputs/runs/r12_erasing"
        Run-Predict @("outputs/runs/r2_r18", "outputs/runs/r12_erasing", "outputs/runs/r2_b0") @(0.25, 0.50, 0.25) "submission_round12_full.csv"
    }
    "15" {
        Run-Train "configs/round15_face_b0.yaml" "outputs/runs/r15_face_b0"
        Run-Predict @("outputs/runs/r12_erasing", "outputs/runs/r15_face_b0") @(0.5, 0.5) "submission_round15_deploy.csv"
    }
    "17" {
        Run-Train "configs/round17_rexnet.yaml" "outputs/runs/r17_rexnet"
        Run-Predict @("outputs/runs/r15_face_b0", "outputs/runs/r17_rexnet") @(0.5, 0.5) "submission_round17_deploy.csv"
    }
    "all" {
        & $PSCommandPath -Round 1
        & $PSCommandPath -Round 2
        & $PSCommandPath -Round 7
        & $PSCommandPath -Round 12 -Refit
        & $PSCommandPath -Round 15 -Refit
        & $PSCommandPath -Round 17 -Refit
    }
    default { throw "未知轮次：$Round（可选 1/2/7/12/15/17/all）" }
}

Write-Host "完成。提交文件位于项目根目录，校验记录在 reports/ 下。" -ForegroundColor Green
