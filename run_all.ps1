<#
.SYNOPSIS
    Reproduce every result from a clean clone.

.DESCRIPTION
    Ordered so that a shippable predictions.csv exists after stage 3, before
    anything slow or speculative runs. If the machine dies at stage 5 there is
    still a valid submission on disk.

    Stage 1  tests            ~1 min   leakage + contracts + model properties
    Stage 2  diagnostics      ~1 min   censoring, anchor selection
    Stage 3  baseline         ~20 min  LR + GBDT + ensemble -> predictions.csv
    Stage 4  transformer      ~2 h     the 2x2 arms, 1 seed each
    Stage 5  seeds            ~5 h     3 seeds of the winning arm

.PARAMETER Smoke
    Run every stage at toy size (~3 min total). Exercises every code path.

.PARAMETER Stages
    Which stages to run, e.g. -Stages 1,2,3

.EXAMPLE
    .\run_all.ps1 -Smoke
    .\run_all.ps1 -Stages 1,2,3
#>
[CmdletBinding()]
param(
    [switch]$Smoke,
    [int[]]$Stages = @(1, 2, 3, 4, 5)
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Step {
    param([int]$N, [string]$Name, [scriptblock]$Body)
    if ($Stages -notcontains $N) {
        Write-Host "[$N] $Name -- skipped" -ForegroundColor DarkGray
        return
    }
    Write-Host ""
    Write-Host "=== [$N] $Name ===" -ForegroundColor Cyan
    $t = [Diagnostics.Stopwatch]::StartNew()
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "stage $N ($Name) failed with exit code $LASTEXITCODE" }
    Write-Host ("    done in {0:n1} min" -f $t.Elapsed.TotalMinutes) -ForegroundColor DarkGray
}

# wandb is opt-in: even offline it spawns a service that has killed a run
# mid-flight. Everything lands in outputs/metrics.jsonl regardless.
if (-not $env:WANDB) { Write-Host "wandb disabled (set `$env:WANDB=1 to enable)" -ForegroundColor DarkGray }

Write-Host "M31 patient timeline forecasting" -ForegroundColor Green
Write-Host ("smoke = {0}   stages = {1}" -f $Smoke, ($Stages -join ","))

Step 1 "Tests: leakage, contracts, model properties" {
    python -m pytest tests/ -q
}

Step 2 "Diagnostics: censoring and anchor selection" {
    python -m src.diagnostics
}

Step 3 "Baseline: LR + GBDT + ensemble, writes predictions.csv" {
    if ($Smoke) { python -m src.run_baseline --smoke }
    else        { python -m src.run_baseline }
}

Step 4 "Transformer: the four coherent arms" {
    foreach ($arm in @("P4", "P3", "P1", "P2")) {
        Write-Host "  arm $arm" -ForegroundColor Yellow
        if ($Smoke) { python -m src.train_finetune --arm $arm --smoke }
        else        { python -m src.train_finetune --arm $arm --epochs 30 }
        if ($LASTEXITCODE -ne 0) { throw "arm $arm failed" }
    }
}

Step 5 "Seed variance: 3 seeds of the control arm" {
    # Run FIRST in spirit, last in wall-clock: nothing in stage 4 is
    # interpretable without this spread, so it is reported alongside it.
    if ($Smoke) { python -m src.train_finetune --arm P3 --seeds 0 1 --smoke }
    else        { python -m src.train_finetune --arm P3 --seeds 0 1 2 --epochs 30 }
}

Write-Host ""
Write-Host "Artifacts:" -ForegroundColor Green
foreach ($f in @("outputs/predictions.csv", "outputs/predictions_meta.json",
                 "outputs/per_code_val.csv", "outputs/metrics.jsonl")) {
    if (Test-Path $f) {
        Write-Host ("  {0,-34} {1,8:n0} bytes" -f $f, (Get-Item $f).Length)
    } else {
        Write-Host ("  {0,-34} MISSING" -f $f) -ForegroundColor Red
    }
}
Write-Host ""
Write-Host "wandb runs are offline by default; upload with: wandb sync wandb/offline-run-*" -ForegroundColor DarkGray
