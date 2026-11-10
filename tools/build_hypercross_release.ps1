<#
  Build a clean, normalized release tree for the HyperCross scientific-HSI project.
  Source (read-only workspace) : G:\Datasets\HyperCross
  Destination (staged release) : F:\ZJU\科研\Hyperspectral\论文\HyperCross-Release
  Excludes: training checkpoints, recovery journals, bulk run trees, dataset-derived caches.
#>
$ErrorActionPreference = 'Stop'

$src = 'G:\Datasets\HyperCross'
$dst = 'F:\ZJU\科研\Hyperspectral\论文\HyperCross-Release'
$guard = 'F:\ZJU\科研\Hyperspectral\论文\'

if (-not $dst.StartsWith($guard)) { throw "refusing to write outside $guard : $dst" }
if (-not (Test-Path -LiteralPath $src)) { throw "source workspace missing: $src" }

function Copy-Filtered {
    param(
        [Parameter(Mandatory)][string]$From,
        [Parameter(Mandatory)][string]$To,
        [string[]]$ExtraExcludeDirs = @(),
        [string[]]$ExtraExcludeNames = @()
    )
    if (-not (Test-Path -LiteralPath $From)) { Write-Warning "MISSING: $From"; return }
    $root = (Resolve-Path -LiteralPath $From).Path
    $exDirs = @('__pycache__', '.pytest_cache', '.ipynb_checkpoints') + $ExtraExcludeDirs
    $exNames = @('.DS_Store') + $ExtraExcludeNames
    $n = 0
    Get-ChildItem -LiteralPath $root -Recurse -File -Force | ForEach-Object {
        $rel = $_.FullName.Substring($root.Length).TrimStart('\')
        $parts = $rel -split '\\'
        $skip = $false
        foreach ($p in $parts) { if ($exDirs -contains $p) { $skip = $true; break } }
        if (-not $skip -and ($exNames -contains $_.Name)) { $skip = $true }
        if (-not $skip -and ($_.Extension -in @('.pyc', '.pyo', '.pyd'))) { $skip = $true }
        if ($skip) { return }
        $target = Join-Path $To $rel
        $tdir = Split-Path -Parent $target
        if (-not (Test-Path -LiteralPath $tdir)) { New-Item -ItemType Directory -Force -Path $tdir | Out-Null }
        Copy-Item -LiteralPath $_.FullName -Destination $target -Force
        $n++
    }
    Write-Output ("  {0,-42} {1,6} files -> {2}" -f (Split-Path -Leaf $From), $n, $To)
}

if (Test-Path -LiteralPath $dst) { Remove-Item -LiteralPath $dst -Recurse -Force }
New-Item -ItemType Directory -Force -Path $dst | Out-Null

Write-Output '== code =='
Copy-Filtered "$src\benchmark"  "$dst\benchmark"
Copy-Filtered "$src\experiments" "$dst\experiments" -ExtraExcludeNames @('cache_leaf_cube_small.npy', 'cache_leaf_spec_rgb.npz')
Copy-Filtered "$src\configs"    "$dst\configs"
Copy-Filtered "$src\tests"      "$dst\tests"
Copy-Filtered "$src\scripts"    "$dst\scripts"
Copy-Filtered "$src\docs"       "$dst\docs"
Copy-Filtered "$src\external"   "$dst\external"
Copy-Filtered "$src\src"        "$dst\src"
Copy-Filtered "$src\artifacts"   "$dst\artifacts"
Copy-Filtered "$src\.kiro\specs\cvpr2027-hyperspectral-benchmark" "$dst\specs" -ExtraExcludeNames @('tasks.meta.json', '.config.kiro')

Write-Output '== packaging metadata =='
Copy-Item -LiteralPath "$src\pyproject.toml" -Destination "$dst\pyproject.toml" -Force
Write-Output '  pyproject.toml'

Write-Output '== results: governed =='
$gov = @(
    'runs_v2\README.md'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\latest-task-24.2-summary.json'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\task-24.2-report.json'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\claim-and-promotion.json'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\environment-lock.json'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\frozen-plan.json'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\supersession-links.jsonl'
    'runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\negative-evidence.jsonl'
    'runs_v2\screening\hyperleaf\latest-task-14.12-summary.json'
    'runs_v2\screening\hyperleaf\executions\task-14-12-20261025T173540205337Z\task-14.12-report.json'
    'runs_v2\screening\histology\latest-task-15.12-summary.json'
    'runs_v2\screening\histology\executions\task-15-12-20261025T183726955895Z\task-15.12-report.json'
    'runs_v2\strict\hyperleaf\latest-task-14.13-summary.json'
    'runs_v2\strict\hyperleaf\executions\task-14-13-20261026T003830741428Z\task-14.13-report.json'
    'runs_v2\strict\histology\latest-task-15.13-summary.json'
    'runs_v2\strict\histology\executions\task-15-13-20261026T134933767825Z\task-15.13-report.json'
    'runs_v2\strict\histology\executions\task-15-13-20261026T134933767825Z\task-15.13-report-supplement.json'
    'runs_v2\strict\hyperleaf-nuisance-alignment\v1\deterministic-float32\task-25-1-20261108T130712621781Z\task-25.1-report.json'
    'runs_v2\strict\hyperleaf-nuisance-alignment\v1\deterministic-float32\task-25-1-20261108T130712621781Z\task-25.1-summary.md'
    'runs_v2\strict\hyperleaf-nuisance-alignment\v1\smoke\task-25-1-20261108T141414395253Z\task-25.1-report.json'
    'runs_v2\strict\hyperleaf-nuisance-alignment\v1\smoke\task-25-1-20261108T141414395253Z\task-25.1-summary.md'
    'runs_v2\optimization\hyperleaf\v1\deterministic-float32\latest-task-24.4-summary.json'
    'runs_v2\optimization\hyperleaf\v1\deterministic-float32\executions\task-24-4-20261028T055249572925Z\task-24.4-report.json'
    'runs_v2\baselines\hyperleaf\executions\task-14-10-20261025T135046189282Z\task-14.10-report.json'
    'runs_v2\baselines\histology\executions\task-15-11-20261025T140007060730Z\task-15.11-report.json'
)
$n = 0
foreach ($rel in $gov) {
    $from = Join-Path $src $rel
    if (-not (Test-Path -LiteralPath $from)) { Write-Warning "MISSING: $rel"; continue }
    $to = Join-Path "$dst\results\governed" $rel
    $tdir = Split-Path -Parent $to
    if (-not (Test-Path -LiteralPath $tdir)) { New-Item -ItemType Directory -Force -Path $tdir | Out-Null }
    Copy-Item -LiteralPath $from -Destination $to -Force
    $n++
}
Write-Output ("  governed report/summary files: {0}" -f $n)

Copy-Filtered "$src\runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\aggregates" `
              "$dst\results\governed\runs_v2\remediation\v1\deterministic-float32\hyperleaf-nested\executions\task-24-2-20261027T101633949873Z\aggregates"
Copy-Filtered "$src\runs_v2\strict\histology\executions\task-15-13-20261026T134933767825Z\lopo\aggregates" `
              "$dst\results\governed\runs_v2\strict\histology\executions\task-15-13-20261026T134933767825Z\aggregates"

Write-Output '== results: exploratory =='
Copy-Filtered "$src\runs_v3" "$dst\results\exploratory\runs_v3"

Write-Output '== results: legacy =='
$legacy = @(
    'README.md', 'CCF_A_COMPARISON.md', 'EXPERIMENT_DECISION_LOG.md', 'top_models_research.md',
    'benchmark_summary.json', 'benchmark_table.csv', 'balanced_cv_metrics.json',
    'hist_ensemble_results.json', 'leaf_multiseed.json', 'mil_eval.json', 'modern_best_summary.json',
    'patient_cv_metrics.json'
)
New-Item -ItemType Directory -Force -Path "$dst\results\legacy" | Out-Null
$n = 0
foreach ($name in $legacy) {
    $from = Join-Path $src $name
    if (-not (Test-Path -LiteralPath $from)) { Write-Warning "MISSING: $name"; continue }
    Copy-Item -LiteralPath $from -Destination (Join-Path "$dst\results\legacy" $name) -Force
    $n++
}
Write-Output ("  legacy files: {0}" -f $n)

Write-Output '== size =='
$s = Get-ChildItem -LiteralPath $dst -Recurse -File | Measure-Object Length -Sum
Write-Output ("  release tree: {0} files, {1} MB" -f $s.Count, [math]::Round($s.Sum / 1MB, 2))



