<#
  Portability patch: retarget the hard-coded data root.

  The audited code in this repository uses absolute paths under `G:\Datasets`
  (the machine on which every reported number was produced). Run this script to
  point the source at your own dataset root.

  This is a PORTABILITY PATCH. Applying it makes the working tree differ from the
  audited revision; note that difference when reporting results.

  Example:
      powershell -ExecutionPolicy Bypass -File tools/relocate_data_root.ps1 -NewRoot D:\datasets
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$NewRoot,
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = 'Stop'
$old = 'G:\Datasets'
$new = $NewRoot.TrimEnd('\')

if (-not (Test-Path -LiteralPath $RepoRoot)) { throw "repo root not found: $RepoRoot" }
if ($new -eq $old) { throw "NewRoot is already the audited root: $old" }

$changed = 0
$hits = 0
Get-ChildItem -LiteralPath $RepoRoot -Recurse -File -Filter *.py |
    ForEach-Object {
        if ($_.FullName -match '\\external\\' -or $_.FullName -match '\\\.git\\') { return }
        $text = Get-Content -LiteralPath $_.FullName -Raw -Encoding utf8
        if ($null -ne $text -and $text.Contains($old)) {
            $count = ([regex]::Matches($text, [regex]::Escape($old))).Count
            $text = $text.Replace($old, $new)
            Set-Content -LiteralPath $_.FullName -Value $text -Encoding utf8 -NoNewline
            $hits += $count
            $changed++
            Write-Output ("  patched {0,3} occurrence(s)  {1}" -f $count, $_.FullName.Substring($RepoRoot.Length).TrimStart('\'))
        }
    }

Write-Output ""
Write-Output ("Rewrote '{0}' -> '{1}' in {2} file(s), {3} occurrence(s)." -f $old, $new, $changed, $hits)
Write-Output "Remember: downstream artefacts (runs_v2/, results/) still record the original paths."
