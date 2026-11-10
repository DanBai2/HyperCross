[CmdletBinding()]
param(
    [ValidateSet("Default", "Unit", "Property", "Integration", "Regression", "Smoke", "Audit", "GpuSmoke", "Slow", "Strict", "Collection")]
    [string]$Suite = "Default",

    [string]$PythonExecutable = "python",

    [string]$CondaEnvironment = "",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$AdditionalArguments = @()
)

$PytestArguments = @("-m", "pytest")

switch ($Suite) {
    "Unit"        { $PytestArguments += @("-m", "unit") }
    "Property"    { $PytestArguments += @("-m", "property") }
    "Integration" { $PytestArguments += @("-m", "integration") }
    "Regression"  { $PytestArguments += @("-m", "regression") }
    "Smoke"       { $PytestArguments += @("-m", "smoke and not gpu") }
    "Audit"       { $PytestArguments += @("-m", "audit and not gpu") }
    "GpuSmoke" {
        $PytestArguments += @("-m", "gpu and smoke", "--require-audited-environment")
        if ([string]::IsNullOrWhiteSpace($CondaEnvironment)) {
            $CondaEnvironment = "TCFM"
        }
    }
    "Slow"        { $PytestArguments += @("-m", "slow and not strict") }
    "Strict"      { $PytestArguments += @("-m", "strict") }
    "Collection"  { $PytestArguments += @("--collect-only", "-q") }
    "Default"     { }
}

$PytestArguments += $AdditionalArguments

if ([string]::IsNullOrWhiteSpace($CondaEnvironment)) {
    & $PythonExecutable @PytestArguments
}
else {
    & conda run --no-capture-output -n $CondaEnvironment $PythonExecutable @PytestArguments
}

exit $LASTEXITCODE
