param(
    [int]$Port = 8770,
    [string]$Open = "",          # label_project.json (or its folder) to reopen
    [switch]$NoBrowser
)

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $here ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing QA virtual environment: $python"
}
$arguments = @("$here\server.py", "--port", $Port)
if ($Open) { $arguments += @("--open", $Open) }
if (-not $NoBrowser) {
    Start-Job -ScriptBlock { param($url) Start-Sleep -Seconds 6; Start-Process $url } -ArgumentList "http://127.0.0.1:$Port/" | Out-Null
}
$env:PYTHONIOENCODING = "utf-8"
& $python @arguments
