$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$env:PYTHONPATH = "D:\Code\AIDetector-evidence-research\data\cache\py3langid-screen-20260907\site;$env:PYTHONPATH"

& "D:\Anaconda\envs\lab\python.exe" ".\run_roberta_server.py"
