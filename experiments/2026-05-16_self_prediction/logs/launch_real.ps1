$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$logFile = 'experiments\2026-05-16_self_prediction\logs\run.log'
"=== REAL RUN LAUNCH $(Get-Date -Format o) ===" | Out-File -FilePath $logFile -Append -Encoding utf8
& python experiments\2026-05-16_self_prediction\run.py --real --rollout-workers 6 --max-workers 6 *>> $logFile
"=== REAL RUN EXIT $(Get-Date -Format o) (exit=$LASTEXITCODE) ===" | Out-File -FilePath $logFile -Append -Encoding utf8
