$ErrorActionPreference = 'Continue'
$listPath = 'experiments\2026-05-16_self_prediction\logs\image_list.txt'
$logPath  = 'experiments\2026-05-16_self_prediction\logs\pull_images.log'
$images = Get-Content $listPath | Where-Object { $_ -and $_.Trim().Length -gt 0 }
"=== Starting pull of $($images.Count) images at $(Get-Date -Format o) ===" | Out-File -FilePath $logPath -Encoding utf8

$images | ForEach-Object -Parallel {
    $img = $_
    $log = $using:logPath
    $start = Get-Date
    "[START] $img at $($start.ToString('HH:mm:ss'))" | Out-File -FilePath $log -Append -Encoding utf8
    $out = & docker pull $img 2>&1 | Out-String
    $end = Get-Date
    $dur = [int]($end - $start).TotalSeconds
    if ($LASTEXITCODE -eq 0) {
        "[DONE]  $img in ${dur}s" | Out-File -FilePath $log -Append -Encoding utf8
    } else {
        "[FAIL]  $img in ${dur}s (exit=$LASTEXITCODE)" | Out-File -FilePath $log -Append -Encoding utf8
        $out | Out-File -FilePath $log -Append -Encoding utf8
    }
} -ThrottleLimit 4

"=== Finished at $(Get-Date -Format o) ===" | Out-File -FilePath $logPath -Append -Encoding utf8
