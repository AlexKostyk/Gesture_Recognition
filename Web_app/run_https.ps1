param(
    [string]$BindHost = "0.0.0.0",
    [int]$Port = 8443
)

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $root
$certFile = Join-Path $root "certs\dev.crt"
$keyFile = Join-Path $root "certs\dev.key"
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (!(Test-Path $pythonExe)) {
    Write-Host "Не найден Python из общей среды проекта: $pythonExe" -ForegroundColor Red
    Write-Host "Создайте или активируйте окружение в корне проекта." -ForegroundColor Yellow
    exit 1
}

if (!(Test-Path $certFile) -or !(Test-Path $keyFile)) {
    Write-Host "Не найдены сертификаты: $certFile или $keyFile" -ForegroundColor Red
    Write-Host "Сначала сгенерируйте сертификат:" -ForegroundColor Yellow
    Write-Host "python tools\generate_dev_cert.py --hosts <IP_ПК>,localhost,127.0.0.1"
    exit 1
}

Push-Location $root
try {
    & $pythonExe -m uvicorn backend.app:app --reload --host $BindHost --port $Port --ssl-certfile $certFile --ssl-keyfile $keyFile
}
finally {
    Pop-Location
}
