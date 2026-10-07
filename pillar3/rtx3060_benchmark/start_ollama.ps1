<#
Inicia o servidor Ollama na configuração da sessão final do run de 969 (P3 atual).

Configuração (ver runtime/ollama_server_config.json):
  - `ollama serve` direto (NÃO o app de bandeja, que se auto-atualiza);
  - todas as variáveis OLLAMA_* / LLAMA_ARG_* nos padrões do Ollama 0.32.15
    (no PC original nenhuma estava definida no ambiente do usuário/máquina),
    exceto OLLAMA_MODELS, que só diz onde estão os modelos;
  - LLAMA_ARG_CACHE_RAM=0 (cache de prompt em RAM desligado), como na sessão final.
O stderr do servidor vai para results\ollama-serve.log — o benchmark lê esse log para
confirmar contexto, slots, offload e flash attention efetivos.

Uso:
  powershell -ExecutionPolicy Bypass -File .\start_ollama.ps1 [-OllamaExe C:\caminho\ollama.exe]
#>
param([string]$OllamaExe = "")

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$results = Join-Path $here "results"
New-Item -ItemType Directory -Force $results | Out-Null

if (-not $OllamaExe) {
    if ($env:OLLAMA_EXE) { $OllamaExe = $env:OLLAMA_EXE }
    elseif (Get-Command ollama -ErrorAction SilentlyContinue) { $OllamaExe = (Get-Command ollama).Source }
    elseif (Test-Path "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe") { $OllamaExe = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
    else { throw "ollama.exe não encontrado. Passe -OllamaExe." }
}
Write-Host "ollama.exe: $OllamaExe"

# Para qualquer Ollama em execução (inclusive o app de bandeja, que pode se auto-atualizar).
Get-Process "ollama app", ollama, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
Start-Sleep -Seconds 3

# Padrões: remove overrides do processo (não toca nas variáveis persistentes do usuário).
Get-ChildItem Env: | Where-Object { ($_.Name -like "OLLAMA_*" -and $_.Name -ne "OLLAMA_MODELS") -or $_.Name -like "LLAMA_ARG_*" } |
    ForEach-Object { Write-Host "removendo do processo: $($_.Name)=$($_.Value)"; Remove-Item "Env:$($_.Name)" }
$env:LLAMA_ARG_CACHE_RAM = "0"

$log = Join-Path $results "ollama-serve.log"
$out = Join-Path $results "ollama-serve.out"
if (Test-Path $log) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    Move-Item $log (Join-Path $results "ollama-serve.$stamp.log")
    if (Test-Path $out) { Move-Item $out (Join-Path $results "ollama-serve.$stamp.out") }
}
Start-Process -FilePath $OllamaExe -ArgumentList "serve" -WindowStyle Hidden `
    -RedirectStandardError $log -RedirectStandardOutput $out

$v = $null
foreach ($i in 1..30) {
    Start-Sleep -Seconds 1
    try { $v = (Invoke-RestMethod http://localhost:11434/api/version).version; break } catch {}
}
if (-not $v) { throw "servidor não respondeu em 30 s; veja $log" }
Write-Host "Ollama no ar: versão $v"
if ($v -ne "0.32.15") {
    Get-Process ollama, llama-server -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:$false
    throw "versão $v != 0.32.15 — servidor parado. NÃO rode o benchmark."
}
Write-Host "Log do servidor: $log"
