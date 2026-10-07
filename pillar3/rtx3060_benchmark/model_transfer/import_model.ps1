<#
Instala no Ollama do PC de destino EXATAMENTE o modelo do experimento
(qwen3-vl:8b-instruct, tag digest 0533d743...c96319), sem passar pelo registry.

Este bundle já traz o manifesto e os blobs pequenos (config, params, licença).
Falta só o blob de pesos (6.140.392.672 bytes), que precisa vir do PC original:
  C:\Users\REPRO_USER\.ollama\models\blobs\sha256-1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466

Uso (com o Ollama PARADO):
  powershell -ExecutionPolicy Bypass -File .\model_transfer\import_model.ps1 -WeightsBlob E:\sha256-1329cd5a...
Destino: $env:OLLAMA_MODELS, ou %USERPROFILE%\.ollama\models.

Nada é sobrescrito: se um arquivo já existe no destino, ele é conferido por sha256.
#>
param([Parameter(Mandatory = $true)][string]$WeightsBlob)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = if ($env:OLLAMA_MODELS) { $env:OLLAMA_MODELS } else { Join-Path $env:USERPROFILE ".ollama\models" }
$weightsName = "sha256-1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466"
$manifestRel = "manifests\registry.ollama.ai\library\qwen3-vl\8b-instruct"
$tagDigest = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"

function Hash($p) { (Get-FileHash -Algorithm SHA256 $p).Hash.ToLower() }

function Put($src, $dst, $want) {
    $h = Hash $src
    if ($h -ne $want) { throw "origem com sha256 errado: $src ($h != $want)" }
    if (Test-Path $dst) {
        $hd = Hash $dst
        if ($hd -ne $want) { throw "destino JÁ EXISTE com conteúdo diferente: $dst ($hd). Não sobrescrevo." }
        Write-Host "ok (já existia): $dst"
        return
    }
    New-Item -ItemType Directory -Force (Split-Path -Parent $dst) | Out-Null
    Copy-Item $src $dst
    if ((Hash $dst) -ne $want) { throw "cópia corrompida: $dst" }
    Write-Host "copiado: $dst"
}

if (Get-Process ollama, llama-server -ErrorAction SilentlyContinue) { throw "pare o Ollama antes de importar." }
Write-Host "destino: $dest"
Write-Host "conferindo o blob de pesos (6,1 GB)..."
Put $WeightsBlob (Join-Path $dest "blobs\$weightsName") $weightsName.Substring(7)
foreach ($b in Get-ChildItem (Join-Path $here "blobs")) {
    Put $b.FullName (Join-Path $dest "blobs\$($b.Name)") $b.Name.Substring(7)
}
Put (Join-Path $here $manifestRel) (Join-Path $dest $manifestRel) $tagDigest
Write-Host "Modelo importado. Inicie o servidor com start_ollama.ps1 e rode verify_environment.py."
