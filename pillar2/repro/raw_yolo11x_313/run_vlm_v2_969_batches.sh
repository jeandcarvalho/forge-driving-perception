#!/usr/bin/env bash
# Roda run_vlm_v2_969.py em lotes, reiniciando o Ollama entre eles.
# Motivo: o llama-server acumula memória privada ao longo das chamadas e a
# primeira sessão foi morta por falta de RAM em 472/969. O checkpoint
# (vlm_v2_969/phase1.jsonl) garante que nenhuma caixa é refeita nem pulada.
set -u
FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${FORGE_PYTHON:-$FORGE/.venv/Scripts/python.exe}"
SCRIPT="$FORGE/pillar2/repro/raw_yolo11x_313/run_vlm_v2_969.py"
BATCH=${1:-150}

# Causa real da falta de RAM (medida em 10/09/2026): o llama-server do Ollama
# 0.32 guarda cada prompt (~490 MiB com as duas imagens) num cache em RAM de até
# 8 GiB (`--cache-ram`, padrão 8192). Aqui ele não reaproveita nada — cada caixa
# tem imagem própria. LLAMA_ARG_CACHE_RAM=0 desliga: working set 8,4 → 1,1 GB,
# vereditos idênticos (78/78 contra o checkpoint).
restart_ollama() {
  powershell.exe -NoProfile -Command \
    "Get-Process llama-server, ollama -ErrorAction SilentlyContinue | Stop-Process -Force -Confirm:\$false; Start-Sleep -Seconds 3; \$env:LLAMA_ARG_CACHE_RAM='0'; Start-Process -FilePath \"\$env:LOCALAPPDATA\\Programs\\Ollama\\ollama.exe\" -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep -Seconds 6" \
    > /dev/null 2>&1
}

for i in 1 2 3 4 5 6 7 8; do
  echo "=== lote $i ==="
  restart_ollama
  out=$("$PY" -u "$SCRIPT" --limit "$BATCH" 2>&1)
  code=$?
  echo "$out" | grep -v "^  [0-9]*/" | tail -12
  echo "$out" | grep "^  [0-9]*/" | tail -1
  if [ $code -ne 0 ]; then echo "lote $i saiu com codigo $code"; fi
  # só ASCII no padrão: o stdout do Python no Windows chega aqui em cp1252
  if echo "$out" | grep -q "; 0 a fazer"; then
    echo "=== completo ==="
    exit 0
  fi
done
echo "=== limite de lotes atingido sem completar ==="
exit 1
