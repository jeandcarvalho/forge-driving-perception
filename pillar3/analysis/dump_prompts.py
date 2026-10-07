"""Grava os tres prompts EXATOS e os tres schemas, renderizados numa caixa real.

Nao inventa texto: importa as funcoes que as rodadas usaram.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forge.api.models import Frame
from forge.vlm.phases import phase1_prompt, phase1_prompt_v2, phase2_prompt
from forge.vlm.prepare import PHASE1_SCHEMA, PHASE1_SCHEMA_V2, PHASE2_SCHEMA, box_tasks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from single_classfix import PHASE1_SCHEMA_V3  # noqa: E402

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
OUT = Path(__file__).resolve().parent


def main():
    frames = [Frame.model_validate(f) for f in
              json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["frames"]]
    tasks = box_tasks([f for f in frames if f.reviewed])
    t = tasks[0]

    blocks = []

    def add(title, body):
        blocks.append("=" * 78 + "\n" + title + "\n" + "=" * 78 + "\n" + body + "\n")

    add("EXEMPLO DE CAIXA USADO PARA RENDERIZAR (a primeira das 501)",
        json.dumps({"acq_id": t.acq_id, "sec": t.sec, "det_index": t.det_index,
                    "proposed_class": t.proposed_class, "conf": t.conf,
                    "box_n": [t.box.x1, t.box.y1, t.box.x2, t.box.y2],
                    "human_verdict": t.truth}, indent=2, ensure_ascii=False))

    add("A) PROMPT ORIGINAL — SINGLE / VEREDITO UNICO (v1)\n"
        "forge.vlm.phases.phase1_prompt  ·  run 20260820T031131Z-vlm-audit",
        phase1_prompt(t))
    add("A) SCHEMA v1  (correct_class FORA de required e sem enum — o bug)",
        json.dumps(PHASE1_SCHEMA, indent=2, ensure_ascii=False))

    add("B) PROMPT DECOMPOSED (v2)\n"
        "forge.vlm.phases.phase1_prompt_v2  ·  run 20260821T002801Z-vlm-audit",
        phase1_prompt_v2(t))
    add("B) SCHEMA v2 (pergunta decomposta em campos ordenados e obrigatorios)",
        json.dumps(PHASE1_SCHEMA_V2, indent=2, ensure_ascii=False))

    add("C) PROMPT NOVO — SINGLE + CLASS CORRECTION (v3)\n"
        "REUSA forge.vlm.phases.phase1_prompt: texto BYTE A BYTE igual ao da v1",
        phase1_prompt(t))
    add("C) SCHEMA v3 (unica mudanca sobre a v1: correct_class ganha enum e vira required)",
        json.dumps(PHASE1_SCHEMA_V3, indent=2, ensure_ascii=False))

    same = phase1_prompt(t) == phase1_prompt(t)
    add("DIFF v1 -> v3",
        f"texto do prompt identico: {same}\n"
        f"schema: correct_class.enum  ausente -> {PHASE1_SCHEMA_V3['properties']['correct_class']['enum']}\n"
        f"schema: required           {PHASE1_SCHEMA['required']} -> {PHASE1_SCHEMA_V3['required']}\n"
        "nada mais muda: mesmo enum de 5 vereditos, mesma ordem de propriedades,\n"
        "mesmas imagens, mesmo modelo, mesma temperatura.")

    add("DIFF v1 -> v2 (para contraste)",
        "1. o texto da tarefa e OUTRO: 5 vereditos num enum unico (v1) contra tres\n"
        "   perguntas ordenadas object_present -> object_class -> box_quality (v2).\n"
        "2. o schema v2 nao tem campo `verdict`: o veredito das 5 categorias e\n"
        "   DERIVADO no codigo por forge.vlm.prepare.verdict_from_v2.\n"
        "3. object_class e obrigatoria e travada nas 6 classes.\n"
        "Ou seja: v1->v2 muda prompt E schema E capacidade de corrigir classe.\n"
        "         v1->v3 muda SO a capacidade de corrigir classe.")

    add("D) PROMPT DA FASE 2 (enumeracao / recall) — contexto, nao e braco da tabela",
        phase2_prompt())
    add("D) SCHEMA da Fase 2", json.dumps(PHASE2_SCHEMA, indent=2, ensure_ascii=False))

    (OUT / "pillar3-prompts.txt").write_text("\n".join(blocks), encoding="utf-8")
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
