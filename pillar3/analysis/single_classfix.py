"""Quarto braco: VLM SINGLE VERDICT + CLASS CORRECTION.

Controle pedido pela revisao: a v1 (single/holistico) nao podia corrigir classe
porque `correct_class` estava FORA de `required` no schema e o decodificador
restrito o omitia. A v2 podia. Logo o ganho da v2 confunde dois fatores:
decomposicao da pergunta E capacidade de corrigir classe.

Este braco isola: prompt de veredito unico IDENTICO ao da v1 (reusa
`forge.vlm.phases.phase1_prompt`, sem uma palavra alterada), enum de 5 vereditos
IDENTICO, e a UNICA mudanca e no schema:

    v1  "correct_class": {"type": "string"}            , required: [verdict, reason]
    v3  "correct_class": {"type": "string", "enum": 6} , required: [verdict, correct_class, reason]

`verdict` continua sendo a primeira propriedade declarada, entao a gramatica do
llama.cpp o gera ANTES de `correct_class` existir: a previsao e que os vereditos
reproduzam a v1 caixa a caixa, e a unica diferenca seja o rotulo passar a vir
preenchido em `class-changed`. O script mede essa previsao explicitamente.

Mesmo modelo, mesma temperatura, mesmas imagens, mesmas 501 propostas, mesmo GT.
Escreve num run NOVO. Nao toca run antigo, nao escreve no banco.

CHECKPOINT (04/09/2026): a primeira tentativa morreu com a maquina travada e
perdeu 100% do trabalho, porque o phase1.json so era escrito no fim. Agora cada
caixa e gravada na hora em CKPT (JSONL, append, flush+fsync). Relancar o script
pula as caixas ja respondidas e NAO reinfere nada. Nada disso toca prompt,
schema, modelo, temperatura ou a ordem de visita - so persistencia.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forge.api.models import Frame
from forge.config import VLM_IMAGE_MAX_SIDE, VLM_MODEL, VLM_TEMPERATURE, VLM_URL
from forge.core.classes import normalize_class
from forge.vlm.client import VlmClient, VlmError, load_frame_image
from forge.vlm.phases import Phase1Row, phase1_prompt
from forge.vlm.prepare import box_tasks, crop_around, draw_box

ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
OUT = Path(__file__).resolve().parent
COLLECTION_ID = "<CARCARA_COLLECTION_ID>"
CLASSES = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
#: fora de qualquer run: sobrevive a run nova e nao polui artefato fechado.
CKPT = ROOT / "runs" / "_v3_classfix_cache" / "phase1.jsonl"

#: v1 + duas mudancas minimas, e SO elas: enum das 6 e campo obrigatorio.
PHASE1_SCHEMA_V3: dict = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["kept", "corrected-minor", "corrected-redrawn",
                     "class-changed", "rejected"],
        },
        "correct_class": {
            "type": "string",
            "enum": CLASSES,
            "description": "Só quando verdict == class-changed: a classe certa.",
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "correct_class", "reason"],
    "additionalProperties": False,
}


def image_path(acq_id: str, sec: int) -> Path:
    return ROOT / "images" / acq_id / f"{sec:06d}.jpg"


def main() -> None:
    frames = [Frame.model_validate(f) for f in
              json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["frames"]]
    reviewed = [f for f in frames if f.reviewed]
    tasks = box_tasks(reviewed)
    print(f"{len(reviewed)} frames revisados · {len(tasks)} caixas (regua 501)")

    missing = [t for t in tasks if not image_path(t.acq_id, t.sec).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} imagens faltando em disco")

    by_frame: dict[tuple[str, int], list] = {}
    for t in tasks:
        by_frame.setdefault((t.acq_id, t.sec), []).append(t)

    # ── retomada ────────────────────────────────────────────────────────────
    # Uma linha JSONL por caixa ja respondida. Chave = task.key. Nada aqui muda
    # o que o modelo ve; so evita reinferir o que ja foi pago.
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    done_rows: dict[str, dict] = {}
    if CKPT.is_file():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # Uma chamada que falhou NAO e um resultado: fica de fora do "done"
            # para ser reinferida. A v1 fechou com zero falhas; herdar uma falha
            # de timeout causada por falta de memoria quebraria a comparacao.
            if r.get("vlm") is None:
                continue
            done_rows[r["key"]] = r  # ultima ocorrencia vence
        print(f"checkpoint: {len(done_rows)} caixas ja respondidas em {CKPT}")

    todo = [t for g in by_frame.values() for t in g if t.key not in done_rows]
    print(f"a inferir agora: {len(todo)} de {len(tasks)}")

    t0 = time.time()
    usage = {}
    if todo:
        vlm = VlmClient(model=VLM_MODEL, url=VLM_URL, temperature=VLM_TEMPERATURE)
        with CKPT.open("a", encoding="utf-8") as ck, vlm as client:
            print("ollama", client.health(), "· modelo", client.model, flush=True)
            done = 0
            for (acq_id, sec), group in by_frame.items():
                pend = [t for t in group if t.key not in done_rows]
                if not pend:
                    continue
                img = load_frame_image(image_path(acq_id, sec))
                for task in pend:
                    scene = draw_box(img, task.box, task.proposed_class)
                    crop = crop_around(img, task.box)
                    row = Phase1Row(
                        key=task.key, acq_id=acq_id, sec=sec, det_index=task.det_index,
                        proposed_class=task.proposed_class, conf=task.conf,
                        truth=task.truth, vlm=None, vlm_class=None, reason="",
                    )
                    try:
                        ans = client.ask(phase1_prompt(task), [scene, crop], PHASE1_SCHEMA_V3)
                        verdict = str(ans.get("verdict", "")).strip()
                        row.vlm = verdict
                        row.reason = str(ans.get("reason", ""))[:400]
                        cc = ans.get("correct_class")
                        if verdict == "class-changed" and isinstance(cc, str):
                            row.vlm_class = normalize_class(cc)
                    except VlmError as exc:
                        row.error = str(exc)[:200]
                    d = {s: getattr(row, s) for s in row.__slots__}
                    done_rows[row.key] = d
                    ck.write(json.dumps(d, ensure_ascii=False) + chr(10))
                    ck.flush()
                    os.fsync(ck.fileno())  # a maquina ja travou uma vez
                    done += 1
                    if done % 25 == 0 or done == len(todo):
                        el = time.time() - t0
                        print(f"  {done}/{len(todo)}  {el/done:.2f}s/caixa  "
                              f"eta {(len(todo)-done)*el/done/60:.1f} min", flush=True)
            usage = client.usage.as_dict()
    else:
        print("nada a inferir: checkpoint ja cobre as 501.")

    faltando = [t.key for t in tasks if t.key not in done_rows]
    if faltando:
        raise SystemExit(f"{len(faltando)} caixas sem resposta; rode de novo para retomar")

    # ordem canonica = ordem da regua, nao a ordem em que o checkpoint encheu
    rows = [done_rows[t.key] for t in tasks]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-vlm-audit-v3-single-classfix"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    print("run:", run_dir)

    (run_dir / "phase1.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "command": "vlm-audit (variante v3 fora do CLI: paper/pillar23/single_classfix.py)",
        "root": str(run_dir),
        "created_at": datetime.now(UTC).isoformat(),
        "params": {
            "model": VLM_MODEL, "phase": "1", "schema": "v3-single-classfix",
            "temperature": VLM_TEMPERATURE, "image_max_side": VLM_IMAGE_MAX_SIDE,
            "reviewed_frames": len(reviewed), "boxes": len(tasks),
            "collection_id": COLLECTION_ID,
            "prompt_fn": "forge.vlm.phases.phase1_prompt (identica a v1)",
            "schema_json": PHASE1_SCHEMA_V3,
        },
        "results": {"usage": usage,
                    "usage_note": ("usage cobre apenas as caixas inferidas na ULTIMA "
                                   "sessao; se houve retomada, o custo total esta "
                                   "espalhado entre as sessoes."),
                    "checkpoint": str(CKPT)},
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")
    print("\nCUSTO:", json.dumps(usage, ensure_ascii=False))
    print("gravado em", run_dir)


if __name__ == "__main__":
    main()
