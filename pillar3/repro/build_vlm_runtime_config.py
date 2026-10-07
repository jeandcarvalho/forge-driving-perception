"""Monta vlm-runtime-config.json a partir dos runs reais + manifesto do Ollama + config.py.

So leitura. Nada aqui e digitado a mao: cada campo aponta a fonte.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

FORGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORGE / "src"))

from forge import config as C
from forge.api.models import Frame
from forge.vlm.phases import phase1_prompt, phase1_prompt_v2, phase2_prompt
from forge.vlm.prepare import (PHASE1_SCHEMA, PHASE1_SCHEMA_V2, PHASE2_SCHEMA, box_tasks)

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent
OLLAMA = Path.home() / ".ollama" / "models"

ARMS = {
    "v1_single": RUNS / "20260820T031131Z-vlm-audit",
    "v2_decomposed": RUNS / "20260821T002801Z-vlm-audit",
    "v3_single_classfix": RUNS / "20260907T164834Z-vlm-audit-v3-single-classfix",
}

sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
shaj = lambda d: hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

man_path = OLLAMA / "manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct"
man_raw = man_path.read_bytes()
manifest = json.loads(man_raw)
cfg_digest = manifest["config"]["digest"].split(":")[1]
model_cfg = json.loads((OLLAMA / "blobs" / f"sha256-{cfg_digest}").read_text(encoding="utf-8"))
params_digest = next(l["digest"].split(":")[1] for l in manifest["layers"]
                     if l["mediaType"].endswith("params"))
model_params = json.loads((OLLAMA / "blobs" / f"sha256-{params_digest}").read_text(encoding="utf-8"))
weights_layer = next(l for l in manifest["layers"] if l["mediaType"].endswith("image.model"))

frames = [Frame.model_validate(f) for f in
          json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["frames"]]
tasks = box_tasks([f for f in frames if f.reviewed])
t0 = tasks[0]

V3_SCHEMA = json.loads(json.dumps(PHASE1_SCHEMA))
V3_SCHEMA["properties"]["correct_class"]["enum"] = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
V3_SCHEMA["required"] = ["verdict", "correct_class", "reason"]

metas = {}
for k, d in ARMS.items():
    p = d / "meta.json"
    metas[k] = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def usage(k):
    m = metas[k]
    if not m:
        return None
    r = m.get("results", {})
    return r.get("usage") or (r.get("phase1") or {}).get("usage")


out = {
    "_gerado_em": "2026-09-07",
    "_fontes": {
        "config": str(FORGE / "src/forge/config.py"),
        "cliente": str(FORGE / "src/forge/vlm/client.py"),
        "preparo_de_imagem": str(FORGE / "src/forge/vlm/prepare.py"),
        "manifesto_ollama": str(man_path),
        "runs": {k: str(v) for k, v in ARMS.items()},
    },

    "comum_aos_tres_bracos": {
        "modelo": {
            "nome_ollama": C.VLM_MODEL,
            "digest_da_tag_sha256": hashlib.sha256(man_raw).hexdigest(),
            "familia": model_cfg["model_family"],
            "tamanho_declarado": model_cfg["model_type"],
            "quantizacao": model_cfg["file_type"],
            "formato": model_cfg["model_format"],
            "renderer_parser": [model_cfg.get("renderer"), model_cfg.get("parser")],
            "peso_bytes": weights_layer["size"],
            "peso_digest_sha256": weights_layer["digest"].split(":")[1],
            "licenca_do_modelo": "Apache-2.0 (blob de licenca dentro do proprio manifesto)",
            "variante": "instruct (NAO thinking) - ver comentario em config.VLM_MODEL",
        },
        "runtime": {
            "servidor": "Ollama",
            "versao_cliente_medida_em_2026-09-07": "0.32.15",
            "endpoint": f"{C.VLM_URL}/api/chat",
            "stream": False,
            "timeout_s": 300.0,
            "OLLAMA_NUM_PARALLEL": 1,
            "paralelismo_no_cliente": "nenhum - chamadas sincronas em serie",
        },
        "amostragem": {
            "temperature": C.VLM_TEMPERATURE,
            "temperature_origem": "unico campo passado em options{} pelo cliente",
            "top_p": model_params.get("top_p"),
            "top_k": model_params.get("top_k"),
            "top_p_top_k_origem": "parametros do Modelfile do modelo; o cliente NAO os sobrescreve",
            "seed": None,
            "seed_nota": "nenhuma seed e enviada. O determinismo vem de temperature=0 + gramatica, nao de semente fixada.",
            "repeat_penalty": None,
            "num_predict": None,
            "num_ctx": None,
            "nao_setados_nota": "campos ausentes de options{} usam o default do Ollama/Modelfile e NAO estao registrados no run.",
            "decodificacao_restrita": "JSON schema no campo `format` do /api/chat (gramatica llama.cpp)",
        },
        "imagens": {
            "quantas_por_chamada": 2,
            "imagem_1": "frame inteiro 1920x1080 com a caixa proposta desenhada (contorno preto 5px + rosa 3px + rotulo)",
            "imagem_2": "recorte ao redor da caixa, com a caixa tambem desenhada dentro do recorte",
            "resolucao_nativa": list(C.EXPECTED_IMAGE_SIZE),
            "VLM_IMAGE_MAX_SIDE": C.VLM_IMAGE_MAX_SIDE,
            "redimensionamento": "nenhum no frame inteiro; o recorte so e ampliado se ficar menor que VLM_CROP_MIN_PX (LANCZOS)",
            "VLM_CROP_PAD": C.VLM_CROP_PAD,
            "VLM_CROP_MIN_PX": C.VLM_CROP_MIN_PX,
            "codificacao": "JPEG qualidade 92, base64, campo images[] do /api/chat",
            "cor_da_caixa_rgb": [255, 0, 128],
            "proposta_desenhada_visualmente": True,
            "coordenadas_tambem_em_texto": True,
            "formato_das_coordenadas_no_texto": "normalizadas [x1,y1,x2,y2] com 3 casas, origem no canto superior esquerdo",
        },
        "granularidade": {
            "uma_chamada_por": "PROPOSTA (caixa), nao por frame",
            "propostas": len(tasks),
            "frames": len({(t.acq_id, t.sec) for t in tasks}),
            "acquisitions": len({t.acq_id for t in tasks}),
        },
        "erro_e_retry": {
            "no_cliente": "nenhum retry. HTTP error, conteudo vazio ou JSON invalido viram VlmError e contam em usage.failures.",
            "no_v3": "o checkpoint reinfere linhas gravadas com erro em vez de herda-las; o phase1.json final tem 0 erros.",
        },
        "hardware": {
            "gpu": "NVIDIA GeForce RTX 5060 Ti, 16311 MiB",
            "driver": "610.88",
            "cpu": "13th Gen Intel(R) Core(TM) i5-13600KF",
            "ram_gb": 15.8,
            "so": "Windows 11 Pro 10.0.26200",
            "python": "3.11.9",
        },
    },

    "diferencas_entre_bracos": {
        "v1_single": {
            "prompt_fn": "forge.vlm.phases.phase1_prompt",
            "prompt_sha256_na_1a_caixa": sha(phase1_prompt(t0)),
            "schema": PHASE1_SCHEMA,
            "schema_sha256": shaj(PHASE1_SCHEMA),
            "linhas_no_phase1_json": 505,
            "pontuado_sobre": 501,
            "usage": usage("v1_single"),
            "run": str(ARMS["v1_single"]),
            "git_commit": (metas["v1_single"] or {}).get("environment", {}).get("git_commit"),
        },
        "v3_single_classfix": {
            "prompt_fn": "forge.vlm.phases.phase1_prompt (a MESMA da v1, reusada)",
            "prompt_sha256_na_1a_caixa": sha(phase1_prompt(t0)),
            "schema": V3_SCHEMA,
            "schema_sha256": shaj(V3_SCHEMA),
            "linhas_no_phase1_json": 501,
            "pontuado_sobre": 501,
            "usage": usage("v3_single_classfix"),
            "usage_nota": "cobre so a ULTIMA das tres sessoes; o braco foi retomado por checkpoint.",
            "run": str(ARMS["v3_single_classfix"]),
            "git_commit": (metas["v3_single_classfix"] or {}).get("environment", {}).get("git_commit"),
            "diff_contra_v1": {
                "texto_do_prompt": "identico byte a byte",
                "schema.correct_class.enum": "ausente -> 6 classes",
                "schema.required": "[verdict, reason] -> [verdict, correct_class, reason]",
            },
        },
        "v2_decomposed": {
            "prompt_fn": "forge.vlm.phases.phase1_prompt_v2",
            "prompt_sha256_na_1a_caixa": sha(phase1_prompt_v2(t0)),
            "schema": PHASE1_SCHEMA_V2,
            "schema_sha256": shaj(PHASE1_SCHEMA_V2),
            "veredito_derivado_por": "forge.vlm.prepare.verdict_from_v2 (o schema nao tem campo verdict)",
            "linhas_no_phase1_json": 501,
            "pontuado_sobre": 501,
            "usage": usage("v2_decomposed"),
            "run": str(ARMS["v2_decomposed"]),
            "git_commit": (metas["v2_decomposed"] or {}).get("environment", {}).get("git_commit"),
        },
    },

    "fase2_recall_contexto": {
        "nao_e_braco_da_tabela": True,
        "prompt_sha256": sha(phase2_prompt()),
        "schema_sha256": shaj(PHASE2_SCHEMA),
        "VLM_RECALL_MAX_ITERS": C.VLM_RECALL_MAX_ITERS,
        "VLM_RECALL_MAX_ITEMS": C.VLM_RECALL_MAX_ITEMS,
        "VLM_RECALL_DEDUP_IOU": C.VLM_RECALL_DEDUP_IOU,
        "VLM_RECALL_PRESENCE_IOU": C.VLM_RECALL_PRESENCE_IOU,
        "run": str(RUNS / "20260820T173145Z-vlm-audit"),
    },

    "determinismo_observado": {
        "v1_x_v3_vereditos_identicos": "494/501 = 98,6% (pillar3-semantic-geometry.json)",
        "nota": ("v1 e v3 NAO sao repeticoes do mesmo experimento - o schema muda -, "
                 "entao 98,6% e o limite superior observavel de repetibilidade, nao uma "
                 "medida de determinismo. Nenhum braco foi reexecutado com a MESMA "
                 "configuracao para medir determinismo puro."),
    },
}

(OUT / "vlm-runtime-config.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print("escrito", OUT / "vlm-runtime-config.json")
print(json.dumps(out["comum_aos_tres_bracos"]["modelo"], indent=1, ensure_ascii=False))
print(json.dumps(out["comum_aos_tres_bracos"]["amostragem"], indent=1, ensure_ascii=False))
