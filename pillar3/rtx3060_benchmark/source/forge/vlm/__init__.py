"""Experimento A1 — VLM como curador.

Mede o quão bom um VLM local é substituindo o humano na curadoria, contra o
mesmo gabarito dos 313 frames. Não treina nada e não escreve no gabarito: gera
um veredito paralelo e compara.

O gabarito é verdade absoluta. Quando o VLM discorda, o VLM errou.
"""

from forge.vlm.client import Usage, VlmClient, VlmError
from forge.vlm.phases import Phase1Row, Phase2Row, run_phase1, run_phase2
from forge.vlm.prepare import BoxTask, box_tasks, human_verdict
from forge.vlm.report import agreement, curation_metrics, recall_report

__all__ = [
    "BoxTask",
    "Phase1Row",
    "Phase2Row",
    "Usage",
    "VlmClient",
    "VlmError",
    "agreement",
    "box_tasks",
    "curation_metrics",
    "human_verdict",
    "recall_report",
    "run_phase1",
    "run_phase2",
]
