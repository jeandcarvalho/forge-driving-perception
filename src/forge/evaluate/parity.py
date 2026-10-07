"""Auditoria da curadoria — porte 1:1 de `curation/stats.ts`.

ISTO NÃO É AVALIAÇÃO DE DETECTOR. Não confunda com evaluate/matcher.py.

A API não faz matching geométrico entre predição e gabarito: ela conta por
*veredito de curadoria*, usando o vínculo `ref_det_id` que liga cada detecção
pré-anotada à ação humana que caiu sobre ela. O IoU 0.7 que aparece aqui compara
a caixa ORIGINAL com a CORREÇÃO DELA MESMA — a pergunta é "o curador empurrou a
caixa ou refez a caixa?", não "a predição achou o objeto?".

    tp = kept + corrected(mesma classe, IoU(orig, correção) ≥ 0.7)
    fp = corrected_box_major + corrected_class + rejected
    fn = added + corrected_box_major + corrected_class
    gt = detections − rejected + added

Serve para uma coisa só, e ela é essencial: provar que a dobra do gabarito no
Forge é idêntica à da API. Se os números baterem, todo o resto do pipeline está
lendo o mesmo gabarito que a plataforma. Se não baterem, para tudo.

Não "melhore" a aritmética daqui. Ela existe para reproduzir outra implementação,
não para estar certa segundo algum livro.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

from forge.api.models import Frame
from forge.config import PARITY_IOU
from forge.core.classes import normalize_class
from forge.core.geometry import iou

#: Bloco publicado em PROMPT.md §5 para a coleção-alvo. É o gabarito do gabarito.
EXPECTED_TARGET_COLLECTION = {
    "reviewed_frames": 313,
    "detections": 505,
    "ground_truth": 735,
    "tp": 333,
    "fp": 172,
    "fn": 402,
    "kept": 249,
    "corrected_box_minor": 84,
    "corrected_box_major": 93,
    "corrected_class": 55,
    "rejected": 24,
    "added": 254,
}


def js_round(x: float) -> float:
    """`Math.round` do JS: meio arredonda para +infinito (Python arredonda para par)."""
    return math.floor(x + 0.5)


def _ratio(n: int, total: int) -> float | None:
    return js_round((n / total) * 1000) / 10 if total > 0 else None


@dataclass(slots=True)
class ParityReport:
    total_frames: int
    reviewed_frames: int
    iou_threshold: float

    detections: int
    ground_truth: int
    tp: int
    fp: int
    fn: int
    precision: float | None
    recall: float | None
    f1: float | None

    fp_localization: int
    fp_classification: int
    fp_background: int
    fn_undetected: int
    fn_from_errors: int

    kept: int
    corrected_box_minor: int
    corrected_box_major: int
    corrected_class: int
    rejected: int
    added: int

    model_boxes: int = 0
    #: Divergências entre o que derivei de stored_detections/ground_truth e o
    #: `curation_counts` que a própria API mandou. Deve ser vazio.
    count_mismatches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def diff_against(self, expected: dict[str, int]) -> dict[str, tuple[int, int]]:
        """Campos que divergem: {campo: (obtido, esperado)}."""
        got = self.as_dict()
        return {k: (got[k], v) for k, v in expected.items() if got.get(k) != v}


def curation_parity(frames: list[Frame]) -> ParityReport:
    """Recalcula o bloco de `curation/stats` a partir do manifest.

    Conta só sobre frames revisados — o resto do conjunto ainda não tem veredito,
    e misturá-los inflaria `unreviewed` dentro de `detections`.
    """
    total_frames = 0
    reviewed_frames = 0

    model_boxes = 0
    kept = 0
    box_minor = 0  # mesma classe, caixa só empurrada
    box_major = 0  # mesma classe, caixa refeita (ou sem caixa para comparar)
    class_changed = 0  # o modelo errou o QUE é
    rejected = 0
    added = 0

    mismatches: list[str] = []

    for f in frames:
        total_frames += 1
        if not f.reviewed:
            continue
        reviewed_frames += 1

        model_boxes += len(f.stored_detections)

        f_kept = sum(1 for d in f.stored_detections if d.verdict == "kept")
        f_corrected = sum(1 for d in f.stored_detections if d.verdict == "corrected")
        f_rejected = sum(1 for d in f.stored_detections if d.verdict == "rejected")
        f_unreviewed = sum(1 for d in f.stored_detections if d.verdict == "unreviewed")
        # `added` não está em stored_detections (não tem original de onde pendurar):
        # vive só no gabarito resolvido.
        f_added = sum(1 for g in (f.ground_truth or []) if g.origin == "added")

        kept += f_kept
        rejected += f_rejected
        added += f_added

        # Derivei tudo de stored_detections + ground_truth; a API também mandou
        # curation_counts. Os dois caminhos têm que concordar — se não, um dos
        # dois lados mudou e o silêncio seria pior que o erro.
        cc = f.curation_counts
        if cc is not None:
            for label, mine, theirs in (
                ("model", len(f.stored_detections), cc.model),
                ("kept", f_kept, cc.kept),
                ("corrected", f_corrected, cc.corrected),
                ("rejected", f_rejected, cc.rejected),
                ("added", f_added, cc.added),
                ("unreviewed", f_unreviewed, cc.unreviewed),
            ):
                if mine != theirs:
                    mismatches.append(f"{f.key} {label}: derivei {mine}, curation_counts diz {theirs}")

        # `corrected` ainda precisa da comparação original × correção para separar
        # erro de classe de erro de caixa (e caixa empurrada de caixa refeita).
        for d in f.stored_detections:
            if d.verdict != "corrected" or d.correction is None:
                continue
            same_class = (
                isinstance(d.cls_name, str)
                and isinstance(d.correction.cls_name, str)
                and normalize_class(d.cls_name) == normalize_class(d.correction.cls_name)
            )
            if not same_class:
                class_changed += 1
                continue
            a = d.box_n
            b = d.correction.box_n
            if a is not None and b is not None and iou(a.to_box(), b.to_box()) >= PARITY_IOU:
                box_minor += 1
            else:
                box_major += 1

    # ── Aritmética do stats.ts, campo a campo ────────────────────────────────
    detections = kept + box_minor + box_major + class_changed + rejected
    tp = kept + box_minor
    fp_localization = box_major
    fp_classification = class_changed
    fp_background = rejected
    fp = fp_localization + fp_classification + fp_background
    ground_truth = detections - fp_background + added
    fn_undetected = added
    fn_from_errors = fp_localization + fp_classification
    fn = fn_undetected + fn_from_errors

    precision = _ratio(tp, detections)
    recall = _ratio(tp, ground_truth)
    # f1 sai dos valores JÁ ARREDONDADOS de p e r — é assim no stats.ts, e
    # recalcular "direito" daria um número diferente do da API.
    f1 = (
        js_round(((2 * precision * recall) / (precision + recall)) * 10) / 10
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )

    return ParityReport(
        total_frames=total_frames,
        reviewed_frames=reviewed_frames,
        iou_threshold=PARITY_IOU,
        detections=detections,
        ground_truth=ground_truth,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        fp_localization=fp_localization,
        fp_classification=fp_classification,
        fp_background=fp_background,
        fn_undetected=fn_undetected,
        fn_from_errors=fn_from_errors,
        kept=kept,
        corrected_box_minor=box_minor,
        corrected_box_major=box_major,
        corrected_class=class_changed,
        rejected=rejected,
        added=added,
        model_boxes=model_boxes,
        count_mismatches=mismatches,
    )


def parity_detection_labels(frames: list[Frame]) -> dict[str, str]:
    """Rótulo que a contagem por veredito dá a CADA detecção armazenada.

    Chave: "<acq_id>:<sec>|<det_id>". Só frames revisados — os outros não têm
    veredito e não entram na contagem.

    É a mesma aritmética de `curation_parity`, exposta por detecção em vez de
    agregada, para que evaluate/diff.py possa comparar item a item com o que o
    matcher geométrico decidiu.
    """
    labels: dict[str, str] = {}
    for f in frames:
        if not f.reviewed:
            continue
        for d in f.stored_detections:
            if d.det_id is None:
                continue
            key = f"{f.key}|{d.det_id}"
            if d.verdict == "kept":
                labels[key] = "tp"
            elif d.verdict == "rejected":
                labels[key] = "fp_background"
            elif d.verdict == "corrected" and d.correction is not None:
                same_class = (
                    isinstance(d.cls_name, str)
                    and isinstance(d.correction.cls_name, str)
                    and normalize_class(d.cls_name) == normalize_class(d.correction.cls_name)
                )
                if not same_class:
                    labels[key] = "fp_classification"
                else:
                    a, b = d.box_n, d.correction.box_n
                    ok = (
                        a is not None
                        and b is not None
                        and iou(a.to_box(), b.to_box()) >= PARITY_IOU
                    )
                    labels[key] = "tp" if ok else "fp_localization"
            # `unreviewed` não recebe rótulo: não há veredito.
    return labels


def gt_class_histogram(frames: list[Frame]) -> dict[str, int]:
    """Distribuição de classes no gabarito. PROMPT.md §3 diz o que tem que dar."""
    hist: dict[str, int] = {}
    for f in frames:
        for g in f.ground_truth or []:
            if g.cls_name is None:
                continue
            key = normalize_class(g.cls_name)
            hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: (-kv[1], kv[0])))
