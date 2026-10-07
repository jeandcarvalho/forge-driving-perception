"""Pilar 3 — separacao SEMANTICA x GEOMETRIA, e o experimento de coordenadas.

Tudo offline: le manifest, phase1.json dos runs de VLM e o cache do vlm-flow.
Nao chama modelo, nao escreve run antigo.

Semantica  : presenca (existe objeto?) e classe (qual objeto?).
Geometria  : qualidade da caixa (kept / needs_nudge / badly_placed do VLM
             contra kept / corrected-minor / corrected-redrawn do humano).
Coordenada : substituir a caixa por aquela do YOLO11x fresco que mais sobrepoe,
             onde o VLM pediu correcao — com IoU antes/depois contra o gabarito.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forge.api.models import Frame
from forge.config import GT_EXCLUDED_CLASSES, PARITY_IOU
from forge.core.classes import normalize_class
from forge.core.geometry import BoxN, iou
from forge.vlm.flow import best_overlap, box_is_correct
from forge.vlm.phases import Phase1Row
from forge.vlm.prepare import box_tasks

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
RUNS = ROOT / "runs"
CACHE = RUNS / "_flow_cache"
OUT = Path(__file__).resolve().parent

ARMS = {
    "vlm_single_v1": RUNS / "20260820T031131Z-vlm-audit",
    "vlm_decomposed_v2": RUNS / "20260821T002801Z-vlm-audit",
}
# o quarto braco entra se ja tiver rodado
for d in sorted(RUNS.glob("*-vlm-audit-v3-single-classfix")):
    ARMS["vlm_single_classfix_v3"] = d

_TRUTH_OK = frozenset({"kept", "corrected-minor"})
#: leitura de "o VLM acha que a caixa esta boa"
_VLM_BOX_OK = frozenset({"kept", "class-changed"})


def load_frames():
    return [Frame.model_validate(f) for f in
            json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["frames"]]


def truth_index(frames):
    """key -> (classe do gabarito, caixa do gabarito) para a proposta ligada."""
    out = {}
    for f in frames:
        for i, d in enumerate(f.stored_detections):
            key = f"{f.acq_id}:{f.sec}:{i}"
            if d.verdict == "kept" and d.box_n is not None and isinstance(d.cls_name, str):
                out[key] = (normalize_class(d.cls_name), d.box_n.to_box())
            elif d.verdict == "corrected" and d.correction is not None \
                    and d.correction.box_n is not None \
                    and isinstance(d.correction.cls_name, str):
                out[key] = (normalize_class(d.correction.cls_name),
                            d.correction.box_n.to_box())
    return out


def main():
    frames = load_frames()
    reviewed = [f for f in frames if f.reviewed]
    tasks = box_tasks(reviewed)
    by_key = {t.key: t for t in tasks}
    truth = truth_index(frames)
    orig_box = {f"{f.acq_id}:{f.sec}:{i}": d.box_n.to_box()
                for f in frames for i, d in enumerate(f.stored_detections)
                if d.box_n is not None}

    out = {"_meta": {
        "manifest": str(ROOT / "manifest.json"),
        "arms": {k: str(v) for k, v in ARMS.items()},
        "flow_cache": str(CACHE),
        "parity_iou": PARITY_IOU,
        "gt_excluded": sorted(GT_EXCLUDED_CLASSES),
        "semantics_vs_geometry": {
            "semantica": "presenca (object_present / rejected) e classe",
            "geometria": "qualidade da caixa: kept x corrected-minor x corrected-redrawn",
            "aviso": ("nenhum braco de Fase 1 devolve COORDENADA. Geometria so entra "
                      "no placar via o veredito humano: corrected-redrawn nunca vira TP. "
                      "O julgamento geometrico do VLM nao move a Table 6 por si."),
        },
    }, "arms": {}}

    for label, run in ARMS.items():
        p1 = run / "phase1.json"
        if not p1.is_file():
            continue
        rows = [Phase1Row(**r) for r in json.loads(p1.read_text(encoding="utf-8"))]
        rows = [r for r in rows if r.key in by_key]
        out["arms"][label] = arm_semantics_geometry(rows, truth, by_key)
        out["arms"][label]["run_dir"] = str(run)

    # ── concordancia de veredito entre bracos (isola o efeito do schema) ──
    loaded = {}
    for label, run in ARMS.items():
        p1 = run / "phase1.json"
        if p1.is_file():
            loaded[label] = {r.key: r for r in
                             (Phase1Row(**r) for r in json.loads(p1.read_text(encoding="utf-8")))
                             if r.key in by_key}
    pairs = {}
    names = list(loaded)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = set(loaded[a]) & set(loaded[b])
            same = sum(1 for k in common if loaded[a][k].vlm == loaded[b][k].vlm)
            cm = defaultdict(Counter)
            for k in common:
                cm[loaded[a][k].vlm or "ERRO"][loaded[b][k].vlm or "ERRO"] += 1
            pairs[f"{a} x {b}"] = {
                "n_common": len(common), "identical_verdicts": same,
                "pct_identical": round(100 * same / len(common), 1) if common else None,
                "matrix": {k: dict(v) for k, v in cm.items()},
            }
    out["verdict_agreement_between_arms"] = pairs

    # ── experimento de coordenadas ──
    out["coordinate_adjustment"] = coordinate_experiment(rows_of(ARMS["vlm_single_v1"], by_key),
                                                         truth, orig_box)

    (OUT / "pillar3-semantic-geometry.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


def rows_of(run: Path, by_key):
    rows = [Phase1Row(**r) for r in json.loads((run / "phase1.json").read_text(encoding="utf-8"))]
    return [r for r in rows if r.key in by_key]


def arm_semantics_geometry(rows, truth, by_key):
    # ---------- SEMANTICA: presenca ----------
    pres = Counter()
    for r in rows:
        v = r.vlm or "kept"
        vlm_present = v != "rejected"
        real = r.truth != "rejected"
        pres[("real" if real else "background") + "__vlm_" +
             ("present" if vlm_present else "absent")] += 1
    tp_p = pres["background__vlm_absent"]      # FP de fundo corretamente rejeitado
    fn_p = pres["background__vlm_present"]     # lixo aceito
    fp_p = pres["real__vlm_absent"]            # objeto real rejeitado (dano)
    tn_p = pres["real__vlm_present"]
    presence = {
        "counts": dict(sorted(pres.items())),
        "background_correctly_rejected": tp_p,
        "background_wrongly_kept": fn_p,
        "real_object_wrongly_rejected": fp_p,
        "real_object_correctly_kept": tn_p,
        "rejection_precision_of_the_vlm":
            round(100 * tp_p / (tp_p + fp_p), 1) if (tp_p + fp_p) else None,
        "rejection_recall_over_24_background":
            round(100 * tp_p / (tp_p + fn_p), 1) if (tp_p + fn_p) else None,
        "good_boxes_destroyed_kept_or_minor": sum(
            1 for r in rows if (r.vlm or "kept") == "rejected" and r.truth in _TRUTH_OK),
    }

    # ---------- SEMANTICA: classe ----------
    attempts = [r for r in rows if (r.vlm or "kept") == "class-changed"]
    by_truth = Counter(r.truth for r in attempts)
    right = wrong_target = wrong_class = no_class = 0
    for r in attempts:
        if r.truth != "class-changed":
            wrong_target += 1     # pediu troca onde o humano nao trocou
            continue
        want = truth.get(r.key, (None, None))[0]
        if r.vlm_class is None:
            no_class += 1
        elif r.vlm_class == want:
            right += 1
        else:
            wrong_class += 1
    human_class_errors = sum(1 for r in rows if r.truth == "class-changed")
    missed = sum(1 for r in rows if r.truth == "class-changed"
                 and (r.vlm or "kept") != "class-changed")
    klass = {
        "human_class_errors_in_501": human_class_errors,
        "attempts": len(attempts),
        "attempts_by_human_verdict": dict(by_truth),
        "right": right,
        "wrong_class_on_a_real_class_error": wrong_class,
        "no_class_emitted_schema_bug": no_class,
        "attempted_where_human_did_not_relabel": wrong_target,
        "human_class_errors_missed": missed,
        "precision_of_attempts": round(100 * right / len(attempts), 1) if attempts else None,
        "recall_over_human_class_errors":
            round(100 * right / human_class_errors, 1) if human_class_errors else None,
        "note": ("o numero publicado '43 de 55' e RECALL sobre os 55 erros de classe "
                 "do humano, nao acuracia das tentativas do VLM."),
    }

    # ---------- GEOMETRIA ----------
    mat = defaultdict(Counter)
    for r in rows:
        mat[r.truth][r.vlm or "ERRO"] += 1
    geom_rows = {}
    for t in ("kept", "corrected-minor", "corrected-redrawn"):
        row = mat.get(t, Counter())
        n = sum(row.values())
        ok = sum(row[v] for v in _VLM_BOX_OK)
        geom_rows[t] = {
            "n": n,
            "vlm_says_box_is_fine": ok,
            "vlm_says_box_needs_work": row["corrected-minor"] + row["corrected-redrawn"],
            "vlm_rejects_outright": row["rejected"],
            "pct_says_fine": round(100 * ok / n, 1) if n else None,
        }
    good = geom_rows["kept"]["n"] + geom_rows["corrected-minor"]["n"]
    good_ok = geom_rows["kept"]["vlm_says_box_is_fine"] + geom_rows["corrected-minor"]["vlm_says_box_is_fine"]
    bad = geom_rows["corrected-redrawn"]["n"]
    bad_ok = geom_rows["corrected-redrawn"]["vlm_says_box_is_fine"]
    geometry = {
        "human_x_vlm_full_matrix": {k: dict(v) for k, v in mat.items()},
        "by_human_geometry_verdict": geom_rows,
        "headline": {
            "bad_boxes_accepted_as_fine": bad_ok,
            "bad_boxes_total": bad,
            "pct_bad_accepted": round(100 * bad_ok / bad, 1) if bad else None,
            "good_boxes_flagged_or_rejected": good - good_ok,
            "good_boxes_total": good,
            "pct_good_flagged": round(100 * (good - good_ok) / good, 1) if good else None,
            "separation_pp": (round(100 * good_ok / good - 100 * bad_ok / bad, 1)
                              if good and bad else None),
            "reading": ("separation_pp e a diferenca entre P(VLM diz 'caixa ok' | humano "
                        "aceitou a caixa) e P(mesmo | humano refez a caixa). Zero = o "
                        "julgamento geometrico do VLM nao carrega informacao."),
        },
        "minor_vs_redraw_performance": {
            "on_corrected_minor": dict(mat.get("corrected-minor", Counter())),
            "on_corrected_redrawn": dict(mat.get("corrected-redrawn", Counter())),
            "caveat": ("o VLM nunca ve a distincao minor x redrawn: ela e derivada "
                       "no codigo por IoU(original, correcao) >= 0.7, depois do fato."),
        },
    }
    return {"semantics": {"presence": presence, "class_correction": klass},
            "geometry": geometry}


def coordinate_experiment(rows, truth, orig_box):
    """Substitui a caixa pela deteccao YOLO11x fresca que mais sobrepoe,
    onde o VLM pediu correcao. Mede IoU antes/depois contra o gabarito."""
    fresh_raw = json.loads((CACHE / "detections.json").read_text(encoding="utf-8"))
    fresh = {k: [(c, f, BoxN(*b)) for c, f, b in v] for k, v in fresh_raw.items()}

    changed, improved, worsened, unchanged, no_candidate = [], 0, 0, 0, 0
    tp_before = tp_after = 0
    deltas = []
    for r in rows:
        v = r.vlm or "kept"
        if v == "rejected":
            continue
        label = r.vlm_class if v == "class-changed" and r.vlm_class else r.proposed_class
        box = orig_box.get(r.key)
        if box is None:
            continue
        t = truth.get(r.key)
        before_ok = box_is_correct(label, box, t)
        tp_before += before_ok

        new_box = box
        if v in {"corrected-minor", "corrected-redrawn"}:
            nb = best_overlap(box, fresh.get(f"{r.acq_id}:{r.sec}", []))
            if nb is None:
                no_candidate += 1
            else:
                new_box = nb
                iou_before = iou(box, t[1]) if t else 0.0
                iou_after = iou(nb, t[1]) if t else 0.0
                deltas.append(iou_after - iou_before)
                changed.append({
                    "key": r.key, "human_verdict": r.truth, "vlm_verdict": v,
                    "iou_before": round(iou_before, 4), "iou_after": round(iou_after, 4),
                    "delta": round(iou_after - iou_before, 4),
                    "was_tp": bool(before_ok),
                    # mesma regra de acerto antes e depois: rotulo E IoU >= 0.7
                    "is_tp_after": bool(box_is_correct(label, nb, t)),
                })
                if iou_after > iou_before + 1e-9:
                    improved += 1
                elif iou_after < iou_before - 1e-9:
                    worsened += 1
                else:
                    unchanged += 1
        after_ok = box_is_correct(label, new_box, truth.get(r.key))
        tp_after += after_ok

    broke = sum(1 for c in changed if c["was_tp"] and not c["is_tp_after"])
    fixed = sum(1 for c in changed if not c["was_tp"] and c["is_tp_after"])
    still_wrong = sum(1 for c in changed if not c["was_tp"] and not c["is_tp_after"])
    still_right = sum(1 for c in changed if c["was_tp"] and c["is_tp_after"])
    deltas_sorted = sorted(deltas)
    n = len(deltas_sorted)
    return {
        "source_arm": "vlm_single_v1 (o unico braco em que este experimento foi rodado)",
        "scoring_rule": ("box_is_correct: rotulo igual ao do gabarito E IoU(caixa, "
                         "caixa do gabarito) >= 0.7. Regra GEOMETRICA, diferente da "
                         "contagem por veredito da Table 6 — mas reproduz 329/172 no base."),
        "replacement_source": "YOLO11x fresco @conf 0.25 (runs/_flow_cache/detections.json)",
        "corrections_requested_by_vlm": len(changed) + no_candidate,
        "geometry_replaced": len(changed),
        "no_overlapping_candidate": no_candidate,
        "tp_before": tp_before, "tp_after": tp_after, "net_delta_tp": tp_after - tp_before,
        "boxes_improved_iou": improved,
        "boxes_worsened_iou": worsened,
        "boxes_unchanged_iou": unchanged,
        "good_box_broken_tp_to_fp": broke,
        "bad_box_fixed_fp_to_tp": fixed,
        "stayed_tp": still_right,
        "stayed_fp": still_wrong,
        "net_from_transitions": fixed - broke,
        "iou_delta_mean": round(sum(deltas) / n, 4) if n else None,
        "iou_delta_median": round(deltas_sorted[n // 2], 4) if n else None,
        "iou_before_mean": round(sum(c["iou_before"] for c in changed) / len(changed), 4)
                           if changed else None,
        "iou_after_mean": round(sum(c["iou_after"] for c in changed) / len(changed), 4)
                          if changed else None,
        "per_box": changed,
        "published_claim_check": {
            "numeros_csv_says": "tp_antes 395 -> tp_depois 387 (marcado 'conferir antes do texto')",
            "measured_here": f"tp_antes {tp_before} -> tp_depois {tp_after}",
            "395_387_found_in_artifacts": False,
        },
    }


if __name__ == "__main__":
    main()
