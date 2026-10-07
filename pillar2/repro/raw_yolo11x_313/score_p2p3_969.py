"""P2 × P3 no conjunto congelado de 969: YOLO11x puro contra YOLO11x + VLM v2.

Os dois braços usam as MESMAS 969 caixas e os MESMOS 731 objetos. O VLM só
decide se a caixa sobrevive e com que rótulo; ele não move caixa. Quem decide se
a caixa final está certa é o gabarito.

  braço YOLO      : as 969, rótulo do detector
  braço YOLO+VLM  : tira `rejected`; `class-changed` troca o rótulo pelo do VLM;
                    kept / corrected-minor / corrected-redrawn mantêm a caixa

Duas leituras, as mesmas do REPORT §3:
  C1  matcher padrão (one-to-one, confiança decrescente, mesma classe), IoU 0.50 e 0.75
  C2  vínculo geométrico τ=0.10 fixado nas caixas (a geometria não muda com o VLM);
      correta = vinculada E rótulo FINAL == classe do objeto E IoU >= 0.70.
      Mais estrita que o `score_phase1` antigo, que dava acerto a uma correção de
      classe sem olhar a geometria.

Somente leitura. Escreve `p2p3_969_results.json` e `p2p3_969_examples.json`.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORGE = HERE.parents[2]
sys.path.insert(0, str(FORGE / "src"))
sys.path.insert(0, str(HERE))

from analyze_raw_p2p3 import (  # noqa: E402
    Box, by_key, link, load_frames, load_gt, load_old, pct, q,
)
from forge.config import PARITY_IOU, EvalClassScope  # noqa: E402
from forge.core.geometry import BoxN, iou  # noqa: E402
from forge.evaluate.matcher import GtInstance, Label, Prediction, match  # noqa: E402

CORRECT_IOU = PARITY_IOU  # 0.70
FOCUS = ("car", "truck")


def load():
    frozen = json.loads((HERE / "p2p3_frozen_969.json").read_text(encoding="utf-8"))["proposals"]
    rows = {r["key"]: r for r in json.loads(
        (HERE / "vlm_v2_969" / "phase1.json").read_text(encoding="utf-8"))}
    boxes, vlm = [], {}
    for p in frozen:
        b = Box(f"{p['acq_id']}:{p['sec']}", p["proposal_id"], p["class"], p["conf"],
                BoxN(*p["box_n"]))
        boxes.append(b)
        r = rows.get(f"{p['acq_id']}:{p['sec']}:{p['rank_in_frame']}")
        if r is None or not r.get("vlm"):
            raise SystemExit(f"ERRO: sem veredito VLM para {p['proposal_id']}")
        vlm[b.uid] = r
    return boxes, vlm


def apply_vlm(boxes: list[Box], vlm: dict) -> list[Box]:
    out = []
    for b in boxes:
        r = vlm[b.uid]
        if r["vlm"] == "rejected":
            continue
        lab = r["vlm_class"] if r["vlm"] == "class-changed" and r["vlm_class"] else b.cls
        out.append(Box(b.key, b.uid, lab, b.conf, b.b))
    return out


def c1(arm: list[Box], gt: list[Box]) -> dict:
    preds = [Prediction(b.key, b.cls, b.conf, b.b, b.uid) for b in arm]
    gts = [GtInstance(g.key, g.cls, g.b, g.extra[0], g.uid) for g in gt]
    out = {}
    for t in (0.5, 0.75):
        res = match(preds, gts, iou_threshold=t, class_scope=EvalClassScope.GT_PRESENT)
        k = res.counters
        ious = [m.best_iou for m in res.predictions if m.label is Label.TP]
        row = {"detections": k["detections"], "tp": k["tp"], "fp": k["fp"], "fn": k["fn"],
               "precision": pct(k["tp"], k["detections"]), "recall": pct(k["tp"], k["ground_truth"]),
               "f1": round(200 * k["tp"] / (k["detections"] + k["ground_truth"]), 1),
               "tp_iou_mean": round(sum(ious) / len(ious), 4) if ious else None,
               "tide": {x: k[x] for x in ("fp_localization", "fp_classification", "fp_both",
                                          "fp_duplicate", "fp_background")},
               "per_class": {}}
        for c in FOCUS:
            det = sum(1 for m in res.predictions if m.pred.cls_name == c)
            tp = sum(1 for m in res.predictions if m.pred.cls_name == c and m.label is Label.TP)
            g = sum(1 for m in res.ground_truth if m.gt.cls_name == c)
            gtp = sum(1 for m in res.ground_truth if m.gt.cls_name == c and m.matched_by)
            row["per_class"][c] = {"gt": g, "detections": det, "tp": tp, "fp": det - tp,
                                   "fn": g - gtp, "precision": pct(tp, det),
                                   "recall": pct(gtp, g),
                                   "f1": round(200 * tp / (det + g), 1) if det + g else None}
        out[str(t)] = row
    return out


def c2(arm: list[Box], gt: list[Box], p2g: dict, reach: int) -> dict:
    ok = [b for b in arm if b.uid in p2g and b.cls == p2g[b.uid].cls
          and iou(b.b, p2g[b.uid].b) >= CORRECT_IOU]
    out = {"retained": len(arm), "correct": len(ok), "incorrect": len(arm) - len(ok),
           "not_recovered": len(gt) - len(ok), "gt_total": len(gt),
           "reachable_by_detector": reach, "not_proposed": len(gt) - reach,
           "precision": pct(len(ok), len(arm)), "coverage_of_gt": pct(len(ok), len(gt)),
           "reachable_coverage": pct(len(ok), reach), "per_class": {}}
    for c in FOCUS:
        gc = [g for g in gt if g.cls == c]
        ret_c = [b for b in arm if b.cls == c]
        ok_c = [b for b in ok if b.cls == c]
        reach_c = sum(1 for b in p2g.values() if b.cls == c)
        out["per_class"][c] = {"gt": len(gc), "reachable": reach_c, "retained": len(ret_c),
                               "correct": len(ok_c), "precision": pct(len(ok_c), len(ret_c)),
                               "coverage_of_gt": pct(len(ok_c), len(gc)),
                               "reachable_coverage": pct(len(ok_c), reach_c)}
    return out


def diff(a: dict, b: dict, keys: list[str]) -> dict:
    out = {}
    for k in keys:
        x, y = a.get(k), b.get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            out[k] = {"yolo": x, "yolo_vlm": y, "abs": round(y - x, 1),
                      "rel_pct": round(100 * (y - x) / x, 1) if x else None}
    return out


def truth_of(b: Box, p2g: dict) -> str:
    g = p2g.get(b.uid)
    if g is None:
        return "rejected"
    if b.cls != g.cls:
        return "class-changed"
    return "correct" if iou(b.b, g.b) >= CORRECT_IOU else "redrawn"


def vlm_actions(boxes: list[Box], vlm: dict, p2g: dict) -> dict:
    act = Counter(vlm[b.uid]["vlm"] for b in boxes)
    cross = Counter((truth_of(b, p2g), vlm[b.uid]["vlm"]) for b in boxes)
    cc = [b for b in boxes if vlm[b.uid]["vlm"] == "class-changed"]
    cc_right = sum(1 for b in cc if b.uid in p2g and vlm[b.uid]["vlm_class"] == p2g[b.uid].cls)
    rej = [b for b in boxes if vlm[b.uid]["vlm"] == "rejected"]
    return {
        "verdicts": dict(act.most_common()),
        "box_retained": sum(v for k, v in act.items() if k != "rejected"),
        "rejected": act["rejected"], "class_changed": act["class-changed"],
        "class_changed_to": dict(Counter(f"{b.cls}->{vlm[b.uid]['vlm_class']}" for b in cc).most_common()),
        "class_changed_to_gt_class": cc_right,
        "class_changed_wrong_or_unlinked": len(cc) - cc_right,
        "rejected_by_geometric_truth": dict(Counter(truth_of(b, p2g) for b in rej)),
        "truth_x_vlm": {f"{t} | {v}": n for (t, v), n in sorted(cross.items())},
    }


def gt_near_duplicates(gt: list[Box], thr: float = 0.7) -> list[tuple[Box, Box, float]]:
    """Pares de caixas do gabarito no mesmo frame com IoU >= thr (qualquer classe).

    O gabarito nasceu corrigindo as propostas antigas, e duas propostas
    co-localizadas (ex.: `train` + `truck` no mesmo veículo) viraram duas
    instâncias. Não se mexe no gabarito (política), mas com matching one-to-one
    a segunda caixa vira FN forçado sempre que só uma proposta cobre o objeto —
    e isso precisa estar medido, não escondido.
    """
    out = []
    G = by_key(gt)
    for k, gs in G.items():
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                ov = iou(gs[i].b, gs[j].b)
                if ov >= thr:
                    out.append((gs[i], gs[j], ov))
    return out


def _bn(b: BoxN) -> list[float]:
    return [round(v, 5) for v in (b.x1, b.y1, b.x2, b.y2)]


def examples(boxes, vlm, gt, p2g, g2p, raw_all, raw_tail, old, dup_gt_uids):
    """Candidatos para os 4 painéis. Só lista; não desenha nada.

    Critérios, todos para que a frase da legenda seja literalmente verdadeira:
      miss      objeto sem NENHUMA proposta crua conf>=0.25 (qualquer das 80
                classes) a IoU >= 0.3, e que não é caixa duplicada do gabarito
      class fix detector errou a classe do objeto vinculado, VLM trocou para a
                classe do gabarito, e nenhuma outra das 969 cobre o objeto
      bad box   mesma classe, IoU < 0.70 com o objeto, VLM manteve a caixa sem
                pedir redesenho (kept ou corrected-minor), nenhuma outra
                proposta cobre bem o objeto; prioriza caixa do gabarito
                desenhada por humano (origin != kept, que é geometria de Kalman)
      fp        sem objeto do gabarito a IoU > 0, VLM rejeitou; ordenado por
                tamanho, porque caixa de 20 px não se lê numa figura
    """
    B = by_key(boxes)
    RA = by_key(raw_all)
    RT = by_key(raw_tail)
    OLD = by_key(old)
    area = lambda b: (b.x2 - b.x1) * (b.y2 - b.y1)  # noqa: E731

    def others_on(g, uid, thr=0.3):
        return [(o.uid, o.cls, round(o.conf, 3), round(iou(o.b, g.b), 3))
                for o in B.get(g.key, []) if o.uid != uid and iou(o.b, g.b) >= thr]

    miss = []
    for g in gt:
        if g.uid in g2p or g.uid in dup_gt_uids:
            continue
        near = [(r.cls, round(r.conf, 3), round(iou(r.b, g.b), 3))
                for r in RA.get(g.key, []) if iou(r.b, g.b) >= 0.3]
        if near:
            continue
        tail_best = max(((iou(r.b, g.b), r) for r in RT.get(g.key, []) if iou(r.b, g.b) >= 0.5),
                        key=lambda t: t[1].conf, default=None)
        miss.append({"frame": g.key, "gt_class": g.cls, "gt_origin": g.extra[0],
                     "gt_box_n": _bn(g.b), "gt_area_frac": round(area(g.b), 4),
                     "best_tail_pred_iou>=0.5": (None if tail_best is None else
                         {"class": tail_best[1].cls, "conf": round(tail_best[1].conf, 3),
                          "iou": round(tail_best[0], 3)}),
                     "claim": "no proposal (conf>=0.25) for this object"})
    miss.sort(key=lambda d: -d["gt_area_frac"])

    fix = []
    for b in boxes:
        g = p2g.get(b.uid)
        r = vlm[b.uid]
        if g is None or b.cls == g.cls or r["vlm"] != "class-changed" or r["vlm_class"] != g.cls:
            continue
        oth = others_on(g, b.uid)
        if oth:
            continue
        fix.append({"frame": b.key, "proposal_id": b.uid, "yolo_class": b.cls,
                    "conf": round(b.conf, 3), "vlm_class": r["vlm_class"], "gt_class": g.cls,
                    "gt_origin": g.extra[0], "iou_with_gt": round(iou(b.b, g.b), 3),
                    "box_area_frac": round(area(b.b), 4),
                    "proposal_box_n": _bn(b.b), "gt_box_n": _bn(g.b), "vlm_reason": r["reason"]})
    fix.sort(key=lambda d: (-d["box_area_frac"] * (d["iou_with_gt"] >= 0.8), -d["iou_with_gt"]))

    bad = []
    for b in boxes:
        g = p2g.get(b.uid)
        r = vlm[b.uid]
        if g is None or b.cls != g.cls or r["vlm"] not in ("kept", "corrected-minor"):
            continue
        ov = iou(b.b, g.b)
        if ov >= CORRECT_IOU or others_on(g, b.uid, 0.5):
            continue
        bad.append({"frame": b.key, "proposal_id": b.uid, "class": b.cls,
                    "conf": round(b.conf, 3), "vlm_verdict": r["vlm"],
                    "iou_with_gt": round(ov, 3), "gt_origin": g.extra[0],
                    "box_area_frac": round(area(b.b), 4), "gt_area_frac": round(area(g.b), 4),
                    "proposal_box_n": _bn(b.b), "gt_box_n": _bn(g.b), "vlm_reason": r["reason"]})
    bad.sort(key=lambda d: (d["gt_origin"] == "kept", d["box_area_frac"] < 0.004,
                            d["iou_with_gt"]))

    fp = []
    for b in boxes:
        if b.uid in p2g or vlm[b.uid]["vlm"] != "rejected":
            continue
        max_gt = max((iou(b.b, g.b) for g in gt if g.key == b.key), default=0.0)
        human_rej = any(o.extra[0] == "rejected" and iou(o.b, b.b) >= 0.5
                        for o in OLD.get(b.key, []))
        fp.append({"frame": b.key, "proposal_id": b.uid, "class": b.cls,
                   "conf": round(b.conf, 3), "max_iou_any_gt": round(max_gt, 3),
                   "same_region_explicitly_rejected_by_human": human_rej,
                   "box_area_frac": round(area(b.b), 4), "proposal_box_n": _bn(b.b),
                   "vlm_reason": vlm[b.uid]["reason"]})
    fp.sort(key=lambda d: (not d["same_region_explicitly_rejected_by_human"],
                           d["max_iou_any_gt"] > 0, -d["box_area_frac"]))
    return {"detector_miss": {"n": len(miss), "top": miss[:8]},
            "class_correction": {"n": len(fix), "top": fix[:8]},
            "bad_box_accepted": {"n": len(bad), "top": bad[:8]},
            "false_positive_rejected": {"n": len(fp), "top": fp[:8]}}


def main() -> int:
    frames = load_frames()
    gt = load_gt(frames)
    boxes, vlm = load()
    assert len(boxes) == 969 and len(gt) == 731
    p2g, g2p = link(boxes, gt, 0.1)
    reach = len(g2p)
    arm_vlm = apply_vlm(boxes, vlm)

    r = {"_meta": {"proposals_frozen": len(boxes), "gt": len(gt), "link_iou": 0.1,
                   "correct_iou_c2": CORRECT_IOU,
                   "vlm_run": str(HERE / "vlm_v2_969"),
                   "rows_with_error": sum(1 for v in vlm.values() if v.get("error"))}}
    r["C1"] = {"yolo": c1(boxes, gt), "yolo_vlm": c1(arm_vlm, gt)}
    r["C2"] = {"yolo": c2(boxes, gt, p2g, reach), "yolo_vlm": c2(arm_vlm, gt, p2g, reach)}
    keys1 = ["detections", "tp", "fp", "fn", "precision", "recall", "f1"]
    keys2 = ["retained", "correct", "incorrect", "precision", "coverage_of_gt", "reachable_coverage"]
    r["diff"] = {
        "C1@0.5": diff(r["C1"]["yolo"]["0.5"], r["C1"]["yolo_vlm"]["0.5"], keys1),
        "C1@0.75": diff(r["C1"]["yolo"]["0.75"], r["C1"]["yolo_vlm"]["0.75"], keys1),
        "C2": diff(r["C2"]["yolo"], r["C2"]["yolo_vlm"], keys2),
    }
    for c in FOCUS:
        r["diff"][f"C1@0.5_{c}"] = diff(r["C1"]["yolo"]["0.5"]["per_class"][c],
                                        r["C1"]["yolo_vlm"]["0.5"]["per_class"][c], keys1)
        r["diff"][f"C2_{c}"] = diff(r["C2"]["yolo"]["per_class"][c],
                                    r["C2"]["yolo_vlm"]["per_class"][c],
                                    ["retained", "correct", "precision", "coverage_of_gt",
                                     "reachable_coverage"])
    r["vlm_actions"] = vlm_actions(boxes, vlm, p2g)

    # Caixas quase duplicadas no gabarito e quanto do FN elas explicam.
    dups = gt_near_duplicates(gt)
    dup_uids = {x.uid for a, b_, _ in dups for x in (a, b_)}
    preds = [Prediction(b.key, b.cls, b.conf, b.b, b.uid) for b in boxes]
    gts = [GtInstance(g.key, g.cls, g.b, g.extra[0], g.uid) for g in gt]
    res = match(preds, gts, iou_threshold=0.5, class_scope=EvalClassScope.GT_PRESENT)
    fn_uids = {m.gt.det_id for m in res.ground_truth if not m.matched_by}
    r["gt_near_duplicates_iou070"] = {
        "pairs": len(dups), "gt_boxes_involved": len(dup_uids),
        "same_class_pairs": sum(1 for a, b_, _ in dups if a.cls == b_.cls),
        "fn_yolo_c1_050_that_are_dup_boxes": len(fn_uids & dup_uids),
        "fn_yolo_c1_050_total": len(fn_uids),
        "pairs_detail": [{"frame": a.key, "a": [a.cls, a.extra[0]], "b": [b_.cls, b_.extra[0]],
                          "iou": round(ov, 3)} for a, b_, ov in dups],
        "note": ("Gabarito não é alterado (política). Com matching one-to-one, a segunda "
                 "caixa de um par só casa se houver duas propostas no mesmo objeto."),
    }
    (HERE / "p2p3_969_results.json").write_text(json.dumps(r, indent=1, ensure_ascii=False),
                                                encoding="utf-8")

    from analyze_raw_p2p3 import load_raw
    raw_all = load_raw(HERE / "raw_predictions_tail.json", 0.25)
    raw_tail = load_raw(HERE / "raw_predictions_tail.json", 0.0)
    ex = examples(boxes, vlm, gt, p2g, g2p, raw_all, raw_tail, load_old(frames), dup_uids)
    (HERE / "p2p3_969_examples.json").write_text(json.dumps(ex, indent=1, ensure_ascii=False),
                                                 encoding="utf-8")
    print(json.dumps({k: v for k, v in r["gt_near_duplicates_iou070"].items()
                      if k != "pairs_detail"}, indent=1, ensure_ascii=False))
    print({k: v["n"] for k, v in ex.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
