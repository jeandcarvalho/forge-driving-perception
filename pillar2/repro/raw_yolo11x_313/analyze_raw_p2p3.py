"""Partes C, D, E e F: raw YOLO11x contra o gabarito humano, e contra as 505 antigas.

Somente leitura. Lê o que `run_raw_yolo11x.py` gravou nesta pasta, o
`manifest.json` do dataset (gabarito + stored_detections antigas) e o
`instances_test.json`. Escreve `analysis.json` e `analysis_tables.md` aqui.
Não roda YOLO nem VLM.

Regimes, para não confundir
───────────────────────────
C1  avaliação padrão de detector: `forge.evaluate.matcher.match` (one-to-one,
    greedy por confiança decrescente, mesma classe, escopo GT_PRESENT), IoU 0.50
    e 0.75; AP/mAP pelo pycocotools sobre a cauda conf >= 0.001.
C2  aritmética compatível com a antiga (veredito por caixa), mas com o vínculo
    proposta→objeto feito por GEOMETRIA, porque as propostas novas não têm
    `ref_det_id`. A regra é validada primeiro sobre as 501 antigas, onde o
    vínculo verdadeiro é conhecido.
D   raw × antigas: auditoria interna, não resultado.
E   o que da revisão humana antiga continua valendo.
F   prontidão do VLM: monta os BoxTask das propostas novas e renderiza, sem
    chamar o modelo.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

FORGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORGE / "src"))

from forge.api.models import Frame  # noqa: E402
from forge.config import GT_EXCLUDED_CLASSES, PARITY_IOU, EvalClassScope  # noqa: E402
from forge.core.classes import class_name, normalize_class  # noqa: E402
from forge.core.geometry import BoxN, iou  # noqa: E402
from forge.evaluate.matcher import GtInstance, Label, Prediction, match  # noqa: E402
from forge.vlm.prepare import BoxTask, box_tasks, crop_around, draw_box, human_verdict  # noqa: E402
from forge.vlm.phases import RECALL_CLASSES, phase1_prompt_v2  # noqa: E402

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
HERE = Path(__file__).resolve().parent
GT_JSON = ROOT / "annotations" / "instances_test.json"
OLD_INFER = ROOT / "runs" / "20260816T220716Z-infer" / "predictions.json"

PROPOSAL_CONF = 0.25
SENS_CONF = 0.20
#: IoU que a curadoria antiga usava para separar minor de redrawn. Aqui é o
#: critério de "caixa correta" em C2 — é exatamente o que `kept + corrected-minor`
#: significava: mesma classe e IoU(proposta, caixa humana final) >= 0.70.
CORRECT_IOU = PARITY_IOU
LINK_IOUS = (0.1, 0.3, 0.5)
NEAR_EXACT = {"iou": 0.90, "dconf": 0.05}
V2_SECONDS_PER_CALL = 4.88  # RELATORIO-REPRODUCIBILITY-DETAILS.md §6.3


# ── carga ────────────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class Box:
    """Caixa genérica para os três conjuntos (raw, antiga, gabarito)."""

    key: str          # acq:sec
    uid: str          # proposal_id / det_id / gt det_id
    cls: str
    conf: float | None
    b: BoxN
    extra: tuple = ()


def load_frames() -> list[Frame]:
    man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    fr = [Frame.model_validate(f) for f in man["frames"] if f["split"] == "test"]
    return sorted(fr, key=lambda f: (f.acq_id, f.sec))


def load_gt(frames: list[Frame]) -> list[Box]:
    out = []
    for f in frames:
        for i, g in enumerate(f.ground_truth or []):
            c = normalize_class(g.cls_name)
            if c in GT_EXCLUDED_CLASSES:
                continue
            out.append(Box(f.key, g.det_id or f"{f.key}:gt{i}", c, None, g.box_n.to_box(),
                           (g.origin, g.ref_det_id)))
    return out


def load_old(frames: list[Frame]) -> list[Box]:
    """As 505 stored_detections; extra = (verdict5, track_id, det_index, in_501)."""
    in501 = {t.key for t in box_tasks(frames)}
    out = []
    for f in frames:
        for i, d in enumerate(f.stored_detections):
            out.append(Box(f.key, d.det_id, normalize_class(d.cls_name), d.conf,
                           d.box_n.to_box(),
                           (human_verdict(d), d.track_id, i, f"{f.key}:{i}" in in501)))
    return out


def load_raw(path: Path, conf_min: float) -> list[Box]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [Box(f"{r['acq_id']}:{r['sec']}", r["proposal_id"], normalize_class(r["class"]),
                r["conf"], BoxN(*r["box_n"])) for r in rows if r["conf"] >= conf_min]


def by_key(boxes: list[Box]) -> dict[str, list[Box]]:
    d: dict[str, list[Box]] = defaultdict(list)
    for b in boxes:
        d[b.key].append(b)
    return d


def q(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    pick = lambda p: s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]  # noqa: E731
    return {"n": len(s), "mean": round(st.fmean(s), 4), "median": round(st.median(s), 4),
            "p05": round(pick(0.05), 4), "p25": round(pick(0.25), 4),
            "p75": round(pick(0.75), 4), "p95": round(pick(0.95), 4),
            "min": round(s[0], 4), "max": round(s[-1], 4)}


def pct(a: int, b: int) -> float | None:
    return round(100 * a / b, 1) if b else None


# ── C1 ───────────────────────────────────────────────────────────────────────


def c1(raw: list[Box], gt: list[Box], frames: list[Frame]) -> dict:
    preds = [Prediction(r.key, r.cls, r.conf, r.b, r.uid) for r in raw]
    gts = [GtInstance(g.key, g.cls, g.b, g.extra[0], g.uid) for g in gt]
    keys = [f.key for f in frames]
    per_frame_all = Counter(r.key for r in raw)
    in_scope = {g.cls for g in gt}
    per_frame_scope = Counter(r.key for r in raw if r.cls in in_scope)
    out = {"proposals_all_80_classes": len(raw),
           "proposals_in_gt_scope": sum(per_frame_scope.values()),
           "proposals_per_frame_all": round(len(raw) / len(keys), 3),
           "proposals_per_frame_in_scope": round(sum(per_frame_scope.values()) / len(keys), 3),
           "empty_frames_all": sum(1 for k in keys if per_frame_all[k] == 0),
           "empty_frames_in_scope": sum(1 for k in keys if per_frame_scope[k] == 0),
           "masked_out_of_scope_by_class": dict(Counter(
               r.cls for r in raw if r.cls not in in_scope).most_common()),
           "by_iou": {}}
    for t in (0.5, 0.75):
        res = match(preds, gts, iou_threshold=t, class_scope=EvalClassScope.GT_PRESENT)
        k = res.counters
        tp_ious = [m.best_iou for m in res.predictions if m.label is Label.TP]
        row = {"tp": k["tp"], "fp": k["fp"], "fn": k["fn"], "detections": k["detections"],
               "ground_truth": k["ground_truth"],
               "precision": pct(k["tp"], k["detections"]), "recall": pct(k["tp"], k["ground_truth"]),
               "f1": round(200 * k["tp"] / (k["detections"] + k["ground_truth"]), 1),
               "tp_iou": q(tp_ious), "tide": {x: k[x] for x in (
                   "fp_localization", "fp_classification", "fp_both", "fp_duplicate",
                   "fp_background", "fn_from_errors", "fn_undetected")},
               "per_class": {}}
        for c in sorted(in_scope, key=lambda c: -sum(g.cls == c for g in gt)):
            ptp = sum(1 for m in res.predictions if m.label is Label.TP and m.pred.cls_name == c)
            pdet = sum(1 for m in res.predictions if m.pred.cls_name == c)
            ggt = sum(1 for g in res.ground_truth if g.gt.cls_name == c)
            gtp = sum(1 for g in res.ground_truth if g.gt.cls_name == c and g.matched_by)
            ious = [m.best_iou for m in res.predictions if m.label is Label.TP and m.pred.cls_name == c]
            row["per_class"][c] = {"gt": ggt, "detections": pdet, "tp": ptp, "fp": pdet - ptp,
                                   "fn": ggt - gtp, "precision": pct(ptp, pdet),
                                   "recall": pct(gtp, ggt),
                                   "f1": round(200 * ptp / (pdet + ggt), 1) if pdet + ggt else None,
                                   "tp_iou_mean": round(st.fmean(ious), 4) if ious else None,
                                   "tp_iou_median": round(st.median(ious), 4) if ious else None}
        out["by_iou"][str(t)] = row
    return out


def coco_ap(pred_json: Path) -> dict:
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    preds = json.loads(pred_json.read_text(encoding="utf-8"))
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(str(GT_JSON))
    cats = sorted({a["category_id"] for a in cg.dataset["annotations"]})
    preds = [p for p in preds if p["category_id"] in set(cats)]
    with contextlib.redirect_stdout(io.StringIO()):
        cd = cg.loadRes(preds)
        ev = COCOeval(cg, cd, iouType="bbox")
        ev.params.catIds = cats
        ev.evaluate(); ev.accumulate(); ev.summarize()
    P = ev.eval["precision"]  # [T,R,K,A,M]
    per = {}
    for k, cid in enumerate(cats):
        def ap(ti):
            v = P[ti, :, k, 0, 2]; v = v[v > -1]
            return round(float(v.mean()), 4) if v.size else None
        allv = P[:, :, k, 0, 2]; allv = allv[allv > -1]
        per[class_name(cid)] = {"AP50": ap(0), "AP75": ap(5),
                                "AP50_95": round(float(allv.mean()), 4) if allv.size else None}
    return {"mAP50_95": round(float(ev.stats[0]), 4), "AP50": round(float(ev.stats[1]), 4),
            "AP75": round(float(ev.stats[2]), 4), "per_class": per,
            "predictions_in_scope": len(preds), "maxDets": 100,
            "note": "cauda conf>=0.001, escopo GT_PRESENT, pycocotools, média simples entre as 6 classes"}


def old_infer_as_boxes() -> list[Box]:
    old = json.loads(OLD_INFER.read_text(encoding="utf-8"))
    out = []
    for i, p in enumerate(old):
        acq, sec = divmod(p["image_id"], 100_000)
        x, y, w, h = p["bbox"]
        out.append(Box(f"{acq}:{sec}", f"infer16aug:{i}", class_name(p["category_id"]),
                       p["score"], BoxN(x / 1920, y / 1080, (x + w) / 1920, (y + h) / 1080)))
    return out


def crosscheck_old_infer(raw_tail: list[Box], gt: list[Box], frames: list[Frame]) -> dict:
    """A inferência nova (batch 1) contra a de 16/08 (batch 8, bbox arredondada a 3 casas).

    Mesmos pesos, mesmo imgsz/conf/NMS: a diferença esperada é só ruído de ponto
    flutuante da GPU entre tamanhos de batch. Casamento com tolerância, não por
    igualdade de string.
    """
    old = old_infer_as_boxes()
    O, N = by_key(old), by_key(raw_tail)
    hit = 0
    dconf, ious = [], []
    for k in O:
        used = set()
        for o in sorted(O[k], key=lambda b: -b.conf):
            best = None
            for n in N.get(k, []):
                if n.uid in used or n.cls != o.cls:
                    continue
                ov = iou(o.b, n.b)
                if ov >= 0.98 and (best is None or ov > best[0]):
                    best = (ov, n)
            if best is not None:
                used.add(best[1].uid)
                hit += 1
                ious.append(best[0]); dconf.append(abs(best[1].conf - o.conf))
    old25 = c1([b for b in old if b.conf >= PROPOSAL_CONF], gt, frames)["by_iou"]["0.5"]
    return {"old_run": str(OLD_INFER), "old_predictions": len(old),
            "new_predictions": len(raw_tail),
            "matched_same_class_iou>=0.98": hit, "matched_pct_of_old": pct(hit, len(old)),
            "matched_iou": q(ious), "matched_abs_dconf": q(dconf),
            "old_run_sweep_line_conf025_iou050": {k: old25[k] for k in (
                "detections", "tp", "fp", "fn", "precision", "recall")},
            "published_sweep_line": {"detections": 969, "tp": 601, "fp": 368, "fn": 130,
                                     "precision": 62.0, "recall": 82.2}}


def added_gt_recovery(raw: list[Box], gt: list[Box]) -> dict:
    """Dos objetos que o humano teve de desenhar (origin=added), quantos o YOLO cru acha?"""
    preds = [Prediction(r.key, r.cls, r.conf, r.b, r.uid) for r in raw]
    gts = [GtInstance(g.key, g.cls, g.b, g.extra[0], g.uid) for g in gt]
    res = match(preds, gts, iou_threshold=0.5, class_scope=EvalClassScope.GT_PRESENT)
    added = [m for m in res.ground_truth if m.gt.origin == "added"]
    _, g2p = link(raw, gt, 0.1)
    return {"added_gt": len(added),
            "added_matched_same_class_iou050": sum(1 for m in added if m.matched_by),
            "added_linked_any_class_iou010": sum(1 for g in gt if g.extra[0] == "added"
                                                 and g.uid in g2p),
            "added_by_class_matched": dict(Counter(m.gt.cls_name for m in added
                                                   if m.matched_by).most_common())}


def speed_vs_dconf(old: list[Box], raw: list[Box], frames: list[Frame]) -> dict:
    """Se as antigas saíram do MESMO pixel com o mesmo modelo, conf deveria bater
    (no tracker do ultralytics `score` é o da detecção). Se bater só com o carro
    parado, o pixel que o tracker viu é outro quadro do mesmo segundo."""
    speed = {f.key: (f.telemetry.speed if f.telemetry else None) for f in frames}
    buckets: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for x, y, ov in pair(old, raw):
        s = speed.get(x.key)
        if s is None:
            b = "sem velocidade"
        elif s < 1:
            b = "parado (<1 km/h)"
        elif s < 30:
            b = "1-30 km/h"
        else:
            b = ">=30 km/h"
        buckets[b].append((abs(x.conf - y.conf), ov))
    return {b: {"pairs": len(v), "median_abs_dconf": round(st.median(d for d, _ in v), 4),
                "median_iou": round(st.median(o for _, o in v), 4),
                "pairs_abs_dconf<0.01": sum(d < 0.01 for d, _ in v)}
            for b, v in sorted(buckets.items())}


# ── C2: aritmética antiga com vínculo geométrico ─────────────────────────────


def link(props: list[Box], gt: list[Box], tau: float) -> tuple[dict[str, Box], dict[str, Box]]:
    """Vínculo one-to-one proposta→objeto, por frame.

    Estágio 1: mesma classe, greedy por IoU decrescente, IoU >= tau.
    Estágio 2: o que sobrou, qualquer classe, greedy por IoU decrescente.
    Mesma prioridade do matcher (classe certa primeiro); a diferença é que aqui
    o objetivo é reconstruir "a que objeto o curador ligaria esta caixa", não
    decidir acerto — acerto é decidido depois, pela régua de 0.70.
    """
    p2g: dict[str, Box] = {}
    g2p: dict[str, Box] = {}
    P, G = by_key(props), by_key(gt)
    for k in sorted(set(P) | set(G)):
        ps, gs = P.get(k, []), G.get(k, [])
        for same in (True, False):
            pairs = sorted(((iou(p.b, g.b), -(p.conf or 0), p.uid, g.uid, p, g)
                            for p in ps for g in gs
                            if (p.cls == g.cls) == same or not same),
                           key=lambda t: (-t[0], t[1], t[2], t[3]))
            for ov, _, _, _, p, g in pairs:
                if ov < tau or p.uid in p2g or g.uid in g2p:
                    continue
                p2g[p.uid] = g
                g2p[g.uid] = p
    return p2g, g2p


def geometric_verdict(p: Box, g: Box | None) -> str:
    if g is None:
        return "rejected"
    if p.cls != g.cls:
        return "class-changed"
    return "corrected-minor-or-kept" if iou(p.b, g.b) >= CORRECT_IOU else "corrected-redrawn"


def c2_table(props: list[Box], gt: list[Box], tau: float) -> dict:
    p2g, g2p = link(props, gt, tau)
    v = Counter(geometric_verdict(p, p2g.get(p.uid)) for p in props)
    tp = v["corrected-minor-or-kept"]
    reach = len(g2p)
    out = {"link_iou": tau, "proposals": len(props), "gt_total": len(gt),
           "reachable_by_detector": reach, "not_proposed": len(gt) - reach,
           "correct_tp": tp, "verdicts_geometric": dict(v),
           "proposal_precision": pct(tp, len(props)),
           "coverage_of_gt": pct(tp, len(gt)),
           "reachable_coverage": pct(tp, reach), "per_class": {}}
    for c in ("car", "truck", "bus", "motorcycle", "person", "bicycle"):
        gc = [g for g in gt if g.cls == c]
        pc = [p for p in props if p.cls == c]
        r_c = sum(1 for g in gc if g.uid in g2p)
        tp_c = sum(1 for p in pc if geometric_verdict(p, p2g.get(p.uid)) == "corrected-minor-or-kept")
        out["per_class"][c] = {"gt": len(gc), "proposals_of_class": len(pc),
                               "reachable": r_c, "not_proposed": len(gc) - r_c, "correct": tp_c,
                               "precision": pct(tp_c, len(pc)), "coverage": pct(tp_c, len(gc)),
                               "reachable_coverage": pct(tp_c, r_c)}
    return out


def validate_c2_on_old(old: list[Box], gt: list[Box]) -> dict:
    """A regra geométrica reproduz a contagem antiga, onde o vínculo real é conhecido?"""
    old501 = [o for o in old if o.extra[3]]
    truth_link = {g.extra[1]: g for g in gt if g.extra[1]}   # ref_det_id -> gt
    ver = Counter(o.extra[0] for o in old501)
    added = sum(1 for g in gt if g.extra[0] == "added")
    official = {"proposals": len(old501), "gt_total": len(gt), "reachable": len(gt) - added,
                "not_proposed": added, "correct_tp": ver["kept"] + ver["corrected-minor"],
                "verdicts": dict(ver)}
    rows = {}
    for tau in LINK_IOUS:
        t = c2_table(old501, gt, tau)
        p2g, _ = link(old501, gt, tau)
        agree = sum(1 for o in old501 if o.uid in truth_link
                    and p2g.get(o.uid) is truth_link[o.uid])
        wrong_obj = sum(1 for o in old501 if o.uid in p2g and o.uid in truth_link
                        and p2g[o.uid] is not truth_link[o.uid])
        miss = sum(1 for o in old501 if o.uid in truth_link and o.uid not in p2g)
        spurious = sum(1 for o in old501 if o.uid not in truth_link and o.uid in p2g)
        vmat = Counter((o.extra[0], geometric_verdict(o, p2g.get(o.uid))) for o in old501)
        rows[str(tau)] = {**{k: t[k] for k in ("reachable_by_detector", "not_proposed",
                                               "correct_tp", "proposal_precision",
                                               "coverage_of_gt", "reachable_coverage")},
                          "true_links": len([o for o in old501 if o.uid in truth_link]),
                          "links_recovered_exactly": agree, "linked_to_other_object": wrong_obj,
                          "true_link_missed": miss, "linked_but_human_rejected": spurious,
                          "verdict_confusion_human_x_geometric": {
                              f"{a} -> {b}": n for (a, b), n in sorted(vmat.items())}}
    return {"official_from_ref_det_id": official, "geometric_rule": rows}


# ── D: raw × antigas ─────────────────────────────────────────────────────────


def pair(a: list[Box], b: list[Box], tau: float = 0.3) -> list[tuple[Box, Box, float]]:
    """One-to-one por frame, qualquer classe, greedy por IoU decrescente."""
    out = []
    A, B = by_key(a), by_key(b)
    for k in sorted(set(A) & set(B)):
        pairs = sorted(((iou(x.b, y.b), x.uid, y.uid, x, y) for x in A[k] for y in B[k]),
                       key=lambda t: (-t[0], t[1], t[2]))
        ua, ub = set(), set()
        for ov, _, _, x, y in pairs:
            if ov < tau or x.uid in ua or y.uid in ub:
                continue
            ua.add(x.uid); ub.add(y.uid)
            out.append((x, y, ov))
    return out


def bins(vals: list[float]) -> dict:
    edges = [(0.95, 1.01), (0.90, 0.95), (0.80, 0.90), (0.70, 0.80), (0.50, 0.70), (0.30, 0.50)]
    return {f"[{lo:.2f},{min(hi, 1):.2f}{']' if hi > 1 else ')'}": sum(lo <= v < hi for v in vals)
            for lo, hi in edges}


def audit_raw_vs_old(old: list[Box], raw_tail: list[Box], label: str) -> dict:
    res = {}
    for cmin in (PROPOSAL_CONF, SENS_CONF, 0.001):
        rr = [r for r in raw_tail if r.conf >= cmin]
        pr = pair(old, rr)
        m_old = {x.uid for x, _, _ in pr}
        m_raw = {y.uid for _, y, _ in pr}
        ious = [ov for *_, ov in pr]
        same = sum(1 for x, y, _ in pr if x.cls == y.cls)
        dconf = [abs(x.conf - y.conf) for x, y, _ in pr]
        # fusões/duplicatas: uma caixa de um lado cobrindo >=2 do outro a IoU>=0.5
        O, R = by_key(old), by_key(rr)
        old_multi = sum(1 for k in O for x in O[k]
                        if sum(iou(x.b, y.b) >= 0.5 for y in R.get(k, [])) >= 2)
        raw_multi = sum(1 for k in R for y in R[k]
                        if sum(iou(x.b, y.b) >= 0.5 for x in O.get(k, [])) >= 2)
        per_frame = Counter()
        for x, _, _ in pr:
            per_frame[x.key] += 1
        frames_all_old_matched = sum(1 for k in O if per_frame[k] == len(O[k]))
        res[f"raw_conf>={cmin}"] = {
            "raw_count": len(rr), "old_count": len(old), "pairs_iou>=0.3": len(pr),
            "old_only": len(old) - len(m_old), "raw_only": len(rr) - len(m_raw),
            "frames_where_every_old_box_has_a_raw_pair": frames_all_old_matched,
            "same_class_pct": pct(same, len(pr)), "different_class": len(pr) - same,
            "iou": q(ious), "iou_bins": bins(ious),
            "abs_dconf": q(dconf),
            "abs_dconf_bins": {"<0.01": sum(d < 0.01 for d in dconf),
                               "[0.01,0.05)": sum(0.01 <= d < 0.05 for d in dconf),
                               "[0.05,0.10)": sum(0.05 <= d < 0.10 for d in dconf),
                               ">=0.10": sum(d >= 0.10 for d in dconf)},
            "near_exact(same cls, IoU>=0.90, |dconf|<=0.05)": sum(
                1 for x, y, ov in pr if x.cls == y.cls and ov >= NEAR_EXACT["iou"]
                and abs(x.conf - y.conf) <= NEAR_EXACT["dconf"]),
            "old_box_covering_>=2_raw(IoU>=0.5)": old_multi,
            "raw_box_covering_>=2_old(IoU>=0.5)": raw_multi,
            "old_only_by_class": dict(Counter(o.cls for o in old if o.uid not in m_old).most_common()),
            "old_only_conf": q([o.conf for o in old if o.uid not in m_old]),
        }
    return {"label": label, **res}


# ── E: impacto no P2 ─────────────────────────────────────────────────────────


def impact_p2(old: list[Box], raw: list[Box], gt: list[Box], tau_link: float) -> dict:
    old501 = [o for o in old if o.extra[3]]
    pr = pair(old501, raw)
    by_old = {x.uid: (y, ov) for x, y, ov in pr}
    near, changed = [], []
    for o in old501:
        if o.uid in by_old:
            y, ov = by_old[o.uid]
            if o.cls == y.cls and ov >= NEAR_EXACT["iou"] and abs(o.conf - y.conf) <= NEAR_EXACT["dconf"]:
                near.append((o, y, ov))
                continue
        changed.append(o)
    # Veredito transferido x veredito geométrico recalculado contra o GT, nos quase-exatos.
    p2g, _ = link(raw, gt, tau_link)
    collapse = lambda v: "corrected-minor-or-kept" if v in ("kept", "corrected-minor") else v  # noqa: E731
    agree = Counter()
    for o, y, _ in near:
        agree[(collapse(o.extra[0]), geometric_verdict(y, p2g.get(y.uid)))] += 1
    # Propostas novas sem objeto no gabarito: quem já foi julgado por um humano?
    rejected_old = [o for o in old if o.extra[0] == "rejected"]
    RJ = by_key(rejected_old)
    in_scope = {g.cls for g in gt}
    unlinked = [p for p in raw if p.uid not in p2g]
    covered_by_rejection = [p for p in unlinked
                            if any(iou(p.b, r.b) >= 0.5 for r in RJ.get(p.key, []))]
    OLDK = by_key(old)
    near_any_old = [p for p in unlinked if p not in covered_by_rejection
                    and any(iou(p.b, o.b) >= 0.3 for o in OLDK.get(p.key, []))]
    need = [p for p in unlinked if p not in covered_by_rejection and p not in near_any_old]
    # Viés de geometria: 249 caixas do gabarito SÃO a caixa antiga (kept).
    kept_gt = [g for g in gt if g.extra[0] == "kept"]
    kept_iou = []
    R = by_key(raw)
    for g in kept_gt:
        same = [iou(g.b, p.b) for p in R.get(g.key, []) if p.cls == g.cls]
        kept_iou.append(max(same, default=0.0))
    return {
        "old_501_by_verdict": dict(Counter(o.extra[0] for o in old501)),
        "near_exact_counterpart": len(near),
        "near_exact_by_verdict": dict(Counter(o.extra[0] for o, _, _ in near)),
        "materially_changed_or_missing": len(changed),
        "changed_by_verdict": dict(Counter(o.extra[0] for o in changed)),
        "transferred_vs_geometric_on_near_exact": {f"{a} -> {b}": n for (a, b), n in sorted(agree.items())},
        "raw_unlinked_to_gt": len(unlinked),
        "raw_unlinked_in_scope": sum(p.cls in in_scope for p in unlinked),
        "raw_unlinked_covered_by_old_human_rejection": len(covered_by_rejection),
        "raw_unlinked_overlapping_an_old_nonrejected_box(IoU>=0.3)": len(near_any_old),
        "raw_needing_new_human_audit": len(need),
        "raw_needing_new_human_audit_in_scope": sum(p.cls in in_scope for p in need),
        "raw_needing_new_human_audit_by_class": dict(Counter(p.cls for p in need).most_common()),
        "raw_needing_new_human_audit_conf": q([p.conf for p in need if p.cls in in_scope]),
        "kept_gt_boxes_equal_to_old_tracker_box": len(kept_gt),
        "kept_gt_best_same_class_raw_iou": q(kept_iou),
        "kept_gt_raw_iou_bins": bins(kept_iou),
    }


# ── F: prontidão do VLM ──────────────────────────────────────────────────────


def vlm_readiness(raw: list[Box], gt: list[Box], tau_link: float) -> dict:
    from PIL import Image

    p2g, _ = link(raw, gt, tau_link)
    scope6 = set(RECALL_CLASSES)
    confusable = scope6 | {"train", "airplane"}
    tasks = []
    for p in raw:
        acq, sec = p.key.split(":")
        truth = geometric_verdict(p, p2g.get(p.uid))
        tasks.append((p, BoxTask(acq_id=acq, sec=int(sec), det_index=int(p.uid.rsplit(":", 1)[1]),
                                 det_id=p.uid, proposed_class=p.cls, conf=p.conf, box=p.b,
                                 truth=truth)))
    t0 = next(t for p, t in tasks if p.cls in scope6)
    img = Image.open(ROOT / "images" / t0.acq_id / f"{t0.sec:06d}.jpg").convert("RGB")
    scene, crop = draw_box(img, t0.box, t0.proposed_class), crop_around(img, t0.box)
    n6 = sum(p.cls in scope6 for p, _ in tasks)
    n8 = sum(p.cls in confusable for p, _ in tasks)
    return {
        "box_task_constructible": True,
        "example_key": t0.key, "example_proposal_id": t0.det_id,
        "example_render_sizes": {"scene": list(scene.size), "crop": list(crop.size)},
        "example_prompt_v2_sha256": hashlib.sha256(phase1_prompt_v2(t0).encode()).hexdigest(),
        "tasks_all_classes": len(tasks), "tasks_6_gt_classes": n6,
        "tasks_6_gt_classes_plus_train_airplane": n8,
        "estimated_minutes_v2_at_4.88s": {"6_classes": round(n6 * V2_SECONDS_PER_CALL / 60, 1),
                                          "8_classes": round(n8 * V2_SECONDS_PER_CALL / 60, 1)},
        "geometric_truth_distribution_8_classes": dict(Counter(
            t.truth for p, t in tasks if p.cls in confusable)),
    }


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    frames = load_frames()
    gt = load_gt(frames)
    old = load_old(frames)
    tail_rows = json.loads((HERE / "raw_predictions_tail.json").read_text(encoding="utf-8"))
    raw_tail = load_raw(HERE / "raw_predictions_tail.json", 0.0)
    raw = [r for r in raw_tail if r.conf >= PROPOSAL_CONF]
    raw20 = [r for r in raw_tail if r.conf >= SENS_CONF]

    # consistência com o COCO do gabarito
    coco = json.loads(GT_JSON.read_text(encoding="utf-8"))
    assert len(gt) == len(coco["annotations"]) == 731, (len(gt), len(coco["annotations"]))
    assert len(frames) == 313 and len(old) == 505

    out = {"_meta": {"ground_truth": 731, "frames": 313, "old_proposals": 505,
                     "old_scored_501": sum(o.extra[3] for o in old),
                     "raw_tail": len(raw_tail), "raw_conf025": len(raw), "raw_conf020": len(raw20),
                     "correct_iou_c2": CORRECT_IOU, "link_ious": LINK_IOUS}}
    out["C0_crosscheck_vs_16aug_run"] = crosscheck_old_infer(raw_tail, gt, frames)
    out["C1_standard"] = {"conf025": c1(raw, gt, frames), "conf020": c1(raw20, gt, frames)}
    out["C1_coco_ap_tail"] = coco_ap(HERE / "predictions_coco_tail.json")
    val = validate_c2_on_old(old, gt)
    out["C2_validation_on_old_501"] = val
    out["C2_raw"] = {f"conf{c}": {str(t): c2_table(rr, gt, t) for t in LINK_IOUS}
                     for c, rr in (("025", raw), ("020", raw20))}
    # τ de vínculo primário = o que melhor reproduz o vínculo real nas antigas
    best = max(LINK_IOUS, key=lambda t: (val["geometric_rule"][str(t)]["links_recovered_exactly"]
                                          - val["geometric_rule"][str(t)]["linked_but_human_rejected"]))
    out["_meta"]["primary_link_iou"] = best
    # Escopo de proposta em C2. As antigas não tinham semáforo/placa; o protocolo
    # de anotação nunca considerou essas classes, então chamá-las de "rejeitadas"
    # seria atribuir ao humano uma decisão que ele não tomou.
    scope6 = {"car", "truck", "bus", "motorcycle", "person", "bicycle"}
    scope8 = scope6 | {"train", "airplane"}
    out["C2_raw_scopes_primary_link"] = {
        name: c2_table([r for r in rr if r.cls in sc], gt, best)
        for name, rr, sc in (("conf025_8cls", raw, scope8), ("conf025_6cls", raw, scope6),
                             ("conf020_8cls", raw20, scope8))}
    out["C2_old501_classes"] = dict(Counter(o.cls for o in old if o.extra[3]).most_common())
    out["C3_added_gt_recovered_by_raw"] = added_gt_recovery(raw, gt)
    out["D_speed_vs_dconf"] = speed_vs_dconf(old, raw, frames)
    out["D_raw_vs_old_imgsz640"] = audit_raw_vs_old(old, raw_tail, "imgsz 640 (principal)")
    d1280 = HERE / "diagnostic_imgsz1280" / "raw_predictions_tail.json"
    if d1280.exists():
        out["D_raw_vs_old_imgsz1280_diagnostic"] = audit_raw_vs_old(
            old, load_raw(d1280, 0.0), "imgsz 1280 (diagnostico)")
    out["E_impact_p2"] = impact_p2(old, raw, gt, best)
    out["F_vlm_readiness"] = vlm_readiness(raw, gt, best)

    (HERE / "analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    # Hashes finais de tudo que esta pasta produziu, por caminho relativo.
    sums = []
    for p in sorted(HERE.rglob("*")):
        if p.is_file() and p.suffix in {".json", ".csv", ".py", ".md"} and p.name != "SHA256SUMS.txt":
            sums.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(HERE).as_posix()}\n")
    (HERE / "SHA256SUMS.txt").write_text("".join(sums), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("C0_crosscheck_vs_16aug_run", "C2_raw_scopes_primary_link",
                                          "C2_old501_classes", "C3_added_gt_recovered_by_raw",
                                          "D_speed_vs_dconf")}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
