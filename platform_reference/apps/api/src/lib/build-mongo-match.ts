// Constrói o $match do MongoDB a partir dos filtros de busca.
// Extraído pra ser usado por SearchBigService e SearchAcqIdsService
// sem duplicação de código.

import {
  type SearchQuery,
  splitList,
  parseRangeToken,
  VEHICLE_NORMALIZATION,
} from "@carcara/shared";

export function buildMongoMatch(q: SearchQuery): Record<string, any> {
  const and: any[] = [];

  // ---- blocks: vehicle ----
  const bVehiclesRaw = splitList(q["b.vehicle"]);
  if (bVehiclesRaw.length) {
    const bVehicles = bVehiclesRaw.map((v) => {
      const key = v.toLowerCase();
      return VEHICLE_NORMALIZATION[key] ?? v;
    });
    and.push({ "block.vehicle": { $in: bVehicles } });
  }

  // ---- blocks: period ----
  const bPeriods = splitList(q["b.period"]);
  if (bPeriods.length) {
    and.push({ "block.meteo.period": { $in: bPeriods } });
  }

  // ---- blocks: condition ----
  const bConditions = splitList(q["b.condition"]);
  if (bConditions.length) {
    and.push({ "block.meteo.condition": { $in: bConditions } });
  }

  // ---- CAN: velocidade ----
  if (q["c.v"]) {
    const [mn, mx] = parseRangeToken(q["c.v"]);
    const cond: any = {};
    if (mn !== null) cond.$gte = mn;
    if (mx !== null) cond.$lte = mx;
    if (Object.keys(cond).length) and.push({ "can.VehicleSpeed": cond });
  }

  // ---- CAN: SWA (vários ranges em OR) ----
  if (q["c.swa"]) {
    const parts = splitList(q["c.swa"]);
    const ors: any[] = [];
    for (const p of parts) {
      const [mn, mx] = parseRangeToken(p);
      const cond: any = {};
      if (mn !== null) cond.$gte = mn;
      if (mx !== null) cond.$lte = mx;
      if (Object.keys(cond).length) ors.push({ "can.SteeringWheelAngle": cond });
    }
    if (ors.length === 1) and.push(ors[0]);
    else if (ors.length > 1) and.push({ $or: ors });
  }

  // ---- CAN: freio ----
  const rawBrakes = splitList(q["c.brakes"]);
  if (rawBrakes.length) {
    const brakes = rawBrakes.map((v) => {
      const m = v.match(/^b'(.*)'$/);
      return m ? m[1] : v;
    });
    and.push({ "can.BrakeInfoStatus": { $in: brakes } });
  }

  // ---- SemSeg: building ----
  if (q["s.building"]) {
    const parts = splitList(q["s.building"]);
    const ors: any[] = [];
    for (const p of parts) {
      const [mn, mx] = parseRangeToken(p);
      const cond: any = {};
      if (mn !== null) cond.$gte = mn;
      if (mx !== null) cond.$lte = mx;
      if (Object.keys(cond).length) ors.push({ "semseg.building": cond });
    }
    if (ors.length === 1) and.push(ors[0]);
    else if (ors.length > 1) and.push({ $or: ors });
  }

  // ---- SemSeg: vegetation ----
  if (q["s.vegetation"]) {
    const parts = splitList(q["s.vegetation"]);
    const ors: any[] = [];
    for (const p of parts) {
      const [mn, mx] = parseRangeToken(p);
      const cond: any = {};
      if (mn !== null) cond.$gte = mn;
      if (mx !== null) cond.$lte = mx;
      if (Object.keys(cond).length) ors.push({ "semseg.vegetation": cond });
    }
    if (ors.length === 1) and.push(ors[0]);
    else if (ors.length > 1) and.push({ $or: ors });
  }

  // ---- SemSeg: sidewalk (binary yes/no — left OR right > 2%) ----
  // DB investigation: bimodal at 0 (57% left, 50% right) + real cluster above 2%.
  // The 0–2% zone is noise/transition; threshold=2 gives clean 41% "has sidewalk".
  const sSidewalks = splitList(q["s.sidewalk"]);
  if (sSidewalks.length && !(sSidewalks.includes("yes") && sSidewalks.includes("no"))) {
    const THRESH = 2;
    if (sSidewalks.includes("yes")) {
      and.push({ $or: [
        { "semseg.sidewalk_left":  { $gt: THRESH } },
        { "semseg.sidewalk_right": { $gt: THRESH } },
      ]});
    } else if (sSidewalks.includes("no")) {
      and.push({
        "semseg.sidewalk_left":  { $lte: THRESH },
        "semseg.sidewalk_right": { $lte: THRESH },
      });
    }
  }

  // ---- YOLO: classes ----
  const yClasses = splitList(q["y.class"]);
  if (yClasses.length) {
    and.push({ yolo: { $elemMatch: { class: { $in: yClasses } } } });
  }

  // ---- YOLO: rel_to_ego ----
  const yRel = splitList(q["y.rel"]);
  if (yRel.length) {
    and.push({ yolo: { $elemMatch: { rel_to_ego: { $in: yRel } } } });
  }

  // ---- YOLO: conf ----
  if (q["y.conf"]) {
    const parts = splitList(q["y.conf"]);
    const ors: any[] = [];
    for (const p of parts) {
      const [mn, mx] = parseRangeToken(p);
      const cond: any = {};
      if (mn !== null) cond.$gte = mn;
      if (mx !== null) cond.$lte = mx;
      if (Object.keys(cond).length) ors.push({ conf: cond });
    }
    if (ors.length) {
      and.push({ yolo: { $elemMatch: { $or: ors } } });
    }
  }

  // ---- YOLO: dist_m ----
  if (q["y.dist_m"]) {
    const [mn, mx] = parseRangeToken(q["y.dist_m"]);
    const cond: any = {};
    if (mn !== null) cond.$gte = mn;
    if (mx !== null) cond.$lte = mx;
    if (Object.keys(cond).length) {
      and.push({ yolo: { $elemMatch: { dist_m: cond } } });
    }
  }

  // ---- YOLO: scene volume (total detection count per second) ----
  const yVolumes = splitList(q["y.volume"]);
  if (yVolumes.length) {
    const sizeExpr = { $size: { $ifNull: ["$yolo", []] } };
    const ors: any[] = [];
    if (yVolumes.includes("low"))  ors.push({ $expr: { $lte: [sizeExpr, 3] } });
    if (yVolumes.includes("mid"))  ors.push({ $expr: { $and: [{ $gte: [sizeExpr, 4] }, { $lte: [sizeExpr, 9] }] } });
    if (yVolumes.includes("high")) ors.push({ $expr: { $gte: [sizeExpr, 10] } });
    if (ors.length === 1) and.push(ors[0]);
    else if (ors.length > 1) and.push({ $or: ors });
  }

  // ---- Build final match ----
  if (!and.length) return {};
  if (and.length === 1) return and[0];
  return { $and: and };
}
