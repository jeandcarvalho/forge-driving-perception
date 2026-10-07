"""Telemetria → parquet.

Uma linha por frame, inclusive os não revisados: telemetria não depende de
curadoria, e é ela que alimenta a análise de cobertura (período, condição,
velocidade) por trás dos braços do Experimento 1.

`brake` é categórico (BrakeInfoStatus), fica como string. `semseg` é um dicionário
de proporções por classe — vira colunas `semseg_<classe>` para o parquet ficar
consultável sem desempacotar JSON.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from forge.api.models import Frame
from forge.data.layout import DatasetLayout


def write_telemetry(layout: DatasetLayout, frames: list[Frame]) -> tuple[Path, int]:
    semseg_keys: set[str] = set()
    for f in frames:
        if f.telemetry and f.telemetry.semseg:
            semseg_keys.update(f.telemetry.semseg.keys())
    semseg_cols = sorted(semseg_keys)

    cols: dict[str, list] = {
        "acq_id": [], "sec": [], "split": [], "reviewed": [],
        "gt_instances": [], "stored_detections": [],
        "lat": [], "lon": [], "alt": [],
        "speed": [], "steering": [], "brake": [],
        "vehicle": [], "period": [], "condition": [],
        "drive_id": [],
        **{f"semseg_{k}": [] for k in semseg_cols},
    }

    for f in frames:
        t = f.telemetry
        gps = t.gps if t else None
        cols["acq_id"].append(f.acq_id)
        cols["sec"].append(f.sec)
        cols["split"].append(f.split)
        cols["reviewed"].append(f.reviewed)
        # None (não []) quando não revisado — o parquet preserva a distinção.
        cols["gt_instances"].append(len(f.ground_truth) if f.ground_truth is not None else None)
        cols["stored_detections"].append(len(f.stored_detections))
        cols["lat"].append(gps.lat if gps else None)
        cols["lon"].append(gps.lon if gps else None)
        cols["alt"].append(gps.alt if gps else None)
        cols["speed"].append(t.speed if t else None)
        cols["steering"].append(t.steering if t else None)
        cols["brake"].append(str(t.brake) if t and t.brake is not None else None)
        cols["vehicle"].append(t.vehicle if t else None)
        cols["period"].append(t.period if t else None)
        cols["condition"].append(t.condition if t else None)
        cols["drive_id"].append(f.image.drive_id if f.image else None)
        semseg = (t.semseg if t else None) or {}
        for k in semseg_cols:
            cols[f"semseg_{k}"].append(semseg.get(k))

    schema = pa.schema(
        [
            ("acq_id", pa.string()), ("sec", pa.int32()), ("split", pa.string()),
            ("reviewed", pa.bool_()), ("gt_instances", pa.int32()),
            ("stored_detections", pa.int32()),
            ("lat", pa.float64()), ("lon", pa.float64()), ("alt", pa.float64()),
            ("speed", pa.float64()), ("steering", pa.float64()), ("brake", pa.string()),
            ("vehicle", pa.string()), ("period", pa.string()), ("condition", pa.string()),
            ("drive_id", pa.string()),
        ]
        + [(f"semseg_{k}", pa.float64()) for k in semseg_cols]
    )

    table = pa.Table.from_pydict(cols, schema=schema)
    path = layout.telemetry_path
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd")
    return path, table.num_rows
