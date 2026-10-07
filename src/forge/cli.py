"""CLI do CarCará Forge."""

from __future__ import annotations

import contextlib
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from forge import __version__
from forge.api.client import ApiError, CarcaraClient
from forge.api.manifest import fetch_manifest, merge_frames
from forge.api.models import Frame, SampleResponse
from forge.api.sample import sample_frames, sample_to_frames
from forge.config import (
    EVAL_CLASS_SCOPE_DEFAULT,
    INFER_CONF,
    INFER_IMGSZ,
    INFER_NMS_IOU,
    LABEL_CONF,
    LABEL_IMGSZ,
    LABEL_NMS_IOU,
    PARITY_IOU,
    SPLITS,
    STUDENT_WEIGHTS,
    TEACHER_WEIGHTS,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_WORKERS,
    EvalClassScope,
    Settings,
)
from forge.data.download import download_frames, write_checksums
from forge.data.layout import DatasetLayout, frame_stem
from forge.data.verify import verify as run_verify
from forge.data.view import materialize_view
from forge.data.writers.coco import write_coco
from forge.data.writers.telemetry import write_telemetry
from forge.data.writers.yolo import (
    write_auto_split_list,
    write_data_yaml,
    write_split_list,
    write_yolo_labels,
)
from forge.evaluate.parity import (
    EXPECTED_TARGET_COLLECTION,
    curation_parity,
    gt_class_histogram,
)
from forge.runs.registry import Run

# O console do Windows costuma estar em cp1252, que não tem `≥`, `→` nem `≠`.
# Sem isto, um relatório inteiro morre em UnicodeEncodeError na hora de imprimir
# — e o `diff`, que existe para ser lido e colado, é o que mais usa esses sinais.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(Exception):
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Materializa o dataset de percepção da plataforma CarCará.",
)
console = Console()


def _settings(api_url: str | None) -> Settings:
    s = Settings.from_env()
    return Settings(api_url=api_url.rstrip("/"), data_root=s.data_root, collection_id=s.collection_id) if api_url else s


def _load_local_manifest(layout: DatasetLayout) -> list[Frame]:
    if not layout.manifest_path.is_file():
        console.print(f"[red]manifest não encontrado em {layout.manifest_path}[/]")
        console.print("Rode `forge pull` primeiro.")
        raise typer.Exit(1)
    doc = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
    return [Frame.model_validate(f) for f in doc["frames"]]


# ─────────────────────────────────────────────────────────────────────────────


@app.command()
def pull(
    collection_id: Annotated[str, typer.Argument(help="id da coleção")] = "",
    split: Annotated[str, typer.Option(help="all | test | train | discarded")] = "all",
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
    workers: Annotated[int, typer.Option(help="downloads simultâneos")] = 6,
    fast: Annotated[bool, typer.Option("--fast", help="CDN 1920x1080, porém JPEG recomprimido")] = False,
    force: Annotated[bool, typer.Option("--force", help="rebaixa imagens já em disco")] = False,
    no_images: Annotated[bool, typer.Option("--no-images", help="só manifest e rótulos")] = False,
) -> None:
    """Baixa manifest + imagens e materializa o dataset em disco."""
    s = _settings(api_url)
    cid = collection_id or s.collection_id
    data_root = out or s.data_root

    console.print(f"[bold]API[/] {s.api_url}   [bold]coleção[/] {cid}   [bold]split[/] {split}")

    with CarcaraClient(s.api_url) as client:
        try:
            frames, meta, raws = fetch_manifest(client, cid, split=split)
        except ApiError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from None

    layout = DatasetLayout.for_collection(data_root, meta.collection.name, cid)
    layout.ensure_dirs()
    console.print(f"[bold]destino[/] {layout.root}")

    for i, raw in enumerate(raws, start=1):
        layout.raw_page_path(split, i).write_text(
            json.dumps(raw, ensure_ascii=False), encoding="utf-8"
        )

    # ── manifest: pull parcial ACUMULA, não substitui ────────────────────────
    # `--split train` sobrescrevendo o manifest apagaria os 313 frames do
    # gabarito, e com eles o instances_test.json que o `eval` usa. O pull por
    # split é o caminho normal (ninguém baixa 4.364 imagens para treinar em
    # 1.406), então o destrutivo aqui seria o padrão, não a exceção.
    previous: list[Frame] = []
    if layout.manifest_path.is_file():
        prev = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
        previous = [Frame.model_validate(d) for d in prev.get("frames", [])]
    all_frames = merge_frames(frames, previous)
    kept = len(all_frames) - len(frames)
    splits_on_disk = sorted({f.split for f in all_frames})

    layout.manifest_path.write_text(
        json.dumps(
            {
                "collection": meta.collection.model_dump(),
                "split_requested": split,
                "splits_on_disk": splits_on_disk,
                "totals": meta.totals.model_dump(),
                "frames": [f.model_dump(by_alias=True) for f in all_frames],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    n_labeled = sum(1 for f in frames if f.is_labeled)
    console.print(
        f"manifest: {len(frames)} frames  ·  {n_labeled} com gabarito  ·  "
        f"{len(frames) - n_labeled} sem"
    )
    if kept:
        console.print(
            f"manifest acumulado: +{kept} frames de pulls anteriores "
            f"= {len(all_frames)} ({', '.join(splits_on_disk)})"
        )

    # ── imagens ──────────────────────────────────────────────────────────────
    dims: dict[str, tuple[int, int]] = {}
    if no_images:
        console.print("[yellow]--no-images: pulando download[/]")
    else:
        if fast:
            console.print(
                "[yellow]--fast: JPEG recomprimido pelo CDN. Não use para número de paper.[/]"
            )
        report = download_frames(layout, frames, workers=workers, fast=fast, force=force)
        dims = report.dims_by_key
        write_checksums(layout, report)

        counts = report.counts
        console.print(f"download: {counts}")
        for status, label in (
            ("bad_size", "dimensão diferente de 1920x1080"),
            ("failed", "falha de download"),
            ("no_image", "sem link de imagem"),
        ):
            bad = report.by_status(status)
            if bad:
                console.print(f"[red]{len(bad)} × {label}[/]")
                for o in bad[:5]:
                    console.print(f"   {o.key}: {o.error}")
                if len(bad) > 5:
                    console.print(f"   … e mais {len(bad) - 5}")

    # ── rótulos, listas, COCO, telemetria ────────────────────────────────────
    table = Table("split", "frames", "com rótulo", "vazios", "instâncias", "na lista")
    for sp in SPLITS:
        in_split = [f for f in all_frames if f.split == sp]
        if not in_split:
            continue
        res = write_yolo_labels(layout, all_frames, sp, only_downloaded=not no_images)
        # Só vira lista treinável se todo item dela tiver rótulo em disco. Split
        # sem gabarito nenhum sai como `_pool`: inventário do que dá para rotular
        # depois, e é dele que o `forge label` lê a lista de trabalho.
        trainable = res.labels_written > 0 and sp != "discarded"
        listed = res.listed_images if trainable else res.inventory_images
        path = write_split_list(layout, listed, sp, trainable=trainable)
        table.add_row(
            sp,
            str(len(in_split)),
            str(res.labels_written),
            str(res.empty_labels),
            str(res.instances),
            f"{len(listed)} → {path.name}",
        )
    console.print(table)

    # O COCO só é reescrito quando este pull trouxe frames de teste: as caixas em
    # px saem das dimensões MEDIDAS no download, e reescrever a partir de um pull
    # de treino cairia no declarado pelo manifest sem ganhar nada.
    if any(f.split == "test" for f in frames):
        coco = write_coco(layout, all_frames, "test", dims_by_key=dims)
        console.print(
            f"COCO: {coco.path.name} — {coco.images} imagens, {coco.annotations} instâncias, "
            f"{coco.empty_images} vazias"
        )
    else:
        console.print(f"COCO: preservado ({layout.coco_path('test').name}) — pull sem teste")

    tpath, nrows = write_telemetry(layout, all_frames)
    console.print(f"telemetria: {tpath.name} — {nrows} linhas")

    layout.provenance_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "forge_version": __version__,
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "api_url": s.api_url,
                "collection_id": cid,
                "split_requested": split,
                "splits_on_disk": splits_on_disk,
                "frames_pulled": len(frames),
                "frames": len(all_frames),
                "frames_labeled": n_labeled,
                "image_source": "cdn" if fast else "drive_original",
                "totals": meta.totals.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]pronto[/] → {layout.root}")


@app.command()
def verify(
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    collection_id: Annotated[str, typer.Argument()] = "",
    skip_hashes: Annotated[bool, typer.Option("--skip-hashes")] = False,
) -> None:
    """Revalida checksums, dimensões, geometria das caixas e listas de split."""
    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)
    frames = _load_local_manifest(layout)

    rep = run_verify(layout, frames, check_hashes=not skip_hashes)
    console.print(f"imagens conferidas: {rep.checked_images}   frames: {rep.checked_frames}")

    if rep.ok:
        console.print("[green]tudo íntegro[/]")
        return

    for kind, items in rep.failures.items():
        console.print(f"[red]{kind}: {len(items)}[/]")
        for item in items[:5]:
            console.print(f"   {item}")
        if len(items) > 5:
            console.print(f"   … e mais {len(items) - 5}")
    console.print(f"detalhes em {layout.verify_failures_path}")
    raise typer.Exit(1)


@app.command()
def audit(
    collection_id: Annotated[str, typer.Argument()] = "",
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
    local: Annotated[bool, typer.Option("--local", help="usa o manifest em disco")] = False,
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
) -> None:
    """Confere que a dobra do gabarito no Forge é idêntica à da API.

    Porte de curation/stats.ts. Contagem por veredito de curadoria — NÃO é
    avaliação de detector (isso é `forge eval`).
    """
    s = _settings(api_url)
    cid = collection_id or s.collection_id

    if local:
        layout = _resolve_layout(out or s.data_root, cid)
        frames = _load_local_manifest(layout)
    else:
        with CarcaraClient(s.api_url) as client:
            frames, _, _ = fetch_manifest(client, cid, split="test")

    rep = curation_parity(frames)
    expected = EXPECTED_TARGET_COLLECTION if cid == Settings.from_env().collection_id else None

    table = Table("campo", "obtido", "esperado", "")
    got = rep.as_dict()
    fields = list(EXPECTED_TARGET_COLLECTION) if expected else [
        "reviewed_frames", "detections", "ground_truth", "tp", "fp", "fn",
    ]
    for k in fields:
        exp = expected.get(k) if expected else None
        mark = "" if exp is None else ("[green]ok[/]" if got[k] == exp else "[red]DIVERGE[/]")
        table.add_row(k, str(got[k]), "—" if exp is None else str(exp), mark)
    console.print(table)
    console.print(f"precision {rep.precision}  recall {rep.recall}  f1 {rep.f1}")

    if rep.count_mismatches:
        console.print(f"[red]{len(rep.count_mismatches)} divergência(s) vs curation_counts[/]")
        for m in rep.count_mismatches[:5]:
            console.print(f"   {m}")

    hist = gt_class_histogram(frames)
    console.print("gabarito: " + " · ".join(f"{k} {v}" for k, v in hist.items()))
    console.print(f"total de instâncias: {sum(hist.values())}")

    diff = rep.diff_against(expected) if expected else {}
    if diff or rep.count_mismatches:
        console.print("[red]PARIDADE FALHOU — a dobra do gabarito diverge da API.[/]")
        raise typer.Exit(1)
    console.print("[green]paridade exata com a API[/]")


@app.command()
def sample(
    mode: Annotated[str, typer.Option(help="random | query")] = "random",
    n: Annotated[int, typer.Option("-n")] = 1000,
    seed: Annotated[int, typer.Option(help="publique este número no paper")] = 42,
    dedup_m: Annotated[float | None, typer.Option("--dedup-m")] = None,
    exclude_mode: Annotated[
        str,
        typer.Option(
            "--exclude-mode",
            help="acqs (antivazamento) | acqs_all (braço de controle) | frames | none",
        ),
    ] = "acqs",
    spread: Annotated[
        bool,
        typer.Option("--spread", help="rodízio entre gravações (diversidade de aquisição)"),
    ] = False,
    filters: Annotated[
        list[str] | None,
        typer.Option("--filter", "-f", help="filtro de busca, ex.: -f y.class=truck (repetível)"),
    ] = None,
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
    save: Annotated[Path | None, typer.Option(help="grava o JSON da resposta")] = None,
) -> None:
    """Amostra frames para um braço do Experimento 1 (sempre com antivazamento).

    `--exclude-mode acqs` (padrão) tira do pool as gravações que doaram frame ao
    gabarito: é a garantia de antivazamento, e responde "posso treinar com isto?".

    `--exclude-mode acqs_all` tira TODAS as gravações da coleção, doadoras ou não.
    É o braço de controle da eficiência de curadoria, e responde outra pergunta:
    "o que a curadoria acrescenta sobre o acervo?". Com `acqs`, 31% do braço sai
    de gravação curada e o contraste sai encolhido — ver o comentário longo em
    `apps/api/src/controllers/dataset-sample.ts`.
    """
    parsed: dict[str, str] = {}
    for item in filters or []:
        key, sep, value = item.partition("=")
        if not sep:
            console.print(f"[red]--filter {item!r}: esperado chave=valor (ex.: y.class=truck)[/]")
            raise typer.Exit(1)
        parsed[key.strip()] = value.strip()

    # `mode=query` sem filtro nenhum é o braço aleatório com outro nome — e os
    # dois apareceriam no paper como se fossem tratamentos diferentes.
    if mode == "query" and not parsed:
        console.print("[red]mode=query sem --filter é o mesmo que mode=random.[/]")
        raise typer.Exit(1)

    s = _settings(api_url)
    with CarcaraClient(s.api_url) as client:
        resp = sample_frames(
            client,
            mode=mode,  # type: ignore[arg-type]
            n=n,
            seed=seed,
            exclude_collection=s.collection_id,
            dedup_m=dedup_m,
            exclude_mode=exclude_mode,
            spread=spread,
            filters=parsed or None,
        )

    console.print(
        f"pool: {resp.pool_matched} casaram → {resp.pool_after_exclusion} após exclusão "
        f"→ {resp.pool_after_dedup} após dedup"
    )
    console.print(f"devolvidos: {resp.n_returned} de {resp.n_requested} pedidos (seed {resp.seed})")
    if resp.acqs_sampled:
        # A estatística que decidiu o Experimento 1: o gabarito tem 2,5 frames
        # por gravação, e a amostra que mais se parecer com ele leva vantagem.
        console.print(
            f"diversidade: {resp.acqs_sampled} gravações de {resp.acqs_in_pool} no pool  ·  "
            f"{resp.frames_per_acq_mean} frames por gravação (maior: {resp.frames_per_acq_max})"
        )
    if resp.n_returned < resp.n_requested:
        console.print(
            f"[yellow]o pool não comporta N={resp.n_requested}. "
            "Os três braços precisam do MESMO N — use este teto nos três.[/]"
        )
    if save:
        save.parent.mkdir(parents=True, exist_ok=True)
        # Os filtros não voltam na resposta, e sem eles o arquivo não identifica o
        # braço: `mode=query` com `y.class=truck` e com `b.period=night` são
        # tratamentos diferentes e produziriam JSONs indistinguíveis.
        doc = {**json.loads(resp.model_dump_json()), "filters": parsed}
        save.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"registro do braço → {save}")


@app.command()
def infer(
    weights: Annotated[str, typer.Argument(help="pesos, ex.: yolo11n.pt")] = "yolo11n.pt",
    split: Annotated[str, typer.Option(help="split a inferir")] = "test",
    conf: Annotated[float, typer.Option(help="baixo de propósito: sem cauda não há mAP")] = INFER_CONF,
    iou: Annotated[float, typer.Option(help="IoU do NMS")] = INFER_NMS_IOU,
    imgsz: Annotated[int, typer.Option()] = INFER_IMGSZ,
    device: Annotated[str | None, typer.Option(help="0 para GPU, cpu para CPU")] = None,
    batch: Annotated[int, typer.Option()] = 16,
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    collection_id: Annotated[str, typer.Argument()] = "",
) -> None:
    """Roda o detector localmente sobre os JPEGs e grava predições em COCO results."""
    from forge.detect.infer import run_inference

    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)
    list_path = layout.split_list_path(split, trainable=True)
    if not list_path.is_file():
        console.print(f"[red]{list_path} não existe — rode `forge pull` antes.[/]")
        raise typer.Exit(1)

    images = [Path(p) for p in list_path.read_text(encoding="utf-8").split() if p.strip()]
    console.print(f"{len(images)} imagens de {list_path.name}")

    run = Run.create(
        layout.root / "runs",
        "infer",
        params={
            "weights": str(weights), "split": split, "conf": conf, "iou": iou,
            "imgsz": imgsz, "device": device, "batch": batch, "images": len(images),
        },
        weights=Path(weights),
    )
    res = run_inference(
        weights, images, run.path("predictions.json"),
        conf=conf, iou=iou, imgsz=imgsz, device=device, batch=batch,
    )
    run.results = {
        "images": res.images,
        "detections": res.detections,
        "classes_seen": res.classes_seen,
    }
    run.save()

    console.print(f"{res.detections} detecções em {res.images} imagens")
    top = list(res.classes_seen.items())[:8]
    console.print("classes: " + " · ".join(f"{k} {v}" for k, v in top))
    console.print(f"[green]{run.path('predictions.json')}[/]")


@app.command()
def label(
    teacher: Annotated[str, typer.Argument(help="pesos do professor")] = TEACHER_WEIGHTS,
    split: Annotated[str, typer.Option(help="pool local a rotular")] = "train",
    sample_json: Annotated[
        Path | None,
        typer.Option("--sample", help="braço salvo pelo `forge sample` (baixa o que faltar)"),
    ] = None,
    conf: Annotated[
        float, typer.Option(help="alto de propósito: aqui a cauda vira rótulo errado")
    ] = LABEL_CONF,
    iou: Annotated[float, typer.Option(help="IoU do NMS")] = LABEL_NMS_IOU,
    imgsz: Annotated[int, typer.Option()] = LABEL_IMGSZ,
    device: Annotated[str | None, typer.Option(help="0 para GPU, cpu para CPU")] = None,
    batch: Annotated[int, typer.Option()] = 16,
    tag: Annotated[str | None, typer.Option(help="padrão: <professor>_c<corte>")] = None,
    drop_empty: Annotated[
        bool, typer.Option("--drop-empty", help="ablação: descarta frame sem detecção")
    ] = False,
    limit: Annotated[int, typer.Option(help="rotula só os N primeiros (ensaio)")] = 0,
    workers: Annotated[int, typer.Option(help="downloads simultâneos")] = 6,
    no_download: Annotated[bool, typer.Option("--no-download")] = False,
    download_only: Annotated[
        bool,
        typer.Option(
            "--download-only",
            help="baixa as imagens do braço e para; a rotulagem fica para depois",
        ),
    ] = False,
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    collection_id: Annotated[str, typer.Option("--collection-id")] = "",
) -> None:
    """Rotula frames sem gabarito com o detector-professor (insumo dos experimentos).

    O professor é ferramenta de método, não resultado — ele não aparece em tabela
    nenhuma. A saída é um diretório de rótulos e uma lista treinável, ambos
    nomeados pelo `tag` (professor + corte), para que as duas pernas da ablação
    0.25 × 0.5 coexistam em disco.
    """
    from forge.detect.label import default_tag, generate_labels

    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)
    base = default_tag(teacher, conf)
    source_split: str | None = None

    if sample_json is not None:
        resp = SampleResponse.model_validate_json(
            Path(sample_json).read_text(encoding="utf-8")
        )
        console.print(
            f"braço [bold]{Path(sample_json).name}[/]: {resp.n_returned} frames "
            f"(mode={resp.mode} seed={resp.seed} dedup_m={resp.dedup_m})"
        )
        # Sem exclusão de aquisição, os frames do braço podem vir das mesmas
        # aquisições do gabarito — e aí o treino contamina a avaliação. É erro
        # de amostragem, não dá para consertar aqui.
        if not resp.exclude_collection:
            console.print(
                "[red]este braço foi amostrado sem `exclude_collection` — "
                "treinar nele contamina a avaliação. Refaça o `forge sample`.[/]"
            )
            raise typer.Exit(1)

        frames = sample_to_frames(resp)
        if not no_download:
            report = download_frames(layout, frames, workers=workers)
            console.print(f"download: {report.counts}")
            for o in report.by_status("failed")[:5]:
                console.print(f"   [red]{o.key}: {o.error}[/]")
        images = [
            p
            for p in (layout.image_path(f.acq_id, f.sec) for f in frames)
            if p.is_file()
        ]
        tag = tag or f"{Path(sample_json).stem}_{base}"
    else:
        if split == "discarded":
            console.print(
                "[red]`discarded` nunca treina — são frames de aquisições que "
                "doaram frame ao teste. Rotulá-los não desfaz o vazamento.[/]"
            )
            raise typer.Exit(1)
        pool = layout.split_list_path(split, trainable=False)
        if not pool.is_file():
            pool = layout.split_list_path(split, trainable=True)
        if not pool.is_file():
            console.print(f"[red]{pool} não existe — rode `forge pull --split {split}`.[/]")
            raise typer.Exit(1)
        images = [Path(p) for p in pool.read_text(encoding="utf-8").split() if p.strip()]
        console.print(f"{len(images)} imagens de {pool.name}")
        source_split = split
        tag = tag or (base if split == "train" else f"{split}_{base}")

    if download_only:
        # O download é I/O e a rotulagem é GPU; separar os dois deixa baixar um
        # braço enquanto outro treina.
        on_disk = [p for p in images if p.is_file()]
        console.print(f"[green]{len(on_disk)} imagens em disco[/] (de {len(images)})")
        console.print("rotule depois com o mesmo comando, sem --download-only")
        return

    missing = [p for p in images if not p.is_file()]
    if missing:
        console.print(f"[yellow]{len(missing)} imagens da lista não estão em disco — ignoradas[/]")
        images = [p for p in images if p.is_file()]
    if not images:
        console.print("[red]nenhuma imagem para rotular.[/]")
        raise typer.Exit(1)
    if limit:
        images = images[:limit]
        console.print(f"[yellow]--limit {limit}: ensaio, não use o resultado no paper[/]")

    console.print(
        f"professor [bold]{teacher}[/]  corte [bold]{conf}[/]  tag [bold]{tag}[/]  "
        f"{len(images)} imagens"
    )

    run = Run.create(
        layout.root / "runs",
        "label",
        params={
            "teacher": str(teacher), "conf": conf, "iou": iou, "imgsz": imgsz,
            "device": device, "batch": batch, "tag": tag, "keep_empty": not drop_empty,
            "source": str(sample_json) if sample_json else f"split:{split}",
            "images": len(images),
        },
        weights=Path(teacher),
    )

    res = generate_labels(
        teacher, images, layout.auto_labels_dir(tag),
        conf=conf, iou=iou, imgsz=imgsz, device=device, batch=batch,
        keep_empty=not drop_empty,
    )
    list_path = write_auto_split_list(layout, res.listed_images, tag, source_split=source_split)

    console.print(
        f"{res.instances} instâncias em {res.labels_written} rótulos "
        f"({res.instances_per_image:.2f} por imagem)"
    )
    # O número que descreve o conjunto: quantos frames o professor achou vazios.
    # No acervo cru isso costuma passar de metade — é o desperdício que a
    # plataforma existe para evitar, e ele precisa aparecer no relatório.
    share = res.empty_labels / res.images if res.images else 0.0
    console.print(
        f"sem detecção acima do corte: {res.empty_labels} de {res.images} ({share:.1%}) — "
        + ("mantidos como background" if not drop_empty else "[yellow]descartados (--drop-empty)[/]")
    )
    if res.classes_seen:
        top = list(res.classes_seen.items())[:8]
        console.print("classes: " + " · ".join(f"{k} {v}" for k, v in top))
    if res.dropped_boxes:
        console.print(f"[yellow]{len(res.dropped_boxes)} caixas descartadas[/]")
        for d in res.dropped_boxes[:5]:
            console.print(f"   {d}")

    run.results = {
        "labels_written": res.labels_written,
        "empty_labels": res.empty_labels,
        "empty_share": round(share, 4),
        "instances": res.instances,
        "classes_seen": res.classes_seen,
        "dropped_boxes": len(res.dropped_boxes),
        "list": str(list_path),
    }
    run.save()

    console.print(f"[green]{list_path}[/]  ·  rótulos em {res.labels_dir}")
    console.print(f"treine com: forge train --tag {tag}")


@app.command()
def train(
    tag: Annotated[str, typer.Option(help="tag da rodada de `forge label`")],
    weights: Annotated[str, typer.Option(help="pesos iniciais do aluno")] = STUDENT_WEIGHTS,
    epochs: Annotated[int, typer.Option()] = TRAIN_EPOCHS,
    imgsz: Annotated[int, typer.Option()] = TRAIN_IMGSZ,
    batch: Annotated[int, typer.Option()] = TRAIN_BATCH,
    seed: Annotated[int, typer.Option(help="publique no paper; 3 sementes por braço")] = 0,
    device: Annotated[str | None, typer.Option(help="0 para GPU, cpu para CPU")] = None,
    val_split: Annotated[str, typer.Option(help="split do gabarito humano")] = "test",
    workers: Annotated[
        int, typer.Option(help="dataloader; no Windows cada um é um processo com torch")
    ] = TRAIN_WORKERS,
    patience: Annotated[int, typer.Option(help="0 = sem early stopping")] = 0,
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    collection_id: Annotated[str, typer.Option("--collection-id")] = "",
) -> None:
    """Treina o aluno (YOLO11n) nos rótulos de um `tag`, validando no gabarito humano.

    Treino e validação rodam sobre uma view por hardlink, montada de um mapa
    explícito imagem → rótulo: o layout do acervo não é o pareamento por caminho
    que o ultralytics faz sozinho, e um pareamento errado não estoura — treina em
    silêncio com o split inteiro contado como background.
    """
    from forge.detect.train import train_detector

    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)

    train_list = layout.auto_split_list_path(tag)
    if not train_list.is_file():
        console.print(f"[red]{train_list} não existe — rode `forge label` com esse tag.[/]")
        raise typer.Exit(1)
    val_list = layout.split_list_path(val_split, trainable=True)
    if not val_list.is_file():
        console.print(
            f"[red]{val_list} não existe. A validação é o GABARITO HUMANO — "
            f"rode `forge pull --split {val_split}`.[/]"
        )
        raise typer.Exit(1)

    train_imgs = [Path(p) for p in train_list.read_text(encoding="utf-8").split() if p.strip()]
    val_imgs = [Path(p) for p in val_list.read_text(encoding="utf-8").split() if p.strip()]

    view_root = layout.view_dir(tag)
    tv = materialize_view(
        view_root, "train", train_imgs,
        {p: layout.auto_labels_dir(tag) / f"{frame_stem(p)}.txt" for p in train_imgs},
    )
    vv = materialize_view(
        view_root, "val", val_imgs,
        {p: layout.label_path(val_split, p.parent.name, int(p.stem)) for p in val_imgs},
    )
    for name, v in (("train", tv), ("val", vv)):
        console.print(f"view {name}: {v.linked} linkadas, {v.reused} reaproveitadas → {v.root}")
        if v.skipped_missing_label:
            # Imagem sem rótulo não pode entrar: no ultralytics ela vira
            # background silencioso, e o treino aprende "aqui não há objeto".
            console.print(
                f"[yellow]{len(v.skipped_missing_label)} imagens fora da view "
                "por falta de rótulo[/]"
            )
            for p in v.skipped_missing_label[:5]:
                console.print(f"   {p}")

    if tv.linked + tv.reused == 0:
        console.print("[red]view de treino vazia — nada para treinar.[/]")
        raise typer.Exit(1)

    data_yaml = write_data_yaml(
        layout,
        train_list=tv.images_dir,
        val_list=vv.images_dir,
        path=view_root / "data.yaml",
    )
    console.print(f"[bold]data.yaml[/] {data_yaml}")
    console.print(
        f"aluno [bold]{weights}[/] · {epochs} épocas · imgsz {imgsz} · batch {batch} · "
        f"seed [bold]{seed}[/]"
    )

    run = Run.create(
        layout.root / "runs",
        "train",
        params={
            "tag": tag, "weights": str(weights), "epochs": epochs, "imgsz": imgsz,
            "batch": batch, "seed": seed, "device": device, "workers": workers,
            "patience": patience, "val_split": val_split,
            "data_yaml": str(data_yaml),
            "train_images": tv.linked + tv.reused,
            "val_images": vv.linked + vv.reused,
        },
        weights=Path(weights),
    )

    res = train_detector(
        data_yaml=data_yaml, weights=weights, epochs=epochs, imgsz=imgsz, batch=batch,
        seed=seed, device=device, project=run.root, workers=workers, patience=patience,
    )

    table = Table("métrica", "valor")
    for k, v in res.metrics.items():
        if k in ("mAP50_95", "mAP50", "precision", "recall"):
            table.add_row(k, f"{v:.4f}")
    console.print(table)

    run.results = {
        "metrics": res.metrics,
        "epochs_ran": res.epochs_ran,
        "weights_best": str(res.weights_best) if res.weights_best else None,
        "weights_last": str(res.weights_last) if res.weights_last else None,
        "output_dir": str(res.output_dir),
    }
    run.save()

    if res.weights_best is None:
        console.print("[red]treino terminou sem best.pt — veja o log do ultralytics.[/]")
        raise typer.Exit(1)

    console.print(f"[green]{res.weights_best}[/]")
    # O `val` do ultralytics roda no espaço de 80 classes e sem o mascaramento
    # do `EvalClassScope` — o número do paper é o do `forge eval`, não o de cima.
    console.print(
        f"para o número do paper: forge infer {res.weights_best} && forge eval <predictions.json>"
    )


@app.command()
def eval(  # o nome do subcomando é `eval` de propósito
    predictions: Annotated[Path, typer.Argument(help="COCO results do `forge infer`")],
    iou: Annotated[float, typer.Option(help="IoU do matcher (decomposição de erro)")] = 0.5,
    decomp_conf: Annotated[
        float,
        typer.Option(
            "--decomp-conf",
            help="ponto de operação da decomposição de erro (o mAP usa a curva inteira)",
        ),
    ] = 0.25,
    scope: Annotated[str, typer.Option(help="gt_present | all_coco")] = EVAL_CLASS_SCOPE_DEFAULT.value,
    split: Annotated[str, typer.Option()] = "test",
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    collection_id: Annotated[str, typer.Argument()] = "",
) -> None:
    """mAP via pycocotools + decomposição de erro estilo TIDE.

    Avaliação de detector por matching GEOMÉTRICO. Para conferir a dobra do
    gabarito contra a plataforma use `forge audit`; para ver onde os dois
    algoritmos discordam, `forge diff`.
    """
    from forge.evaluate.coco_map import evaluate_map, predictions_to_matcher_input
    from forge.evaluate.matcher import match

    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)
    gt_json = layout.coco_path(split)
    if not gt_json.is_file():
        console.print(f"[red]{gt_json} não existe — rode `forge pull` antes.[/]")
        raise typer.Exit(1)

    class_scope = EvalClassScope(scope)
    console.print(
        f"escopo de classe: [bold]{class_scope.value}[/] "
        + (
            "(detecção de classe ausente do gabarito é descartada, não vira FP)"
            if class_scope is EvalClassScope.GT_PRESENT
            else "(classe ausente do gabarito vira FP de fundo)"
        )
    )

    m = evaluate_map(gt_json, predictions, class_scope=class_scope)
    table = Table("métrica", "valor")
    for k, v in m.as_dict().items():
        if k.startswith("mAP"):
            table.add_row(k, f"{v:.4f}")
    console.print(table)

    if m.per_class:
        console.print(
            "AP@50:95 por classe: "
            + " · ".join(f"{k} {v:.3f}" for k, v in m.per_class.items())
        )

    preds, gts = predictions_to_matcher_input(predictions, gt_json)

    # mAP e decomposição querem coisas opostas do mesmo conjunto de predições.
    # O mAP precisa da cauda de baixa confiança inteira: é ela que desenha a
    # curva precision-recall. A decomposição precisa de um PONTO DE OPERAÇÃO —
    # sem corte, ela só relata que a cauda é grande (dezenas de milhares de FP
    # de fundo com score 0.002), o que não diz nada sobre o detector.
    n_all = len(preds)
    preds = [p for p in preds if p.conf >= decomp_conf]
    res = match(preds, gts, iou_threshold=iou, class_scope=class_scope)
    console.print(
        f"\ndecomposição de erro @ IoU {iou}, conf ≥ {decomp_conf} "
        f"({len(preds)} de {n_all} predições; o mAP acima usou todas):"
    )
    dec = Table("contador", "valor")
    for k, v in res.counters.items():
        dec.add_row(k, str(v))
    console.print(dec)
    if res.masked_predictions:
        top = sorted(res.masked_classes.items(), key=lambda kv: -kv[1])[:8]
        console.print(
            f"[yellow]{res.masked_predictions} predições mascaradas[/] "
            + " · ".join(f"{k} {v}" for k, v in top)
        )

    run = Run.create(
        layout.root / "runs",
        "eval",
        params={
            "predictions": str(Path(predictions).resolve()), "gt": str(gt_json),
            "iou": iou, "class_scope": class_scope.value, "split": split,
            "decomp_conf": decomp_conf,
        },
    )
    run.results = {"map": m.as_dict(), "decomposition": res.counters}
    run.save()
    console.print(f"[green]{run.path('meta.json')}[/]")


@app.command()
def candidates(
    predictions: Annotated[Path, typer.Argument(help="COCO results do `forge infer`")],
    min_conf: Annotated[
        float, typer.Option("--min-conf", help="só detecção que o detector afirma com força")
    ] = 0.5,
    iou: Annotated[float, typer.Option(help="IoU do matcher")] = 0.5,
    scope: Annotated[str, typer.Option(help="gt_present | all_coco")] = EVAL_CLASS_SCOPE_DEFAULT.value,
    split: Annotated[str, typer.Option()] = "test",
    limit: Annotated[int, typer.Option(help="quantos listar (0 = todos)")] = 20,
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    collection_id: Annotated[str, typer.Argument()] = "",
) -> None:
    """Fila da segunda passada de curadoria: objetos que o gabarito talvez não tenha.

    O gabarito nasceu editando a saída de um detector, então pode ter herdado o
    ponto cego dele. Uma inferência local é um observador diferente: o que ela vê
    com confiança alta e o gabarito não tem é candidato a objeto que ninguém
    procurou. Não é medida — é lista de trabalho, e quem decide é a curadoria.
    """
    from forge.evaluate.candidates import find_candidates
    from forge.evaluate.coco_map import predictions_to_matcher_input

    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)
    gt_json = layout.coco_path(split)
    if not gt_json.is_file():
        console.print(f"[red]{gt_json} não existe — rode `forge pull` antes.[/]")
        raise typer.Exit(1)

    preds, gts = predictions_to_matcher_input(predictions, gt_json)
    rep = find_candidates(
        preds, gts, min_conf=min_conf, iou_threshold=iou, class_scope=EvalClassScope(scope)
    )

    console.print(
        f"{rep.predictions_considered} predições com conf ≥ {min_conf} contra "
        f"{rep.ground_truth} instâncias de gabarito"
    )
    console.print(
        f"[bold]{len(rep.candidates)} candidatos[/] em {rep.frames_touched} frames: "
        + (" · ".join(f"{k} {v}" for k, v in rep.by_class.items()) or "nenhum")
    )
    if rep.out_of_scope_by_class:
        total = sum(rep.out_of_scope_by_class.values())
        top = list(rep.out_of_scope_by_class.items())[:8]
        # Classe que o gabarito nunca teve não é ponto cego, é escopo: ampliar o
        # contrato da anotação é decisão de projeto, não de frame.
        console.print(
            f"[yellow]{total} detecções de classe fora do gabarito[/] (escopo, não "
            "ponto cego): " + " · ".join(f"{k} {v}" for k, v in top)
        )

    if rep.candidates:
        console.print("\n[bold]fila[/]  (frame · classe · conf · maior IoU com o gabarito)")
        shown = rep.candidates if limit == 0 else rep.candidates[:limit]
        for c in shown:
            console.print(
                f"  {c.frame_key:<22} {c.cls_name:<12} {c.conf:.3f}  IoU {c.best_iou:.3f}"
            )
        if limit and len(rep.candidates) > limit:
            console.print(f"  … e mais {len(rep.candidates) - limit} (use --limit 0)")

    run = Run.create(
        layout.root / "runs",
        "candidates",
        params={
            "predictions": str(Path(predictions).resolve()), "gt": str(gt_json),
            "min_conf": min_conf, "iou": iou, "class_scope": scope, "split": split,
        },
    )
    path = rep.to_json(run.path("candidates.json"))
    run.results = {
        "candidates": len(rep.candidates),
        "frames": rep.frames_touched,
        "by_class": rep.by_class,
        "out_of_scope": sum(rep.out_of_scope_by_class.values()),
    }
    run.save()
    console.print(f"\n[green]{path}[/]")


@app.command()
def diff(
    iou: Annotated[float, typer.Option(help="IoU do matcher")] = PARITY_IOU,
    collection_id: Annotated[str, typer.Argument()] = "",
    api_url: Annotated[str | None, typer.Option("--api-url")] = None,
    local: Annotated[bool, typer.Option("--local", help="usa o manifest em disco")] = False,
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
    limit: Annotated[int, typer.Option(help="quantos casos listar (0 = todos)")] = 20,
) -> None:
    """Onde `audit` e `eval` discordam sobre as MESMAS detecções.

    Entregável, não log: a lista de casos é o material do parágrafo de método.
    O matcher roda em `all_coco` aqui — a contagem por veredito não mascara
    classe nenhuma, então mascarar de um lado só inventaria divergência.
    """
    from forge.evaluate.diff import audit_vs_eval

    s = _settings(api_url)
    cid = collection_id or s.collection_id
    layout = _resolve_layout(out or s.data_root, cid)

    if local:
        frames = _load_local_manifest(layout)
    else:
        with CarcaraClient(s.api_url) as client:
            frames, _, _ = fetch_manifest(client, cid, split="test")

    rep = audit_vs_eval(frames, iou_threshold=iou)

    comp = Table("contador", "audit (veredito)", "eval (geometria)", "")
    for k in ("detections", "ground_truth", "tp", "fp", "fn",
              "fp_localization", "fp_classification", "fp_background",
              "fn_from_errors", "fn_undetected"):
        a = rep.audit_counters.get(k)
        e = rep.eval_counters.get(k)
        mark = "" if a == e else "[yellow]≠[/]"
        comp.add_row(k, str(a), str(e), mark)
    console.print(comp)

    console.print(
        f"\n{rep.compared_detections} detecções julgadas pelos dois · "
        f"[bold]{len(rep.divergences)} divergem[/]"
    )
    if rep.by_transition:
        console.print("\npor transição (rótulo do audit → rótulo do eval):")
        for k, v in rep.by_transition.items():
            console.print(f"   {k:42} {v}")

    if rep.divergences:
        # Texto puro, não Rich table: a lista é o entregável e não pode sair
        # truncada na largura do terminal.
        short = {
            "tp": "tp", "fp_localization": "fp_loc", "fp_classification": "fp_cls",
            "fp_duplicate": "fp_dup", "fp_background": "fp_bg",
        }
        console.print("\n[bold]casos[/]  (frame · classe/conf · audit → eval · IoU · gabarito)")
        shown = rep.divergences if limit == 0 else rep.divergences[:limit]
        for d in shown:
            gt = (
                f"{d.matched_gt_class}[{d.matched_gt_origin}]"
                if d.matched_gt_class
                else "nada por perto"
            )
            console.print(
                f"  {d.frame_key:<22} {d.cls_name:<11} {d.conf:.3f}  "
                f"{short.get(d.audit_label, d.audit_label):>6} → "
                f"{short.get(d.eval_label, d.eval_label):<6}  "
                f"IoU {d.best_iou:.3f}  {gt}"
            )
        if limit and len(rep.divergences) > limit:
            console.print(f"  … e mais {len(rep.divergences) - limit} (use --limit 0)")

    if rep.added_gt_matched_geometrically:
        console.print(
            f"\n[yellow]{len(rep.added_gt_matched_geometrically)} instâncias `added` "
            "casadas geometricamente com uma pré-anotação[/] — a auditoria conta "
            "todas como FN não detectada, por construção."
        )

    run = Run.create(layout.root / "runs", "diff", params={"iou": iou, "source": "local" if local else "api"})
    rep.to_json(run.path("diff.json"))
    run.results = {
        "divergences": len(rep.divergences),
        "by_transition": rep.by_transition,
        "agree": rep.agree,
    }
    run.save()
    console.print(f"\n[green]{run.path('diff.json')}[/]")


@app.command()
def version() -> None:
    """Versão do Forge."""
    console.print(f"carcara-forge {__version__}")


def _resolve_layout(data_root: Path, collection_id: str) -> DatasetLayout:
    """Acha o diretório da coleção sem precisar da API (o slug vem do nome)."""
    direct = DatasetLayout(root=Path(data_root) / collection_id)
    if direct.manifest_path.is_file():
        return direct
    for child in sorted(Path(data_root).glob("*")):
        cand = DatasetLayout(root=child)
        if not cand.provenance_path.is_file():
            continue
        prov = json.loads(cand.provenance_path.read_text(encoding="utf-8"))
        if prov.get("collection_id") == collection_id:
            return cand
    return direct


if __name__ == "__main__":
    app()


@app.command("vlm-audit")
def vlm_audit(
    phase: Annotated[str, typer.Option(help="1 | 2 | all")] = "all",
    v2: Annotated[
        bool,
        typer.Option("--v2", help="Fase 1 com pergunta decomposta e classe travada"),
    ] = False,
    model: Annotated[str | None, typer.Option(help="modelo Ollama")] = None,
    vlm_url: Annotated[str | None, typer.Option("--vlm-url")] = None,
    limit: Annotated[int, typer.Option(help="só as N primeiras (ensaio)")] = 0,
    max_iters: Annotated[int, typer.Option("--max-iters")] = 0,
    collection_id: Annotated[str, typer.Argument()] = "",
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
) -> None:
    """Experimento A1: o VLM como curador, medido contra o gabarito humano.

    Fase 1 dá um veredito nas 5 categorias do Hub para cada caixa que o detector
    propôs. Fase 2 procura o que ficou de fora, e é medida como RECALL DE
    PRESENÇA — o modelo é fraco em geometria, e cobrar caixa apertada dele
    responderia a pergunta errada.

    Não escreve no gabarito e não treina nada. Lê as imagens, chama o modelo
    local, e grava tudo em runs/.
    """
    from tqdm import tqdm

    from forge.config import VLM_MODEL, VLM_RECALL_MAX_ITERS, VLM_URL
    from forge.vlm.client import VlmClient, VlmError
    from forge.vlm.phases import dump_rows, run_phase1, run_phase2
    from forge.vlm.prepare import box_tasks
    from forge.vlm.report import (
        agreement,
        curation_metrics,
        format_matrix,
        recall_report,
    )

    if phase not in {"1", "2", "all"}:
        console.print("[red]--phase deve ser 1, 2 ou all[/]")
        raise typer.Exit(1)

    s = _settings(None)
    cid = collection_id or s.collection_id
    layout = _resolve_layout(out or s.data_root, cid)
    frames = _load_local_manifest(layout)
    reviewed = [f for f in frames if f.reviewed]
    if not reviewed:
        console.print("[red]nenhum frame revisado no manifest — nada a auditar.[/]")
        raise typer.Exit(1)

    def image_of(acq_id: str, sec: int) -> Path:
        return layout.image_path(acq_id, sec)

    missing = [f for f in reviewed if not image_of(f.acq_id, f.sec).is_file()]
    if missing:
        console.print(f"[red]{len(missing)} imagem(ns) de frame revisado faltando em disco.[/]")
        console.print("Rode `forge pull` antes.")
        raise typer.Exit(1)

    tasks = box_tasks(reviewed)
    if limit:
        tasks = tasks[:limit]
        reviewed = reviewed[:limit]

    run = Run.create(
        layout.root / "runs",
        "vlm-audit",
        params={
            "model": model or VLM_MODEL,
            "phase": phase,
            "schema": "v2" if v2 else "v1",
            "reviewed_frames": len(reviewed),
            "boxes": len(tasks),
            "max_iters": max_iters or VLM_RECALL_MAX_ITERS,
            "limit": limit or None,
            "collection_id": cid,
        },
    )

    with VlmClient(
        model=model or VLM_MODEL, url=vlm_url or VLM_URL
    ) as client:
        try:
            version = client.health()
        except VlmError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        console.print(
            f"ollama {version}  ·  modelo [bold]{client.model}[/]  ·  "
            f"{len(reviewed)} frames revisados  ·  {len(tasks)} caixas"
        )

        results: dict[str, object] = {}

        # ── Fase 1 ──
        if phase in {"1", "all"}:
            bar = tqdm(total=len(tasks), desc="fase 1 (veredito)", unit="caixa")
            rows1 = run_phase1(
                client, tasks, image_of, on_step=lambda d, t: bar.update(1), v2=v2
            )
            bar.close()
            dump_rows(run.path("phase1.json"), rows1)

            fine = agreement(rows1)
            coarse = agreement(rows1, coarse=True)
            console.print()
            console.print(format_matrix(fine, "MATRIZ — 5 categorias do Hub"))
            console.print()
            console.print(format_matrix(coarse, "MATRIZ — 3 vias (sem geometria fina)"))

            added = sum(
                1
                for f in reviewed
                for g in (f.ground_truth or [])
                if g.origin == "added"
            )
            m_h = curation_metrics(rows1, added=added, source="truth")
            m_v = curation_metrics(rows1, added=added, source="vlm")
            t = Table("curador", "detecções", "tp", "fp", "fn", "precisão", "recall")
            for name, m in (("humano (gabarito)", m_h), ("VLM", m_v)):
                t.add_row(
                    name, str(m.detections), str(m.tp), str(m.fp), str(m.fn),
                    f"{m.precision}%", f"{m.recall}%",
                )
            console.print()
            console.print(t)

            results["phase1"] = {
                "agreement_5": {
                    "matrix": fine.matrix, "accuracy": fine.accuracy,
                    "per_truth": fine.per_truth(), "errors": fine.errors,
                },
                "agreement_3": {"matrix": coarse.matrix, "accuracy": coarse.accuracy},
                "curation_human": m_h.as_dict(),
                "curation_vlm": m_v.as_dict(),
            }

        # ── Fase 2 ──
        if phase in {"2", "all"}:
            bar = tqdm(total=len(reviewed), desc="fase 2 (recall)", unit="frame")
            rows2 = run_phase2(
                client, reviewed, image_of,
                # Desenho A é passe único: iterar com temperature=0 e entrada
                # idêntica repete a resposta. O teto antigo só volta se pedido.
                max_iters=max_iters or 1,
                on_step=lambda d, t: bar.update(1),
            )
            bar.close()
            dump_rows(run.path("phase2.json"), rows2)

            rec = recall_report(rows2, reviewed)
            console.print()
            console.print(
                f"[bold]FASE 2 — recall de presença[/] (IoU frouxo ou centro dentro)\n"
                f"  caixas desenhadas na mão pelo humano : {rec.human_added}\n"
                f"  encontradas pelo VLM                 : {rec.found}  "
                f"([bold]{rec.recall}%[/])\n"
                f"  propostas do VLM                     : {rec.proposals}  "
                f"(acertaram alvo: {rec.matched_proposals} = {rec.precision}%)\n"
                f"  classe também correta                : {rec.class_correct}"
            )
            if rec.by_class:
                t2 = Table("classe", "humano desenhou", "VLM achou", "%")
                for cls, d in sorted(rec.by_class.items(), key=lambda kv: -kv[1]["human"]):
                    pct = round(100 * d["found"] / d["human"], 1) if d["human"] else None
                    t2.add_row(cls, str(d["human"]), str(d["found"]), f"{pct}%")
                console.print(t2)
            results["phase2"] = rec.as_dict()

        # ── custo ──
        u = client.usage.as_dict()
        console.print()
        console.print(
            f"[bold]CUSTO[/] (local, sem API paga): {u['calls']} chamadas  ·  "
            f"{u['seconds'] / 60:.1f} min  ·  {u['seconds_per_call']}s/chamada  ·  "
            f"{u['prompt_tokens'] + u['eval_tokens']} tokens"
            + (f"  ·  [red]{u['failures']} falhas[/]" if u["failures"] else "")
        )
        results["usage"] = u

    run.results = results
    run.save()
    console.print(f"[green]{run.root}[/]")


@app.command("vlm-score")
def vlm_score(
    run_dir: Annotated[Path | None, typer.Argument(help="run do vlm-audit")] = None,
    phase2_run: Annotated[
        Path | None, typer.Option("--phase2-run", help="se a Fase 2 saiu em outro run")
    ] = None,
    collection_id: Annotated[str, typer.Option("--collection-id")] = "",
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
) -> None:
    """Placar YOLO base × YOLO+VLM contra o gabarito, na aritmética da Table 6.

    Não chama o modelo: lê o que o `vlm-audit` já produziu. A curadoria do VLM
    decide quais caixas sobrevivem; o gabarito decide se cada sobrevivente está
    certa. As duas coisas nunca se misturam.
    """
    import json as _json

    from forge.core.geometry import BoxN as _BoxN
    from forge.vlm.phases import Phase1Row, Phase2Row, Proposal
    from forge.vlm.score import (
        Score,
        baseline_score,
        format_table,
        human_corrected_class,
        score_phase1,
        score_phase2,
    )

    s = _settings(None)
    cid = collection_id or s.collection_id
    layout = _resolve_layout(out or s.data_root, cid)
    frames = [f for f in _load_local_manifest(layout) if f.reviewed]

    runs_root = layout.root / "runs"

    def newest_with(name: str) -> Path | None:
        cands = sorted(
            (d for d in runs_root.glob("*-vlm-audit") if (d / name).is_file()),
            key=lambda d: d.name,
        )
        return cands[-1] if cands else None

    p1_dir = run_dir or newest_with("phase1.json")
    p2_dir = phase2_run or run_dir or newest_with("phase2.json")
    if p1_dir is None or not (p1_dir / "phase1.json").is_file():
        console.print("[red]nenhum run com phase1.json. Rode `forge vlm-audit --phase 1`.[/]")
        raise typer.Exit(1)

    rows1 = [
        Phase1Row(**r)
        for r in _json.loads((p1_dir / "phase1.json").read_text(encoding="utf-8"))
    ]
    # A régua do paper é 501; um run antigo pode ter 505 (antes da correção de
    # GT_EXCLUDED_CLASSES). Refiltra aqui para o placar nunca sair fora de régua.
    from forge.config import GT_EXCLUDED_CLASSES

    dropped = [
        r for r in rows1 if r.truth == "kept" and r.proposed_class in GT_EXCLUDED_CLASSES
    ]
    if dropped:
        rows1 = [r for r in rows1 if r not in dropped]
        console.print(
            f"[yellow]régua: {len(dropped)} caixa(s) de "
            f"{sorted(GT_EXCLUDED_CLASSES)} excluída(s) → {len(rows1)} detecções[/]"
        )

    rows2: list[Phase2Row] = []
    if p2_dir is not None and (p2_dir / "phase2.json").is_file():
        for r in _json.loads((p2_dir / "phase2.json").read_text(encoding="utf-8")):
            props = [
                Proposal(
                    cls_name=p["cls_name"],
                    box=_BoxN(**p["box"]),
                    confidence=p["confidence"],
                    iteration=p["iteration"],
                )
                for p in r.pop("proposals", [])
            ]
            rows2.append(Phase2Row(**r, proposals=props))
    else:
        console.print("[yellow]sem phase2.json — o placar sai só com a Fase 1.[/]")

    added = sum(
        1 for f in frames for g in (f.ground_truth or []) if g.origin == "added"
    )
    rejected = sum(1 for r in rows1 if r.truth == "rejected")
    gt = len(rows1) - rejected + added
    console.print(
        f"gabarito: {gt} instâncias  ({len(rows1)} detecções − {rejected} rejeitadas "
        f"+ {added} desenhadas na mão)"
    )

    corrected = human_corrected_class(frames)
    scores = [baseline_score(rows1, gt)]

    for label, drop in (
        ("YOLO+VLM (principal)", False),
        ("YOLO+VLM (sensib.: corrigidas saem)", True),
    ):
        sc = Score(label=label, gt=gt)
        score_phase1(rows1, corrected, sc, drop_corrections=drop)
        if rows2:
            score_phase2(rows2, frames, sc)
        scores.append(sc)

    if rows2:
        sc = Score(label="YOLO+VLM (só Fase 1, sem adições)", gt=gt)
        score_phase1(rows1, corrected, sc, drop_corrections=False)
        scores.insert(2, sc)

    console.print()
    console.print(format_table(scores))
    console.print()

    base = scores[0]
    for sc in scores[1:]:
        dp = (sc.precision or 0) - (base.precision or 0)
        dr = (sc.recall or 0) - (base.recall or 0)
        console.print(
            f"{sc.label:<38} precisão {dp:+.1f} pp   recall {dr:+.1f} pp"
        )

    console.print()
    for sc in scores[1:]:
        console.print(f"[bold]{sc.label}[/] — de onde veio:")
        for k, v in sorted(sc.detail.items(), key=lambda kv: -kv[1]):
            console.print(f"   {k:<36} {v}")
        console.print()

    dest = p1_dir / "score.json"
    dest.write_text(
        _json.dumps([s.as_dict() for s in scores], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"[green]{dest}[/]")


@app.command("vlm-flow")
def vlm_flow(
    step: Annotated[str, typer.Option(help="candidates | binary | score | all")] = "all",
    phase1_run: Annotated[Path | None, typer.Option("--phase1-run")] = None,
    conf: Annotated[float, typer.Option(help="corte do proponente")] = 0.25,
    weights: Annotated[str, typer.Option(help="pesos do proponente")] = "yolo11x.pt",
    device: Annotated[str | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(help="só os N primeiros candidatos")] = 0,
    collection_id: Annotated[str, typer.Option("--collection-id")] = "",
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
) -> None:
    """Fluxo reordenado: YOLO → proponente → VLM classificador.

    Três passos com cache entre eles, porque o passo 1 quer a GPU para o
    detector e o passo 2 a quer para o VLM — juntos disputam VRAM.
    """
    import json as _json

    from tqdm import tqdm

    from forge.config import GT_EXCLUDED_CLASSES, PARITY_IOU, VLM_MODEL
    from forge.core.classes import normalize_class as _norm
    from forge.core.geometry import BoxN as _BoxN
    from forge.core.geometry import iou as _iou
    from forge.vlm.client import VlmClient, VlmError, load_frame_image
    from forge.vlm.flow import (
        BINARY_SCHEMA,
        Candidate,
        FlowScore,
        best_overlap,
        binary_prompt,
        box_is_correct,
        build_candidates,
        truth_index,
    )
    from forge.vlm.phases import Phase1Row
    from forge.vlm.prepare import crop_around, draw_box, human_added_boxes

    s = _settings(None)
    cid = collection_id or s.collection_id
    layout = _resolve_layout(out or s.data_root, cid)
    frames = [f for f in _load_local_manifest(layout) if f.reviewed]
    runs_root = layout.root / "runs"
    cache = runs_root / "_flow_cache"
    cache.mkdir(parents=True, exist_ok=True)
    det_path, cand_path = cache / "detections.json", cache / "candidates.json"
    ans_path = cache / "binary.json"

    def fkey(acq_id: str, sec: int) -> str:
        return f"{acq_id}:{sec}"

    # ── passo 1: o proponente ────────────────────────────────────────────────
    if step in {"candidates", "all"}:
        from ultralytics import YOLO

        paths, keys = [], []
        for f in frames:
            p = layout.image_path(f.acq_id, f.sec)
            if p.is_file():
                paths.append(str(p))
                keys.append(fkey(f.acq_id, f.sec))
        listfile = cache / "_sources.txt"
        listfile.write_text("\n".join(paths), encoding="utf-8")

        model = YOLO(weights)
        dets: dict[str, list] = {}
        for k, r in zip(
            keys,
            model.predict(source=str(listfile), conf=conf, iou=0.7, imgsz=640,
                          device=device or 0, batch=8, stream=True, verbose=False),
            strict=False,
        ):
            items = []
            if r.boxes is not None:
                for b in r.boxes:
                    xy = b.xyxyn[0].tolist()
                    items.append([_norm(r.names[int(b.cls)]), float(b.conf), xy])
            dets[k] = items
        det_path.write_text(_json.dumps(dets), encoding="utf-8")

        parsed = {k: [(c, cf, _BoxN(*bx)) for c, cf, bx in v] for k, v in dets.items()}
        cands0 = build_candidates(parsed, frames)
        cand_path.write_text(
            _json.dumps([
                {"acq_id": c.acq_id, "sec": c.sec, "index": c.index,
                 "det_class": c.det_class, "conf": c.conf,
                 "box": [c.box.x1, c.box.y1, c.box.x2, c.box.y2]}
                for c in cands0
            ], indent=2),
            encoding="utf-8",
        )
        console.print(
            f"proponente [bold]{weights}[/] @{conf}: "
            f"{sum(len(v) for v in dets.values())} detecções → "
            f"[bold]{len(cands0)}[/] candidatos novos"
        )

    if not cand_path.is_file():
        console.print("[red]sem candidatos. Rode --step candidates.[/]")
        raise typer.Exit(1)

    cands = [
        Candidate(acq_id=c["acq_id"], sec=c["sec"], index=c["index"],
                  det_class=c["det_class"], conf=c["conf"], box=_BoxN(*c["box"]))
        for c in _json.loads(cand_path.read_text(encoding="utf-8"))
    ]
    if limit:
        cands = cands[:limit]
    fresh = {
        k: [(c, cf, _BoxN(*bx)) for c, cf, bx in v]
        for k, v in _json.loads(det_path.read_text(encoding="utf-8")).items()
    }

    # ── passo 2: a fila binária ──────────────────────────────────────────────
    if step in {"binary", "all"}:
        run = Run.create(runs_root, "vlm-flow", params={
            "weights": weights, "conf": conf,
            "candidates": len(cands), "model": VLM_MODEL,
        })
        with VlmClient() as client:
            try:
                client.health()
            except VlmError as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(1) from exc

            grouped: dict[str, list[Candidate]] = {}
            for c in cands:
                grouped.setdefault(fkey(c.acq_id, c.sec), []).append(c)

            bar = tqdm(total=len(cands), desc="fila binaria", unit="cand")
            for fk, group in grouped.items():
                acq, sec = fk.split(":")
                img = load_frame_image(layout.image_path(acq, int(sec)))
                for c in group:
                    scene = draw_box(img, c.box, c.det_class)
                    crop = crop_around(img, c.box)
                    try:
                        a = client.ask(binary_prompt(c), [scene, crop], BINARY_SCHEMA)
                        c.vlm_present = bool(a.get("present"))
                        cl = a.get("class")
                        c.vlm_class = _norm(cl) if isinstance(cl, str) else None
                        c.reason = str(a.get("reason", ""))[:300]
                    except VlmError as exc:
                        c.error = str(exc)[:200]
                    bar.update(1)
            bar.close()
            usage = client.usage.as_dict()
            run.results = {"usage": usage}
            run.save()
            console.print(
                f"[bold]CUSTO[/] {usage['calls']} chamadas · "
                f"{usage['seconds'] / 60:.1f} min · {usage['seconds_per_call']}s/chamada"
            )
        ans_path.write_text(_json.dumps([
            {"key": c.key, "present": c.vlm_present, "class": c.vlm_class,
             "reason": c.reason, "error": c.error} for c in cands
        ], indent=2, ensure_ascii=False), encoding="utf-8")

    if ans_path.is_file():
        ans = {a["key"]: a for a in _json.loads(ans_path.read_text(encoding="utf-8"))}
        for c in cands:
            a = ans.get(c.key)
            if a is not None:
                c.vlm_present, c.vlm_class, c.error = a["present"], a["class"], a["error"]

    if step not in {"score", "all"}:
        return

    # ── passo 3: o placar ────────────────────────────────────────────────────
    p1 = phase1_run or max(
        (d for d in runs_root.glob("*-vlm-audit") if (d / "phase1.json").is_file()),
        key=lambda d: d.name, default=None,
    )
    if p1 is None:
        console.print("[red]sem phase1.json — rode `forge vlm-audit --phase 1`.[/]")
        raise typer.Exit(1)
    rows1 = [Phase1Row(**r)
             for r in _json.loads((p1 / "phase1.json").read_text(encoding="utf-8"))]
    rows1 = [r for r in rows1
             if not (r.truth == "kept" and r.proposed_class in GT_EXCLUDED_CLASSES)]

    truth = truth_index(frames)
    orig = {
        f"{f.acq_id}:{f.sec}:{i}": d.box_n.to_box()
        for f in frames for i, d in enumerate(f.stored_detections) if d.box_n is not None
    }
    added_total = sum(len(human_added_boxes(f)) for f in frames)
    gt = len(rows1) - sum(1 for r in rows1 if r.truth == "rejected") + added_total
    console.print(f"gabarito {gt} · {len(rows1)} detecções · {len(cands)} candidatos")

    def part1(sc: FlowScore, *, fix_geometry: bool) -> None:
        for r in rows1:
            v = r.vlm or "kept"
            if v == "rejected":
                sc.bump("descartou_caixa")
                continue
            label = (r.vlm_class if v == "class-changed" and r.vlm_class
                     else r.proposed_class)
            box = orig.get(r.key)
            if box is None:
                continue
            if fix_geometry and v in {"corrected-minor", "corrected-redrawn"}:
                nb = best_overlap(box, fresh.get(fkey(r.acq_id, r.sec), []))
                if nb is not None:
                    box = nb
                    sc.bump("geometria_substituida")
                else:
                    sc.bump("sem_geometria_para_substituir")
            sc.boxes += 1
            if box_is_correct(label, box, truth.get(r.key)):
                sc.tp += 1
            else:
                sc.bump("caixa_incorreta")

    def part2(sc: FlowScore, *, use_filter: bool, thr: float) -> None:
        grouped: dict[str, list[Candidate]] = {}
        for c in cands:
            grouped.setdefault(fkey(c.acq_id, c.sec), []).append(c)
        for f in frames:
            added = human_added_boxes(f)
            unused = list(range(len(added)))
            for c in grouped.get(fkey(f.acq_id, f.sec), []):
                if use_filter and c.vlm_present is not True:
                    sc.bump("vlm_descartou_candidato")
                    continue
                label = c.vlm_class if (use_filter and c.vlm_class) else c.det_class
                sc.boxes += 1
                hit = next(
                    (i for i in unused
                     if _iou(c.box, added[i][1]) >= thr and label == added[i][0]),
                    None,
                )
                if hit is None:
                    sc.bump("candidato_sem_alvo")
                    continue
                unused.remove(hit)
                sc.tp += 1
                sc.bump("recuperou_objeto")

    scores: list[FlowScore] = []
    base = FlowScore(label="YOLO base", gt=gt)
    for r in rows1:
        box = orig.get(r.key)
        if box is None:
            continue
        base.boxes += 1
        if box_is_correct(r.proposed_class, box, truth.get(r.key)):
            base.tp += 1
    scores.append(base)

    only1 = FlowScore(label="YOLO+VLM so Fase 1 (hoje)", gt=gt)
    part1(only1, fix_geometry=False)
    scores.append(only1)

    geo = FlowScore(label="+ geometria do YOLO11x nas correcoes", gt=gt)
    part1(geo, fix_geometry=True)
    scores.append(geo)

    full = FlowScore(label="FLUXO COMPLETO (VLM filtra candidatos)", gt=gt)
    part1(full, fix_geometry=True)
    part2(full, use_filter=True, thr=PARITY_IOU)
    scores.append(full)

    floor = FlowScore(label="   piso: sem o filtro do VLM", gt=gt)
    part1(floor, fix_geometry=True)
    part2(floor, use_filter=False, thr=PARITY_IOU)
    scores.append(floor)

    head = (f"{'conjunto':<40}{'caixas':>8}{'TP':>6}{'FP':>6}{'FN':>6}"
            f"{'precisao':>11}{'recall':>9}")
    console.print()
    console.print(head)
    console.print("-" * len(head))
    for sc in scores:
        console.print(
            f"{sc.label:<40}{sc.boxes:>8}{sc.tp:>6}{sc.fp:>6}{sc.fn:>6}"
            f"{str(sc.precision) + '%':>11}{str(sc.recall) + '%':>9}"
        )
    console.print()
    for sc in scores[1:]:
        console.print(f"[bold]{sc.label}[/]: " + " · ".join(
            f"{k} {v}" for k, v in sorted(sc.detail.items(), key=lambda kv: -kv[1])))

    dest = cache / "flow_score.json"
    dest.write_text(
        _json.dumps([s.as_dict() for s in scores], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    console.print(f"\n[green]{dest}[/]")


@app.command("vlm-curate")
def vlm_curate(
    tag: Annotated[str, typer.Option(help="tag dos rótulos de origem")] = "trackb-nao_doadora_yolo11x_c025",
    out_tag: Annotated[str, typer.Option("--out-tag", help="tag de destino")] = "trackb-nao_doadora_vlm-v2",
    limit: Annotated[int, typer.Option(help="só os N primeiros frames pendentes")] = 0,
    every: Annotated[int, typer.Option(help="imprime progresso a cada N frames")] = 25,
    fake: Annotated[
        bool,
        typer.Option("--fake", help="não chama o modelo; exercita o laço e o checkpoint"),
    ] = False,
    reset: Annotated[
        bool, typer.Option("--reset", help="apaga o ledger e recomeça do zero")
    ] = False,
    collection_id: Annotated[str, typer.Option("--collection-id")] = "",
    out: Annotated[Path | None, typer.Option(help="raiz do dataset")] = None,
) -> None:
    """Cura pseudo-rótulo de TREINO com o VLM v2. Retomável.

    Grava por frame completo: interromper com Ctrl+C perde no máximo o frame
    corrente. Rodar de novo continua de onde parou — não passe `--reset` a menos
    que queira mesmo refazer tudo.

    Não toca no rótulo de origem, no gabarito, nem em coleção nenhuma.
    """
    import time as _time

    from forge.vlm.client import VlmClient, VlmError
    from forge.vlm.curate import (
        CURATION_CLASSES,
        completed_keys,
        eta,
        iter_curation,
        write_meta,
    )

    s = _settings(None)
    layout = _resolve_layout(out or s.data_root, collection_id or s.collection_id)
    src = layout.auto_labels_dir(tag)
    dst = layout.auto_labels_dir(out_tag)
    if not src.is_dir():
        console.print(f"[red]rótulos de origem não encontrados: {src}[/]")
        raise typer.Exit(1)
    if src == dst:
        console.print("[red]origem e destino iguais — isso sobrescreveria o pseudo-rótulo.[/]")
        raise typer.Exit(1)

    ledger = dst.parent / f"{out_tag}_progress.jsonl"
    meta_path = dst.parent / f"{out_tag}_meta.json"
    if reset and ledger.exists():
        ledger.unlink()
        console.print("[yellow]ledger apagado — recomeçando do zero.[/]")

    total = len(list(src.glob("*.txt")))
    done_before = len(completed_keys(ledger))
    pending = total - done_before
    if limit:
        pending = min(pending, limit)

    console.print(
        f"origem [bold]{tag}[/] → destino [bold]{out_tag}[/]\n"
        f"{total} frames · {done_before} já feitos · [bold]{pending}[/] nesta rodada"
    )
    if pending == 0:
        console.print("[green]nada pendente.[/]")
        write_meta(meta_path, ledger, total)
        return

    class _FakeClient:
        """Substituto do cliente para exercitar laço, checkpoint e retomada
        sem depender do Ollama. Devolve respostas determinísticas."""

        from forge.vlm.client import Usage as _U

        usage = _U()

        def ask(self, prompt: str, images: list, schema: dict) -> dict:
            self.usage.calls += 1
            n = len(prompt)
            if n % 7 == 0:
                return {"object_present": False, "object_class": "car",
                        "box_quality": "tight", "reason": "fake: sem objeto"}
            if n % 5 == 0:
                return {"object_present": True, "object_class": "truck",
                        "box_quality": "tight", "reason": "fake: classe trocada"}
            return {"object_present": True, "object_class": "car",
                    "box_quality": "needs_nudge", "reason": "fake: ajuste leve"}

        def health(self) -> str:
            return "fake"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

    client_cm = _FakeClient() if fake else VlmClient()
    t0 = _time.time()
    processed = 0

    with client_cm as client:
        if not fake:
            try:
                version = client.health()
                console.print(f"ollama {version} · modelo [bold]{client.model}[/]")
            except VlmError as exc:
                console.print(f"[red]{exc}[/]")
                raise typer.Exit(1) from exc

        try:
            for result in iter_curation(
                client, src, dst, layout.image_path,
                ledger=ledger, classes=CURATION_CLASSES, limit=limit,
            ):
                processed += 1
                if processed % every == 0 or processed == pending:
                    spent = _time.time() - t0
                    console.print(
                        f"  {processed}/{pending} nesta rodada · "
                        f"{done_before + processed}/{total} no total · "
                        f"{spent / 60:.1f} min · resta ~{eta(processed, pending, spent)}"
                    )
                    write_meta(meta_path, ledger, total)
        except KeyboardInterrupt:
            console.print("\n[yellow]interrompido. O último frame completo está salvo; "
                          "rode de novo para continuar de onde parou.[/]")
        except VlmError as exc:
            console.print(f"\n[red]{exc}[/]")

    doc = write_meta(meta_path, ledger, total)
    console.print()
    console.print(
        f"[bold]{doc['frames_processed']}/{total} frames[/] · "
        f"{doc['boxes_curated']} caixas curadas · "
        f"{doc['boxes_passthrough']} passthrough · "
        f"{doc['model_calls']} chamadas ({doc['retries']} retries) · "
        f"{doc['error_count']} erros"
    )
    if doc["by_verdict"]:
        console.print("veredito: " + " · ".join(
            f"{k} {v}" for k, v in doc["by_verdict"].items()))
    console.print(f"[green]{ledger}[/]\n[green]{meta_path}[/]")
