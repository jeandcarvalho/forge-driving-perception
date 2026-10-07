"""Baixar os bytes de uma imagem do Google Drive.

Porte de `resolveStreamUrl` em carcara/apps/api/src/controllers/video-proxy.ts.

O Drive não devolve o arquivo direto para tudo: para uma parte dos arquivos ele
responde **200 com HTML** — a página de "virus scan warning" — e o download real
exige repetir a requisição com o `uuid` que está escondido nessa página. Um
cliente ingênuo salva o HTML como se fosse JPEG e só descobre no treino.

Dois caminhos:
  original (padrão) — bytes exatos que foram enviados ao Drive.
  cdn (--fast)      — lh3.googleusercontent.com, entrega 1920x1080 sem a dança
                      do uuid, porém JPEG RECOMPRIMIDO. Serve para prototipar;
                      não use para número que vai no paper.
"""

from __future__ import annotations

import re
import time
from typing import Self

import httpx

MAX_HOPS = 14
MAX_ATTEMPTS = 4
JPEG_MAGIC = b"\xff\xd8\xff"

_UUID_PATTERNS = (
    re.compile(r'name="uuid"\s+value="([^"]+)"'),
    re.compile(r"""name=["']uuid["']\s+value=["']([^"']+)["']"""),
    re.compile(r"[?&]uuid=([A-Za-z0-9_-]+)"),
)
_HREF_PATTERNS = (
    re.compile(r'href="(/download\?[^"]*confirm[^"]*)"'),
    re.compile(r'href="(https://drive\.usercontent\.google\.com/download\?[^"]*confirm[^"]*)"'),
)


class DriveError(RuntimeError):
    pass


def original_url(drive_id: str) -> str:
    return (
        f"https://drive.usercontent.google.com/download"
        f"?id={drive_id}&export=download&confirm=t"
    )


def cdn_url(drive_id: str) -> str:
    return f"https://lh3.googleusercontent.com/d/{drive_id}=w1920-h1080-n"


class DriveFetcher:
    """Cliente com cookie jar próprio. Crie um por thread — httpx.Client é
    thread-safe para requisições, mas a dança de cookies do Drive não é."""

    def __init__(self, *, timeout: float = 120.0, fast: bool = False) -> None:
        self.fast = fast
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0"},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def close(self) -> None:
        self._client.close()

    def fetch(self, drive_id: str) -> bytes:
        last: Exception | None = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                data = self._fetch_once(drive_id)
                if not data.startswith(JPEG_MAGIC):
                    raise DriveError(
                        f"{drive_id}: os bytes não são JPEG (começam com {data[:8]!r}). "
                        "Provavelmente uma página HTML do Drive escapou da resolução."
                    )
                return data
            except (httpx.RequestError, DriveError) as exc:
                last = exc
                if attempt == MAX_ATTEMPTS:
                    break
                time.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))
            except httpx.HTTPStatusError as exc:
                last = exc
                # 403/429 = cota do Drive. Recuar mais que o normal.
                if exc.response.status_code in (403, 429) and attempt < MAX_ATTEMPTS:
                    time.sleep(min(60.0, 5.0 * (2 ** (attempt - 1))))
                    continue
                if attempt == MAX_ATTEMPTS:
                    break
                time.sleep(min(20.0, 1.5 * (2 ** (attempt - 1))))
        raise DriveError(f"{drive_id}: falhou em {MAX_ATTEMPTS} tentativas — {last}")

    def _fetch_once(self, drive_id: str) -> bytes:
        if self.fast:
            resp = self._client.get(cdn_url(drive_id))
            resp.raise_for_status()
            return resp.content

        url = original_url(drive_id)
        for _ in range(MAX_HOPS):
            resp = self._client.get(url)
            if resp.status_code in (403, 429):
                resp.raise_for_status()
            resp.raise_for_status()

            ctype = resp.headers.get("content-type", "")
            if "text/html" not in ctype:
                return resp.content

            # Página de confirmação do antivírus: extrai e repete.
            html = resp.text
            uuid = _first_match(html, _UUID_PATTERNS)
            if uuid:
                url = f"{original_url(drive_id)}&uuid={uuid}"
                continue

            href = _first_match(html, _HREF_PATTERNS)
            if href:
                href = href.replace("&amp;", "&")
                url = (
                    f"https://drive.usercontent.google.com{href}"
                    if href.startswith("/")
                    else href
                )
                continue

            raise DriveError(
                f"{drive_id}: página de confirmação do Drive não reconhecida — "
                f"{html[:200]!r}"
            )

        raise DriveError(f"{drive_id}: excedeu {MAX_HOPS} saltos de redirecionamento")


def _first_match(text: str, patterns: tuple[re.Pattern[str], ...]) -> str | None:
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1)
    return None
