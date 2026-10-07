"""Cliente do Ollama para o experimento A1 (VLM como curador).

Roda 100% local. Não há chave de API, não há custo por token — o que se mede
aqui é tempo de máquina e número de chamadas, e é isso que o paper reporta na
seção de custo.

A saída é SEMPRE restringida por JSON schema (`format`). Isso não é conveniência
de parsing: é a garantia de método de que o VLM escolhe dentro da taxonomia do
Hub e não inventa uma sexta categoria que depois contaminaria a matriz de
concordância. O `enum` do schema faz no decodificador o que uma instrução no
prompt só pediria por favor.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self

import httpx
from PIL import Image

from forge.config import VLM_MODEL, VLM_TEMPERATURE, VLM_URL


@dataclass(slots=True)
class Usage:
    """Contabilidade da rodada. É o 'quanto custou' do A1.

    Sem preço por token (o modelo é local), então o que importa é: quantas
    chamadas, quanto tempo de relógio, e quantos tokens — este último para que
    alguém consiga estimar o custo de rodar o mesmo desenho numa API paga.
    """

    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    eval_tokens: int = 0
    seconds: float = 0.0

    def add(self, resp: dict[str, Any], elapsed: float) -> None:
        self.calls += 1
        self.prompt_tokens += int(resp.get("prompt_eval_count") or 0)
        self.eval_tokens += int(resp.get("eval_count") or 0)
        self.seconds += elapsed

    def as_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "eval_tokens": self.eval_tokens,
            "seconds": round(self.seconds, 1),
            "seconds_per_call": round(self.seconds / self.calls, 2) if self.calls else None,
        }


def encode_image(img: Image.Image, *, quality: int = 92) -> str:
    """PIL → base64 JPEG, o formato que o `images[]` do Ollama espera."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VlmError(RuntimeError):
    pass


@dataclass(slots=True)
class VlmClient:
    """Chamada única, síncrona, sem paralelismo — e isso é deliberado.

    Medido em 19/08/2026: com 2 slots (`OLLAMA_NUM_PARALLEL=2`) o throughput
    CAIU para 0,86x do sequencial. O gargalo é o encoder de visão, que já satura
    a GPU numa imagem só; um segundo slot disputa o mesmo recurso e ainda soma
    KV cache. Paralelizar aqui piora e arrisca OOM.
    """

    model: str = VLM_MODEL
    url: str = VLM_URL
    temperature: float = VLM_TEMPERATURE
    timeout: float = 300.0
    usage: Usage = field(default_factory=Usage)
    _client: httpx.Client | None = None

    def __enter__(self) -> Self:
        self._client = httpx.Client(base_url=self.url.rstrip("/"), timeout=self.timeout)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def ask(
        self,
        prompt: str,
        images: list[Image.Image],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Uma pergunta, uma resposta já validada contra o schema.

        Levanta `VlmError` em vez de devolver lixo: um veredito ausente tem que
        aparecer como falha contada em `usage.failures`, nunca como uma
        categoria plausível inventada para tapar buraco.
        """
        if self._client is None:
            raise VlmError("use o cliente dentro de `with VlmClient() as c:`")

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [encode_image(im) for im in images],
                }
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        t0 = time.time()
        try:
            resp = self._client.post("/api/chat", json=body)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            self.usage.failures += 1
            raise VlmError(
                f"Ollama não respondeu ({exc}). O servidor está de pé em {self.url}? "
                f"E o modelo `{self.model}` foi puxado (`ollama pull {self.model}`)?"
            ) from exc

        elapsed = time.time() - t0
        self.usage.add(payload, elapsed)

        content = (payload.get("message") or {}).get("content") or ""
        if not content.strip():
            self.usage.failures += 1
            raise VlmError(
                "resposta vazia. Se o modelo for a variante *thinking*, o conteúdo "
                "sai fora de `message.content` — use a variante instruct."
            )
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            self.usage.failures += 1
            raise VlmError(f"schema não respeitado: {content[:200]!r}") from exc

    def health(self) -> str:
        """Confere servidor e modelo ANTES de começar uma rodada de horas."""
        if self._client is None:
            raise VlmError("use dentro de `with`")
        try:
            version = self._client.get("/api/version").json().get("version", "?")
            tags = self._client.get("/api/tags").json().get("models", [])
        except httpx.HTTPError as exc:
            raise VlmError(f"Ollama não responde em {self.url}: {exc}") from exc
        names = {m.get("name", "") for m in tags}
        if self.model not in names:
            raise VlmError(
                f"modelo `{self.model}` não está instalado. "
                f"Rode: ollama pull {self.model}\nInstalados: {sorted(names)}"
            )
        return version


def load_frame_image(path: Path) -> Image.Image:
    """Lê o JPEG do frame. Sem redimensionar — ver VLM_IMAGE_MAX_SIDE."""
    with Image.open(path) as im:
        return im.convert("RGB").copy()
