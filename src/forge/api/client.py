"""Cliente HTTP da API CarCará. Todas as rotas usadas são públicas, sem token."""

from __future__ import annotations

import time
from typing import Any, Self

import httpx

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 5


class ApiError(RuntimeError):
    pass


class CarcaraClient:
    def __init__(self, base_url: str, *, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "carcara-forge/0.1"},
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        clean = {k: v for k, v in (params or {}).items() if v is not None}
        last: Exception | None = None

        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = self._client.get(path, params=clean)
            except httpx.RequestError as exc:
                last = exc
                if attempt == MAX_ATTEMPTS:
                    break
                time.sleep(_backoff(attempt))
                continue

            if resp.status_code in RETRY_STATUS and attempt < MAX_ATTEMPTS:
                time.sleep(_retry_after(resp) or _backoff(attempt))
                continue

            if resp.status_code >= 400:
                raise ApiError(
                    f"{resp.status_code} em GET {path} — {resp.text[:300]}"
                )
            return resp.json()

        raise ApiError(
            f"GET {path} falhou depois de {MAX_ATTEMPTS} tentativas. "
            f"A API está de pé em {self.base_url}? (`npm run dev:api` no repo carcara). "
            f"Último erro: {last}"
        )

    def health(self) -> bool:
        try:
            self._client.get("/", timeout=5.0)
            return True
        except httpx.RequestError:
            return False


def _backoff(attempt: int) -> float:
    return min(30.0, 0.75 * (2 ** (attempt - 1)))


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("retry-after")
    if not raw:
        return None
    try:
        return min(60.0, float(raw))
    except ValueError:
        return None
