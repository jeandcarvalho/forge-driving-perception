from __future__ import annotations

import json
from pathlib import Path

import pytest

from forge.api.models import Frame, ManifestPage

FIXTURES = Path(__file__).parent / "fixtures"


def _page(name: str) -> ManifestPage:
    return ManifestPage.model_validate(
        json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    )


@pytest.fixture(scope="session")
def test_page() -> ManifestPage:
    """313 frames do split de teste, congelados da API em 2026-08-15."""
    return _page("manifest_test_full.json")


@pytest.fixture(scope="session")
def test_frames(test_page: ManifestPage) -> list[Frame]:
    return test_page.frames


@pytest.fixture(scope="session")
def train_frames() -> list[Frame]:
    return _page("manifest_train_head.json").frames


@pytest.fixture(scope="session")
def discarded_frames() -> list[Frame]:
    return _page("manifest_discarded_head.json").frames
