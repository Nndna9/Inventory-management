"""Configuration loader. All paths are resolved relative to the project root."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def load_config() -> dict:
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def path(key: str) -> Path:
    """Absolute path for a key under `paths:` in settings.yaml."""
    return ROOT / load_config()["paths"][key]
