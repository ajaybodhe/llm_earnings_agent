"""Filesystem JSON cache keyed by SHA1 of agent inputs.

Cache layout: data/cache/{key[:2]}/{key}.json

A key collapses (ticker, agent, prompt_version, model, input_hash) into a stable
SHA1. Bumping the prompt version or changing the input payload yields a fresh
key automatically, so there's nothing to invalidate by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "cache"


def cache_key(
    *,
    ticker: str,
    agent: str,
    prompt_version: str,
    model: str,
    payload: Any,
) -> str:
    """Stable SHA1 of all cache-affecting inputs."""
    canonical = json.dumps(
        {
            "ticker": ticker.upper(),
            "agent": agent,
            "prompt_version": prompt_version,
            "model": model,
            "payload": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


class JsonCache:
    """Disk-backed JSON cache. Thread-safe enough for single-process use."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else DEFAULT_CACHE_ROOT
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, indent=2, default=str))
        tmp.replace(p)

    def delete(self, key: str) -> bool:
        p = self._path(key)
        if p.exists():
            p.unlink()
            return True
        return False
