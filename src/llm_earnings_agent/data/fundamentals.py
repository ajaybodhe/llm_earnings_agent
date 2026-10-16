"""Fundamentals via subprocess to the `quarterly_results` Go binary.

We shell out instead of re-implementing SEC + Finnhub parsing in Python. The
Go binary already produces a normalized JSON view of one stock and we just
need to read it.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

QUARTERLY_RESULTS_DIR = Path(__file__).resolve().parents[4] / "quarterly_results"
DEFAULT_TIMEOUT_S = 180


@dataclass(frozen=True)
class FundamentalsPayload:
    """Raw JSON from `quarterly_results --symbol X --output json`.

    Kept as a thin wrapper so the agent prompt can pick fields it cares about
    without us locking down a strict schema on the Go side.
    """

    symbol: str
    data: dict[str, Any]


async def fetch_fundamentals(
    symbol: str,
    *,
    quarterly_results_dir: Path | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> FundamentalsPayload:
    """Run the Go binary and return its first JSON result for the symbol."""
    work_dir = quarterly_results_dir or QUARTERLY_RESULTS_DIR
    if not work_dir.exists():
        raise RuntimeError(f"quarterly_results dir not found: {work_dir}")

    proc = await asyncio.create_subprocess_exec(
        "go",
        "run",
        ".",
        "--symbol",
        symbol,
        "--output",
        "json",
        cwd=str(work_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ},
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except TimeoutError:
        proc.kill()
        raise RuntimeError(f"quarterly_results timed out after {timeout_s}s for {symbol}") from None

    if proc.returncode != 0:
        msg = stderr.decode(errors="replace")[:1000]
        raise RuntimeError(f"quarterly_results exit {proc.returncode}: {msg}")

    return parse_payload(stdout.decode(), symbol)


def parse_payload(text: str, symbol: str) -> FundamentalsPayload:
    """Parse quarterly_results JSON output for one symbol.

    Accepts either the CLI's JSON array (the element matching ``symbol`` is
    picked, else the first) or a single result object — the form
    quarterly_results sends on stdin with ``--payload -``.
    """
    text = text.strip()
    if not text:
        raise RuntimeError(f"quarterly_results produced no output for {symbol}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        # Some runs may include log noise before the JSON. Try to locate the
        # first '[' or '{' and parse from there.
        starts = [i for i in (text.find("["), text.find("{")) if i >= 0]
        if not starts:
            raise
        payload = json.loads(text[min(starts):])

    if isinstance(payload, list):
        if not payload:
            raise RuntimeError(f"quarterly_results returned empty array for {symbol}")
        target = symbol.strip().upper()
        match = next(
            (
                p
                for p in payload
                if isinstance(p, dict) and str(p.get("symbol", "")).upper() == target
            ),
            payload[0],
        )
        return FundamentalsPayload(symbol=symbol.upper(), data=match)
    if not isinstance(payload, dict):
        raise RuntimeError(f"quarterly_results payload for {symbol} is not a JSON object or array")
    return FundamentalsPayload(symbol=symbol.upper(), data=payload)
