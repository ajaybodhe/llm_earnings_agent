"""CLI entry point — invoked as `llm-earnings-agent`."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Annotated

import typer

from .cache import JsonCache
from .pipeline import analyze_symbol
from .runtime.base import Runtime, auto_select_runtime

app = typer.Typer(add_completion=False, help="LLM-agent earnings reaction analyzer.")


def _runtime_for(name: str | None) -> Runtime:
    if name == "api":
        from .runtime.api import AnthropicAPIRuntime

        return AnthropicAPIRuntime()
    if name == "claude-code":
        from .runtime.claude_code import ClaudeCodeRuntime

        return ClaudeCodeRuntime()
    return auto_select_runtime()


@app.command()
def analyze(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Ticker symbol")],
    runtime: Annotated[str | None, typer.Option("--runtime", help="api | claude-code")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model id override")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="json | jsonl")] = "json",
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Disable filesystem cache")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    use_llm_aggregator: Annotated[bool, typer.Option("--llm-aggregator", help="Run aggregator agent via LLM")] = False,
) -> None:
    """Analyze a single ticker and emit JSON."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    rt = _runtime_for(runtime)
    cache = None if no_cache else JsonCache()

    response = asyncio.run(
        analyze_symbol(
            symbol,
            runtime=rt,
            model=model,
            cache=cache,
            use_llm_aggregator=use_llm_aggregator,
        )
    )

    serialized = response.model_dump_json(indent=None if output == "jsonl" else 2)
    sys.stdout.write(serialized + "\n")


@app.command()
def backtest(
    symbol: Annotated[str, typer.Option("--symbol", "-s", help="Ticker symbol")],
    quarters: Annotated[int, typer.Option("--quarters", "-q", help="Number of past quarters")] = 4,
    runtime: Annotated[str | None, typer.Option("--runtime", help="api | claude-code")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model id override")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Walk-forward backtest across recent earnings reactions."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s")
    from .backtest import run_backtest

    rt = _runtime_for(runtime)
    summary = asyncio.run(run_backtest(symbol, quarters=quarters, runtime=rt, model=model))
    sys.stdout.write(summary.model_dump_json(indent=2) + "\n")


@app.command()
def bulk(
    runtime: Annotated[str | None, typer.Option("--runtime", help="api | claude-code")] = None,
    model: Annotated[str | None, typer.Option("--model", help="Model id override")] = None,
) -> None:
    """Read tickers from stdin (one per line), emit JSONL to stdout."""
    rt = _runtime_for(runtime)
    cache = JsonCache()

    async def _run_all() -> None:
        for line in sys.stdin:
            sym = line.strip()
            if not sym:
                continue
            try:
                resp = await analyze_symbol(sym, runtime=rt, model=model, cache=cache)
                sys.stdout.write(resp.model_dump_json() + "\n")
                sys.stdout.flush()
            except Exception as e:  # noqa: BLE001
                sys.stderr.write(f"failed {sym}: {e}\n")

    asyncio.run(_run_all())


if __name__ == "__main__":
    app()
