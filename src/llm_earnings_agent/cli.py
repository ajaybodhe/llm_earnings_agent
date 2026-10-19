"""CLI entry point — invoked as `llm-earnings-agent`."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

from .cache import JsonCache
from .data.fundamentals import parse_payload
from .pipeline import analyze_symbol, parse_agents
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
    use_llm_aggregator: Annotated[
        bool, typer.Option("--llm-aggregator", help="Run aggregator agent via LLM")
    ] = False,
    agents: Annotated[
        str,
        typer.Option(
            "--agents",
            help="Comma-separated subset of fundamentals,transcript,news,macro,dynamic, or 'all'",
        ),
    ] = "all",
    payload: Annotated[
        str | None,
        typer.Option(
            "--payload",
            help="quarterly_results JSON for this symbol (file path, or '-' for stdin); "
            "skips re-running the Go binary",
        ),
    ] = None,
) -> None:
    """Analyze a single ticker and emit JSON."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    try:
        selected = parse_agents(agents)
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--agents") from None
    fundamentals_payload = None
    if payload is not None:
        text = sys.stdin.read() if payload == "-" else Path(payload).read_text(encoding="utf-8")
        fundamentals_payload = parse_payload(text, symbol).data
    rt = _runtime_for(runtime)
    cache = None if no_cache else JsonCache()

    response = asyncio.run(
        analyze_symbol(
            symbol,
            runtime=rt,
            model=model,
            cache=cache,
            use_llm_aggregator=use_llm_aggregator,
            agents=selected,
            fundamentals_payload=fundamentals_payload,
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
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    from .backtest import run_backtest

    rt = _runtime_for(runtime)
    summary = asyncio.run(run_backtest(symbol, quarters=quarters, runtime=rt, model=model))
    sys.stdout.write(summary.model_dump_json(indent=2) + "\n")


pead_app = typer.Typer(help="Post-earnings-drift research on the quarterly_results event dataset.")
app.add_typer(pead_app, name="pead")


@pead_app.command("text")
def pead_text(
    events: Annotated[
        Path, typer.Option("--events", help="Event JSONL from `go run . --build-dataset`")
    ],
    out: Annotated[Path, typer.Option("--out", help="Output dir (texts/ cache + features)")] = Path(
        "data/pead"
    ),
    lm_dictionary: Annotated[
        Path | None,
        typer.Option(
            "--lm-dictionary",
            help="Loughran-McDonald Master Dictionary CSV (optional; licence-restricted)",
        ),
    ] = None,
    embed: Annotated[
        bool, typer.Option("--embed/--no-embed", help="Local fastembed embeddings")
    ] = True,
    model: Annotated[str, typer.Option("--model", help="fastembed model")] = (
        "BAAI/bge-small-en-v1.5"
    ),
    rps: Annotated[
        float, typer.Option("--rps", help="SEC requests per second (limit is 10)")
    ] = 8.0,
    concurrency: Annotated[
        int, typer.Option("--concurrency", help="Requests in flight (rate stays capped by --rps)")
    ] = 12,
    fetch_only: Annotated[
        bool, typer.Option("--fetch-only", help="Only download press releases")
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Fetch each event's earnings press release from SEC and build text features."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    from .pead.events import load_events
    from .pead.text import (
        build_text_features,
        fastembed_embedder,
        fetch_release_texts,
        load_lm_dictionary,
    )

    evs = load_events(events)
    out.mkdir(parents=True, exist_ok=True)
    stats = asyncio.run(fetch_release_texts(evs, out, rps=rps, concurrency=concurrency))
    sys.stderr.write(f"press releases: {stats}\n")
    if fetch_only:
        return
    lm = load_lm_dictionary(lm_dictionary) if lm_dictionary else None
    embedder = fastembed_embedder(model) if embed else None
    result = build_text_features(evs, out, lm=lm, embed=embedder, model_name=model)
    sys.stderr.write(f"text features: {result} → {out}\n")


@pead_app.command("gate")
def pead_gate(
    events: Annotated[
        Path, typer.Option("--events", help="Event JSONL from `go run . --build-dataset`")
    ],
    by_year_target: Annotated[
        str, typer.Option("--by-year-target", help="Target for the per-year breakdown")
    ] = "drift20_adj",
) -> None:
    """Quarterly IC and decile-spread tests: does any simple signal predict drift?"""
    from .pead.events import load_events
    from .pead.gate import SIGNALS, by_year, format_gate, run_gate

    evs = load_events(events)
    sys.stdout.write(f"{len(evs)} events\n\n{format_gate(run_gate(evs))}\n")
    sys.stdout.write(f"\nBy year — {by_year_target}, decile spread (t):\n")
    for name in ("sue (winsorised ±5)", "reaction_ret_pct", "rules_score"):
        cells = " ".join(
            f"{y}:{r.spread:+.2f}({r.spread_t:+.1f})"
            for y, r in by_year(evs, SIGNALS[name], by_year_target)
            if r.quarters
        )
        sys.stdout.write(f"  {name:<22} {cells}\n")


@pead_app.command("model")
def pead_model(
    events: Annotated[
        Path, typer.Option("--events", help="Event JSONL from `go run . --build-dataset`")
    ],
    tasks: Annotated[
        str,
        typer.Option(
            "--tasks", help="Comma-separated: drift20,drift5,reaction,intraday,intraday_mkt"
        ),
    ] = "drift20,drift5,reaction,intraday,intraday_mkt",
    text_dir: Annotated[
        Path | None,
        typer.Option("--text", help="Dir with embeddings.npy from `pead text` (adds text PCA)"),
    ] = None,
    components: Annotated[int, typer.Option("--components", help="Text PCA components")] = 16,
    kind: Annotated[str, typer.Option("--model", help="gbdt | ridge")] = "gbdt",
) -> None:
    """Walk-forward GBDT per task; compares against single-feature baselines."""
    from .pead.events import load_events
    from .pead.model import TASKS, format_results, walk_forward

    evs = load_events(events)
    from .pead.context import add_context, fetch_daily

    add_context(evs, fetch_daily("SPY"), fetch_daily("^VIX"))
    extra_fn = None
    if text_dir is not None:
        from .pead.textpca import TextPCA

        extra_fn = TextPCA(text_dir, components)
    results = [
        walk_forward(evs, TASKS[t.strip()], extra_fn=extra_fn, kind=kind)[0]
        for t in tasks.split(",")
        if t.strip()
    ]
    label = f" + text PCA({components})" if text_dir else ""
    sys.stdout.write(f"{len(evs)} events{label}\n{format_results(results)}\n")


@pead_app.command("live")
def pead_live(
    events: Annotated[
        Path, typer.Option("--events", help="Historical event JSONL (training/calibration)")
    ],
    payload: Annotated[
        Path,
        typer.Option(
            "--payload", help="quarterly_results --output json for recent releases ('-' = stdin)"
        ),
    ],
    output: Annotated[str, typer.Option("--output", "-o", help="table | json")] = "table",
) -> None:
    """Signal 1 (at the release) and Signal 2 (at the open) for recent releases."""
    import json as _json

    from .pead.context import add_context, fetch_daily
    from .pead.events import load_events
    from .pead.live import events_from_results, report, signals

    history = load_events(events)
    raw = sys.stdin.read() if str(payload) == "-" else payload.read_text(encoding="utf-8")
    new = events_from_results(_json.loads(raw))
    if not new:
        if output == "json":
            sys.stdout.write("[]\n")
            return
        sys.stderr.write("no released reports with a consensus surprise in the payload\n")
        raise typer.Exit(1)
    add_context([*history, *new], fetch_daily("SPY"), fetch_daily("^VIX"))
    if output == "json":
        sys.stdout.write(_json.dumps(signals(history, new)) + "\n")
    else:
        sys.stdout.write(report(history, [e for e in new if e["has_open"]]) + "\n")


@pead_app.command("suetxt")
def pead_suetxt(
    events: Annotated[Path, typer.Option("--events", help="Event JSONL")],
    data: Annotated[Path, typer.Option("--data", help="pead data dir (texts/, prices/)")] = Path(
        "data/pead"
    ),
    refresh: Annotated[bool, typer.Option("--refresh", help="Re-estimate SUE.txt")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Replicate PEAD.txt (SUE.txt) on press releases and test drift + earnings-day use."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    from .pead import suetxt
    from .pead.context import add_context, fetch_daily
    from .pead.events import load_events
    from .pead.longdrift import add_long_drift
    from .pead.suetxt_eval import combined_models, report
    from .pead.text import text_path

    evs = load_events(events)
    spy = fetch_daily("SPY")
    add_context(evs, spy, fetch_daily("^VIX"))
    out = data / "sue_txt.jsonl"
    if refresh or not out.exists():
        spy_close = {d: c for d, (_, c) in spy.items()}
        dates = sorted(spy_close)
        texts, ar = {}, {}
        for e in evs:
            acc = e.get("accession")
            if acc and text_path(data, acc).exists():
                texts[e["event_id"]] = text_path(data, acc).read_text(encoding="utf-8")
            a = suetxt.abnormal_reaction(e, spy_close, dates)
            if a is not None:
                ar[e["event_id"]] = a
        sue = suetxt.rolling_sue_txt(evs, texts, ar, progress=lambda m: sys.stderr.write(m + "\n"))
        suetxt.save(out, sue)
    sue = suetxt.load(out)
    for e in evs:
        if e["event_id"] in sue:
            e["features"]["sue_txt"] = sue[e["event_id"]]
    add_long_drift(evs, data / "prices")
    sys.stdout.write(f"{len(sue)} events with out-of-sample SUE.txt\n\n")
    sys.stdout.write(report(evs) + "\n\n" + combined_models(evs) + "\n")


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
