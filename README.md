# llm-earnings-agent

LLM-agent earnings reaction analyzer. MarketSenseAI 2.0–style multi-agent pipeline
that combines fundamentals, earnings call transcripts, news, sector/macro context,
and price-action setup to emit a Positive / Negative / Neutral rating for an
upcoming (or just-released) earnings report.

Sibling project to `quarterly_results/` (Go). Invokable standalone or from
`quarterly_results` via `--llm-rating`.

## Install

```bash
cd llm_earnings_agent
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
```

## Usage

```bash
# Single ticker
.venv/bin/llm-earnings-agent analyze --symbol AAPL --output json

# Force API runtime (requires ANTHROPIC_API_KEY)
.venv/bin/llm-earnings-agent analyze --symbol AAPL --runtime api

# Walk-forward backtest
.venv/bin/llm-earnings-agent backtest --symbol AAPL --quarters 4
```

## Architecture

### High level

```
                ┌─────────────────────────┐
                │   data fetch (parallel) │
                │  asyncio.gather(...)    │
                ├─────────────────────────┤
                │ • quarterly_results (Go)│  ← fundamentals, sector ETF + returns,
                │                         │    earnings reactions, options, etc.
                │ • Alpha Vantage         │  ← transcript (primary)
                │ • SEC EDGAR 8-K         │  ← transcript fallback
                │ • Finnhub company-news  │  ← headlines
                │ • SEC EDGAR (8-K)       │  ← 8-K material events
                └─────────────────────────┘
                            │
                            ▼ slice payload per agent
   ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
   │ fundamentals │  transcript  │     news     │    macro     │   dynamic    │
   │   (Claude)   │   (Claude)   │   (Claude)   │   (Claude)   │   (Claude)   │
   └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
                            │
                            ▼ five typed sub-analyses
                ┌─────────────────────────┐
                │   aggregator            │
                │   deterministic by      │  Σ wᵢ · scoreᵢ · confᵢ → label, top reasons
                │   default; LLM optional │
                └─────────────────────────┘
                            │
                            ▼
                       AgentResponse JSON
```

### The five specialist agents

Each agent receives a **narrow slice** of the data fetch (defined in
`pipeline.py:43-68`), runs once against the LLM, and returns a strictly typed
Pydantic model (`schemas.py`).

| Agent | Inputs (sliced) | Output schema | Prompt |
|---|---|---|---|
| `fundamentals` | Full Go payload: revenue/EPS YoY, valuation ratios, peer medians, analyst PTs, beat history | `score, confidence, themes[≤5], reasoning` | `prompts/fundamentals.md` |
| `transcript` | Latest earnings-call transcript text + (quarter, year) — Alpha Vantage primary (free tier: 25 req/day, 1 req/sec), SEC EDGAR 8-K exhibit fallback | `score, confidence, sentiment, guidance_direction, themes, management_tone, reasoning` | `prompts/transcript.md` |
| `news` | Finnhub headlines (last 30d) + SEC 8-K material events | `score, confidence, material_dev_count, polarity, reasoning` | `prompts/news.md` |
| `macro` | `earnings_date`, `macro_context` (FOMC/CPI/NFP near date), `sector_etf`, `sector_ret_1m`, `sector_ret_3m`, `currency` | `score, confidence, regime, sector_trend, reasoning` | `prompts/macro.md` |
| `dynamic` | `current_price`, `ret_{1w,1m,6m,1y}`, `rsi14`, `hi_52`/`lo_52` and pct-from, `beat_rate`, `avg_beat_pct`, `implied_vs_hist_ratio`, `earnings_reactions[]` | `score, confidence, momentum, overbought_oversold, reasoning` | `prompts/dynamic.md` |

All five score on the same `[-1, +1]` axis and report a `confidence ∈ [0, 1]`.

#### `macro` agent

Judges the rate / inflation / sector tape into which the report will land. It
asks: *is the backdrop friendly or hostile for this specific stock right now?*

- **Sector momentum** — the sector ETF's 1M and 3M return. Persistent positive
  ⇒ supportive; persistent negative ⇒ hostile. Sector ETF is mapped from the
  company's SIC code by `quarterly_results/sector_momentum.go` (e.g. AAPL → XLK).
- **Scheduled macro events** — FOMC / CPI / NFP / ECB / BoE within ±2 days of
  the earnings date. A hot CPI on the same week can drown out the earnings
  signal regardless of the print.
- **Regime characterisation** — `risk-on | risk-off | mixed | neutral`,
  derived from the combination of the above.
- Confidence falls below 0.3 when both ETF returns are absent and there are no
  scheduled events.

#### `dynamic` agent

Judges *how the tape is positioned* into the report — not the long-run
direction. It asks: *do recent price action and prior reactions argue for a
positive or negative move on this print?*

- **Momentum** — alignment of 1M and 6M returns. Both positive = real edge.
- **52-week position + RSI(14)** — near highs with RSI > 70 = stretched
  (exhaustion risk); near lows with RSI < 30 = oversold (mean-reversion bid).
- **Beat history** — a 4/4 beat record with average beat > 5% tends to
  continue, but the bar is high and the market may already price it in.
- **Prior earnings reactions** — if the last 4 reactions are all positive, a
  miss hurts more. Mixed reactions = lower predictability.
- **Implied vs historical move ratio** > 1.3x = options pricing extra
  uncertainty into the print.

### How agents communicate

**They don't talk to each other.** The pipeline is a pure fan-out / fan-in:

1. The orchestrator (`pipeline.analyze_symbol`) fans out three independent
   data fetches in parallel via `asyncio.gather` (Go subprocess, transcript
   HTTP, news + 8-K HTTP).
2. After the fetch completes, the orchestrator slices the Go payload into
   per-agent inputs and invokes each specialist agent.
3. Each agent returns a typed Pydantic model. The orchestrator collects the
   five sub-analyses and hands them to the aggregator.
4. The aggregator (`agents/aggregator.py:_compute_rating`) does
   `Σ wᵢ · scoreᵢ · confᵢ` deterministically in Python and emits the final
   `Rating`. The exact weights are in code (currently fundamentals 0.30,
   transcript 0.25, dynamic 0.20, news 0.15, macro 0.10).

So "communication" is actually *orchestration*: the pipeline owns all state;
agents are stateless functions of (system_prompt, user_payload) → typed JSON.

### When the LLM is called — and when it isn't

| Step | LLM? | Notes |
|---|---|---|
| Fetch fundamentals (`go run quarterly_results`) | No | Pure subprocess, ~5–60s per ticker |
| Fetch transcript (Alpha Vantage → SEC EDGAR) | No | AV walk-back (25/day, 1s throttle, quota-bail). Disk-backed 7-day negative cache so re-runs skip known-empty quarters. On AV miss or quota: fall back to scanning recent 8-K Item 2.02 exhibits on EDGAR (free, no rate limit). EDGAR hit rate is low in practice (<10% for S&P 500 — most companies file only the press release, not prepared remarks) but it's harmless to try and costs ~2 HTTP calls per filing. |
| Fetch news + 8-K events (Finnhub, SEC EDGAR) | No | Plain HTTP |
| Each specialist agent (5 of them) | **Yes** | One LLM call per agent per ticker per cache miss |
| Cache hit on any agent | No | SHA1 of `(ticker, agent, prompt_version, model, payload)` reads from disk |
| Aggregator (default `use_llm=False`) | No | Deterministic weighted blend in pure Python |
| Aggregator with `--llm-aggregator` | Yes | One extra LLM call to optionally rewrite top_reasons / override label on tight calls |

So a **fresh** run of one ticker is 5 LLM calls (or 6 with `--llm-aggregator`);
a **fully cached** run is 0.

### What is passed to Claude

Each specialist agent is a single-shot, structured-output call. The runtime
constructs:

- **system** = full text of `prompts/{agent}.md` (with the
  `<!-- prompt_version: vN -->` comment stripped). Defines the role, weighing
  rules, score scale, confidence policy, and length limit.
- **user** = `f"Symbol: {SYM}\n\n{agent_specific_label}:\n{json.dumps(slice)}"`
  — the narrow JSON slice for that agent.
- An appended request to *"respond with a single JSON object that matches this
  JSON Schema exactly. No prose, no code fences, just the JSON object."*
  followed by `schema.model_json_schema()` derived from the Pydantic class.

The runtime then JSON-parses the response (stripping ```json fences if the
model wrapped them) and `schema.model_validate(...)`-s it. A schema mismatch
raises and the agent's sub-rating becomes `None` for that run.

### Two runtime modes

`runtime/base.py` defines a `Runtime` protocol with one method:
`async complete(system, user, schema, model) → CompletionResult`.

| Runtime | Pricing | Latency | Where used |
|---|---|---|---|
| `ClaudeCodeRuntime` (`claude_code.py`) | Subscription (free marginal) | Slow — spawns `claude -p` subprocess per call | Dev / single ticker; default when no API key |
| `AnthropicAPIRuntime` (`api.py`) | Per-token (~$0.02–0.05 / ticker on Sonnet) | Fast (HTTPS) | Backtests / bulk; supports concurrent requests |

Auto-selected by `auto_select_runtime()`: `LLM_RUNTIME=api|claude-code`
overrides; otherwise picks API when `ANTHROPIC_API_KEY` is set, else Claude
Code headless.

### Caching

`cache.JsonCache` writes one JSON file per agent invocation under
`data/cache/{first2-of-key}/{key}.json`. The key is
`sha1(ticker + agent + prompt_version + model + payload)`, so:

- Bumping the `<!-- prompt_version: vN -->` comment in any prompt forces a
  refresh for that agent only.
- Changing the input payload (e.g. fresh news headlines) invalidates the
  affected agent automatically.
- Disable with `--no-cache`.

### Can we parallelise the agents?

**Data fetch is already parallel** (`pipeline.py:132`,
`asyncio.gather(_fundamentals(), _transcript(), _news())`).

**The five LLM calls are currently sequential** — `pipeline.py:135-193` awaits
each agent in turn. This is the obvious next optimisation:

```python
fund_t = asyncio.create_task(analyze_fundamentals(...))
tx_t   = asyncio.create_task(analyze_transcript(...))    if transcript_obj else None
news_t = asyncio.create_task(analyze_news(...))          if (headlines or events) else None
macro_t = asyncio.create_task(analyze_macro(...))        if _has_signal(macro_slice) else None
dyn_t  = asyncio.create_task(analyze_dynamic(...))       if _has_signal(dynamic_slice) else None
fund_result, tx_result, news_result, macro_result, dynamic_result = await asyncio.gather(
    fund_t, tx_t or _none(), news_t or _none(), macro_t or _none(), dyn_t or _none(),
    return_exceptions=True,
)
```

Caveats by runtime:

- **`AnthropicAPIRuntime`**: parallel-friendly. Anthropic supports concurrent
  requests up to your account RPM/TPM. Wall-clock drops from `~5×T` to
  `~max(T)`. Total token cost is unchanged.
- **`ClaudeCodeRuntime`**: each call is its own `claude -p` subprocess. Five
  concurrent subprocesses are fine on the OS, but they hit one Claude session
  and may bump into session rate limits / queueing. Worth a `--max-parallel-agents`
  knob (default 5 on API, 1–2 on Claude Code).

Per-ticker sub-agent results are pure functions of disjoint inputs, so there
is no correctness hazard — only resource contention. A small concurrency
limiter (`asyncio.Semaphore`) is enough.

The aggregator must remain serial (it consumes all five sub-analyses); the
`bulk` command already runs ticker-by-ticker, but ticker-level concurrency is
also straightforward to add on top of the same semaphore.

## Caveats

MarketSenseAI 2.0's published 8.0–18.9% excess returns are unreplicated and
based on a short 2023–2024 window. LLM training cutoffs introduce look-ahead
bias on backtested tickers. Treat results as a hypothesis, not a strategy.
