# llm-earnings-agent

LLM-agent earnings reaction analyzer. MarketSenseAI 2.0–style multi-agent pipeline
that combines fundamentals, earnings call transcripts, and news to emit a
Positive / Negative / Neutral rating for an upcoming earnings report.

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

Three sub-agents + a deterministic aggregator:

| Agent | Inputs | Output |
|---|---|---|
| `fundamentals` | Quarterly actuals, valuation ratios, peer medians (from `quarterly_results --output json`) | score / confidence / themes |
| `transcript` | Earnings call transcript (Discounting Cash Flows) | score / sentiment / guidance direction |
| `news` | Finnhub headlines + SEC 8-K filings | score / polarity / material event count |
| `aggregator` | Three sub-agent outputs | label / score / confidence / top reasons |

The aggregator is deterministic (`Σ weight × score × confidence`) by default. Set
`use_llm=True` to ask the model to override on tight calls.

## Runtime modes

| Runtime | Pricing | Latency | Where used |
|---|---|---|---|
| `ClaudeCodeRuntime` | Subscription (free marginal) | Slow (subprocess) | Dev / single ticker |
| `AnthropicAPIRuntime` | Per-token (~$0.02–0.05/ticker) | Fast (HTTP) | Backtests / bulk |

Auto-selected: `LLM_RUNTIME=api|claude-code` overrides; otherwise picks API when
`ANTHROPIC_API_KEY` is set, else Claude Code headless.

## Caveats

MarketSenseAI 2.0's published 8.0–18.9% excess returns are unreplicated and
based on a short 2023–2024 window. LLM training cutoffs introduce look-ahead bias
on backtested tickers. Treat results as a hypothesis, not a strategy.
