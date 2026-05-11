"""Pydantic models for agent outputs and the top-level JSON contract.

The JSON contract is what `quarterly_results` consumes via subprocess. Keep it
stable; bump prompt versions if you change agent prompts, but only change the
schema with a coordinated update on the Go side.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Label = Literal["Positive", "Negative", "Neutral"]
Polarity = Literal["positive", "negative", "neutral", "mixed"]
GuidanceDirection = Literal["raised", "lowered", "reaffirmed", "withdrawn", "none"]


class FundamentalsAnalysis(BaseModel):
    """Output of the fundamentals agent.

    Score in [-1, +1] matches the Go SignalResult convention.
    """

    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    themes: list[str] = Field(default_factory=list, max_length=5)
    reasoning: str


class TranscriptAnalysis(BaseModel):
    """Output of the transcript agent."""

    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    sentiment: Polarity
    guidance_direction: GuidanceDirection
    themes: list[str] = Field(default_factory=list, max_length=5)
    management_tone: str
    reasoning: str


class NewsAnalysis(BaseModel):
    """Output of the news agent."""

    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    material_dev_count: int = Field(ge=0)
    polarity: Polarity
    reasoning: str


class MacroAnalysis(BaseModel):
    """Output of the macroeconomic agent.

    Assesses the rate/inflation regime and sector-tape backdrop in which the
    earnings report will land.
    """

    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    regime: Literal["risk-on", "risk-off", "mixed", "neutral"]
    sector_trend: Polarity
    reasoning: str


class DynamicAnalysis(BaseModel):
    """Output of the dynamic / price-action agent.

    Reads recent returns, 52-week position, RSI, beat history and prior
    earnings reactions to gauge how the tape is positioned into the report.
    """

    score: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    momentum: Polarity
    overbought_oversold: Literal["overbought", "oversold", "neutral"]
    reasoning: str


class Rating(BaseModel):
    """Final aggregated rating. Mirrors `Recommendation` in `quarterly_results/score.go`."""

    label: Label
    score: float = Field(ge=-100.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    top_reasons: list[str] = Field(default_factory=list, max_length=5)


class SubRatings(BaseModel):
    fundamentals: FundamentalsAnalysis | None = None
    transcript: TranscriptAnalysis | None = None
    news: NewsAnalysis | None = None
    macro: MacroAnalysis | None = None
    dynamic: DynamicAnalysis | None = None


class Metadata(BaseModel):
    model: str
    runtime: str
    prompt_version: str
    timestamp: datetime
    cost_estimate_usd: float = Field(ge=0.0)
    transcript_quarter: str | None = None


class AgentResponse(BaseModel):
    """Top-level JSON contract emitted by `llm-earnings-agent analyze`."""

    symbol: str
    asof: date
    rating: Rating
    sub_ratings: SubRatings
    metadata: Metadata

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class BacktestPoint(BaseModel):
    """One reaction in a walk-forward backtest."""

    symbol: str
    period: str
    announcement_date: date
    predicted: Label
    predicted_score: float
    confidence: float
    actual_label: Label
    actual_ret_pct: float
    hit: bool


class BacktestSummary(BaseModel):
    """Aggregate result for `llm-earnings-agent backtest`."""

    symbol: str
    total: int
    directional: int
    hits: int
    hit_rate: float = Field(ge=0.0, le=1.0)
    baseline: float = Field(ge=0.0, le=1.0)
    avg_ret_when_positive: float
    avg_ret_when_negative: float
    avg_ret_when_neutral: float
    points: list[BacktestPoint]
