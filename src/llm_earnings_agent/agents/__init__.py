"""Agents — one module per sub-analyst plus the aggregator."""

from .aggregator import aggregate
from .dynamic import analyze_dynamic
from .fundamentals import analyze_fundamentals
from .macro import analyze_macro
from .news import analyze_news
from .transcript import analyze_transcript

__all__ = [
    "aggregate",
    "analyze_dynamic",
    "analyze_fundamentals",
    "analyze_macro",
    "analyze_news",
    "analyze_transcript",
]
