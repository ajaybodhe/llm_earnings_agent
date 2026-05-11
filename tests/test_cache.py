from pathlib import Path

from llm_earnings_agent.cache import JsonCache, cache_key


def test_cache_key_stable():
    k1 = cache_key(ticker="AAPL", agent="news", prompt_version="v1", model="m", payload={"a": 1, "b": 2})
    k2 = cache_key(ticker="aapl", agent="news", prompt_version="v1", model="m", payload={"b": 2, "a": 1})
    assert k1 == k2  # ticker uppercased, dict key order ignored


def test_cache_key_changes_on_version():
    k1 = cache_key(ticker="X", agent="a", prompt_version="v1", model="m", payload={})
    k2 = cache_key(ticker="X", agent="a", prompt_version="v2", model="m", payload={})
    assert k1 != k2


def test_roundtrip(tmp_path: Path):
    cache = JsonCache(root=tmp_path)
    assert cache.get("missing") is None
    cache.put("abcd1234", {"hello": "world"})
    assert cache.get("abcd1234") == {"hello": "world"}
    assert cache.delete("abcd1234") is True
    assert cache.get("abcd1234") is None
    assert cache.delete("abcd1234") is False
