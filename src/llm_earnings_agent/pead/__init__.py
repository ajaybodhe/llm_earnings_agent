"""Post-earnings-announcement drift (PEAD) research.

Works on the point-in-time event dataset built by quarterly_results
(`go run . --build-dataset …`): one JSON line per historical earnings release
with features known at the reaction-day close and drift labels after it.
"""
