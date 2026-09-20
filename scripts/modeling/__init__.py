"""November Empirical Benchmark: the four-model test of the 5 cover-design claims.

Modules:
  assemble.py  -> data/model_frame.parquet : one row per cover, outcomes + controls +
                  the 5 claim features, ready for statsmodels.

Everything downstream (the A/B/C/D models, the nested A-vs-C test) consumes model_frame.parquet
and the author-grouped split defined alongside it. See docs/pipeline.md for the feature
provenance and build-log.md (November section) for the methodology decisions.
"""
