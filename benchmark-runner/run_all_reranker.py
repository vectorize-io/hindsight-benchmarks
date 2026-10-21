#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "sentence-transformers==5.7.0",
#   "torch==2.14.1",
#   "transformers==5.18.0",
#   "numpy==2.5.3",
#   "rank-bm25==0.2.2",
#   "httpx==0.28.1",
#   "toktok-rs==0.1.3",
# ]
# ///
"""Direct reranking evaluation; use `uv run --no-project --script run_all_reranker.py`."""

from rerank_eval import main

if __name__ == "__main__":
    main()
