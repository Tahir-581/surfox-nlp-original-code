#!/usr/bin/env python3
"""Pre-download local models used by legacy scripts (MiniLM). Remote NLP models are not cached here."""
import os
import sys


def main() -> int:
    model_id = os.getenv(
        "SENTENCE_TRANSFORMER_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    print(f"Downloading SentenceTransformer: {model_id}")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed. Install backend requirements first.", file=sys.stderr)
        return 1

    SentenceTransformer(model_id)
    print(f"Cached successfully: {model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
