"""Shared NLP runtime compatibility shims."""
from __future__ import annotations

import sys


def patch_torch_jit_for_py313() -> None:
    """Avoid transformers DebertaV2 @torch.jit.script parse failure on Python 3.13."""
    if sys.version_info < (3, 13):
        return
    import torch

    torch.jit.script = lambda fn, *args, **kwargs: fn  # type: ignore[assignment]
    torch.jit.trace = lambda fn, *args, **kwargs: fn  # type: ignore[assignment]
