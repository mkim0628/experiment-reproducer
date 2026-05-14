"""Pytest path setup so that `cacheblend` and `eval` are importable from tests.

We deliberately insert this directory at the front of sys.path so the local
``cacheblend`` and ``eval`` packages take precedence regardless of the cwd that
``pytest`` is invoked from.
"""
from __future__ import annotations

import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
