"""``python -m cacheblend --smoke`` entry point.

Delegates to ``scripts.run_smoke`` so the same code is exercised whether the
user invokes the script directly or as a package module.
"""
from __future__ import annotations

import sys
import pathlib

_CODE_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from scripts.run_smoke import main  # noqa: E402

if __name__ == "__main__":
    main()
