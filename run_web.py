#!/usr/bin/env python3
"""
Zero-install entry point for the web demo.

    python3 run_web.py

Then open http://127.0.0.1:8000 in a browser. Uses only the Python
standard library — no pip install required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cancer_dx.web.server import run  # noqa: E402

if __name__ == "__main__":
    run()