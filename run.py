#!/usr/bin/env python3
"""
Zero-install entry point.

Lets anyone try the classifier immediately after cloning, with no pip
install or virtual environment required:

    python3 run.py

Accepts the same flags as the installed `cancer-dx` command, e.g.:

    python3 run.py --he4 180 --afp 6 --ca19-9 22 --name Alice
    python3 run.py --top 3

For the full experience (the `cancer-dx` command itself, running the
test suite, using cancer_dx as a library) see "Full install" in
README.md.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cancer_dx.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())