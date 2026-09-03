import sys
from pathlib import Path

# Allow `import cancer_dx` in tests without installing the package first.
SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))