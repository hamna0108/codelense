import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from main import app  # noqa: E402