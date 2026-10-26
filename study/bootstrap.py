"""Import the pinned vendor without editing its tracked files."""
from pathlib import Path
import sys
ROOT = Path(__file__).absolute().parents[1]
VENDOR = ROOT/'vendor/TinyRecursiveModels'
sys.path.insert(0, str(VENDOR))
import models
models.__path__.append(str(ROOT/'study'))
