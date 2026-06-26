"""Root pytest conftest.

Puts the repository root (for `import src.*`) and the Experiment-1 directory
(for `import config`, `import trajectory`, ...) on `sys.path` so the two test
suites in `pyproject.toml::testpaths` resolve their imports without an install.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.resolve()
_EXP1 = _ROOT / "experiments" / "2026-05-16_self_prediction"

for _p in (_ROOT, _EXP1):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
