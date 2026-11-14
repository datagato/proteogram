"""Shared pytest configuration.

The pipeline entry points under scripts/ are scripts, not an installed
package, so importing their helpers in a test needs their directories on
sys.path. Every script reads config.yml inside an `if __name__ == '__main__'`
guard, so importing one has no side effects beyond its own imports.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

for _script_dir in (REPO_ROOT / 'scripts' / 'v2',
                    REPO_ROOT / 'scripts' / 'utilities'):
    if str(_script_dir) not in sys.path:
        sys.path.insert(0, str(_script_dir))
