"""Entry point: ``python -m crystalline`` (or the ``crystalline`` console script)."""

from __future__ import annotations

import sys


def main() -> int:
    from crystalline.app import run

    return run()


if __name__ == "__main__":
    sys.exit(main())
