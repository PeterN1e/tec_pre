"""Backward-compatible entry point for the unified training pipeline.

Use ``python -m scripts.train --model GA_Predrnn`` for the configurable CLI.
"""

from __future__ import annotations

from core.config import load_config
from core.trainer import run_training


def main() -> int:
    result = run_training(load_config("GA_Predrnn"))
    print(f"checkpoint: {result['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
