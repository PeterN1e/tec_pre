from __future__ import annotations

import argparse
import json

from core.config import load_config
from core.inference import predict_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a TEC checkpoint.")
    parser.add_argument("--model", default="GA_Predrnn", help="Model preset name")
    parser.add_argument("--config", default=None, help="Base YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override in dotted.key=value form",
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(
        model_name=args.model,
        config_path=args.config,
        overrides=args.overrides,
    )
    result = predict_split(
        config,
        split=args.split,
        checkpoint=args.checkpoint,
    )
    print(json.dumps(result["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

