from __future__ import annotations

import argparse
import json

from core.config import load_config
from core.inference import predict_split
from core.trainer import run_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TEC forecasting model.")
    parser.add_argument("--model", default="GA_Predrnn", help="Model preset name")
    parser.add_argument("--config", default=None, help="Base YAML config path")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Config override in dotted.key=value form",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Checkpoint used to initialize model and optimizer state",
    )
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the test split after training",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(
        model_name=args.model,
        config_path=args.config,
        overrides=args.overrides,
    )
    result = run_training(config, resume_from=args.resume)
    output = {
        "model_dir": result["model_dir"],
        "checkpoint": result["checkpoint"],
    }
    if args.evaluate_test:
        evaluation = predict_split(config, split="test")
        output["test_metrics"] = evaluation["metrics"]["aggregate"]
    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

