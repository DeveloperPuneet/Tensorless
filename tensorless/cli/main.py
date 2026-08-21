"""Command-line interface.

    tensorless train ./data
    tensorless run model.tl
    tensorless inspect ./data
    tensorless info model.tl
"""

from __future__ import annotations

import argparse
import json
import sys

from .. import api
from ..errors import TensorlessError
from ..serialization.tl_format import load_tl


def _add_train_parser(subparsers) -> None:
    p = subparsers.add_parser("train", help="Train a model on a dataset (fully automatic by default)")
    p.add_argument("path", help="Path to a dataset file or directory")
    p.add_argument("--out", default=None, help="Output .tl file path (default: model.tl)")
    p.add_argument("--force", action="store_true", help="Force retraining even if a matching model exists")
    p.add_argument("--d-model", type=int, default=None, dest="d_model")
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None, dest="learning_rate")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "tpu", "mps"])
    p.add_argument("--quiet", action="store_true", help="Suppress training logs")


def _add_run_parser(subparsers) -> None:
    p = subparsers.add_parser("run", help="Run a trained .tl model")
    p.add_argument("path", help="Path to a .tl model file")
    p.add_argument("--prompt", default=None, help="Input text/prompt (skips interactive chat)")


def _add_inspect_parser(subparsers) -> None:
    p = subparsers.add_parser("inspect", help="Inspect a dataset without training")
    p.add_argument("path", help="Path to a dataset file or directory")


def _add_info_parser(subparsers) -> None:
    p = subparsers.add_parser("info", help="Show information about a trained .tl model")
    p.add_argument("path", help="Path to a .tl model file")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tensorless", description="Tensorless: ML with maximum automation.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    _add_run_parser(subparsers)
    _add_inspect_parser(subparsers)
    _add_info_parser(subparsers)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            overrides = {}
            if args.out:
                overrides["out"] = args.out
            if args.force:
                overrides["force"] = True
            if args.d_model:
                overrides["d_model"] = args.d_model
            if args.layers:
                overrides["layers"] = args.layers
            if args.heads:
                overrides["heads"] = args.heads
            if args.batch_size:
                overrides["batch_size"] = args.batch_size
            if args.epochs:
                overrides["epochs"] = args.epochs
            if args.learning_rate:
                overrides["learning_rate"] = args.learning_rate
            if args.device:
                overrides["device"] = args.device
            if args.quiet:
                overrides["verbose"] = False
            api.train(args.path, **overrides)
            return 0

        elif args.command == "run":
            api.run(args.path, prompt=args.prompt)
            return 0

        elif args.command == "inspect":
            api.inspect(args.path)
            return 0

        elif args.command == "info":
            payload = load_tl(args.path)
            info = {
                "task": payload["task"],
                "model_type": payload["model_type"],
                "tensorless_version": payload.get("tensorless_version"),
                "tl_format_version": payload.get("tl_format_version"),
                "training_complete": payload.get("training_complete"),
                "metrics": payload.get("metrics"),
                "dataset_fingerprint": (payload.get("dataset_fingerprint") or "")[:16],
            }
            print(json.dumps(info, indent=2, default=str))
            return 0

    except TensorlessError as e:
        print(f"tensorless: error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
