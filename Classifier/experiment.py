"""Unified CLI for train, smoke, resume, and export across A/B/C/D/G/H/I."""
from __future__ import annotations

import argparse
from pathlib import Path

from .engine import export_run, resolve_config, train
from .protocol import FORMAL_SEEDS, GROUPS, run_dir, validate_group_seed


DEFAULT_CONFIG = Path("Classifier/config.json")
DEFAULT_OUTPUT_ROOT = Path("/mnt/hdd8t/Mingle/xyyy/Classifier/outputs")


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode in ("train", "smoke", "export"):
        item = subparsers.add_parser(mode)
        item.add_argument("--group", required=True, choices=tuple(GROUPS))
        item.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
        item.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
        item.add_argument("--device", default="cuda")
    for mode in ("train", "smoke"):
        subparsers.choices[mode].add_argument(
            "--config", type=Path,
            help="Base config for a new run; resume defaults to run/config.json",
        )
    subparsers.choices["train"].add_argument("--resume", action="store_true")
    smoke = subparsers.choices["smoke"]
    smoke.add_argument("--batches", type=int, default=2)
    export = subparsers.choices["export"]
    export.add_argument("--split-dir", type=Path, required=True)
    export.add_argument("--name", default="val_export")
    export.add_argument("--expected-samples", type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    validate_group_seed(args.group, args.seed)
    formal_run = run_dir(args.output_root, args.group, args.seed)
    if args.mode == "export":
        output = formal_run / args.name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite export: {output}")
        export_run(args.group, formal_run, args.split_dir, output,
                   args.device, args.expected_samples)
        return
    config_path = args.config or (
        formal_run / "config.json" if args.mode == "train" and args.resume else DEFAULT_CONFIG
    )
    cfg = resolve_config(config_path, args.group, args.seed)
    if args.mode == "smoke":
        if args.batches <= 0:
            raise ValueError("--batches must be positive")
        output = args.output_root / "_smoke" / args.group / f"seed_{args.seed}"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite smoke output: {output}")
        train(cfg, output, args.device, smoke_batches=args.batches)
        return
    if formal_run.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite formal run: {formal_run}")
    if args.resume and not (formal_run / "last.pt").is_file():
        raise FileNotFoundError(f"Cannot resume without {formal_run / 'last.pt'}")
    train(cfg, formal_run, args.device, resume=args.resume)


if __name__ == "__main__":
    main()
