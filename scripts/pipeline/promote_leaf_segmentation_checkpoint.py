#!/usr/bin/env python3
"""Promueve el best.pt entrenado a la ruta de checkpoint que consume la inferencia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import get_output_root
from src.training.checkpoint_promotion import promote_checkpoint


def build_parser() -> argparse.ArgumentParser:
    """Construye la CLI de promoción."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    """Punto de entrada de la promoción."""
    args = build_parser().parse_args()
    output_root = (args.output_root or get_output_root()).resolve()
    registry = promote_checkpoint(
        output_root,
        source=args.source.resolve() if args.source else None,
        force=args.force,
    )
    print(json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
