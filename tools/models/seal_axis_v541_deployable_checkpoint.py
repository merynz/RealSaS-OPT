from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.axis.deployable_checkpoint_v541 import (
    seal_axis_v541_deployable_checkpoint_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Seal the historical AXIS V5.4.1 three-part FIT lineage into one "
            "self-contained deployable ModelRelease checkpoint."
        )
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--v53-progress", type=Path, required=True)
    parser.add_argument("--v541-composite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = seal_axis_v541_deployable_checkpoint_v1(
        base_checkpoint_path=args.base_checkpoint,
        v53_progress_path=args.v53_progress,
        v541_composite_path=args.v541_composite,
        output_path=args.output,
    )
    print(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
