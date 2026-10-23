from __future__ import annotations

"""Package sealed AXIS V5.4.1 research components into one runtime checkpoint.

This command performs no training and mints no scientific/product authority. It
exists so a ModelRelease can point to one immutable, self-contained inference
artifact instead of depending on notebook-era filesystem composition.
"""

import argparse
import json
from pathlib import Path

from models.axis.runtime_checkpoint_v1 import save_axis_runtime_checkpoint_v1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--v53-progress", required=True)
    parser.add_argument("--v541-composite", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt")
    args = parser.parse_args()

    receipt = save_axis_runtime_checkpoint_v1(
        output_path=args.output,
        base_checkpoint_path=args.base_checkpoint,
        v53_progress_path=args.v53_progress,
        v541_composite_path=args.v541_composite,
    )
    payload = receipt.to_dict()
    if args.receipt:
        path = Path(args.receipt).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
