from __future__ import annotations

"""Build the canonical self-contained MIRA V5.5 runtime checkpoint.

This is packaging only. It does not train, fit, evaluate, promote a ModelRelease,
or mint Product/Skin authority. The canonical recovery tool is deliberately
pinned to the exact sealed V6 + V5.5 component identities; alternate component
pairs require a different/versioned packaging contract rather than CLI overrides.
"""

import argparse
import json
from pathlib import Path

from models.mira.runtime_checkpoint_v1 import (
    MIRA_V55_CANONICAL_BASE_V6_SHA256,
    MIRA_V55_CANONICAL_COMPOSITE_SHA256,
    save_mira_runtime_checkpoint_v1,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-v6-checkpoint", required=True)
    parser.add_argument("--v55-composite", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()

    receipt = save_mira_runtime_checkpoint_v1(
        output_path=args.out,
        base_v6_checkpoint_path=args.base_v6_checkpoint,
        v55_composite_path=args.v55_composite,
        expected_base_v6_sha256=MIRA_V55_CANONICAL_BASE_V6_SHA256,
        expected_v55_composite_sha256=MIRA_V55_CANONICAL_COMPOSITE_SHA256,
    )
    receipt_path = Path(args.receipt).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt.to_dict(), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
