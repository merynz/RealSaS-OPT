from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA = "RealSaS.ArachneMageA0AuthorityBlock.v2"
STATUS = "A0_BLOCKED__P0_V1_BODY_ONLY_TEACHER_RETIRED__FULL_SOURCE_P0_V2_REQUIRED"
SUPERSESSION_AUTHORITY = "canonical/ARACHNE_MAGE_A0_AUTHORITY_BLOCK_V2_20260908.json"
HISTORICAL_EXECUTABLE_COMMIT = "1cc449be4cc6eea81f3e8e9768f857dfb9b72477"

RETIRED_FINGERPRINTS = {
    "teacher_weights_content_sha256": "c15db7b78d272ac22998071e1fb1cec4c65d7366133ef16fb72824f222c852d9",
    "target_npz_sha256": "f3db92194660f12de4425e015f85ac4ac995fcfc44a0272047b049ffb0d36d71",
    "binding_sha256": "cb41eb7055b8e2646628daecdd0e31dfc079d163d5f5adaaa1a92f1ca1dfb994",
    "conditioning_cache_sha256": "12484afc23d5c03cbad8020266ed5b39c3201d979e78f96380e748902152be6e",
}

REQUIRED_NEXT_GATE = (
    "BUILD_DETERMINISTIC_FULL_SOURCE_P0_V2",
    "REPLAY_ALL_950_PROJECTIONS",
    "RECOMPUTE_CONFIDENCE_INVENTORY",
    "SEAL_TARGET_AND_BINDING_HASHES",
    "REBUILD_CONDITIONING_CACHE",
    "REBIND_A0_PREREG_AND_RUNNER",
    "RERUN_CPU_RESUME_TERMINAL_IDEMPOTENCE_REGRESSIONS",
    "BUILD_SELF_CONTAINED_COLAB",
    "RUN_MAIN_A0_CUDA",
)


def _blocked_payload() -> dict:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "a0_optimizer_authorized": False,
        "a1_optimizer_authorized": False,
        "supersession_authority": SUPERSESSION_AUTHORITY,
        "historical_executable_commit": HISTORICAL_EXECUTABLE_COMMIT,
        "retired_fingerprints": RETIRED_FINGERPRINTS,
        "required_next_gate": list(REQUIRED_NEXT_GATE),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed interlock for retired Mage A0 P0 V1 authority. "
            "The executable runner is restored only after full-source P0 V2 is sealed."
        )
    )
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--zero-surface", type=Path)
    parser.add_argument("--camera-dir", type=Path)
    parser.add_argument("--qualified-skeleton", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.parse_args(argv)

    payload = _blocked_payload()
    print("A0_AUTHORITY_BLOCK=" + json.dumps(payload, sort_keys=True), flush=True)
    raise RuntimeError(
        "A0_AUTHORITY_BLOCKED__P0_V1_BODY_ONLY_TEACHER_RETIRED__"
        "FULL_SOURCE_P0_V2_REQUIRED__DO_NOT_RUN_OPTIMIZER"
    )


if __name__ == "__main__":
    raise SystemExit(main())
