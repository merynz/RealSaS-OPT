"""Select the sealed appearance admission after an exact-main deployment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess


REQUEST_PATH = "canonical/SOURCE_RAY_APPEARANCE_ADMISSION_REQUEST_V1.json"
SCHEMA = "RealSaS.SourceRayAppearanceAdmissionRequest.v1"
OPERATION = "ADMIT_SEALED_SPARSE_SOURCE_RAY_APPEARANCE"
NEGATIVE_CLAIMS = (
    "training_authorized", "model_promotion_authorized",
    "product_revision_authorized", "canonical_mesh_binding_authorized",
)


def select_admission(request, *, event, branch, conclusion, changed_paths):
    if request.get("schema") != SCHEMA or request.get("operation") != OPERATION:
        raise ValueError("SOURCE_RAY_ADMISSION_REQUEST_CONTRACT")
    if type(request.get("enabled")) is not bool:
        raise ValueError("SOURCE_RAY_ADMISSION_REQUEST_ENABLED")
    if type(request.get("request_revision")) is not int or request["request_revision"] < 1:
        raise ValueError("SOURCE_RAY_ADMISSION_REQUEST_REVISION")
    if any(request.get("claims", {}).get(key) is not False for key in NEGATIVE_CLAIMS):
        raise ValueError("SOURCE_RAY_ADMISSION_REQUEST_AUTHORITY_ESCALATION")
    if event not in {"workflow_run", "workflow_dispatch"}:
        return "NOOP", "source_ci_only"
    if branch != "main":
        return "NOOP", "non_main_execution"
    if not request["enabled"]:
        return "NOOP", "request_disabled"
    if event == "workflow_dispatch":
        return "RUN", "explicit_main_dispatch"
    if conclusion != "success":
        return "NOOP", "deployment_not_successful"
    if REQUEST_PATH not in changed_paths:
        return "NOOP", "request_not_changed"
    return "RUN", "request_changed_after_successful_main_deployment"


def main():
    expected = os.environ["EXPECTED_SHA"]
    if not re.fullmatch(r"[0-9a-f]{40}", expected):
        raise ValueError("SOURCE_RAY_ADMISSION_EXPECTED_SHA")
    checkout = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if checkout != expected:
        raise ValueError("SOURCE_RAY_ADMISSION_SELECTOR_CHECKOUT_DRIFT")
    event = os.environ["EVENT_NAME"]
    changed = []
    if event == "workflow_run":
        changed = subprocess.check_output(
            ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD^1", "HEAD"],
            text=True,
        ).splitlines()
    request = json.loads(Path(REQUEST_PATH).read_text())
    decision, reason = select_admission(
        request, event=event, branch=os.environ.get("SOURCE_BRANCH", ""),
        conclusion=os.environ.get("DEPLOYMENT_CONCLUSION", ""), changed_paths=changed,
    )
    receipt = {
        "schema": "RealSaS.SourceRayAppearanceAdmissionDecision.v1",
        "decision": decision, "reason": reason, "source_code_sha": expected,
        "request_revision": request["request_revision"],
        "request_path": REQUEST_PATH, "claims": request["claims"],
    }
    Path("source-ray-admission-decision.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        output.write(f"decision={decision}\nreason={reason}\nexpected_sha={expected}\n")
    print(json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    main()
