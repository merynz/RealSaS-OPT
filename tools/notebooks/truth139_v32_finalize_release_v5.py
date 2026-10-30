"""Notebook-facing Truth139 V3.2 finalization release orchestrator v5.

V5 preserves the V4 release contract and adds the second Stage48 canary repair:
the independent raw-source reference must transport evaluated UV/color attributes
onto its isolated exact-teacher triangles.  Historical V1 courts stay untouched.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from tools.notebooks import truth139_v32_finalize_release_v4 as base


class FinalizeReleaseV5(base.FinalizeReleaseV4):
    def static_preflight(self) -> None:
        super().static_preflight()
        supplemental = [
            "tools/truth_foundry/blender_source_ray_appearance_reference_v2.py",
            "tools/corpus/run_truth139_stage48_witness_subject_v2.py",
            "tools/corpus/run_truth139_v32_finalize_remaining_full139_v4.py",
            "tools/notebooks/truth139_v32_finalize_release_v5.py",
        ]
        base.run(
            [sys.executable, "-m", "py_compile", *supplemental],
            cwd=self.repo_dir,
            repo_dir=self.repo_dir,
        )
        print("[PASS] V5 source-reference transport syntax")

    def freeze_runtime(self) -> None:
        super().freeze_runtime()
        tests = [
            "tests/truth_foundry/test_source_ray_reference_v2_attribute_transport_contract_v1.py",
        ]
        base.run(
            [sys.executable, "-m", "pytest", "-q", *tests],
            cwd=self.repo_dir,
            repo_dir=self.repo_dir,
        )

        runtime = base.load_json(self.runtime_fingerprint)
        runtime.pop("runtime_identity_sha256", None)
        runtime["release_revision"] = 5
        runtime["supplemental_release_tests"] = tests
        runtime["supplemental_release_tests_passed"] = True
        runtime["stage48_source_reference_transport"] = {
            "runner": "tools/corpus/run_truth139_stage48_witness_subject_v2.py",
            "reference_adapter": "tools/truth_foundry/blender_source_ray_appearance_reference_v2.py",
            "sealed_v1_reference_logic_changed": False,
            "tolerance_relaxed": False,
        }
        runtime["runtime_identity_sha256"] = base.digest(runtime)
        self.runtime_fingerprint.write_text(
            json.dumps(runtime, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
        print("[PASS] V5 runtime fingerprint augmented =", self.runtime_fingerprint)

    def _run_subject(self, sid: str) -> None:
        original_run = base.run

        def routed_run(cmd, **kwargs):
            routed = []
            for item in cmd:
                text = str(item)
                if text.endswith("tools/corpus/run_truth139_stage48_witness_subject_v1.py"):
                    text = str(
                        self.repo_dir
                        / "tools/corpus/run_truth139_stage48_witness_subject_v2.py"
                    )
                routed.append(text)
            return original_run(routed, **kwargs)

        base.run = routed_run
        try:
            super()._run_subject(sid)
        finally:
            base.run = original_run

    def run_full(self) -> None:
        if self.batch_root is None or self.blender_exe is None or self.response_receipt is None:
            raise RuntimeError("V32_RELEASE_FULL_BEFORE_BATCH")
        canary = self.batch_root / "V32_FINALIZE_TWO_REAL_CANARIES_RECEIPT.json"
        if (
            not canary.is_file()
            or base.load_json(canary).get("status")
            != "PASS_V32_FINALIZE_TWO_REAL_CANARIES"
        ):
            raise RuntimeError("V32_RELEASE_FULL_REQUIRES_CANARIES")
        base.run(
            [
                sys.executable,
                "-m",
                "tools.corpus.run_truth139_v32_finalize_remaining_full139_v4",
                "--work-orders",
                self.work_orders,
                "--prepared-root",
                self.prepared_root,
                "--prepared-rescue-root",
                self.prepared_rescue_root,
                "--base-receipt",
                self.base_receipt,
                "--v2-extension-receipt",
                self.v2_receipt,
                "--response-receipt",
                self.response_receipt,
                "--response-svd-rescue",
                self.response_svd_rescue,
                "--runtime-fingerprint",
                self.runtime_fingerprint,
                "--blender",
                self.blender_exe,
                "--output-root",
                self.batch_root,
                "--continue-on-failure",
            ],
            cwd=self.repo_dir,
            repo_dir=self.repo_dir,
        )
