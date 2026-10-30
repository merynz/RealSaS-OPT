"""Notebook-facing release orchestrator for Truth139 V3.2 finalization v4.

The notebook intentionally stays thin.  This module owns the testable release
logic: sealed prerequisite gates, frozen runtime, Response SVD-rescue provenance,
two real-batch canaries, full resumable execution, and final training-readiness
closure.  No optimizer is executed here.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from urllib.request import urlopen


BASE_PASS = "PASS_FULL139_CANONICAL_V2_COMPOSITE_MATERIALIZATION_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
V2_PASS = "PASS_FULL139_V2_EXTENSION_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
RESPONSE_PASS = "PASS_FULL139_IRIS_RESPONSE_EVIDENCE_COMPLETE__DERIVED_TRUTH_ADMISSION_SEPARATE"
RESPONSE_RESCUE_PASS = "PASS_SINGLE_SUBJECT_SVD_COURT_NUMERIC_RESCUE"
FINAL_PASS = "PASS_TRUTH139_V32_ALL_PRETRAIN_TARGETS_AND_SPLIT_READY"
RESPONSE_RESCUE_SUBJECT = "truth139_0fc13c59558762dccc11"
PREPARED_RESCUE_SUBJECT = "truth139_41666dce2c1ff356219a"
RESPONSE_RESCUE_VALIDATOR_SHA = "90a1ebd246e0254b8e7bac704a0d3a65a6ba0fbe202c2ebc804d87f64c53f39c"
EXPECTED_WORK_ORDER_ID = "7fd1b6e3ac7d980c469a99de116e9896fd8828b3ce76e56494db83be21d0235e"

EXPECTED_PACKAGES = {
    "torch": "2.8.0+cpu",
    "torchvision": "0.23.0+cpu",
    "numpy": "2.1.3",
    "Pillow": "11.3.0",
    "trimesh": "5.2.0",
    "Rtree": "1.4.1",
    "fast-simplification": "0.2.0",
    "pytest": "8.4.2",
}

BLENDER_ARCHIVE_SHA256 = "5583a5588736da8858c522ef17fff5d73be59c47a6fe91ad29c6f3263e22086a"
BLENDER_BINARY_SHA256 = "6fe78c9e0a1cca51a598eb37c3288b975357efbe86dcee18d5e6fdecaaf1847e"

GEOM_PASS = "PASS_V32_EXACT_SOURCE_GEOMETRY_CANDIDATE__INDEPENDENT_COURT_REQUIRED"
GEOM_COURT_PASS = "PASS_V32_EXACT_GEOMETRY_REBINDS_SOURCE_AND_INDEPENDENT_SAMPLES"
RGB_PASS = "PASS_V32_RGB37_PATCH_EVIDENCE_MATERIALIZED"
RGB_COURT_PASS = "PASS_V32_RGB37_PATCH_EVIDENCE_REBINDS_EXACT_SOURCE_PIXELS"
TESSA_PASS = "PASS_V32_MECHANICALLY_QUALIFIED_TESSA_TEACHER"
TESSA_COURT_PASS = "PASS_V32_MECHANICAL_TESSA_TEACHER_REBINDS_EXACT_SOURCE_AND_DECLARED_COURTS"
STAGE48_PASS = "PASS_STAGE48_RENDERER_INTERVENTION_WITNESS__SOURCE_EQUIVALENT_BOUNDED_DOMAIN"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def load_json(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED:" + str(path))
    return value


def run(cmd, *, cwd: Path | None = None, env: dict | None = None, repo_dir: Path | None = None) -> str:
    cmd = [str(x) for x in cmd]
    print("\n[RUN]", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if repo_dir is not None:
        merged["PYTHONPATH"] = str(repo_dir) + os.pathsep + merged.get("PYTHONPATH", "")
    if env:
        merged.update({str(k): str(v) for k, v in env.items()})
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = process.wait()
    text = "".join(lines)
    if code:
        raise RuntimeError(f"COMMAND_FAILED_RC_{code}:{' '.join(cmd)}\n{text[-20000:]}")
    return text


class FinalizeReleaseV4:
    def __init__(self, *, repo_dir: Path, drive_root: Path, source_sha: str) -> None:
        self.repo_dir = Path(repo_dir).resolve()
        self.drive_root = Path(drive_root).resolve()
        self.source_sha = str(source_sha)
        head = subprocess.check_output(
            ["git", "-C", str(self.repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != self.source_sha:
            raise ValueError(f"V32_RELEASE_SOURCE_SHA:{head}:{self.source_sha}")

        self.materialized_root = self.drive_root / "materialized"
        self.source_cache_root = self.drive_root / "inputs/TRUTH139_FULL139_SOURCE_CACHE_V1"
        self.work_orders = self.source_cache_root / "metadata/TRUTH139_MATERIALIZATION_WORK_ORDERS_V1.json"
        self.prepared_root = self.materialized_root / "prepared_full139/936dd528105c84d2974fc512"
        self.prepared_rescue_root = self.materialized_root / "prepared_full139_rescue/af2a36f11a8a141cde9b501b"
        self.base_receipt = self.materialized_root / "full139_rescue_batches/af2a36f11a8a141cde9b501b/FULL139_COMPOSITE_BATCH_RECEIPT.json"
        self.v2_receipt = self.materialized_root / "v2_extension_batches/6f7909378525a54251d0d91e/FULL139_V2_EXTENSION_RECEIPT.json"
        self.response_batch_root = self.materialized_root / "response_evidence_batches"
        self.cache_root = self.drive_root / "cache"
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root = Path("/content/truth139_v32_finalize_runtime")
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.requirements = self.repo_dir / "requirements/truth139-v32-finalize.txt"
        self.contract = self.repo_dir / "canonical/TRUTH139_V32_REMAINING_TRUTHS_V1.json"

        self.base = None
        self.v2 = None
        self.work = None
        self.response = None
        self.response_receipt = None
        self.response_svd_rescue = None
        self.rescue = None
        self.packet_map = None
        self.runtime_fingerprint = None
        self.blender_exe = None
        self.batch_identity = None
        self.batch_id = None
        self.batch_root = None

    def _receipt_gate(self, path: Path, status: str) -> dict:
        value = load_json(path)
        if value.get("status") != status or int(value.get("completed_subject_count", -1)) != 139:
            raise ValueError("V32_RELEASE_PREREQUISITE_NOT_CLOSED:" + str(path))
        return value

    def gate_base_v2(self) -> None:
        self.base = self._receipt_gate(self.base_receipt, BASE_PASS)
        self.v2 = self._receipt_gate(self.v2_receipt, V2_PASS)
        if not (self.work_orders.is_file() and self.prepared_root.is_dir() and self.prepared_rescue_root.is_dir()):
            raise FileNotFoundError("V32_RELEASE_BASE_INPUTS")
        print("[PASS] Base139 + V2 Extension sealed 139/139")

    def static_preflight(self) -> None:
        if not self.requirements.is_file():
            raise FileNotFoundError(self.requirements)
        sources = [
            "tools/corpus/build_truth139_v32_exact_geometry_v1.py",
            "tools/corpus/validate_truth139_v32_exact_geometry_v1.py",
            "tools/corpus/build_truth139_v32_rgb_patch_cache_v1.py",
            "tools/corpus/validate_truth139_v32_rgb_patch_cache_v1.py",
            "tools/training/truth139_v32_spatial_index_v1.py",
            "tools/training/build_truth139_v32_mechanical_tessa_teacher_v1.py",
            "tools/training/build_truth139_v32_mechanical_tessa_teacher_v2.py",
            "tools/training/validate_truth139_v32_mechanical_tessa_teacher_v1.py",
            "tools/training/validate_truth139_v32_mechanical_tessa_teacher_v2.py",
            "tools/corpus/run_truth139_stage48_witness_subject_v1.py",
            "tools/corpus/run_truth139_v32_finalize_remaining_full139_v2.py",
            "tools/corpus/run_truth139_v32_finalize_remaining_full139_v3.py",
            "tools/notebooks/truth139_v32_finalize_release_v4.py",
        ]
        run([sys.executable, "-m", "py_compile", *sources], cwd=self.repo_dir, repo_dir=self.repo_dir)
        print("[PASS] release source syntax")

    def freeze_runtime(self) -> None:
        if sys.platform.startswith("linux"):
            run([
                "bash", "-lc",
                "sudo apt-get -o Acquire::Retries=6 update -qq && "
                "sudo apt-get -o Acquire::Retries=6 install -y --no-install-recommends "
                "libxi6 libxrender1 libxfixes3 libxkbcommon0 libsm6 libgl1 libspatialindex6",
            ], repo_dir=self.repo_dir)
        run([
            sys.executable, "-m", "pip", "install", "--no-input", "--progress-bar", "off",
            "-r", self.requirements,
        ], cwd=self.repo_dir, repo_dir=self.repo_dir)
        packages = {name: metadata.version(name) for name in EXPECTED_PACKAGES}
        if packages != EXPECTED_PACKAGES:
            raise ValueError("V32_RELEASE_PACKAGE_DRIFT:" + repr(packages))

        archive = self.cache_root / "blender-4.0.2-linux-x64.tar.xz"
        home = self.runtime_root / "blender-4.0.2-linux-x64"
        blender = home / "blender"
        if not archive.exists() or sha256_file(archive) != BLENDER_ARCHIVE_SHA256:
            tmp = archive.with_suffix(".partial")
            url = "https://download.blender.org/release/Blender4.0/blender-4.0.2-linux-x64.tar.xz"
            with urlopen(url, timeout=120) as response, tmp.open("wb") as stream:
                while True:
                    chunk = response.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    stream.write(chunk)
            tmp.replace(archive)
        if sha256_file(archive) != BLENDER_ARCHIVE_SHA256:
            raise ValueError("V32_RELEASE_BLENDER_ARCHIVE_DRIFT")
        if home.exists():
            shutil.rmtree(home)
        run(["tar", "-xJf", archive, "-C", self.runtime_root], repo_dir=self.repo_dir)
        if subprocess.check_output([blender, "--version"], text=True).splitlines()[0] != "Blender 4.0.2":
            raise ValueError("V32_RELEASE_BLENDER_VERSION")
        if sha256_file(blender) != BLENDER_BINARY_SHA256:
            raise ValueError("V32_RELEASE_BLENDER_BINARY_DRIFT")
        self.blender_exe = blender

        # Exercise the real optional dependency, not only a mocked unit test.
        code = (
            "from tools.training.truth139_v32_spatial_index_v1 import new_rtree3;"
            "i=new_rtree3();i.insert(7,(0,0,0,1,1,1));"
            "assert list(i.intersection((.25,.25,.25,.75,.75,.75)))==[7]"
        )
        run([sys.executable, "-c", code], cwd=self.repo_dir, repo_dir=self.repo_dir)

        tests = [
            "tests/corpus/test_truth139_stage48_standalone_attempt_v1.py",
            "tests/training/test_truth139_v32_spatial_index_v1.py",
            "tests/corpus/test_truth139_v32_finalize_v3_response_rescue_binding.py",
        ]
        run([sys.executable, "-m", "pytest", "-q", *tests], cwd=self.repo_dir, repo_dir=self.repo_dir)

        runtime = {
            "schema": "RealSaS.Truth139V32FinalizeRuntimeFingerprint.v1",
            "status": "PASS_TRUTH139_V32_FINALIZE_RUNTIME_FROZEN",
            "source_commit_sha": self.source_sha,
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation(),
                "platform": platform.platform(),
            },
            "packages": packages,
            "requirements": {"path": str(self.requirements), "sha256": sha256_file(self.requirements)},
            "blender": {
                "version": "4.0.2",
                "archive_sha256": BLENDER_ARCHIVE_SHA256,
                "binary_sha256": BLENDER_BINARY_SHA256,
            },
            "rtree_3d_smoke_passed": True,
            "release_tests": tests,
            "release_tests_passed": True,
        }
        runtime["runtime_identity_sha256"] = digest(runtime)
        path = self.cache_root / f"TRUTH139_V32_FINALIZE_RUNTIME_{runtime['runtime_identity_sha256'][:24]}.json"
        path.write_text(json.dumps(runtime, sort_keys=True, indent=2, allow_nan=False) + "\n")
        self.runtime_fingerprint = path
        print("[PASS] frozen runtime =", path)

    def bind_response_and_packets(self) -> None:
        work = load_json(self.work_orders)
        if work.get("schema") != "RealSaS.Truth139MaterializationWorkOrders.v1":
            raise ValueError("V32_RELEASE_WORK_SCHEMA")
        if work.get("work_order_identity_sha256") != EXPECTED_WORK_ORDER_ID:
            raise ValueError("V32_RELEASE_WORK_ID")
        rows = list(work.get("subjects") or [])
        if int(work.get("subject_count", -1)) != 139 or len(rows) != 139:
            raise ValueError("V32_RELEASE_DENOMINATOR")
        self.work = work

        candidates = []
        for path in sorted(self.response_batch_root.glob("*/FULL139_IRIS_RESPONSE_EVIDENCE_RECEIPT.json")):
            try:
                value = load_json(path)
                bound = (value.get("work_orders") or {}).get("identity_sha256")
                if (
                    value.get("status") == RESPONSE_PASS
                    and int(value.get("completed_subject_count", -1)) == 139
                    and int(value.get("subject_count", -1)) == 139
                    and not value.get("failed_jobs_by_subject")
                    and bound == EXPECTED_WORK_ORDER_ID
                ):
                    candidates.append((value.get("completed_at_utc") or "", path, value))
            except Exception:
                pass
        if not candidates:
            raise RuntimeError("V32_RELEASE_NO_CLOSED_RESPONSE_139")
        candidates.sort(key=lambda row: (row[0], str(row[1])))
        _, response_path, response = candidates[-1]
        if len(response.get("subject_output_bindings") or {}) != 139:
            raise ValueError("V32_RELEASE_RESPONSE_BINDING_COUNT")
        if any(response.get(key) is not False for key in ("training_performed", "split_created", "qualification_minted")):
            raise ValueError("V32_RELEASE_RESPONSE_AUTHORITY_DRIFT")

        rescue_path = response_path.parent / "FULL139_RESPONSE_SVD_COURT_RESCUE_RECEIPT.json"
        rescue = load_json(rescue_path)
        if rescue.get("status") != RESPONSE_RESCUE_PASS or rescue.get("subject_id") != RESPONSE_RESCUE_SUBJECT:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_STATUS")
        if rescue.get("base_repository_commit") != response.get("code_commit_sha"):
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_BASE")
        if rescue.get("patched_validator_sha256") != RESPONSE_RESCUE_VALIDATOR_SHA:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_VALIDATOR")
        diagnosis = dict(rescue.get("diagnosis") or {})
        if diagnosis.get("producer_truth_changed") is not False or diagnosis.get("teacher_truth_changed") is not False or diagnosis.get("tolerance_relaxed") is not False:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_AUTHORITY")
        if diagnosis.get("new_independent_formulation") != "REST_TANGENT_SVD_GENERALIZED_PRINCIPAL_STRETCH":
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_FORMULATION")
        if float((rescue.get("court") or {}).get("maximum_nonrigid_abs_error", 1.0)) >= 1.0e-8:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_NUMERIC")

        binding = response["subject_output_bindings"][RESPONSE_RESCUE_SUBJECT]
        chain_path = Path(binding["chain_path"])
        if not chain_path.is_file() or sha256_file(chain_path) != binding["chain_sha256"]:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_CHAIN")
        chain = load_json(chain_path)
        court = dict(rescue["court"])
        if chain["outputs"]["response_court"]["sha256"] != court["sha256"]:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_COURT_SHA")
        if chain["bindings"]["response_court_identity_sha256"] != court["court_identity_sha256"]:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_COURT_ID")
        if sha256_file(Path(court["path"])) != court["sha256"]:
            raise ValueError("V32_RELEASE_RESPONSE_RESCUE_COURT_BYTES")

        packet_map = {}
        base_count = rescue_count = 0
        for index, row in enumerate(rows, 1):
            sid = str(row["subject_id"])
            found = []
            for root, lineage in ((self.prepared_root, "PREPARED_BASE"), (self.prepared_rescue_root, "PREPARED_RESCUE")):
                for packet in sorted(root.glob(f"{sid}/objects/*/packet/{sid}")):
                    if packet.is_dir():
                        found.append((packet.resolve(), lineage))
            if len(found) != 1:
                raise ValueError(f"V32_RELEASE_PACKET_CARDINALITY:{sid}:{len(found)}")
            packet, lineage = found[0]
            for name in ("REST_SURFACE_GEOMETRY.npz", "IRIS_OBSERVATION_ADMISSION.json", "SOURCE_REST_RENDER_CANDIDATE.json", "SOURCE_SCENE_SNAPSHOT.json", "camera_bundle.json"):
                if not (packet / name).is_file():
                    raise FileNotFoundError(f"{sid}:{lineage}:{name}")
            for view in range(8):
                if not (packet / f"V{view}.png").is_file():
                    raise FileNotFoundError(f"{sid}:{lineage}:V{view}")
            packet_map[sid] = (packet, lineage)
            base_count += int(lineage == "PREPARED_BASE")
            rescue_count += int(lineage == "PREPARED_RESCUE")
            if index % 20 == 0 or index == 139:
                print(f"[PREFLIGHT {index:03d}/139] base={base_count} rescue={rescue_count}")
        if (base_count, rescue_count) != (138, 1) or packet_map[PREPARED_RESCUE_SUBJECT][1] != "PREPARED_RESCUE":
            raise ValueError("V32_RELEASE_PREPARED_LINEAGE")

        self.response = response
        self.response_receipt = response_path
        self.response_svd_rescue = rescue_path
        self.rescue = rescue
        self.packet_map = packet_map
        print("[PASS] Response 139/139 + SVD rescue rebound")

    def freeze_batch(self) -> None:
        required = (self.base, self.v2, self.work, self.response, self.rescue, self.packet_map, self.runtime_fingerprint, self.blender_exe)
        if any(value is None for value in required):
            raise RuntimeError("V32_RELEASE_PHASE_ORDER")
        identity = {
            "schema": "RealSaS.Truth139V32FinalizeBatchIdentity.v4",
            "source_commit_sha": self.source_sha,
            "work_order_identity_sha256": EXPECTED_WORK_ORDER_ID,
            "work_order_sha256": sha256_file(self.work_orders),
            "base_receipt_sha256": sha256_file(self.base_receipt),
            "v2_receipt_sha256": sha256_file(self.v2_receipt),
            "response_receipt_sha256": sha256_file(self.response_receipt),
            "response_svd_rescue_sha256": sha256_file(self.response_svd_rescue),
            "runtime_fingerprint_sha256": sha256_file(self.runtime_fingerprint),
            "prepared_base_id": "936dd528105c84d2974fc512",
            "prepared_rescue_id": "af2a36f11a8a141cde9b501b",
            "contract_sha256": sha256_file(self.contract),
            "blender_binary_sha256": sha256_file(self.blender_exe),
        }
        batch_id = digest(identity)[:24]
        root = self.materialized_root / "v32_finalize_batches" / batch_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "V32_FINALIZE_BATCH_IDENTITY_V4.json").write_text(
            json.dumps({**identity, "batch_id": batch_id}, sort_keys=True, indent=2) + "\n"
        )
        self.batch_identity = identity
        self.batch_id = batch_id
        self.batch_root = root
        print("BATCH_ID =", batch_id)
        print("BATCH_ROOT =", root)

    @staticmethod
    def _status(path: Path) -> str | None:
        return load_json(path).get("status") if Path(path).is_file() else None

    def run_canaries(self) -> None:
        if self.batch_root is None or self.packet_map is None or self.response is None or self.blender_exe is None:
            raise RuntimeError("V32_RELEASE_PHASE_ORDER")
        for sid in (RESPONSE_RESCUE_SUBJECT, PREPARED_RESCUE_SUBJECT):
            packet, lineage = self.packet_map[sid]
            chain_path = Path(self.response["subject_output_bindings"][sid]["chain_path"])
            out = self.batch_root / "subjects" / sid

            geom_dir = out / "geometry"
            geom = geom_dir / "V32_EXACT_GEOMETRY_MANIFEST.json"
            geom_court = geom_dir / "V32_EXACT_GEOMETRY_COURT.json"
            if self._status(geom) != GEOM_PASS:
                run([sys.executable, self.repo_dir / "tools/corpus/build_truth139_v32_exact_geometry_v1.py", "--subject-id", sid, "--geometry-npz", packet / "REST_SURFACE_GEOMETRY.npz", "--source-scene-snapshot", packet / "SOURCE_SCENE_SNAPSHOT.json", "--camera-bundle", packet / "camera_bundle.json", "--output-dir", geom_dir], cwd=self.repo_dir, repo_dir=self.repo_dir)
            if self._status(geom_court) != GEOM_COURT_PASS:
                run([sys.executable, self.repo_dir / "tools/corpus/validate_truth139_v32_exact_geometry_v1.py", "--geometry-manifest", geom, "--output", geom_court], cwd=self.repo_dir, repo_dir=self.repo_dir)

            rgb_dir = out / "rgb"
            rgb = rgb_dir / "V32_RGB_PATCH_EVIDENCE_MANIFEST.json"
            rgb_court = rgb_dir / "V32_RGB_PATCH_EVIDENCE_COURT.json"
            if self._status(rgb) != RGB_PASS:
                run([sys.executable, self.repo_dir / "tools/corpus/build_truth139_v32_rgb_patch_cache_v1.py", "--subject-id", sid, "--packet-root", packet, "--output-dir", rgb_dir], cwd=self.repo_dir, repo_dir=self.repo_dir)
            if self._status(rgb_court) != RGB_COURT_PASS:
                run([sys.executable, self.repo_dir / "tools/corpus/validate_truth139_v32_rgb_patch_cache_v1.py", "--rgb-manifest", rgb, "--output", rgb_court], cwd=self.repo_dir, repo_dir=self.repo_dir)

            tessa_dir = out / "tessa_teacher"
            tessa = tessa_dir / "V32_MECHANICAL_TESSA_TEACHER_MANIFEST.json"
            tessa_court = tessa_dir / "V32_MECHANICAL_TESSA_TEACHER_COURT.json"
            if self._status(tessa) != TESSA_PASS:
                run([sys.executable, self.repo_dir / "tools/training/build_truth139_v32_mechanical_tessa_teacher_v2.py", "--subject-id", sid, "--geometry-manifest", geom, "--response-chain", chain_path, "--camera-bundle", packet / "camera_bundle.json", "--output-dir", tessa_dir], cwd=self.repo_dir, repo_dir=self.repo_dir)
            if self._status(tessa_court) != TESSA_COURT_PASS:
                run([sys.executable, self.repo_dir / "tools/training/validate_truth139_v32_mechanical_tessa_teacher_v2.py", "--teacher-manifest", tessa, "--output", tessa_court], cwd=self.repo_dir, repo_dir=self.repo_dir)

            stage48_dir = out / "stage48"
            stage48 = stage48_dir / "STAGE48_RENDERER_WITNESS_MANIFEST.json"
            if self._status(stage48) != STAGE48_PASS:
                run([sys.executable, self.repo_dir / "tools/corpus/run_truth139_stage48_witness_subject_v1.py", "--subject-id", sid, "--response-chain", chain_path, "--blender", self.blender_exe, "--output-dir", stage48_dir], cwd=self.repo_dir, repo_dir=self.repo_dir)

            expected = ((geom, GEOM_PASS), (geom_court, GEOM_COURT_PASS), (rgb, RGB_PASS), (rgb_court, RGB_COURT_PASS), (tessa, TESSA_PASS), (tessa_court, TESSA_COURT_PASS), (stage48, STAGE48_PASS))
            for path, wanted in expected:
                if self._status(path) != wanted:
                    raise ValueError(f"V32_RELEASE_CANARY:{sid}:{path.name}:{self._status(path)}")
            print(f"[CANARY PASS] {sid} lineage={lineage}")
        print("[PASS] both end-to-end canaries reusable by full runner")

    def run_full(self) -> None:
        if self.batch_root is None or self.runtime_fingerprint is None or self.response_receipt is None or self.response_svd_rescue is None or self.blender_exe is None:
            raise RuntimeError("V32_RELEASE_PHASE_ORDER")
        run([
            sys.executable, "-m", "tools.corpus.run_truth139_v32_finalize_remaining_full139_v3",
            "--work-orders", self.work_orders,
            "--prepared-root", self.prepared_root,
            "--prepared-rescue-root", self.prepared_rescue_root,
            "--base-receipt", self.base_receipt,
            "--v2-extension-receipt", self.v2_receipt,
            "--response-receipt", self.response_receipt,
            "--response-svd-rescue", self.response_svd_rescue,
            "--runtime-fingerprint", self.runtime_fingerprint,
            "--blender", self.blender_exe,
            "--output-root", self.batch_root,
            "--continue-on-failure",
        ], cwd=self.repo_dir, repo_dir=self.repo_dir)

    def close(self) -> dict:
        if self.batch_root is None or self.batch_identity is None or self.response_receipt is None or self.response_svd_rescue is None or self.runtime_fingerprint is None:
            raise RuntimeError("V32_RELEASE_PHASE_ORDER")
        receipt_path = self.batch_root / "V32_TRAINING_READINESS_RECEIPT.json"
        receipt = load_json(receipt_path)
        if receipt.get("status") != FINAL_PASS or int(receipt.get("subject_count", -1)) != 139 or len(receipt.get("subjects") or {}) != 139:
            raise ValueError("V32_RELEASE_FINAL_RECEIPT")
        if receipt.get("code_commit_sha") != self.source_sha or receipt.get("runner_revision") != 3:
            raise ValueError("V32_RELEASE_FINAL_CODE_BINDING")
        if receipt.get("training_performed") is not False or receipt.get("training_authorized") is not True or receipt.get("qualification_minted") is not False:
            raise ValueError("V32_RELEASE_FINAL_AUTHORITY")
        if receipt.get("compiler_phase_e_watertight_court_pending_training") is not True:
            raise ValueError("V32_RELEASE_FINAL_PHASE_E")
        if receipt["prior_truth_receipts"]["response"]["sha256"] != sha256_file(self.response_receipt):
            raise ValueError("V32_RELEASE_FINAL_RESPONSE_BINDING")
        if receipt["runtime_fingerprint"]["sha256"] != sha256_file(self.runtime_fingerprint):
            raise ValueError("V32_RELEASE_FINAL_RUNTIME_BINDING")
        supplemental = receipt["supplemental_response_numeric_rescue"]
        if supplemental.get("status") != RESPONSE_RESCUE_PASS or supplemental.get("sha256") != sha256_file(self.response_svd_rescue) or supplemental.get("patched_validator_sha256") != RESPONSE_RESCUE_VALIDATOR_SHA:
            raise ValueError("V32_RELEASE_FINAL_RESCUE_BINDING")

        split_path = Path(receipt["split"]["path"])
        if sha256_file(split_path) != receipt["split"]["sha256"]:
            raise ValueError("V32_RELEASE_SPLIT_SHA")
        split = load_json(split_path)
        train = set(split["train_subjects"])
        unseen = set(split["unseen_subjects"])
        if not train or not unseen or train & unseen or len(train) + len(unseen) != 139:
            raise ValueError("V32_RELEASE_SPLIT_PARTITION")
        if split.get("optimizer_may_read_unseen") is not False or split.get("checkpoint_selection_may_read_unseen") is not False or split.get("unseen_bytes_opened_for_fit") is not False:
            raise ValueError("V32_RELEASE_SPLIT_LEAKAGE")

        validation_path = Path(receipt["train_internal_validation"]["path"])
        if sha256_file(validation_path) != receipt["train_internal_validation"]["sha256"]:
            raise ValueError("V32_RELEASE_VALIDATION_SHA")
        validation = load_json(validation_path)
        if validation.get("unseen_subjects_included") is not False or set(validation.get("subjects") or {}) != train:
            raise ValueError("V32_RELEASE_VALIDATION_SPLIT")
        for lanes in validation["subjects"].values():
            for row in lanes.values():
                if len(row["parent_sha256"]) != 64 or int(row["row_count"]) < 0 or not all(0 <= i < row["row_count"] for i in row["selected_indices"]):
                    raise ValueError("V32_RELEASE_VALIDATION_ROWS")

        notebook_receipt = {
            "schema": "RealSaS.Truth139V32FinalizeNotebookReceipt.v4",
            "status": "PASS_V32_FINALIZE_NOTEBOOK_HARD_CLOSURE",
            "source_commit_sha": self.source_sha,
            "batch_id": self.batch_id,
            "batch_identity_sha256": digest(self.batch_identity),
            "base_receipt_sha256": sha256_file(self.base_receipt),
            "v2_receipt_sha256": sha256_file(self.v2_receipt),
            "response_receipt_sha256": sha256_file(self.response_receipt),
            "response_svd_rescue_sha256": sha256_file(self.response_svd_rescue),
            "runtime_fingerprint_sha256": sha256_file(self.runtime_fingerprint),
            "training_readiness_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "identity_sha256": receipt["ledger_identity_sha256"]},
            "train_subject_count": len(train),
            "unseen_subject_count": len(unseen),
            "training_performed": False,
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        notebook_receipt["receipt_identity_sha256"] = digest(notebook_receipt)
        notebook_path = self.batch_root / "V32_FINALIZE_NOTEBOOK_RECEIPT_V4.json"
        notebook_path.write_text(json.dumps(notebook_receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")

        print("\n" + "=" * 80)
        print("PASS — TRUTH139 V3.2 PRETRAIN TARGETS READY")
        print("worker SHA     :", self.source_sha)
        print("batch          :", self.batch_id)
        print("train subjects :", len(train))
        print("unseen subjects:", len(unseen))
        print("readiness      :", receipt_path)
        print("next           : IRIS/TESSA V3.2 FULL139 A100 TRAIN Run-All")
        print("=" * 80)
        return receipt
