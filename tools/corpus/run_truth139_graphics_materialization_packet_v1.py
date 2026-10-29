#!/usr/bin/env python3
"""Run the canonical Truth139 Graphics chain from one verified prepared observation packet.

The packet is first re-qualified by the existing observation verifier and the response-independent
rest-surface packet verifier.  Only then is it mapped into the lower-level subject worker.  This is
the intended notebook call surface: ``raw source + prepared packet + frozen work orders``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.corpus.prepare_iris_observations_v1 import (
    SURFACE_GEOMETRY_NAME,
    verify_surface_geometry_packet,
)
from tools.corpus.run_truth139_graphics_materialization_subject_v1 import (
    REPO_ROOT,
    build_command_plan,
    execute_plan,
    file_sha256,
    seal_chain,
    validate_request,
)
from tools.training.prepare_iris_cohort_v1 import verify_observations


def packet_paths(packet: Path, subject_id: str) -> dict[str, Path]:
    packet = Path(packet).resolve()
    if packet.name != subject_id:
        raise ValueError(f"TRUTH139_GRAPHICS_PACKET_SUBJECT_DIRECTORY:{packet.name}!={subject_id}")
    required = {
        "geometry_npz": packet / SURFACE_GEOMETRY_NAME,
        "source_scene_snapshot": packet / "SOURCE_SCENE_SNAPSHOT.json",
        "camera_bundle": packet / "camera_bundle.json",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError("TRUTH139_GRAPHICS_PACKET_REQUIRED_FILE:" + name + ":" + str(path))
    for view in range(8):
        for suffix in (".png", ".mask.bin"):
            path = packet / f"V{view}{suffix}"
            if not path.is_file():
                raise FileNotFoundError("TRUTH139_GRAPHICS_PACKET_VIEW_FILE:" + str(path))
    return {**required, "observation_dir": packet}


def qualified_request(
    *,
    work_orders_path: Path,
    subject_id: str,
    raw_source_path: Path,
    prepared_packet: Path,
    output_root: Path,
    code_commit_sha: str,
) -> dict:
    work_orders_path = Path(work_orders_path).resolve()
    work = json.loads(work_orders_path.read_text())
    matches = [row for row in work.get("subjects") or () if str(row.get("subject_id") or "") == subject_id]
    if len(matches) != 1:
        raise ValueError("TRUTH139_GRAPHICS_PACKET_WORK_ORDER_SUBJECT")
    subject = matches[0]
    source_row = dict(subject.get("source") or {})
    roster_row = {"subject_key": subject_id, "source": source_row}
    paths = packet_paths(prepared_packet, subject_id)

    # Existing courts are deliberately reused instead of inventing notebook-side trust.
    folder, proof, observation_binding = verify_observations(Path(prepared_packet).resolve().parent, roster_row)
    if folder.resolve() != Path(prepared_packet).resolve():
        raise ValueError("TRUTH139_GRAPHICS_PACKET_OBSERVATION_FOLDER_DRIFT")
    surface = verify_surface_geometry_packet(folder, roster_row)
    if proof.get("source_sha256") != source_row.get("sha256"):
        raise ValueError("TRUTH139_GRAPHICS_PACKET_SOURCE_BINDING")
    request = validate_request(
        work_orders_path=work_orders_path,
        subject_id=subject_id,
        raw_source_path=raw_source_path,
        geometry_npz=paths["geometry_npz"],
        source_scene_snapshot=paths["source_scene_snapshot"],
        camera_bundle=paths["camera_bundle"],
        observation_dir=paths["observation_dir"],
        output_root=output_root,
        code_commit_sha=code_commit_sha,
    )
    request["prepared_packet"] = {
        "path": str(folder),
        "observation_binding_sha256": observation_binding,
        "observation_admission_sha256": file_sha256(folder / "IRIS_OBSERVATION_ADMISSION.json"),
        "surface_geometry_sha256": surface["geometry_sha256"],
        "surface_geometry_manifest_sha256": surface["manifest_sha256"],
    }
    return request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-orders", required=True, type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--raw-source", required=True, type=Path)
    parser.add_argument("--prepared-packet", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--code-commit-sha", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    request = qualified_request(
        work_orders_path=args.work_orders,
        subject_id=args.subject_id,
        raw_source_path=args.raw_source,
        prepared_packet=args.prepared_packet,
        output_root=args.output_root,
        code_commit_sha=args.code_commit_sha,
    )
    plan = build_command_plan(request, subject_id=args.subject_id, device=args.device)
    if args.plan_only:
        print(json.dumps({
            "status": "PASS_TRUTH139_GRAPHICS_PREPARED_PACKET_PLAN_ONLY",
            "subject_id": args.subject_id,
            "prepared_packet": request["prepared_packet"],
            "bindings": request["bindings"],
            "steps": [{**step, "expected_output": str(step["expected_output"])} for step in plan],
        }, sort_keys=True, default=str), flush=True)
        return
    steps = execute_plan(plan)
    chain_path = args.output_root.resolve() / "GRAPHICS_MATERIALIZATION_CHAIN.json"
    chain = seal_chain(request=request, subject_id=args.subject_id, steps=steps, output=chain_path)
    chain["prepared_packet"] = request["prepared_packet"]
    print(json.dumps({
        "status": chain["status"],
        "subject_id": args.subject_id,
        "chain_path": str(chain_path),
        "chain_sha256": file_sha256(chain_path),
        "prepared_packet_observation_binding_sha256": request["prepared_packet"]["observation_binding_sha256"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
