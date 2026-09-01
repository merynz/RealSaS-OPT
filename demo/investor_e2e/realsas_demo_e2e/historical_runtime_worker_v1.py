from __future__ import annotations

"""Worker executed with PYTHONPATH pointed at the exact historical v0.5 compiler.

Do not import current RealSaS compiler modules here.  The boundary is JSON only.
"""

from pathlib import Path
from types import SimpleNamespace
import json
import sys

from realsas_reference_runtime.deformation_evaluator import ReferenceDeformationEvaluator


def ns(**kwargs):
    return SimpleNamespace(**kwargs)


def build_evaluator(req: dict) -> ReferenceDeformationEvaluator:
    controls = tuple(
        ns(control_id=str(row["joint_id"]), anchor=tuple(map(float, row["anchor_xy"])), metadata={})
        for row in req["skeleton"]["joints"]
    )
    edges = tuple(
        ns(child_control_id=str(row["joint_id"]), parent_control_id=str(row["parent_id"]))
        for row in req["skeleton"]["joints"]
        if row.get("parent_id") is not None
    )
    rig = ns(controls=controls, edges=edges)

    mesh_row = req["mesh"]
    mesh = ns(
        mesh_id="DEMO_VIEW_MESH",
        vertices=tuple(tuple(map(float, p)) for p in mesh_row["vertices_xy"]),
        triangles=tuple(tuple(map(int, tri)) for tri in mesh_row["triangles"]),
        metadata={"final_render_authority": True},
    )
    mesh_plan = ns(meshes=(mesh,))

    binding = ns(
        mesh_id=mesh.mesh_id,
        per_vertex_weights=tuple(
            {str(k): float(v) for k, v in row.items()}
            for row in mesh_row["weights"]
        ),
        metadata={},
    )
    weight_plan = ns(bindings=(binding,))

    render_order_plan = ns(
        ordered_part_ids_by_view={str(req.get("view_index", 0)): [mesh.mesh_id]},
        transitions=(),
    )

    arap_enabled = bool(req.get("deformation", {}).get("arap_enabled", True))
    contract = ns(
        contract_id="DEMO_HISTORICAL_ARAP_CONTRACT",
        mesh_id=mesh.mesh_id,
        arap_enabled=arap_enabled,
        primary_skinning_policy="linear_blend_skinning",
        corrective_backend="production_arap_local_global_2d",
        executable_schema_version="realSaS.DeformationContract.demo_bridge.v1",
        control_ids=tuple(c.control_id for c in controls),
        arap_anchor_vertex_indices_by_control={},
        arap_anchor_weight=float(req.get("deformation", {}).get("arap_anchor_weight", 100000.0)),
        arap_iterations=int(req.get("deformation", {}).get("arap_iterations", 4)),
        secondary_simulation_enabled=False,
        secondary_domains=(),
        contact_domains=(),
    )
    deformation_plan = ns(contracts=(contract,))
    return ReferenceDeformationEvaluator(
        rig=rig,
        mesh_plan=mesh_plan,
        weight_plan=weight_plan,
        render_order_plan=render_order_plan,
        deformation_plan=deformation_plan,
    )


def main(request_path: str, output_path: str) -> None:
    req = json.loads(Path(request_path).read_text(encoding="utf-8"))
    evaluator = build_evaluator(req)
    frames = []
    for index, local_states in enumerate(req["frames"]):
        frame = evaluator.evaluate_local_states(
            local_states,
            clip_intent=str(req.get("clip_intent", "demo_motion")),
            time01=float(index / max(1, len(req["frames"]) - 1)),
            semantic_context={"demo_frame_index": index, "historical_runtime_bridge": True},
        )
        frames.append({
            "frame_index": index,
            "vertices_xy": [list(p) for p in frame.mesh_vertices_by_id["DEMO_VIEW_MESH"]],
            "control_pose": frame.control_pose.world_state_summaries,
            "deformation_execution": frame.metadata.get("deformation_execution_by_mesh", {}),
            "semantic_sha256": frame.semantic_sha256,
        })
    out = {
        "status": "PASS",
        "historical_evaluator_semantic_version": evaluator.semantic_version,
        "frame_count": len(frames),
        "frames": frames,
        "compiler_artifacts_mutated": False,
    }
    Path(output_path).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: historical_runtime_worker_v1.py REQUEST.json OUTPUT.json")
    main(sys.argv[1], sys.argv[2])
