from __future__ import annotations

"""Knight research adapter for rigid equipment roles.

Consumes the existing continuous-skin RUN witness and upgrades equipment
components to explicit rigid handslot attachments. It also emits a single
presentation loadout mask so alternative source equipment variants are not
rendered on top of one another.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.rigid_attachment_v1 import (
    RigidAttachmentBindingV1IR,
    apply_rigid_attachment_weights_v1,
    build_rigid_loadout_face_mask_v1,
    rigid_attachment_edge_report_v1,
)

SCHEMA = "RealSaS.TESSAKnightRigidEquipmentWitness.v1"
EQUIPMENT_GROUP = {
    "1H_Sword": "WEAPON",
    "1H_Sword_Offhand": "WEAPON",
    "2H_Sword": "WEAPON",
    "Spike_Shield": "SHIELD",
    "Badge_Shield": "SHIELD",
    "Round_Shield": "SHIELD",
    "Rectangle_Shield": "SHIELD",
}
DEFAULT_WEAPON = "1H_Sword"
DEFAULT_SHIELD = "Round_Shield"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(value: bool, code: str) -> None:
    if not value:
        raise RuntimeError(code)


def connected_components(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    parent = np.arange(int(vertex_count), dtype=np.int64)

    def find(x: int) -> int:
        x = int(x)
        while int(parent[x]) != x:
            parent[x] = parent[int(parent[x])]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for a, b, c in np.asarray(faces, dtype=np.int64):
        union(int(a), int(b))
        union(int(b), int(c))
        union(int(c), int(a))

    groups: dict[int, list[int]] = {}
    for vi in range(int(vertex_count)):
        groups.setdefault(find(vi), []).append(vi)
    return tuple(
        np.asarray(rows, dtype=np.int64)
        for _, rows in sorted(groups.items(), key=lambda item: min(item[1]))
    )


def component_object_map(*, audit: dict, source_faces: np.ndarray, vertex_count: int):
    owner = np.empty(int(vertex_count), dtype=object)
    owner[:] = None
    ranges = {}
    for raw in tuple(audit.get("mesh_objects") or ()):
        row = dict(raw)
        name = str(row["name"])
        start = int(row["vertex_start"])
        end = start + int(row["vertex_count"])
        require(0 <= start < end <= len(owner), "RIGID_WITNESS_OBJECT_RANGE_INVALID:" + name)
        require(all(value is None for value in owner[start:end]), "RIGID_WITNESS_OBJECT_RANGE_OVERLAP:" + name)
        owner[start:end] = name
        ranges[name] = (start, end)
    require(all(value is not None for value in owner), "RIGID_WITNESS_OBJECT_ACCOUNTING_INCOMPLETE")

    comps = connected_components(vertex_count, source_faces)
    vertex_component = np.full(vertex_count, -1, dtype=np.int64)
    object_by_component = {}
    for ci, comp in enumerate(comps):
        names = {str(owner[int(v)]) for v in comp}
        require(len(names) == 1, "RIGID_WITNESS_COMPONENT_CROSSES_OBJECT")
        object_by_component[int(ci)] = next(iter(names))
        vertex_component[comp] = int(ci)

    source_face_component = np.asarray([
        int(vertex_component[int(face[0])]) for face in np.asarray(source_faces, dtype=np.int64)
    ])
    for fi, face in enumerate(np.asarray(source_faces, dtype=np.int64)):
        require(
            len({int(vertex_component[int(v)]) for v in face}) == 1,
            "RIGID_WITNESS_FACE_CROSSES_COMPONENT:" + str(fi),
        )
    return comps, object_by_component, ranges, source_face_component


def build_equipment_bindings(*, audit, source_faces, source_skin, bone_names, tessa_vci, tessa_fci):
    comps, object_by_component, ranges, source_face_component = component_object_map(
        audit=audit,
        source_faces=source_faces,
        vertex_count=len(source_skin),
    )
    component_ids = sorted(set(map(int, np.asarray(tessa_vci).tolist())))
    require(component_ids == list(range(len(comps))), "RIGID_WITNESS_TESSA_COMPONENT_ID_DRIFT")
    for ci, comp in enumerate(comps):
        source_counts = (int(len(comp)), int(np.sum(source_face_component == ci)))
        tessa_counts = (
            int(np.sum(np.asarray(tessa_vci) == ci)),
            int(np.sum(np.asarray(tessa_fci) == ci)),
        )
        require(source_counts == tessa_counts, "RIGID_WITNESS_COMPONENT_TOPOLOGY_DRIFT:" + str(ci))

    bindings = []
    joint_index = {str(name): i for i, name in enumerate(bone_names)}
    for object_name, group in sorted(EQUIPMENT_GROUP.items()):
        require(object_name in ranges, "RIGID_WITNESS_EQUIPMENT_OBJECT_MISSING:" + object_name)
        start, end = ranges[object_name]
        rows = np.asarray(source_skin[start:end], dtype=np.float64)
        active = np.flatnonzero(np.max(rows, axis=0) > 1.0e-8)
        require(len(active) == 1, "RIGID_WITNESS_EQUIPMENT_NOT_RIGID:" + object_name)
        bi = int(active[0])
        require(float(np.max(np.abs(rows[:, bi] - 1.0))) <= 1.0e-8, "RIGID_WITNESS_EQUIPMENT_WEIGHT_DRIFT:" + object_name)
        joint_id = str(bone_names[bi])
        require(joint_id in ("handslot.l", "handslot.r"), "RIGID_WITNESS_SLOT_INVALID:" + object_name)
        comp_ids = tuple(ci for ci, name in object_by_component.items() if name == object_name)
        bindings.append(
            RigidAttachmentBindingV1IR(
                attachment_id=object_name,
                joint_id=joint_id,
                component_ids=comp_ids,
                variant_group_id=group,
                metadata={
                    "deformable_skin_interpolation_allowed": False,
                    "source_teacher_object_range": [int(start), int(end)],
                    "source_teacher_joint_index": int(joint_index[joint_id]),
                },
            )
        )
    return tuple(bindings), object_by_component


def recompute_poses(vertices: np.ndarray, weights: np.ndarray, skin_matrices: np.ndarray) -> np.ndarray:
    homo = np.concatenate((np.asarray(vertices, dtype=np.float64), np.ones((len(vertices), 1))), axis=1)
    out = []
    for matrices in np.asarray(skin_matrices, dtype=np.float64):
        transformed = np.einsum("jab,nb->jna", matrices, homo, optimize=True)[:, :, :3]
        out.append(np.einsum("nj,jna->na", weights, transformed, optimize=True))
    poses = np.asarray(out)
    require(np.isfinite(poses).all(), "RIGID_WITNESS_POSE_NONFINITE")
    return poses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-witness-npz", required=True)
    ap.add_argument("--teacher-npz", required=True)
    ap.add_argument("--teacher-audit-json", required=True)
    ap.add_argument("--tessa-mesh-npz", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--weapon-object", default=DEFAULT_WEAPON)
    ap.add_argument("--shield-object", default=DEFAULT_SHIELD)
    args = ap.parse_args(argv)

    base_path = Path(args.base_witness_npz).resolve()
    teacher_path = Path(args.teacher_npz).resolve()
    audit_path = Path(args.teacher_audit_json).resolve()
    tessa_path = Path(args.tessa_mesh_npz).resolve()

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    bone_names = tuple(map(str, audit["bone_names"]))
    require(len(bone_names) == 41, "RIGID_WITNESS_BONE_COUNT_DRIFT")

    with np.load(teacher_path, allow_pickle=False) as z:
        source_faces = np.asarray(z["faces"], dtype=np.int64)
        source_skin = np.asarray(z["skin"], dtype=np.float64)
    with np.load(tessa_path, allow_pickle=False) as z:
        tessa_vci = np.asarray(z["vertex_component_indices"], dtype=np.int64)
        tessa_fci = np.asarray(z["face_component_indices"], dtype=np.int64)
    with np.load(base_path, allow_pickle=False) as z:
        vertices = np.asarray(z["vertices_rest"], dtype=np.float64)
        faces = np.asarray(z["faces"], dtype=np.int64)
        weights_continuous = np.asarray(z["weights_continuous"], dtype=np.float64)
        skin_matrices = np.asarray(z["skin_matrices"], dtype=np.float64)
        times = np.asarray(z["times"], dtype=np.float64)
        joint_positions = np.asarray(z["joint_positions"], dtype=np.float64)

    require(weights_continuous.shape == (len(vertices), 23), "RIGID_WITNESS_BASE_WEIGHT_SHAPE_DRIFT")
    bindings, _ = build_equipment_bindings(
        audit=audit,
        source_faces=source_faces,
        source_skin=source_skin,
        bone_names=bone_names,
        tessa_vci=tessa_vci,
        tessa_fci=tessa_fci,
    )
    joint_index = {str(name): i for i, name in enumerate(bone_names[:23])}
    weights = apply_rigid_attachment_weights_v1(
        weights_continuous,
        vertex_component_ids=tessa_vci,
        bindings=bindings,
        joint_index_by_id=joint_index,
    )
    poses = recompute_poses(vertices, weights, skin_matrices)

    weapon = str(args.weapon_object)
    shield = str(args.shield_object)
    selected = (weapon, shield)
    require(EQUIPMENT_GROUP.get(weapon) == "WEAPON", "RIGID_WITNESS_WEAPON_INVALID")
    require(EQUIPMENT_GROUP.get(shield) == "SHIELD", "RIGID_WITNESS_SHIELD_INVALID")
    face_mask = build_rigid_loadout_face_mask_v1(
        faces,
        vertex_component_ids=tessa_vci,
        bindings=bindings,
        selected_attachment_ids=selected,
    )

    by_id = {binding.attachment_id: binding for binding in bindings}
    rigidity = {
        aid: rigid_attachment_edge_report_v1(
            vertices,
            poses,
            faces,
            vertex_component_ids=tessa_vci,
            binding=by_id[aid],
        )
        for aid in selected
    }
    require(all(row["rigid_edge_preservation_pass"] for row in rigidity.values()), "RIGID_WITNESS_DYNAMIC_RIGIDITY_FAIL")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "TESSA_KNIGHT_RIGID_EQUIPMENT_RUN_WITNESS.npz",
        vertices_rest=vertices,
        faces=faces,
        poses=poses,
        weights_mechanical=weights,
        skin_matrices=skin_matrices,
        joint_positions=joint_positions,
        times=times,
        vertex_component_indices=tessa_vci,
        face_component_indices=tessa_fci,
        presentation_face_visible=face_mask,
    )
    report = {
        "schema": SCHEMA,
        "status": "PASS_RESEARCH_WITNESS__NOT_PRODUCT_AUTHORITY",
        "input_sha256": {
            "base_witness_npz": sha256(base_path),
            "teacher_npz": sha256(teacher_path),
            "teacher_audit_json": sha256(audit_path),
            "tessa_mesh_npz": sha256(tessa_path),
        },
        "role_contract": {
            "deformable_surface": "CONTINUOUS_SKIN_FIELD",
            "rigid_attachment": "ONE_HOT_QUALIFIED_JOINT",
            "control_only": "MOTION_EVIDENCE__NOT_SKIN_TARGET",
        },
        "equipment_bindings": [binding.to_dict() for binding in bindings],
        "presentation_loadout": {
            "weapon": weapon,
            "shield": shield,
            "selection_authority": "RESEARCH_PRESENTATION_ONLY__NOT_PRODUCT_AUTHORITY",
            "visible_face_count": int(np.sum(face_mask)),
            "hidden_face_count": int(len(face_mask) - np.sum(face_mask)),
        },
        "dynamic_rigidity": rigidity,
        "claims": {
            "equipment_skin_interpolation_used": False,
            "equipment_variant_multiplexing_rendered": False,
            "product_authority": False,
            "generalization": False,
        },
    }
    report_path = out / "TESSA_KNIGHT_RIGID_EQUIPMENT_RUN_WITNESS.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("TESSA_KNIGHT_RIGID_EQUIPMENT_RUN_WITNESS_PASS", report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
