from __future__ import annotations

from math import pi, sin
from typing import Any

from realsas_compiler_core.types import QualifiedEditableMeshIR, QualifiedMeshSkinIR, QualifiedSkeletonIR


def _camera(camera: dict) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    if camera.get("contract") != "realsas.level_orthographic_z_orbit.v1":
        raise ValueError("CAMERA_CONTRACT_DRIFT")
    right = tuple(float(x) for x in camera["right"])
    up = tuple(float(x) for x in camera["screen_up"])
    half = float(camera["half_extent"])
    if half <= 0:
        raise ValueError("CAMERA_HALF_EXTENT_NONPOSITIVE")
    return right, up, half


def project_point_to_normalized_raster(P: tuple[float, float, float], camera: dict) -> tuple[float, float]:
    right, up, half = _camera(camera)
    x = sum(float(P[i]) * right[i] for i in range(3)) / half
    y = -sum(float(P[i]) * up[i] for i in range(3)) / half
    return float(x), float(y)


def _mesh_rest_xy(mesh: QualifiedEditableMeshIR) -> tuple[list[list[float]], dict[str, int]]:
    rows=[]; index={}
    for i,vertex in enumerate(mesh.vertices):
        xy=(vertex.metadata or {}).get("raster_xy")
        if xy is None or len(xy)!=2:
            raise ValueError(f"RUNTIME_BRIDGE_MESH_VERTEX_MISSING_RASTER:{vertex.canonical_mesh_vertex_id}")
        index[vertex.canonical_mesh_vertex_id]=i
        rows.append([float(xy[0]),float(xy[1])])
    return rows,index


def build_generic_articulation_sweep(
    skeleton: QualifiedSkeletonIR,
    camera: dict,
    *,
    frame_count: int = 48,
    root_translation_fraction: float = 0.012,
    max_rotation_deg: float = 10.0,
) -> list[dict[str, dict[str, float]]]:
    """Generic topology/geometry-conditioned motion used only to demonstrate editability.

    It has no joint-name table and no specimen-specific constants.  Branches on
    opposite sides of the projected root receive opposite phase; depth attenuates
    motion so the sweep remains bounded on arbitrary valid trees.
    """
    if frame_count < 2:
        raise ValueError("frame_count must be >=2")
    by_id={j.canonical_joint_id:j for j in skeleton.joints}
    if skeleton.root_id not in by_id:
        raise ValueError("qualified skeleton root missing")
    root_xy=project_point_to_normalized_raster(by_id[skeleton.root_id].position,camera)
    children={jid:[] for jid in by_id}
    for j in skeleton.joints:
        if j.parent_canonical_id is not None:
            children[j.parent_canonical_id].append(j.canonical_joint_id)
    depth={skeleton.root_id:0};queue=[skeleton.root_id]
    while queue:
        p=queue.pop(0)
        for c in sorted(children[p]):
            depth[c]=depth[p]+1;queue.append(c)
    max_depth=max(depth.values()) if depth else 0
    frames=[]
    for fi in range(frame_count):
        t=fi/(frame_count-1)
        phase=2*pi*t
        state={}
        for jid in sorted(by_id):
            j=by_id[jid]
            xy=project_point_to_normalized_raster(j.position,camera)
            if jid==skeleton.root_id:
                state[jid]={"translation_x":0.0,"translation_y":root_translation_fraction*sin(phase*2.0),"rotation":2.0*sin(phase)}
                continue
            side=-1.0 if xy[0] < root_xy[0] else 1.0
            depth_frac=depth.get(jid,1)/max(1,max_depth)
            amp=max_rotation_deg*(0.35+0.65*depth_frac)
            state[jid]={"rotation":float(side*amp*sin(phase + side*pi/2.0)),"translation_x":0.0,"translation_y":0.0}
        frames.append(state)
    return frames


def build_historical_runtime_request(
    *,
    skeleton: QualifiedSkeletonIR,
    mesh: QualifiedEditableMeshIR,
    mesh_skin: QualifiedMeshSkinIR,
    camera: dict,
    frames: list[dict[str, dict[str, float]]],
    arap_enabled: bool = True,
) -> dict[str, Any]:
    if mesh.mesh_lineage_hash != mesh_skin.mesh_binding_hash:
        raise ValueError("RUNTIME_BRIDGE_MESH_SKIN_LINEAGE_MISMATCH")
    vertices,index=_mesh_rest_xy(mesh)
    triangles=[]
    for face in mesh.faces:
        if len(face)!=3:
            continue
        triangles.append([index[x] for x in face])
    if not triangles:
        raise ValueError("RUNTIME_BRIDGE_REQUIRES_TRIANGLES")
    skin_by_vertex={row.canonical_mesh_vertex_id:dict(row.influences) for row in mesh_skin.rows}
    weights=[]
    for vertex in mesh.vertices:
        row=skin_by_vertex.get(vertex.canonical_mesh_vertex_id)
        if not row:
            raise ValueError(f"RUNTIME_BRIDGE_MISSING_WEIGHT_ROW:{vertex.canonical_mesh_vertex_id}")
        weights.append({str(k):float(v) for k,v in row.items()})
    joints=[]
    for joint in skeleton.joints:
        joints.append({
            "joint_id":joint.canonical_joint_id,
            "parent_id":joint.parent_canonical_id,
            "anchor_xy":list(project_point_to_normalized_raster(joint.position,camera)),
        })
    return {
        "schema":"RealSaS.DemoHistoricalRuntimeRequest.v1",
        "view_index":int(mesh.view_index),
        "skeleton":{"root_id":skeleton.root_id,"joints":joints},
        "mesh":{"vertices_xy":vertices,"triangles":triangles,"weights":weights},
        "frames":frames,
        "clip_intent":"generic_articulation_sweep",
        "deformation":{"arap_enabled":bool(arap_enabled),"arap_anchor_weight":100000.0,"arap_iterations":4},
        "generalization_claim":False,
        "manual_output_injection":False,
    }
