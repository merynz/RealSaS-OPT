from __future__ import annotations

"""One-surface geometry identity for StdGEN++ -> TESSA -> AXIS/MIRA.

This is a fail-closed topology ADDRESS and gradient bridge, not a replacement
high-resolution extractor, not a learned Topology-V3.2 network and not a
ProductRevision admission.  Geometry enters only ONCE, from IRIS semantic
extraction (e.g. pinned FlexiCubes). TESSA can keep its source vertex/face
identities, or subdivide the SAME indexed surface *before* AXIS/MIRA decode.
New vertices carry explicit barycentric parent support, preserving gradients
to the original shared H(x) field and excluding a second independently
generated visual/rig mesh. No post-skin remeshing is authorized.
"""

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class TESSASeededSurfaceV1:
    xyz: Tensor                # [1,V,3] derived from ONE IRIS witness
    faces: Tensor              # [F,3] original or one-to-four subdivision
    vertex_component_ids: Tensor
    face_component_ids: Tensor
    # Each V row has exact parent pairs. (i,i) denotes original seed vertex.
    parent_seed_edges: tuple[tuple[int, int], ...]
    parent_seed_weights: tuple[tuple[float, float], ...]
    original_vertex_count: int
    original_face_count: int
    original_faces: Tensor      # original M0 indexed triangles, before TESSA edits
    scene_sha256: str
    original_surface_sha256: str
    source_authority: str = "ONE_IRIS_SURFACE_TESSA_REFINEMENT_CANDIDATE"
    canonical_mesh_authority: bool = False

    def validate(self) -> None:
        n = self.xyz.shape[1] if self.xyz.ndim == 3 else -1
        if (self.xyz.ndim != 3 or self.xyz.shape[0] != 1
                or self.xyz.shape[2] != 3 or n < 3
                or not bool(torch.isfinite(self.xyz).all())
                or self.faces.ndim != 2 or self.faces.shape[1] != 3
                or self.faces.dtype != torch.long or not self.faces.numel()
                or bool((self.faces < 0).any())
                or int(self.faces.max()) >= n
                or self.vertex_component_ids.shape != (n,)
                or self.face_component_ids.shape != (len(self.faces),)
                or self.canonical_mesh_authority
                or self.source_authority != "ONE_IRIS_SURFACE_TESSA_REFINEMENT_CANDIDATE"
                or len(self.parent_seed_edges) != n
                or len(self.parent_seed_weights) != n
                or self.original_faces.shape != (self.original_face_count, 3)
                or self.original_faces.dtype != torch.long
                or self.original_faces.device != self.faces.device
                or bool((self.original_faces < 0).any())
                or int(self.original_faces.max()) >= self.original_vertex_count):
            raise ValueError("TESSA_SEEDED_SURFACE_DOMAIN_INVALID")
        for (a, b), (wa, wb) in zip(
            self.parent_seed_edges, self.parent_seed_weights
        ):
            if (not 0 <= a < self.original_vertex_count
                    or not 0 <= b < self.original_vertex_count
                    or (wa, wb) not in ((1.0, 0.0), (0.5, 0.5))
                    or (a == b) != (wa == 1.0)):
                raise ValueError("TESSA_SEEDED_PARENT_VERTEX_SUPPORT_DRIFT")
        if (self.original_vertex_count < 3 or self.original_face_count < 1
                or any(type(s) is not str or len(s) != 64
                       or any(c not in "0123456789abcdef" for c in s)
                       for s in (self.scene_sha256, self.original_surface_sha256))):
            raise ValueError("TESSA_SEEDED_SCENE_OR_SURFACE_ID_INVALID")


def build_tessa_seeded_surface_v1(
    *,
    iris_surface_xyz: Tensor,
    iris_surface_faces: Tensor,
    component_ids_by_vertex: Tensor,
    scene_sha256: str,
    original_surface_sha256: str,
    one_to_four_refine: bool = False,
) -> TESSASeededSurfaceV1:
    """Return an exact indexed surface, without *any* independent TESSA mesh.

    Small structural subdivision is a topologically safe executable identity
    proof for seeded refinement, NOT StdGEN++ high-resolution sparse extraction.
    Its parent addresses are fixed before AXIS/MIRA and all positions are
    differentiable functions of the original IRIS vertex positions. Distinct
    semantic component boundaries are NEVER welded.
    """
    v = iris_surface_xyz.shape[1] if iris_surface_xyz.ndim == 3 else -1
    faces = iris_surface_faces
    if (iris_surface_xyz.ndim != 3 or iris_surface_xyz.shape[0] != 1
            or iris_surface_xyz.shape[-1] != 3 or v < 3
            or not bool(torch.isfinite(iris_surface_xyz).all())
            or faces.ndim != 2 or faces.shape[-1] != 3
            or faces.dtype != torch.long or not faces.numel()
            or bool((faces < 0).any()) or int(faces.max()) >= v
            or component_ids_by_vertex.shape != (v,)
            or component_ids_by_vertex.dtype != torch.long
            or faces.device != iris_surface_xyz.device
            or component_ids_by_vertex.device != faces.device
            or type(one_to_four_refine) is not bool):
        raise ValueError("TESSA_SEEDED_INPUT_DOMAIN_INVALID")
    if not bool((
        component_ids_by_vertex[faces]
        == component_ids_by_vertex[faces[:, 0:1]]
    ).all()):
        raise ValueError("TESSA_SEEDED_CROSS_COMPONENT_TRIANGLE")
    if bool(torch.any(
        (faces[:, 0] == faces[:, 1]) |
        (faces[:, 1] == faces[:, 2]) |
        (faces[:, 0] == faces[:, 2])
    )):
        raise ValueError("TESSA_SEEDED_DUPLICATE_TRIANGLE_CORNER")
    tri = iris_surface_xyz[0, faces]
    area = torch.linalg.vector_norm(
        torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1),
        dim=-1,
    )
    if not bool((area > 1e-11).all()):
        raise ValueError("TESSA_SEEDED_DEGENERATE_REST_TRIANGLE")
    parents = [(i, i) for i in range(v)]
    weights = [(1.0, 0.0) for _ in range(v)]
    if not one_to_four_refine:
        result = TESSASeededSurfaceV1(
            xyz=iris_surface_xyz, faces=faces,
            vertex_component_ids=component_ids_by_vertex,
            face_component_ids=component_ids_by_vertex[faces[:, 0]],
            parent_seed_edges=tuple(parents), parent_seed_weights=tuple(weights),
            original_vertex_count=v, original_face_count=len(faces),
            original_faces=faces,
            scene_sha256=scene_sha256,
            original_surface_sha256=original_surface_sha256,
        )
        result.validate()
        return result

    # Uniform one-to-four topology ensures no T-junctions at neighboring faces.
    # The high-resolution production path will use an adaptive SAME-surface
    # refinement that must prove analogous boundary and parent-address rules.
    face_indices = faces.detach().cpu().tolist()
    indexed_edges = sorted({
        tuple(sorted((a, b))) for a, b, c in face_indices
        for a, b in ((a, b), (b, c), (c, a))
    })
    mid_id = {edge: v + i for i, edge in enumerate(indexed_edges)}
    pairs = torch.tensor(indexed_edges, dtype=torch.long, device=faces.device)
    mid = 0.5 * (iris_surface_xyz[:, pairs[:, 0]] +
                 iris_surface_xyz[:, pairs[:, 1]])
    xyz = torch.cat((iris_surface_xyz, mid), dim=1)
    comp_mid = component_ids_by_vertex[pairs[:, 0]]
    vertex_components = torch.cat((component_ids_by_vertex, comp_mid))
    out_faces = []
    for a, b, c in face_indices:
        ab = mid_id[tuple(sorted((a, b)))]
        bc = mid_id[tuple(sorted((b, c)))]
        ca = mid_id[tuple(sorted((c, a)))]
        out_faces.extend((
            (a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca),
        ))
    final_faces = torch.tensor(
        out_faces, dtype=torch.long, device=faces.device,
    )
    parents.extend(indexed_edges)
    weights.extend([(0.5, 0.5) for _ in indexed_edges])
    result = TESSASeededSurfaceV1(
        xyz=xyz, faces=final_faces,
        vertex_component_ids=vertex_components,
        face_component_ids=vertex_components[final_faces[:, 0]],
        parent_seed_edges=tuple(parents), parent_seed_weights=tuple(weights),
        original_vertex_count=v, original_face_count=len(faces),
        original_faces=faces,
        scene_sha256=scene_sha256,
        original_surface_sha256=original_surface_sha256,
    )
    result.validate()
    return result


def tessa_adaptive_split_edges_v1(
    seed: TESSASeededSurfaceV1,
    *,
    importance_logits: Tensor,
    max_added_vertices: int,
    length_gate_ratio: float = 1.0,
) -> TESSASeededSurfaceV1:
    """Conforming per-edge adaptive refinement with EXACT original parent IDs.

    Discrete decisions are NOT differentiated. Once chosen, each new position
    is a differentiable midpoint of the same source surface. All incident
    triangles share one split vertex, so no T-junctions are created.
    Topology operations finish before AXIS/MIRA, UV packing or canonical seal.
    Neither welding nor collapsing a semantic garment shell is permitted.
    """
    seed.validate()
    n = seed.xyz.shape[1]
    if (
        n != seed.original_vertex_count
        or importance_logits.shape != (1, n)
        or importance_logits.device != seed.xyz.device
        or not bool(torch.isfinite(importance_logits).all())
        or type(max_added_vertices) is not int
        or not 0 <= max_added_vertices <= 32768
        or type(length_gate_ratio) not in (int, float)
        or not 0.5 <= length_gate_ratio <= 4.0
    ):
        raise ValueError("TESSA_ADAPTIVE_SOURCE_OR_POLICY")
    if max_added_vertices == 0:
        return seed
    faces = seed.faces.detach().cpu().tolist()
    edges = sorted({
        tuple(sorted((a,b)))
        for a,b,c in faces for a,b in ((a,b),(b,c),(c,a))
    })
    pairs = torch.tensor(edges,dtype=torch.long,device=seed.xyz.device)
    lengths = torch.linalg.vector_norm(
        seed.xyz[0,pairs[:,0]]-seed.xyz[0,pairs[:,1]],dim=-1,
    )
    if not bool((lengths > 1.e-10).all()):
        raise ValueError("TESSA_ADAPTIVE_DEGENERATE_EDGE")
    # Global scale is an explicit, source-independent mesh metric, not ground
    # truth. Trained logits are a bounded local density preference only.
    baseline = torch.median(lengths.detach())
    preference = torch.sigmoid(
        0.5 * (importance_logits[0,pairs[:,0]] +
               importance_logits[0,pairs[:,1]])
    ).detach()
    target = baseline * float(length_gate_ratio) * (
        1.5 - preference
    )
    eligible = torch.nonzero(lengths.detach() > target,as_tuple=False).flatten()
    if not len(eligible):
        return seed
    # Prefer worst relative edge excess. Stable index is a tie break.
    order = sorted(
        eligible.detach().cpu().tolist(),
        key=lambda i:(-float((lengths[i]/target[i]).item()),edges[i]),
    )[:max_added_vertices]
    selected = sorted(edges[i] for i in order)
    index = {edge:n+i for i,edge in enumerate(selected)}
    picked = torch.tensor(selected,dtype=torch.long,device=seed.xyz.device)
    midpoint = (seed.xyz[:,picked[:,0]]+seed.xyz[:,picked[:,1]])*0.5
    xyz = torch.cat((seed.xyz,midpoint),dim=1)
    comp = torch.cat((
        seed.vertex_component_ids,
        seed.vertex_component_ids[picked[:,0]],
    ))
    triangles = []
    for a,b,c in faces:
        ab = index.get(tuple(sorted((a,b))))
        bc = index.get(tuple(sorted((b,c))))
        ca = index.get(tuple(sorted((c,a))))
        if ab is None and bc is None and ca is None:
            triangles.append((a,b,c))
        elif ab is not None and bc is None and ca is None:
            triangles.extend(((a,ab,c),(ab,b,c)))
        elif ab is None and bc is not None and ca is None:
            triangles.extend(((a,b,bc),(a,bc,c)))
        elif ab is None and bc is None and ca is not None:
            triangles.extend(((a,b,ca),(b,c,ca)))
        elif ab is not None and bc is not None and ca is None:
            triangles.extend(((a,ab,c),(ab,bc,c),(ab,b,bc)))
        elif ab is None and bc is not None and ca is not None:
            triangles.extend(((a,b,ca),(b,bc,ca),(bc,c,ca)))
        elif ab is not None and bc is None and ca is not None:
            triangles.extend(((a,ab,ca),(ab,b,c),(ab,c,ca)))
        else:
            triangles.extend((
                (a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca),
            ))
    output_faces = torch.tensor(
        triangles,dtype=torch.long,device=seed.xyz.device,
    )
    normals = torch.cross(
        xyz[:,output_faces[:,1]]-xyz[:,output_faces[:,0]],
        xyz[:,output_faces[:,2]]-xyz[:,output_faces[:,0]],
        dim=-1,
    )
    # All triangles share the original winding; numerical degeneracy ABSTAINS.
    if not bool((torch.linalg.vector_norm(normals,dim=-1) > 1.e-11).all()):
        raise ValueError("TESSA_ADAPTIVE_TRIANGLE_DEGENERACY")
    result = TESSASeededSurfaceV1(
        xyz=xyz,faces=output_faces,
        vertex_component_ids=comp,
        face_component_ids=comp[output_faces[:,0]],
        parent_seed_edges=seed.parent_seed_edges+tuple(selected),
        parent_seed_weights=seed.parent_seed_weights
                           +tuple((0.5,0.5) for _ in selected),
        original_vertex_count=seed.original_vertex_count,
        original_face_count=seed.original_face_count,
        original_faces=seed.original_faces,
        scene_sha256=seed.scene_sha256,
        original_surface_sha256=seed.original_surface_sha256,
    )
    result.validate()
    return result


def tessa_seed_from_flexicubes_v1(
    *,
    witness,
    scene_triplanes: Tensor,
) -> TESSASeededSurfaceV1:
    """One actual IRIS FlexiCubes extraction as the ONLY topology seed.

    Stable connected components are computed by face incidence, NOT XYZ
    proximity: identical body/garment positions never weld. This is a
    source-INDEPENDENT student prediction, not canonical/source authority.
    The two SHA256 bind the exact scene and extracted candidate bytes.
    """
    from hashlib import sha256
    import numpy as np
    from .flexicubes_surface_evidence import RSSvNextFlexiCubesSurfaceEvidence

    if (not isinstance(witness, RSSvNextFlexiCubesSurfaceEvidence)
            or witness.evidence_authority
                != "UNQUALIFIED_FLEXICUBES_SURFACE_WITNESS"
            or scene_triplanes.ndim != 5 or scene_triplanes.shape[0] != 1
            or witness.xyz.device != scene_triplanes.device
            or witness.faces.device != scene_triplanes.device
            or witness.xyz.dtype != scene_triplanes.dtype):
        raise ValueError("TESSA_SEED_IRIS_WITNESS_OR_SCENE_REQUIRED")
    if (witness.xyz.shape[1] < 3
            or witness.faces.ndim != 2 or not witness.faces.numel()
            or witness.faces.dtype != torch.long):
        raise ValueError("TESSA_SEED_INDEXED_FACE_DOMAIN")

    def _hash_tensor(t: Tensor) -> str:
        t0 = t.detach().contiguous().cpu()
        v = t0.numpy()
        digest = sha256()
        digest.update(str((str(v.dtype), v.shape)).encode())
        digest.update(v.tobytes())
        return digest.hexdigest()

    # Union-find on EXACT integer connectivity, no geometric nearest search.
    v = witness.xyz.shape[1]
    parent = list(range(v))
    rank = [0] * v

    def root(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def join(a: int, b: int) -> None:
        a, b = root(a), root(b)
        if a == b:
            return
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1

    faces = witness.faces.detach().cpu().numpy().astype(np.int64, copy=False)
    if faces.size == 0 or int(faces.min()) < 0 or int(faces.max()) >= v:
        raise ValueError("TESSA_SEED_FACE_VERTEX_RANGE")
    for a, b, c in faces:
        if a == b or b == c or a == c:
            raise ValueError("TESSA_SEED_DEGENERATE_INDEXED_FACE")
        join(int(a), int(b))
        join(int(b), int(c))
    groups = {}
    component_ids = []
    # Isolated vertices are components too; admission can reject them later.
    for i in range(v):
        k = root(i)
        if k not in groups:
            groups[k] = len(groups)
        component_ids.append(groups[k])
    comps = torch.as_tensor(
        component_ids, dtype=torch.long, device=witness.faces.device,
    )
    result = build_tessa_seeded_surface_v1(
        iris_surface_xyz=witness.xyz,
        iris_surface_faces=witness.faces,
        component_ids_by_vertex=comps,
        scene_sha256=_hash_tensor(scene_triplanes),
        original_surface_sha256=sha256((
            _hash_tensor(witness.xyz) + _hash_tensor(witness.faces)
            + _hash_tensor(comps)
        ).encode()).hexdigest(),
    )
    return result


__all__ = [
    "TESSASeededSurfaceV1",
    "build_tessa_seeded_surface_v1",
    "tessa_seed_from_flexicubes_v1",
    "tessa_adaptive_split_edges_v1",
]
