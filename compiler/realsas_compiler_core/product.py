from __future__ import annotations
from .types import (
    RiggingSurfaceIR,
    QualifiedSkeletonIR,
    QualifiedSkinIR,
    QualifiedEditableMeshIR,
    QualifiedMeshSkinIR,
    CanonicalPuppetGraph,
    CanonicalPuppetGraphV2,
    ProofFrame,
    RuntimePackageIR,
    QualificationError,
)
from .hashing import content_sha256
from .mesh_binding import validate_qualified_mesh, validate_qualified_mesh_skin

ProductGraph = CanonicalPuppetGraph | CanonicalPuppetGraphV2


def assemble_product(surface:RiggingSurfaceIR,skeleton:QualifiedSkeletonIR,skin:QualifiedSkinIR,*,parent_state_hash:str|None=None,deformation_state=None,contact_state=None,motion_bindings=None,editable_metadata=None)->CanonicalPuppetGraph:
    """Legacy V1 product assembly retained while the typed MWB seam is promoted."""
    if skin.surface_binding_hash!=surface.geometry_lineage_hash: raise QualificationError("skin/surface lineage mismatch")
    if skin.skeleton_binding_hash!=skeleton.skeleton_lineage_hash: raise QualificationError("skin/skeleton lineage mismatch")
    ledger=(
        {"stage":"SURFACE","hash":surface.geometry_lineage_hash},
        {"stage":"SKELETON_QUALIFICATION","hash":skeleton.skeleton_lineage_hash,"report":content_sha256(skeleton.qualification_report)},
        {"stage":"SKIN_QUALIFICATION","hash":skin.skin_lineage_hash,"report":content_sha256(skin.qualification_report)},
    )
    payload={"parent":parent_state_hash,"surface":surface.geometry_lineage_hash,"skeleton":skeleton.skeleton_lineage_hash,"skin":skin.skin_lineage_hash,"ledger":ledger,"deformation":deformation_state or {},"contact":contact_state or {},"motion":motion_bindings or {},"editable":editable_metadata or {}}
    state=content_sha256(payload); lineage="PUPPET:"+content_sha256({"genesis":state,"parent":parent_state_hash})[:24]
    return CanonicalPuppetGraph(lineage,state,parent_state_hash,surface.geometry_lineage_hash,skeleton.skeleton_lineage_hash,skin.skin_lineage_hash,ledger,deformation_state or {},contact_state or {},motion_bindings or {},editable_metadata or {})


def assemble_product_v2(
    surface: RiggingSurfaceIR,
    skeleton: QualifiedSkeletonIR,
    skin: QualifiedSkinIR,
    mesh: QualifiedEditableMeshIR,
    mesh_skin: QualifiedMeshSkinIR,
    *,
    parent_state_hash: str | None = None,
    deformation_state=None,
    contact_state=None,
    motion_bindings=None,
    editable_metadata=None,
) -> CanonicalPuppetGraphV2:
    """Assemble the single mesh-aware canonical product state defined by MWB-0.

    This function validates typed lineage and admissibility invariants only. It does
    not generate mesh topology, synthesize skin semantics, or promote historical
    numerical solvers.
    """
    if skin.surface_binding_hash != surface.geometry_lineage_hash:
        raise QualificationError("skin/surface lineage mismatch")
    if skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
        raise QualificationError("skin/skeleton lineage mismatch")

    validate_qualified_mesh(mesh, surface)
    validate_qualified_mesh_skin(
        mesh_skin,
        surface=surface,
        skeleton=skeleton,
        skin=skin,
        mesh=mesh,
    )

    ledger = (
        {"stage":"SURFACE","hash":surface.geometry_lineage_hash},
        {"stage":"SKELETON_QUALIFICATION","hash":skeleton.skeleton_lineage_hash,"report":content_sha256(skeleton.qualification_report)},
        {"stage":"SKIN_QUALIFICATION","hash":skin.skin_lineage_hash,"report":content_sha256(skin.qualification_report)},
        {"stage":"MESH_QUALIFICATION","hash":mesh.mesh_lineage_hash,"report":content_sha256(mesh.qualification_report)},
        {"stage":"MESH_SKIN_QUALIFICATION","hash":mesh_skin.mesh_skin_lineage_hash,"report":content_sha256(mesh_skin.qualification_report)},
    )
    payload = {
        "schema":"RealSaS.CanonicalPuppetGraph.v2",
        "parent":parent_state_hash,
        "surface":surface.geometry_lineage_hash,
        "skeleton":skeleton.skeleton_lineage_hash,
        "skin":skin.skin_lineage_hash,
        "mesh":mesh.mesh_lineage_hash,
        "mesh_skin":mesh_skin.mesh_skin_lineage_hash,
        "ledger":ledger,
        "deformation":deformation_state or {},
        "contact":contact_state or {},
        "motion":motion_bindings or {},
        "editable":editable_metadata or {},
    }
    state = content_sha256(payload)
    lineage = "PUPPETV2:" + content_sha256({"genesis":state,"parent":parent_state_hash})[:24]
    return CanonicalPuppetGraphV2(
        lineage,
        state,
        parent_state_hash,
        surface.geometry_lineage_hash,
        skeleton.skeleton_lineage_hash,
        skin.skin_lineage_hash,
        mesh.mesh_lineage_hash,
        mesh_skin.mesh_skin_lineage_hash,
        ledger,
        deformation_state or {},
        contact_state or {},
        motion_bindings or {},
        editable_metadata or {},
    )


def bind_proof(product:ProductGraph, *, probe_plan, measurements, passed:bool, proof_payload=None)->ProofFrame:
    return ProofFrame(product.product_state_hash,content_sha256(probe_plan),content_sha256(measurements),bool(passed),proof_payload or {})


def require_current_proof(product:ProductGraph, proof:ProofFrame, *, require_pass:bool=False)->None:
    if proof.product_state_hash!=product.product_state_hash: raise QualificationError("stale proof frame: product_state_hash mismatch")
    if require_pass and not proof.passed: raise QualificationError("runtime/export requires a passing proof bound to the current product state")


def project_runtime_package(product:ProductGraph, proof:ProofFrame, *, manifest:dict, runtime_payload_ref:str)->RuntimePackageIR:
    # Runtime is a projection of a PROVEN current product. It cannot introduce a second truth.
    require_current_proof(product, proof, require_pass=True)
    proof_hash=content_sha256(proof.to_dict())
    m=dict(manifest); m["source_product_state_hash"]=product.product_state_hash; m["source_proof_hash"]=proof_hash
    return RuntimePackageIR(product.product_state_hash,proof_hash,m,runtime_payload_ref)
