from __future__ import annotations

"""Typed Mage component assembly using dense zero-surface BODY + exact rigid foreground."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from compiler.realsas_compiler_core.component_attachment import (
    ComponentAttachmentEvidenceIR,
    qualify_component_attachment,
    qualify_component_assembly,
)
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.semantic_joint_roles import (
    canonical_joint_for_attachment_semantic,
    infer_humanoid_joint_roles,
)

import experiments.mage_demo_fit1_v5_p1.fit2_current_authority_io as fit2io
import experiments.mage_demo_fit1_v5_p1.materialize_p1q_current_authority_v1 as legacy_p1q

SCHEMA = "RealSaS.MageFIT2.DenseTypedComponentAssembly.v2"
EXPECTED_SOURCE_TRUTH_SHA256 = "a23565b0904e9683d11083984a87405f4e0a1069984ca11b690ad431f35f5e86"
EXPECTED_DENSE_STATUS = "PASS__FIT2_DENSE_ZERO_SURFACE_DYNAMIC_SAFE_BODY_V0_V7"
EXPECTED_FOREGROUND_STATUS = "PASS__EXACT_OBSERVATION_PLUS_TYPED_RIGID_FOREGROUND_ATLAS_V0_V7"
RUNTIME_TO_FAMILY = {
    "BOOK_FOREGROUND": "BOOK_VARIANTS",
    "STAFF_FOREGROUND": "WAND_STAFF_VARIANTS",
    "HAT_FOREGROUND": "HAT",
    "CAPE_FOREGROUND": "CAPE",
}


def _sha(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _source_semantic(component: dict) -> str:
    value = component.get("source_control_name") or component.get("source_control_semantics")
    if not value:
        raise RuntimeError(f"FIT2_DENSE_ASSEMBLY_SOURCE_SEMANTIC_MISSING:{component.get('component_family')}")
    return str(value)


def run(args) -> dict:
    dense_dir, foreground_dir, output_dir = Path(args.dense_body_dir), Path(args.foreground_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_truth_path = Path(args.source_truth)
    if _sha(source_truth_path) != EXPECTED_SOURCE_TRUTH_SHA256:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_SOURCE_TRUTH_SHA_DRIFT")
    source_truth = json.loads(source_truth_path.read_text(encoding="utf-8"))
    source_by_family = {str(r["component_family"]): r for r in source_truth["components"]}

    dense_manifest_path = dense_dir / "FIT2_DENSE_ZERO_SURFACE_BODY_MANIFEST.json"
    foreground_manifest_path = foreground_dir / "RUNTIME_FOREGROUND_ATLAS_MANIFEST.json"
    dense_sha, foreground_sha = _sha(dense_manifest_path), _sha(foreground_manifest_path)
    if args.expected_dense_body_manifest and dense_sha != args.expected_dense_body_manifest:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_BODY_MANIFEST_SHA_DRIFT")
    if args.expected_foreground_manifest and foreground_sha != args.expected_foreground_manifest:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_FOREGROUND_MANIFEST_SHA_DRIFT")
    dense_manifest = json.loads(dense_manifest_path.read_text(encoding="utf-8"))
    foreground_manifest = json.loads(foreground_manifest_path.read_text(encoding="utf-8"))
    if dense_manifest.get("status") != EXPECTED_DENSE_STATUS:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_BODY_STATUS_INVALID")
    if dense_manifest.get("surface_lineage_hash") != fit2io.EXPECTED_SURFACE_LINEAGE:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_SURFACE_LINEAGE_DRIFT")
    if dense_manifest.get("skeleton_lineage_hash") != fit2io.EXPECTED_SKELETON_LINEAGE:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_SKELETON_LINEAGE_DRIFT")
    if dense_manifest.get("skin_lineage_hash") != fit2io.EXPECTED_SKIN_LINEAGE:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_SKIN_LINEAGE_DRIFT")
    if dense_manifest.get("P1_or_P1Q_topology_used") is not False or dense_manifest.get("historical_weight_transfer_used") is not False:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_BODY_AUTHORITY_FIREWALL")
    if foreground_manifest.get("status") != EXPECTED_FOREGROUND_STATUS:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_FOREGROUND_STATUS_INVALID")

    surface, skeleton, skin, _mechanical = fit2io.build_exact_mechanical(
        Path(args.fit2_surface), Path(args.skeleton), Path(args.fit2_skin)
    )
    role_binding = infer_humanoid_joint_roles(skeleton)
    _write_json(output_dir / "FIT2_DENSE_HUMANOID_JOINT_ROLE_BINDING.json", role_binding.to_dict())

    dense_rows = {int(r["view"]): r for r in dense_manifest.get("views") or ()}
    fg_rows = {int(r["view"]): r for r in foreground_manifest.get("views") or ()}
    if set(dense_rows) != set(range(8)) or set(fg_rows) != set(range(8)):
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_REQUIRES_8_VIEWS")

    active_joints, body_mesh_hashes, body_skin_hashes, body_refs = set(), [], [], []
    for view in range(8):
        row = dense_rows[view]
        mesh_hash = str(row["mesh_lineage_hash"])
        mesh_skin = legacy_p1q._load_mesh_skin(dense_dir / row["files"]["skin"], mesh_hash)
        if mesh_skin.skin_binding_hash != skin.skin_lineage_hash:
            raise RuntimeError(f"FIT2_DENSE_ASSEMBLY_BODY_SKIN_LINEAGE_DRIFT_V{view}")
        if mesh_skin.surface_binding_hash != surface.geometry_lineage_hash or mesh_skin.skeleton_binding_hash != skeleton.skeleton_lineage_hash:
            raise RuntimeError(f"FIT2_DENSE_ASSEMBLY_BODY_BINDING_DRIFT_V{view}")
        for skin_row in mesh_skin.rows:
            active_joints.update(str(jid) for jid, weight in skin_row.influences if float(weight) > 1e-8)
        body_mesh_hashes.append(mesh_hash); body_skin_hashes.append(str(mesh_skin.mesh_skin_lineage_hash))
        body_refs.append(f"FIT2_DENSE_ZERO_SURFACE:V{view}:{mesh_hash}")
    if len(active_joints) < 2:
        raise RuntimeError("FIT2_DENSE_ASSEMBLY_BODY_MULTI_JOINT_SUPPORT_NOT_PROVEN")

    body_geometry_hash = content_sha256({"schema":SCHEMA,"role":"BODY_UNDERLAY","mesh_hashes":body_mesh_hashes})
    body_skin_hash = content_sha256({"schema":SCHEMA,"role":"BODY_UNDERLAY","mesh_skin_hashes":body_skin_hashes})
    body_evidence_hash = content_sha256({"schema":SCHEMA+".Evidence.v1","component_id":"BODY_UNDERLAY","source_truth_sha256":EXPECTED_SOURCE_TRUTH_SHA256,"dense_manifest_sha256":dense_sha,"active_joint_ids":sorted(active_joints),"fit2_skin_lineage":skin.skin_lineage_hash})
    body_evidence = ComponentAttachmentEvidenceIR(
        component_id="BODY_UNDERLAY",
        source_provenance_refs=(EXPECTED_SOURCE_TRUTH_SHA256,dense_sha,skin.skin_lineage_hash),
        mechanical_class="DEFORMABLE_COMPONENT",
        directional_render_membership=tuple((v,"BODY_UNDERLAY") for v in range(8)),
        geometry_membership_refs=tuple(body_refs),
        geometry_lineage_hash=body_geometry_hash,
        skin_deformer_lineage_hash=body_skin_hash,
        canonical_parent_joint_id="",socket_id="",bind_state_authority_hash="",
        detachability_class="FIXED_COMPONENT",visible_required=True,
        one_hot_carry_verified=False,one_hot_carry_joint_id="",visual_only_qualification_hash="",exclusion_reason="",
        qualification_evidence_hash=body_evidence_hash,
        metadata={"logical_role":"CONTINUITY_DEFORMABLE_UNDERLAY","active_canonical_joint_count":len(active_joints),"source_component_family":"BODY_SET","fresh_fit2_skin":True,"dense_zero_surface_topology":True},
    )
    qualified = [qualify_component_attachment(body_evidence,skeleton,qualification_report={"passed":True,"source_authority_verified":True,"classification_inferred_from_filename":False,"deformable_multi_joint_support":True,"active_canonical_joint_count":len(active_joints),"fit2_skin_lineage":skin.skin_lineage_hash,"dense_zero_surface_topology":True})]

    fg_components = {runtime_id: [] for runtime_id in RUNTIME_TO_FAMILY}
    for view in range(8):
        for row in fg_rows[view]["components"]:
            runtime_id = str(row["runtime_component_id"])
            if runtime_id in fg_components:
                fg_components[runtime_id].append((view,row))

    for runtime_id, family in RUNTIME_TO_FAMILY.items():
        source = source_by_family.get(family)
        if source is None: raise RuntimeError(f"FIT2_DENSE_ASSEMBLY_SOURCE_FAMILY_MISSING:{family}")
        semantic = _source_semantic(source); parent_joint = canonical_joint_for_attachment_semantic(role_binding,semantic)
        members = sorted(fg_components[runtime_id],key=lambda x:x[0])
        if tuple(v for v,_ in members) != tuple(range(8)):
            raise RuntimeError(f"FIT2_DENSE_ASSEMBLY_FOREGROUND_VIEW_MEMBERSHIP_INCOMPLETE:{runtime_id}")
        mesh_hashes = [str(r["mesh_lineage_hash"]) for _v,r in members]
        geometry_hash = content_sha256({"schema":SCHEMA,"component_id":runtime_id,"source_component_family":family,"sprite_mesh_hashes":mesh_hashes,"owner_manifest_sha256":foreground_manifest["owner_manifest_sha256"]})
        refs = tuple(f"OWNER:{foreground_manifest['owner_manifest_sha256']}:V{v}:{family}:{r['mesh_lineage_hash']}" for v,r in members)
        bind_hash = content_sha256({"schema":"RealSaS.FIT2DenseRigidAttachmentBindState.v1","source_truth_sha256":EXPECTED_SOURCE_TRUTH_SHA256,"foreground_manifest_sha256":foreground_sha,"owner_manifest_sha256":foreground_manifest["owner_manifest_sha256"],"role_binding_hash":role_binding.binding_hash,"source_component_family":family,"source_attachment_semantic":semantic,"canonical_parent_joint_id":parent_joint,"geometry_lineage_hash":geometry_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash})
        evidence_hash = content_sha256({"schema":SCHEMA+".Evidence.v1","component_id":runtime_id,"source_truth_sha256":EXPECTED_SOURCE_TRUTH_SHA256,"foreground_manifest_sha256":foreground_sha,"semantic":semantic,"parent_joint":parent_joint,"bind_state_authority_hash":bind_hash})
        evidence = ComponentAttachmentEvidenceIR(
            component_id=runtime_id,
            source_provenance_refs=(EXPECTED_SOURCE_TRUTH_SHA256,foreground_sha,str(foreground_manifest["owner_manifest_sha256"])),
            mechanical_class="RIGID_BONE_ATTACHMENT",
            directional_render_membership=tuple((v,runtime_id) for v in range(8)),
            geometry_membership_refs=refs,geometry_lineage_hash=geometry_hash,skin_deformer_lineage_hash="",
            canonical_parent_joint_id=parent_joint,socket_id=semantic,bind_state_authority_hash=bind_hash,
            detachability_class="FIXED_COMPONENT",visible_required=True,one_hot_carry_verified=False,one_hot_carry_joint_id="",visual_only_qualification_hash="",exclusion_reason="",qualification_evidence_hash=evidence_hash,
            metadata={"source_component_family":family,"source_attachment_semantic":semantic,"source_mechanical_class":str(source.get("mechanical_class","")),"product_mechanical_class":"RIGID_BONE_ATTACHMENT","fit2_weight_semantics_used_for_rigid_attachment":False,"runtime_carry_skin_compiled_later":True},
        )
        qualified.append(qualify_component_attachment(evidence,skeleton,qualification_report={"passed":True,"source_authority_verified":True,"classification_inferred_from_filename":False,"explicit_bone_or_socket_binding":True,"semantic_role_resolved_from_source_truth":True,"semantic_role_binding_hash":role_binding.binding_hash}))

    required = ("BODY_UNDERLAY","BOOK_FOREGROUND","STAFF_FOREGROUND","HAT_FOREGROUND","CAPE_FOREGROUND")
    assembly = qualify_component_assembly(tuple(qualified),skeleton,required_visible_component_ids=required,metadata={
        "source_truth_sha256":EXPECTED_SOURCE_TRUTH_SHA256,"dense_body_manifest_sha256":dense_sha,"foreground_manifest_sha256":foreground_sha,"owner_manifest_sha256":foreground_manifest["owner_manifest_sha256"],
        "humanoid_role_binding_hash":role_binding.binding_hash,"surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,
        "historical_fit1_skin_used":False,"P1_or_P1Q_topology_used":False,"dense_zero_surface_topology":True,"rigid_runtime_carry_is_derived_representation_only":True,
    })
    assembly_path = output_dir / "FIT2_DENSE_QUALIFIED_COMPONENT_ASSEMBLY.json"; _write_json(assembly_path,assembly.to_dict())
    for c in assembly.components: _write_json(output_dir/f"{c.component_id}_FIT2_DENSE_ATTACHMENT.json",c.to_dict())

    manifest = {
        "schema":SCHEMA,"status":"PASS__FIT2_DENSE_TYPED_COMPONENT_ASSEMBLY_BODY_PLUS_4_RIGID_FOREGROUND","dense_body_manifest_sha256":dense_sha,"foreground_manifest_sha256":foreground_sha,"source_truth_sha256":EXPECTED_SOURCE_TRUTH_SHA256,
        "surface_lineage_hash":surface.geometry_lineage_hash,"skeleton_lineage_hash":skeleton.skeleton_lineage_hash,"skin_lineage_hash":skin.skin_lineage_hash,"humanoid_role_binding_hash":role_binding.binding_hash,"component_assembly_hash":assembly.component_assembly_hash,
        "required_visible_component_ids":list(required),"body_active_joint_count":len(active_joints),"historical_fit1_skin_used":False,"P1_or_P1Q_topology_used":False,"dense_zero_surface_topology":True,
        "components":[{"component_id":c.component_id,"mechanical_class":c.mechanical_class,"canonical_parent_joint_id":c.canonical_parent_joint_id,"socket_id":c.socket_id,"geometry_lineage_hash":c.geometry_lineage_hash,"skin_deformer_lineage_hash":c.skin_deformer_lineage_hash,"bind_state_authority_hash":c.bind_state_authority_hash,"component_lineage_hash":c.component_lineage_hash} for c in assembly.components],
        "product_pass_claimed":False,
    }
    manifest_path = output_dir / "FIT2_DENSE_COMPONENT_ASSEMBLY_MANIFEST.json"; _write_json(manifest_path,manifest)
    seal = {"schema":SCHEMA+".Seal.v1","status":"SEALED__FIT2_DENSE_TYPED_COMPONENT_ASSEMBLY","assembly_file":assembly_path.name,"assembly_file_sha256":_sha(assembly_path),"manifest":manifest_path.name,"manifest_sha256":_sha(manifest_path),"component_assembly_hash":assembly.component_assembly_hash,"product_pass_claimed":False}
    _write_json(output_dir/"FIT2_DENSE_COMPONENT_ASSEMBLY_SEAL.json",seal)
    print("FIT2_DENSE_COMPONENT_ASSEMBLY="+json.dumps(seal,sort_keys=True),flush=True)
    return {"assembly":assembly,"manifest":manifest,"seal":seal}


def parse_args():
    p=argparse.ArgumentParser(); p.add_argument("--dense-body-dir",required=True); p.add_argument("--foreground-dir",required=True); p.add_argument("--source-truth",required=True); p.add_argument("--fit2-surface",required=True); p.add_argument("--skeleton",required=True); p.add_argument("--fit2-skin",required=True); p.add_argument("--expected-dense-body-manifest",default=""); p.add_argument("--expected-foreground-manifest",default=""); p.add_argument("--output-dir",required=True); return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
