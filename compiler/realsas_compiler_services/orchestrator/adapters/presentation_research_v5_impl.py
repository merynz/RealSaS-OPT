from __future__ import annotations

"""V7 research closure: qualified contacts plus semantic drawing order.

This adapter leaves sealed M/G/W and the motion witness unchanged. Stage37
qualifies contact and semantic presentation-slot evidence, Stage42 preserves
those contacts and compiles semantic order, Stage43 transports the relations
to runtime, and Stage45 proves that the native consumer used them.
"""

from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.visual_contact_v1 import (
    POLICY as CONTACT_POLICY,
    contact_relation_metrics,
    qualify_visual_contacts,
)
from compiler.realsas_compiler_core.visual_contact_motion_v1 import (
    OPERATOR_ID as CONTACT_MOTION_OPERATOR_ID,
    POLICY as CONTACT_MOTION_POLICY,
    compile_contact_projection,
)
from compiler.realsas_compiler_core.visual_semantic_order_v1 import (
    CONTRACT as SEMANTIC_ORDER_CONTRACT,
    OPERATOR_ID as SEMANTIC_ORDER_OPERATOR_ID,
    POLICY as SEMANTIC_ORDER_POLICY,
    compile_semantic_order,
    semantic_face_slots,
    semantic_order_metrics,
    semantic_vertex_slots,
)
from compiler.realsas_compiler_core.visual_presentation_contract_v1 import relational_verdict
from compiler.realsas_compiler_services.orchestrator.adapters import (
    presentation_research_v4_impl as control,
)

base = control.base
S37, S42, S43, S44, S45 = control.S37, control.S42, control.S43, control.S44, control.S45
TOPOLOGY_SCHEMA, PACKAGE_SCHEMA, PLAYBACK_SCHEMA = (
    control.TOPOLOGY_SCHEMA,
    control.PACKAGE_SCHEMA,
    control.PLAYBACK_SCHEMA,
)

playback_stage = base.playback_stage


def _attachment_target_map(topology):
    return {
        int(row["attachment_index"]): int(row["target_slot_raw_index_fit_only"])
        for row in topology["attachments"]["attachments"]
    }


def _relation_path(row):
    ref = dict(row.get("qualified_relations") or {})
    if not ref:
        raise QualificationError("PRESENTATION_V7_RELATION_REF_MISSING")
    return ref


def _load_relation(row):
    return base._load_npz(_relation_path(row))


def _write_topology(path: Path, topology: dict):
    payload = dict(topology)
    payload.pop("topology_hash", None)
    topology["topology_hash"] = content_sha256(payload)
    return base.write_json(
        path,
        topology,
        authority_class="SCOPED_RESEARCH_PRESENTATION_DOMAINS",
        schema=TOPOLOGY_SCHEMA,
    )


def compile_source_domains_stage(ctx):
    """Stage37: qualify deterministic contact and semantic-slot evidence."""
    result = control.compile_source_domains_stage(ctx)
    topology_path = Path(
        next(o["path"] for o in result["outputs"] if o.get("schema") == TOPOLOGY_SCHEMA)
    )
    topology = json.loads(topology_path.read_text())
    witness = base._load_npz(topology["witness"])
    target_map = _attachment_target_map(topology)
    view_summaries = []
    relation_outputs = []

    for row in topology["views"]:
        vi = int(row["view_index"])
        source = base._load_npz(row["mesh"])
        camera = base.qualify_camera_v3(
            row["camera"], view_id=f"V{vi}", view_index=vi
        )
        binding = base.build_domain_binding(
            points_source_xy=source["positions"],
            visual_faces=source["faces"],
            vertex_region_id=source["vertex_region_id"],
            seed_region_labels=source["seed_region_labels"],
            owner_face_index=source["owner_face_index"],
            mechanical_positions_xyz=witness["vertices"],
            mechanical_faces=witness["faces"],
            camera=camera,
        )
        owners = base.visual_vertex_attachment_owners(
            binding, witness["presentation_attachment_vertex_owner"]
        )
        blend = base.build_motion_blend_coefficients(
            binding,
            visual_faces=source["faces"],
            mechanical_weights=witness["canonical_motion_weights"],
        )
        contacts = qualify_visual_contacts(
            view_index=vi,
            rest_positions=source["positions"],
            visual_faces=source["faces"],
            domain_id=binding["domain_id"],
            anchor_vertex=binding["anchor_vertex"],
            anchor_mechanical_vertices=binding["anchor_mechanical_vertices"],
            motion_blend_coefficients=blend,
            vertex_attachment_owner=owners,
            attachment_target_joint_by_owner=target_map,
        )
        semantic_vertex = semantic_vertex_slots(
            motion_blend_coefficients=blend,
            vertex_attachment_owner=owners,
            attachment_target_joint_by_owner=target_map,
        )
        semantic_face = semantic_face_slots(
            visual_faces=source["faces"],
            motion_blend_coefficients=blend,
            vertex_attachment_owner=owners,
            attachment_target_joint_by_owner=target_map,
        )
        relation_path = topology_path.parent / f"V{vi}_qualified_relations.npz"
        relation_sha = base._save_npz(
            relation_path,
            domain_id=np.asarray(binding["domain_id"], dtype=np.int32),
            vertex_attachment_owner=np.asarray(owners, dtype=np.int32),
            motion_blend_coefficients=np.asarray(blend, dtype=np.float64),
            semantic_vertex_slot=np.asarray(semantic_vertex, dtype=np.int32),
            semantic_face_slot=np.asarray(semantic_face, dtype=np.int32),
            contact_pairs=np.asarray(contacts.pairs, dtype=np.int64),
            contact_relation_codes=np.asarray(contacts.relation_codes, dtype=np.int8),
            contact_domain_pairs=np.asarray(contacts.domain_pairs, dtype=np.int32),
            contact_rest_distances_px=np.asarray(contacts.rest_distances_px, dtype=np.float64),
            contact_evidence_scores=np.asarray(contacts.evidence_scores, dtype=np.float64),
        )
        row["qualified_relations"] = {
            "path": str(relation_path),
            "sha256": relation_sha,
            "contact_hash": contacts.contact_hash,
            "semantic_slot_hash": content_sha256(
                {
                    "view_index": vi,
                    "semantic_vertex_slot": semantic_vertex.astype(int).tolist(),
                    "semantic_face_slot": semantic_face.astype(int).tolist(),
                }
            ),
        }
        summary = {
            **contacts.summary(),
            "semantic_vertex_slot_count": int(len(np.unique(semantic_vertex))),
            "semantic_face_slot_count": int(len(np.unique(semantic_face))),
            "relation_npz_sha256": relation_sha,
        }
        view_summaries.append(summary)
        relation_outputs.append(
            {
                "path": str(relation_path),
                "sha256": relation_sha,
                "schema": "application/x-npz",
                "authority_class": "QUALIFIED_PRESENTATION_RELATIONS",
            }
        )

    topology["qualified_contact_contract"] = {
        "schema": "RealSaS.QualifiedVisualContactContract.v1",
        "policy": CONTACT_POLICY,
        "policy_hash": content_sha256(CONTACT_POLICY),
        "view_contact_hashes": [
            row["qualified_relations"]["contact_hash"] for row in topology["views"]
        ],
    }
    topology["qualified_contact_contract"]["contract_hash"] = content_sha256(
        topology["qualified_contact_contract"]
    )
    topology["semantic_presentation_slot_contract"] = {
        "schema": "RealSaS.SemanticPresentationSlotContract.v1",
        "derivation": "FROZEN_SKIN_DOMINANT_OWNER_PLUS_EXPLICIT_TARGET_ATTACHMENT_OWNER",
        "view_slot_hashes": [
            row["qualified_relations"]["semantic_slot_hash"] for row in topology["views"]
        ],
        "categorical_recognition_used": False,
        "model_inference_used": False,
    }
    topology["semantic_presentation_slot_contract"]["contract_hash"] = content_sha256(
        topology["semantic_presentation_slot_contract"]
    )
    topology["relation_qualification_summary"] = view_summaries
    topology["product_authority"] = False

    output = _write_topology(topology_path, topology)
    result["outputs"] = [
        output if o.get("schema") == TOPOLOGY_SCHEMA else o for o in result["outputs"]
    ]
    result["outputs"].extend(relation_outputs)
    result["diagnostics"] = {
        **dict(result.get("diagnostics") or {}),
        "qualified_contact_contract_hash": topology["qualified_contact_contract"][
            "contract_hash"
        ],
        "semantic_presentation_slot_contract_hash": topology[
            "semantic_presentation_slot_contract"
        ]["contract_hash"],
        "qualified_contact_pair_count": int(
            sum(row["contact_pair_count"] for row in view_summaries)
        ),
    }
    return result


def _verify_relation_arrays(*, row, relation, arrays, view_index):
    vi = int(view_index)
    binding = base.domain_binding_from_arrays(arrays, vi)
    checks = (
        ("domain_id", binding["domain_id"]),
        ("vertex_attachment_owner", arrays[f"view_{vi}_vertex_attachment_owner"]),
        ("motion_blend_coefficients", arrays[f"view_{vi}_motion_blend_coefficients"]),
    )
    for key, expected in checks:
        if not np.array_equal(np.asarray(relation[key]), np.asarray(expected)):
            raise QualificationError("PRESENTATION_V7_RELATION_BINDING_DRIFT:" + key)
    if relation["semantic_vertex_slot"].shape != (
        len(arrays[f"view_{vi}_rest_positions"]),
    ):
        raise QualificationError("PRESENTATION_V7_SEMANTIC_VERTEX_SLOT_SHAPE_DRIFT")
    if relation["semantic_face_slot"].shape != (len(arrays[f"view_{vi}_faces"]),):
        raise QualificationError("PRESENTATION_V7_SEMANTIC_FACE_SLOT_SHAPE_DRIFT")


def compile_projection_stage(ctx):
    """Stage42: preserve qualified contacts and compile semantic order."""
    result = control.compile_projection_stage(ctx)
    projection_path = Path(
        next(
            o["path"]
            for o in result["outputs"]
            if o.get("schema") == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"
        )
    )
    projection = base.source_owned_visual_runtime_projection_from_dict(
        json.loads(projection_path.read_text())
    )
    topology = base.stage_output_payload(ctx, S37, TOPOLOGY_SCHEMA)
    arrays = base._load_npz(
        {
            "path": projection.projection_npz_path,
            "sha256": projection.projection_npz_sha256,
        }
    )

    contact_summaries = {}
    order_summaries = {}
    relation_by_view = {}

    for view in projection.views:
        vi = int(view.view_index)
        row = next(r for r in topology["views"] if int(r["view_index"]) == vi)
        relation = _load_relation(row)
        _verify_relation_arrays(
            row=row, relation=relation, arrays=arrays, view_index=vi
        )
        relation_by_view[f"V{vi}"] = {
            "relation_sha256": row["qualified_relations"]["sha256"],
            "contact_hash": row["qualified_relations"]["contact_hash"],
            "semantic_slot_hash": row["qualified_relations"]["semantic_slot_hash"],
        }

        owners = np.asarray(relation["vertex_attachment_owner"], dtype=np.int32)
        pairs = np.asarray(relation["contact_pairs"], dtype=np.int64).reshape(-1, 2)
        if len(pairs):
            body_pair_mask = (owners[pairs[:, 0]] == 0) & (owners[pairs[:, 1]] == 0)
            body_pairs = pairs[body_pair_mask]
        else:
            body_pairs = pairs

        baseline_fields = {}
        repaired_fields = {}
        for clip in projection.clips:
            p = clip.array_prefix
            canonical_depth = np.asarray(
                arrays[f"{p}_view_{vi}_depths"], dtype=np.float64
            ).copy()
            arrays[f"{p}_view_{vi}_canonical_depths"] = canonical_depth
            arrays[f"{p}_view_{vi}_v6_positions"] = np.asarray(
                arrays[f"{p}_view_{vi}_positions"], dtype=np.float64
            ).copy()
            baseline_fields[p] = np.dstack(
                (
                    arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"],
                    canonical_depth,
                )
            )
            repaired_fields[p] = np.dstack(
                (
                    arrays[f"{p}_view_{vi}_positions"],
                    canonical_depth,
                )
            )

        contact_projection = compile_contact_projection(
            domain_id=relation["domain_id"],
            contact_pairs=body_pairs,
            baseline_clip_fields=baseline_fields,
            repaired_clip_fields=repaired_fields,
        )
        contact_summaries[f"V{vi}"] = {
            "qualified_pair_count": int(len(pairs)),
            "constrained_body_pair_count": int(len(body_pairs)),
            "maximum_contact_delta_residual_after_px": contact_projection[
                "maximum_contact_delta_residual_after_px"
            ],
            "maximum_domain_translation_px": contact_projection[
                "maximum_domain_translation_px"
            ],
        }

        per_clip_order = {}
        for clip in projection.clips:
            p = clip.array_prefix
            projected = contact_projection["projected_fields"][p]
            arrays[f"{p}_view_{vi}_positions"] = projected[:, :, :2]
            compiled = compile_semantic_order(
                canonical_depths=arrays[f"{p}_view_{vi}_canonical_depths"],
                semantic_vertex_slot=relation["semantic_vertex_slot"],
            )
            arrays[f"{p}_view_{vi}_depths"] = compiled["effective_depths"]
            arrays[f"{p}_view_{vi}_semantic_slot_ids"] = compiled["slot_ids"]
            arrays[f"{p}_view_{vi}_semantic_rank_rows"] = compiled["rank_rows"]
            arrays[f"{p}_view_{vi}_semantic_slot_medians"] = compiled[
                "canonical_slot_medians"
            ]
            per_clip_order[clip.clip_id] = {
                "order_hash": compiled["order_hash"],
                "transition_count": int(compiled["transition_count"]),
                "slot_count": int(len(compiled["slot_ids"])),
            }
        order_summaries[f"V{vi}"] = per_clip_order

    digest = base._save_npz(Path(projection.projection_npz_path), **arrays)
    relation_contract = {
        "schema": "RealSaS.PresentationRelationsRuntimeContract.v1",
        "qualified_contact_contract_hash": topology["qualified_contact_contract"][
            "contract_hash"
        ],
        "semantic_presentation_slot_contract_hash": topology[
            "semantic_presentation_slot_contract"
        ]["contract_hash"],
        "contact_motion_operator_id": CONTACT_MOTION_OPERATOR_ID,
        "contact_motion_policy_hash": content_sha256(CONTACT_MOTION_POLICY),
        "semantic_order_operator_id": SEMANTIC_ORDER_OPERATOR_ID,
        "semantic_order_policy_hash": content_sha256(SEMANTIC_ORDER_POLICY),
        "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
        "views": relation_by_view,
    }
    relation_contract["presentation_relations_hash"] = content_sha256(
        relation_contract
    )
    metadata = dict(
        projection.metadata,
        qualified_contact_contract_hash=topology["qualified_contact_contract"][
            "contract_hash"
        ],
        contact_motion_operator_id=CONTACT_MOTION_OPERATOR_ID,
        contact_motion_policy=CONTACT_MOTION_POLICY,
        contact_motion_policy_hash=content_sha256(CONTACT_MOTION_POLICY),
        semantic_order_operator_id=SEMANTIC_ORDER_OPERATOR_ID,
        semantic_order_policy=SEMANTIC_ORDER_POLICY,
        semantic_order_policy_hash=content_sha256(SEMANTIC_ORDER_POLICY),
        semantic_order_contract=SEMANTIC_ORDER_CONTRACT,
        semantic_order_encoding="EFFECTIVE_DEPTH_SORT_KEY_V1",
        presentation_relation_contract=relation_contract,
        contact_projection_summary=contact_summaries,
        semantic_order_summary=order_summaries,
        relational_qualification_scope="QUALIFIED_CONTACT_AND_SEMANTIC_ORDER_V1",
    )
    projection = replace(
        projection,
        projection_npz_sha256=digest,
        metadata=metadata,
        projection_hash="",
    )
    projection = replace(
        projection,
        projection_hash=base.source_owned_visual_runtime_projection_hash(projection),
    )
    base.write_ir(
        projection_path,
        projection,
        authority_class="SCOPED_RESEARCH_PRESENTATION_PROJECTION",
    )
    for output in result["outputs"]:
        if output.get("schema") == "application/x-npz":
            output["sha256"] = digest
        elif (
            output.get("schema")
            == "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"
        ):
            output["sha256"] = base.sha256_file(projection_path)
    result["diagnostics"] = {
        **dict(result.get("diagnostics") or {}),
        "projection_hash": projection.projection_hash,
        "qualified_contact_contract_hash": metadata["qualified_contact_contract_hash"],
        "contact_motion_operator_id": CONTACT_MOTION_OPERATOR_ID,
        "semantic_order_operator_id": SEMANTIC_ORDER_OPERATOR_ID,
        "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
    }
    return result


def package_stage(ctx):
    """Stage43: package semantic composition as a required runtime contract."""
    cfg, root = base._guard(ctx)
    projection = base.source_owned_visual_runtime_projection_from_dict(
        base.stage_output_payload(
            ctx, S42, "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"
        )
    )
    contract = dict(projection.metadata.get("presentation_relation_contract") or {})
    if (
        contract.get("semantic_order_contract") != SEMANTIC_ORDER_CONTRACT
        or contract.get("semantic_order_operator_id") != SEMANTIC_ORDER_OPERATOR_ID
        or len(str(contract.get("presentation_relations_hash") or "")) != 64
    ):
        raise QualificationError("PRESENTATION_V7_RUNTIME_RELATION_CONTRACT_MISSING")

    entries = base.build_source_owned_visual_rss_v2_entries(projection)
    relation_bytes = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    relation_entry = "presentation_relations.json"
    entries[relation_entry] = relation_bytes

    manifest = entries["manifest.txt"].decode("utf-8")
    if not manifest.endswith("\n"):
        manifest += "\n"
    manifest += (
        f"semantic_order_contract={SEMANTIC_ORDER_CONTRACT}\n"
        f"semantic_order_operator_id={SEMANTIC_ORDER_OPERATOR_ID}\n"
        "semantic_order_encoding=EFFECTIVE_DEPTH_SORT_KEY_V1\n"
        f"semantic_order_entry={relation_entry}\n"
        f"presentation_relations_hash={contract['presentation_relations_hash']}\n"
        f"qualified_contact_contract_hash={contract['qualified_contact_contract_hash']}\n"
    )
    entries["manifest.txt"] = manifest.encode("utf-8")

    root.mkdir(parents=True, exist_ok=True)
    path = root / "research_presentation.rss"
    result = base.write_rss_v2(path, OrderedDict(entries))
    data = {
        "schema": PACKAGE_SCHEMA,
        "projection_hash": projection.projection_hash,
        "archive": {"path": str(path), "sha256": result["archive_sha256"]},
        "presentation_relations_hash": contract["presentation_relations_hash"],
        "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
        "product_authority": False,
    }
    return {
        "status": "PASS_DEMO_ONLY",
        "outputs": [
            base.write_json(
                root / "package.json",
                data,
                authority_class="SCOPED_RESEARCH_PRESENTATION_PACKAGE",
                schema=PACKAGE_SCHEMA,
            ),
            {
                "path": str(path),
                "sha256": result["archive_sha256"],
                "schema": "application/x-realsas-rss-v2",
                "authority_class": "SCOPED_NATIVE_PACKAGE",
            },
        ],
        "diagnostics": {
            **result,
            "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
            "presentation_relations_hash": contract["presentation_relations_hash"],
        },
    }


def _legacy_v6_arrays(projection, arrays):
    legacy = dict(arrays)
    for view in projection.views:
        vi = int(view.view_index)
        for clip in projection.clips:
            p = clip.array_prefix
            legacy[f"{p}_view_{vi}_positions"] = arrays[
                f"{p}_view_{vi}_v6_positions"
            ]
            legacy[f"{p}_view_{vi}_depths"] = arrays[
                f"{p}_view_{vi}_canonical_depths"
            ]
    return legacy


def _relation_proof(projection, arrays, topology, witness, mesh, dynamic):
    """Replay V6 invariants, then add independent V7 relation predicates."""
    legacy = _legacy_v6_arrays(projection, arrays)
    proof = control._repair_proof(
        projection, legacy, topology, witness, mesh, dynamic
    )

    contact_failed = 0
    contact_pairs = 0
    max_contact_delta = 0.0
    max_contact_growth = 0.0
    semantic_failures = 0
    minimum_semantic_gap = np.inf
    semantic_slots = set()
    temporal_order_transitions = 0
    rows = []

    for view in projection.views:
        vi = int(view.view_index)
        topology_row = next(
            row for row in topology["views"] if int(row["view_index"]) == vi
        )
        relation = _load_relation(topology_row)
        _verify_relation_arrays(
            row=topology_row, relation=relation, arrays=arrays, view_index=vi
        )
        rest = arrays[f"view_{vi}_rest_positions"]
        pairs = np.asarray(relation["contact_pairs"], dtype=np.int64).reshape(-1, 2)
        codes = np.asarray(relation["contact_relation_codes"], dtype=np.int8)

        for clip in projection.clips:
            p = clip.array_prefix
            compiled_metrics = semantic_order_metrics(
                effective_depths=arrays[f"{p}_view_{vi}_depths"],
                semantic_vertex_slot=relation["semantic_vertex_slot"],
                slot_ids=arrays[f"{p}_view_{vi}_semantic_slot_ids"],
                rank_rows=arrays[f"{p}_view_{vi}_semantic_rank_rows"],
            )
            semantic_failures += int(compiled_metrics["semantic_order_failure_count"])
            minimum_semantic_gap = min(
                minimum_semantic_gap,
                float(compiled_metrics["minimum_inter_slot_depth_gap"]),
            )
            semantic_slots.update(
                map(int, arrays[f"{p}_view_{vi}_semantic_slot_ids"].tolist())
            )
            ranks = arrays[f"{p}_view_{vi}_semantic_rank_rows"]
            if len(ranks) > 1:
                temporal_order_transitions += int(
                    np.count_nonzero(np.any(ranks[1:] != ranks[:-1], axis=1))
                )

            for fi in range(int(clip.frame_count)):
                metrics = contact_relation_metrics(
                    rest_positions=rest,
                    baseline_positions=arrays[
                        f"{p}_view_{vi}_motion_safety_baseline_positions"
                    ][fi],
                    posed_positions=arrays[f"{p}_view_{vi}_positions"][fi],
                    pairs=pairs,
                    relation_codes=codes,
                )
                contact_failed += int(metrics["contact_failure_count"])
                contact_pairs += int(metrics["contact_pair_count"])
                max_contact_delta = max(
                    max_contact_delta,
                    float(metrics["maximum_contact_repair_delta_residual_px"]),
                )
                max_contact_growth = max(
                    max_contact_growth,
                    float(metrics["maximum_contact_gap_growth_px"]),
                )
                rows.append(
                    {
                        "clip_id": clip.clip_id,
                        "view_id": view.view_id,
                        "frame_index": fi,
                        "qualified_contacts": metrics,
                        "semantic_order_failure_count": int(
                            compiled_metrics["semantic_order_failure_count"]
                        ),
                    }
                )

    if not np.isfinite(minimum_semantic_gap):
        minimum_semantic_gap = 0.0
    proof.update(
        qualified_contact_relations_passed=contact_failed == 0,
        qualified_contact_pair_observation_count=contact_pairs,
        qualified_contact_failure_count=contact_failed,
        maximum_contact_repair_delta_residual_px=max_contact_delta,
        maximum_contact_gap_growth_px=max_contact_growth,
        semantic_occlusion_passed=semantic_failures == 0,
        semantic_order_failure_count=semantic_failures,
        minimum_inter_slot_depth_gap=float(minimum_semantic_gap),
        semantic_slot_count=len(semantic_slots),
        semantic_order_transition_count=temporal_order_transitions,
        frame0_relations_passed=(
            bool(proof.get("connected_palette_relations_passed"))
            and bool(proof.get("setup_identity_passed"))
            and contact_failed == 0
            and semantic_failures == 0
        ),
        temporal_relations_passed=(
            bool(proof.get("connected_palette_relations_passed"))
            and contact_failed == 0
            and semantic_failures == 0
        ),
        relational_frames=rows,
        missing_qualification_inputs=[],
    )
    return proof


def _native_semantic_token(stdout: str) -> bool:
    return f"semantic_order_consumer={SEMANTIC_ORDER_CONTRACT}" in str(stdout)


def prove_presentation_stage(ctx):
    """Stage45: relation proof plus optimized whole-clip native verification."""
    cfg, root = base._guard(ctx)
    topology = base.stage_output_payload(ctx, S37, TOPOLOGY_SCHEMA)
    projection = base.source_owned_visual_runtime_projection_from_dict(
        base.stage_output_payload(
            ctx, S42, "RealSaS.SourceOwnedVisualRuntimeProjectionIR.v1"
        )
    )
    package = base.stage_output_payload(ctx, S43, PACKAGE_SCHEMA)
    playback = base.stage_output_payload(ctx, S44, PLAYBACK_SCHEMA)
    witness = base._load_npz(topology["witness"])
    mesh = base._numeric_mesh(witness)
    mesh.mesh_lineage_hash = topology["mechanical_mesh_binding_hash"]
    dynamic = control._dynamic(topology, witness)
    arrays = base._load_npz(
        {
            "path": projection.projection_npz_path,
            "sha256": projection.projection_npz_sha256,
        }
    )

    contract = dict(projection.metadata.get("presentation_relation_contract") or {})
    if (
        projection.metadata.get("contact_motion_operator_id")
        != CONTACT_MOTION_OPERATOR_ID
        or projection.metadata.get("semantic_order_operator_id")
        != SEMANTIC_ORDER_OPERATOR_ID
        or projection.metadata.get("semantic_order_contract")
        != SEMANTIC_ORDER_CONTRACT
        or package.get("semantic_order_contract") != SEMANTIC_ORDER_CONTRACT
        or package.get("presentation_relations_hash")
        != contract.get("presentation_relations_hash")
    ):
        raise QualificationError("PRESENTATION_V7_RELATION_CONTRACT_DRIFT")

    proof = _relation_proof(
        projection, arrays, topology, witness, mesh, dynamic
    )

    player, player_sha = base._native_player(ctx)
    if (
        playback["native_player_sha256"] != player_sha
        or playback["projection_hash"] != projection.projection_hash
        or playback["archive_sha256"] != package["archive"]["sha256"]
    ):
        raise QualificationError("PRESENTATION_V7_PLAYBACK_BINDING_DRIFT")

    archive = base.load_file_ref(package["archive"], json_required=False)
    parity_bad = empty = ties = overflow = flipped = edge_bad = 0
    semantic_consumer_bad = 0
    rendered = []
    coverage = {}

    # Important optimization: one executor per clip, not one executor per frame.
    # This keeps all semantics identical while removing hundreds of pool creations.
    for clip in projection.clips:
        requests = [
            {
                "clip_id": clip.clip_id,
                "view_id": view.view_id,
                "frame_index": fi,
            }
            for fi in range(int(clip.frame_count))
            for view in projection.views
        ]
        native_rows = base._run_native_many(
            player=player,
            package=archive,
            requests=requests,
            root=root / "frames",
            max_workers=4,
        )
        cursor = 0
        for fi in range(int(clip.frame_count)):
            for view in projection.views:
                rgba, provenance, owner, stdout = native_rows[cursor]
                cursor += 1
                ref = base._source_owned_visual_reference_frame(
                    projection,
                    arrays,
                    clip=clip,
                    view=view,
                    frame_index=fi,
                )
                mismatch = (
                    rgba.read_bytes() != ref.straight_rgba_u8.tobytes()
                    or provenance.read_bytes()
                    != ref.provenance_code.tobytes()
                    or owner.read_bytes()
                    != ref.owner_face_index.astype("<i4").tobytes()
                )
                parity_bad += int(mismatch)
                empty += int(not np.any(ref.straight_rgba_u8[:, :, 3]))
                ties += int(ref.unresolved_depth_tie_count)
                overflow += int(ref.fragment_overflow_count)
                semantic_consumer_bad += int(not _native_semantic_token(stdout))

                vi = int(view.view_index)
                faces = arrays[f"view_{vi}_faces"]
                geometry_metrics = base._visual_mesh_motion_metrics(
                    arrays[f"view_{vi}_rest_positions"],
                    arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi],
                    faces,
                )
                flipped += int(geometry_metrics["flipped_triangle_count"])
                edge_bad += int(geometry_metrics["edge_gt_4_count"])

                topology_row = next(
                    row
                    for row in topology["views"]
                    if int(row["view_index"]) == vi
                )
                relation = _load_relation(topology_row)
                face_slots = np.asarray(
                    relation["semantic_face_slot"], dtype=np.int32
                )
                visible_faces = ref.owner_face_index[
                    ref.owner_face_index >= 0
                ].astype(np.int64)
                key = (clip.clip_id, view.view_id)
                slot_rows = coverage.setdefault(
                    key,
                    {
                        int(slot): {
                            "source_face_count": int(
                                np.count_nonzero(face_slots == slot)
                            ),
                            "visible_pixel_count": 0,
                        }
                        for slot in np.unique(face_slots)
                    },
                )
                if len(visible_faces):
                    visible_slots = face_slots[visible_faces]
                    values, counts = np.unique(
                        visible_slots, return_counts=True
                    )
                    for slot, count in zip(values.tolist(), counts.tolist()):
                        slot_rows[int(slot)]["visible_pixel_count"] += int(count)

                rendered.append(
                    {
                        "clip_id": clip.clip_id,
                        "view_id": view.view_id,
                        "frame_index": fi,
                        "rgba": {
                            "path": str(rgba),
                            "sha256": base.sha256_file(rgba),
                        },
                        "native_reference_parity": not mismatch,
                        "semantic_runtime_contract_consumed": _native_semantic_token(
                            stdout
                        ),
                    }
                )

    coverage_failures = []
    coverage_rows = []
    for (clip_id, view_id), slots in sorted(coverage.items()):
        for slot, values in sorted(slots.items()):
            passed = (
                values["source_face_count"] == 0
                or values["visible_pixel_count"] > 0
            )
            coverage_rows.append(
                {
                    "clip_id": clip_id,
                    "view_id": view_id,
                    "semantic_slot_id": int(slot),
                    **values,
                    "dynamic_coverage_passed": bool(passed),
                }
            )
            if not passed:
                coverage_failures.append((clip_id, view_id, int(slot)))

    proof["qualified_dynamic_coverage_passed"] = not coverage_failures
    proof["dynamic_coverage_failure_count"] = len(coverage_failures)
    proof["dynamic_coverage_failures"] = coverage_failures
    proof["semantic_runtime_contract_consumed"] = semantic_consumer_bad == 0
    proof["semantic_runtime_contract_failure_frame_views"] = semantic_consumer_bad
    proof["frame0_relations_passed"] = bool(
        proof["frame0_relations_passed"]
        and proof["qualified_dynamic_coverage_passed"]
        and proof["semantic_runtime_contract_consumed"]
    )
    proof["temporal_relations_passed"] = bool(
        proof["temporal_relations_passed"]
        and proof["qualified_dynamic_coverage_passed"]
        and proof["semantic_runtime_contract_consumed"]
    )
    proof.update(relational_verdict(proof))

    passed = (
        all(
            proof[key]
            for key in (
                "domain_coherence_passed",
                "frame_view_matrix_complete",
                "area_condition_passed",
                "attachment_slot_motion_passed",
                "canonical_pose_palette_passed",
                "motion_safety_repair_passed",
                "attachment_ownership_passed",
                "relational_presentation_passed",
                "semantic_runtime_contract_consumed",
            )
        )
        and parity_bad
        == empty
        == ties
        == overflow
        == flipped
        == edge_bad
        == semantic_consumer_bad
        == 0
    )

    data = {
        "schema": "RealSaS.ScopedPresentationProof.v5",
        "status": "PASS_DEMO_ONLY" if passed else "FAIL",
        **proof,
        "canonical_depth_ownership_passed": bool(
            proof["semantic_occlusion_passed"]
            and proof["semantic_runtime_contract_consumed"]
        )
        and ties == overflow == 0,
        "native_reference_byte_parity_passed": parity_bad == 0,
        "native_reference_mismatch_frame_views": parity_bad,
        "empty_frame_views": empty,
        "unresolved_depth_ties": ties,
        "fragment_overflow": overflow,
        "flipped_triangles": flipped,
        "edge_gt_4_count": edge_bad,
        "rendered_frames": rendered,
        "dynamic_coverage_rows": coverage_rows,
        "projection_hash": projection.projection_hash,
        "native_player_sha256": player_sha,
        "presentation_relations_hash": contract[
            "presentation_relations_hash"
        ],
        "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
        "product_authority": False,
        "mechanics_reopened": False,
    }
    output = base.write_json(
        root / "presentation_proof.json",
        data,
        authority_class="SCOPED_RESEARCH_PRESENTATION_PROOF",
        schema=data["schema"],
    )
    frame_outputs = [
        {
            **row["rgba"],
            "schema": "application/x-rgba8",
            "authority_class": "SCOPED_NATIVE_PRESENTATION_FRAME",
        }
        for row in rendered
    ]
    return {
        "status": data["status"],
        "outputs": [output, *frame_outputs],
        "diagnostics": {
            key: value
            for key, value in data.items()
            if key not in ("rendered_frames", "dynamic_coverage_rows", "relational_frames")
        },
        "blockers": []
        if passed
        else [
            "SCOPED_PRESENTATION_FAIL_CLOSED",
            *proof["relational_presentation_blockers"],
        ],
    }
