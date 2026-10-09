from __future__ import annotations

"""Fast downstream-only contact/order court over the frozen V6 evidence artifact.

This court intentionally reuses the exact completed V6 Stage37/Stage42 bytes from
run 37933851760. It never replays M/G/W or the motion witness. The only changed
state is presentation relation compilation: qualified contacts, contact-preserving
post-repair translation, and semantic drawing order. It renders RUN/V6 first so
visual feedback arrives without another full 984 frame-view matrix.
"""

import argparse
from collections import OrderedDict
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.runtime_package_v2 import (
    build_source_owned_visual_rss_v2_entries,
    read_rss_v2,
    write_rss_v2,
)
from compiler.realsas_compiler_core.runtime_visual_authority_v1 import (
    source_owned_visual_runtime_projection_from_dict,
    source_owned_visual_runtime_projection_hash,
)
from compiler.realsas_compiler_core.visual_contact_motion_v1 import (
    OPERATOR_ID as CONTACT_MOTION_OPERATOR_ID,
    POLICY as CONTACT_MOTION_POLICY,
    compile_contact_projection,
)
from compiler.realsas_compiler_core.visual_contact_v1 import (
    POLICY as CONTACT_POLICY,
    contact_relation_metrics,
    qualify_visual_contacts,
)
from compiler.realsas_compiler_core.visual_domain_v2 import domain_binding_from_arrays
from compiler.realsas_compiler_core.visual_semantic_order_v1 import (
    CONTRACT as SEMANTIC_ORDER_CONTRACT,
    OPERATOR_ID as SEMANTIC_ORDER_OPERATOR_ID,
    POLICY as SEMANTIC_ORDER_POLICY,
    compile_semantic_order,
    semantic_face_slots,
    semantic_order_metrics,
    semantic_vertex_slots,
)
from compiler.realsas_compiler_services.orchestrator.adapters.runtime_v2 import (
    _run_native_many,
    _save_npz,
    _source_owned_visual_reference_frame,
)


PRIOR_RUN_ID = 37933851760
PRIOR_CODE_SHA = "59dc7f4e10eeb56abc4f247ec9ba045a46a6a496"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique(root: Path, suffix: str, *, contains=()) -> Path:
    rows = [p for p in root.rglob(suffix) if all(token in p.as_posix() for token in contains)]
    if len(rows) != 1:
        raise RuntimeError(
            f"RELATION_REPLAY_EXPECTED_ONE:{suffix}:{contains}:found={len(rows)}\n"
            + "\n".join(p.as_posix() for p in rows[:20])
        )
    return rows[0]


def _candidate_stage(root: Path, stage: str, filename: str) -> Path:
    rows = [
        p
        for p in root.rglob(filename)
        if "/candidate/" in p.as_posix()
        and "/stage_artifacts/" in p.as_posix()
        and f"/{stage}/" in p.as_posix()
    ]
    if len(rows) != 1:
        raise RuntimeError(
            f"RELATION_REPLAY_STAGE_FILE_AMBIGUOUS:{stage}:{filename}:{len(rows)}\n"
            + "\n".join(p.as_posix() for p in rows[:20])
        )
    return rows[0]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in data.files}


def _attachment_target_map(topology: dict) -> dict[int, int]:
    return {
        int(row["attachment_index"]): int(row["target_slot_raw_index_fit_only"])
        for row in topology["attachments"]["attachments"]
    }


def _extract_textures(prior_rss: Path, projection, out: Path):
    entries = read_rss_v2(prior_rss)
    views = []
    texture_root = out / "textures"
    texture_root.mkdir(parents=True, exist_ok=True)
    for view in projection.views:
        vi = int(view.view_index)
        key = f"visual_texture_v{vi}.png"
        if key not in entries:
            raise RuntimeError("RELATION_REPLAY_TEXTURE_MISSING:" + key)
        path = texture_root / f"V{vi}.png"
        path.write_bytes(entries[key])
        digest = sha256_file(path)
        if digest != str(view.texture_sha256):
            raise RuntimeError("RELATION_REPLAY_TEXTURE_HASH_DRIFT:" + key)
        views.append(replace(view, texture_path=str(path)))
    return tuple(views)


def _relation_compile(topology: dict, projection, arrays: dict[str, np.ndarray]):
    target_map = _attachment_target_map(topology)
    relation_by_view = {}
    contact_summaries = {}
    order_summaries = {}

    for view in projection.views:
        vi = int(view.view_index)
        binding = domain_binding_from_arrays(arrays, vi)
        rest = np.asarray(arrays[f"view_{vi}_rest_positions"], dtype=np.float64)
        faces = np.asarray(arrays[f"view_{vi}_faces"], dtype=np.int64)
        owners = np.asarray(arrays[f"view_{vi}_vertex_attachment_owner"], dtype=np.int32)
        blend = np.asarray(arrays[f"view_{vi}_motion_blend_coefficients"], dtype=np.float64)

        contacts = qualify_visual_contacts(
            view_index=vi,
            rest_positions=rest,
            visual_faces=faces,
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
            visual_faces=faces,
            motion_blend_coefficients=blend,
            vertex_attachment_owner=owners,
            attachment_target_joint_by_owner=target_map,
        )
        arrays[f"view_{vi}_semantic_vertex_slot"] = semantic_vertex
        arrays[f"view_{vi}_semantic_face_slot"] = semantic_face
        arrays[f"view_{vi}_qualified_contact_pairs"] = contacts.pairs
        arrays[f"view_{vi}_qualified_contact_relation_codes"] = contacts.relation_codes

        pairs = np.asarray(contacts.pairs, dtype=np.int64).reshape(-1, 2)
        if len(pairs):
            body_mask = (owners[pairs[:, 0]] == 0) & (owners[pairs[:, 1]] == 0)
            body_pairs = pairs[body_mask]
        else:
            body_pairs = pairs

        baseline_fields = {}
        repaired_fields = {}
        for clip in projection.clips:
            p = clip.array_prefix
            canonical_depth = np.asarray(arrays[f"{p}_view_{vi}_depths"], dtype=np.float64).copy()
            arrays[f"{p}_view_{vi}_canonical_depths"] = canonical_depth
            arrays[f"{p}_view_{vi}_v6_positions"] = np.asarray(
                arrays[f"{p}_view_{vi}_positions"], dtype=np.float64
            ).copy()
            baseline_fields[p] = np.dstack(
                (arrays[f"{p}_view_{vi}_motion_safety_baseline_positions"], canonical_depth)
            )
            repaired_fields[p] = np.dstack(
                (arrays[f"{p}_view_{vi}_positions"], canonical_depth)
            )

        projected = compile_contact_projection(
            domain_id=binding["domain_id"],
            contact_pairs=body_pairs,
            baseline_clip_fields=baseline_fields,
            repaired_clip_fields=repaired_fields,
        )
        contact_summaries[f"V{vi}"] = {
            **contacts.summary(),
            "constrained_body_pair_count": int(len(body_pairs)),
            "maximum_contact_delta_residual_after_px": projected[
                "maximum_contact_delta_residual_after_px"
            ],
            "maximum_domain_translation_px": projected["maximum_domain_translation_px"],
        }

        per_clip = {}
        for clip in projection.clips:
            p = clip.array_prefix
            arrays[f"{p}_view_{vi}_positions"] = projected["projected_fields"][p][:, :, :2]
            order = compile_semantic_order(
                canonical_depths=arrays[f"{p}_view_{vi}_canonical_depths"],
                semantic_vertex_slot=semantic_vertex,
            )
            arrays[f"{p}_view_{vi}_depths"] = order["effective_depths"]
            arrays[f"{p}_view_{vi}_semantic_slot_ids"] = order["slot_ids"]
            arrays[f"{p}_view_{vi}_semantic_rank_rows"] = order["rank_rows"]
            per_clip[clip.clip_id] = {
                "order_hash": order["order_hash"],
                "transition_count": int(order["transition_count"]),
                "slot_count": int(len(order["slot_ids"])),
            }
        order_summaries[f"V{vi}"] = per_clip
        relation_by_view[f"V{vi}"] = {
            "contact_hash": contacts.contact_hash,
            "semantic_slot_hash": content_sha256(
                {
                    "vertex": semantic_vertex.astype(int).tolist(),
                    "face": semantic_face.astype(int).tolist(),
                }
            ),
        }

    contact_contract = {
        "schema": "RealSaS.QualifiedVisualContactContract.v1",
        "policy_hash": content_sha256(CONTACT_POLICY),
        "view_contact_hashes": [relation_by_view[f"V{i}"]["contact_hash"] for i in range(8)],
    }
    contact_contract["contract_hash"] = content_sha256(contact_contract)
    relation_contract = {
        "schema": "RealSaS.PresentationRelationsRuntimeContract.v1",
        "qualified_contact_contract_hash": contact_contract["contract_hash"],
        "contact_motion_operator_id": CONTACT_MOTION_OPERATOR_ID,
        "contact_motion_policy_hash": content_sha256(CONTACT_MOTION_POLICY),
        "semantic_order_operator_id": SEMANTIC_ORDER_OPERATOR_ID,
        "semantic_order_policy_hash": content_sha256(SEMANTIC_ORDER_POLICY),
        "semantic_order_contract": SEMANTIC_ORDER_CONTRACT,
        "views": relation_by_view,
    }
    relation_contract["presentation_relations_hash"] = content_sha256(relation_contract)
    return relation_contract, contact_summaries, order_summaries


def _build_package(projection, relation_contract: dict, out: Path) -> Path:
    entries = build_source_owned_visual_rss_v2_entries(projection)
    relation_entry = "presentation_relations.json"
    entries[relation_entry] = json.dumps(
        relation_contract, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    manifest = entries["manifest.txt"].decode("utf-8")
    if not manifest.endswith("\n"):
        manifest += "\n"
    manifest += (
        f"semantic_order_contract={SEMANTIC_ORDER_CONTRACT}\n"
        f"semantic_order_operator_id={SEMANTIC_ORDER_OPERATOR_ID}\n"
        "semantic_order_encoding=EFFECTIVE_DEPTH_SORT_KEY_V1\n"
        f"semantic_order_entry={relation_entry}\n"
        f"presentation_relations_hash={relation_contract['presentation_relations_hash']}\n"
        f"qualified_contact_contract_hash={relation_contract['qualified_contact_contract_hash']}\n"
    )
    entries["manifest.txt"] = manifest.encode("utf-8")
    path = out / "Knight_V7_relation_court.rss"
    write_rss_v2(path, OrderedDict(entries))
    return path


def _render_run_v6(*, projection, arrays, package: Path, player: Path, out: Path):
    clip = next(row for row in projection.clips if row.clip_id == "demo_run_v1")
    view = next(row for row in projection.views if int(row.view_index) == 6)
    requests = [
        {"clip_id": clip.clip_id, "view_id": view.view_id, "frame_index": fi}
        for fi in range(int(clip.frame_count))
    ]
    native = _run_native_many(
        player=player,
        package=package,
        requests=requests,
        root=out / "native_run_v6",
        max_workers=4,
    )
    frames = []
    parity_bad = ties = overflow = 0
    semantic_token_bad = 0
    visible_slot_rows = []
    face_slots = np.asarray(arrays["view_6_semantic_face_slot"], dtype=np.int32)
    for fi, (rgba, provenance, owner, stdout) in enumerate(native):
        ref = _source_owned_visual_reference_frame(
            projection, arrays, clip=clip, view=view, frame_index=fi
        )
        mismatch = (
            rgba.read_bytes() != ref.straight_rgba_u8.tobytes()
            or provenance.read_bytes() != ref.provenance_code.tobytes()
            or owner.read_bytes() != ref.owner_face_index.astype("<i4").tobytes()
        )
        parity_bad += int(mismatch)
        ties += int(ref.unresolved_depth_tie_count)
        overflow += int(ref.fragment_overflow_count)
        semantic_token_bad += int(
            f"semantic_order_consumer={SEMANTIC_ORDER_CONTRACT}" not in str(stdout)
        )
        visible = ref.owner_face_index[ref.owner_face_index >= 0].astype(np.int64)
        row = {int(slot): 0 for slot in np.unique(face_slots)}
        if len(visible):
            values, counts = np.unique(face_slots[visible], return_counts=True)
            row.update({int(k): int(v) for k, v in zip(values.tolist(), counts.tolist())})
        visible_slot_rows.append(row)
        raw = rgba.read_bytes()
        image = Image.frombytes("RGBA", (int(view.camera["resolution"]), int(view.camera["resolution"])), raw)
        background = Image.new("RGBA", image.size, (34, 38, 46, 255))
        background.alpha_composite(image)
        frames.append(background.convert("RGB").resize((768, 768), Image.Resampling.NEAREST))
    gif = out / "Knight_RUN_V6_V7_RELATION_COURT_768.gif"
    frames[0].save(
        gif,
        save_all=True,
        append_images=frames[1:],
        duration=1000 / 24,
        loop=0,
    )
    return gif, {
        "native_reference_mismatch_frames": parity_bad,
        "unresolved_depth_ties": ties,
        "fragment_overflow": overflow,
        "semantic_runtime_contract_failure_frames": semantic_token_bad,
        "visible_pixels_by_semantic_slot_per_frame": visible_slot_rows,
    }


def _run_v6_relation_metrics(projection, arrays):
    clip = next(row for row in projection.clips if row.clip_id == "demo_run_v1")
    vi = 6
    rest = np.asarray(arrays[f"view_{vi}_rest_positions"], dtype=np.float64)
    pairs = np.asarray(arrays[f"view_{vi}_qualified_contact_pairs"], dtype=np.int64).reshape(-1, 2)
    codes = np.asarray(arrays[f"view_{vi}_qualified_contact_relation_codes"], dtype=np.int8)
    failed = 0
    max_delta = max_growth = 0.0
    for fi in range(int(clip.frame_count)):
        metrics = contact_relation_metrics(
            rest_positions=rest,
            baseline_positions=arrays[f"{clip.array_prefix}_view_{vi}_motion_safety_baseline_positions"][fi],
            posed_positions=arrays[f"{clip.array_prefix}_view_{vi}_positions"][fi],
            pairs=pairs,
            relation_codes=codes,
        )
        failed += int(metrics["contact_failure_count"])
        max_delta = max(max_delta, float(metrics["maximum_contact_repair_delta_residual_px"]))
        max_growth = max(max_growth, float(metrics["maximum_contact_gap_growth_px"]))
    order = semantic_order_metrics(
        effective_depths=arrays[f"{clip.array_prefix}_view_{vi}_depths"],
        semantic_vertex_slot=arrays[f"view_{vi}_semantic_vertex_slot"],
        slot_ids=arrays[f"{clip.array_prefix}_view_{vi}_semantic_slot_ids"],
        rank_rows=arrays[f"{clip.array_prefix}_view_{vi}_semantic_rank_rows"],
    )
    return {
        "qualified_contact_relations_passed": failed == 0,
        "qualified_contact_failure_observations": failed,
        "maximum_contact_repair_delta_residual_px": max_delta,
        "maximum_contact_gap_growth_px": max_growth,
        **order,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--player", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    topology_path = _candidate_stage(
        args.evidence_root, "37_QUALIFIED_PRESENTATION_STRUCTURE", "source_domains.json"
    )
    projection_path = _candidate_stage(
        args.evidence_root, "42_RUNTIME_PROJECTION_AND_CAA_BINDING", "projection.json"
    )
    arrays_path = _candidate_stage(
        args.evidence_root, "42_RUNTIME_PROJECTION_AND_CAA_BINDING", "projection_arrays.npz"
    )
    prior_rss = _candidate_stage(
        args.evidence_root, "43_RSS_MATERIALIZE_COMPACT", "research_presentation.rss"
    )
    topology = json.loads(topology_path.read_text())
    projection = source_owned_visual_runtime_projection_from_dict(
        json.loads(projection_path.read_text())
    )
    arrays = _load_npz(arrays_path)

    projection = replace(
        projection,
        views=_extract_textures(prior_rss, projection, args.out),
        projection_npz_path=str(args.out / "projection_arrays_v7.npz"),
        projection_npz_sha256="0" * 64,
    )
    relation_contract, contact_summary, order_summary = _relation_compile(
        topology, projection, arrays
    )
    arrays_sha = _save_npz(Path(projection.projection_npz_path), **arrays)
    metadata = dict(
        projection.metadata,
        qualified_contact_contract_hash=relation_contract[
            "qualified_contact_contract_hash"
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
        contact_projection_summary=contact_summary,
        semantic_order_summary=order_summary,
        replay_parent_run_id=PRIOR_RUN_ID,
        replay_parent_code_sha=PRIOR_CODE_SHA,
    )
    projection = replace(
        projection,
        projection_npz_sha256=arrays_sha,
        metadata=metadata,
        projection_hash="",
    )
    projection = replace(
        projection,
        projection_hash=source_owned_visual_runtime_projection_hash(projection),
    )
    projection_json = args.out / "projection_v7.json"
    projection_json.write_text(json.dumps(projection.to_dict(), indent=2) + "\n")

    package = _build_package(projection, relation_contract, args.out)
    gif, runtime_metrics = _render_run_v6(
        projection=projection,
        arrays=arrays,
        package=package,
        player=args.player,
        out=args.out,
    )
    relation_metrics = _run_v6_relation_metrics(projection, arrays)
    receipt = {
        "schema": "RealSaS.FastPresentationRelationCourt.v1",
        "status": "PASS_DIAGNOSTIC"
        if relation_metrics["qualified_contact_relations_passed"]
        and relation_metrics["semantic_occlusion_passed"]
        and runtime_metrics["native_reference_mismatch_frames"] == 0
        and runtime_metrics["semantic_runtime_contract_failure_frames"] == 0
        else "FAIL_DIAGNOSTIC",
        "prior_run_id": PRIOR_RUN_ID,
        "prior_code_sha": PRIOR_CODE_SHA,
        "prior_stage37_sha256": sha256_file(topology_path),
        "prior_stage42_projection_sha256": sha256_file(projection_path),
        "prior_stage42_arrays_sha256": sha256_file(arrays_path),
        "candidate_projection_hash": projection.projection_hash,
        "candidate_package_sha256": sha256_file(package),
        "candidate_run_v6_gif": str(gif),
        "candidate_run_v6_gif_sha256": sha256_file(gif),
        "contact_summary": contact_summary.get("V6"),
        "order_summary": order_summary.get("V6"),
        "relation_metrics": relation_metrics,
        "runtime_metrics": runtime_metrics,
        "product_authority": False,
        "mechanics_reopened": False,
    }
    (args.out / "relation_court_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in receipt.items() if k != "runtime_metrics"}, indent=2))
    if receipt["status"] != "PASS_DIAGNOSTIC":
        raise SystemExit("FAST_PRESENTATION_RELATION_COURT_FAILED__EVIDENCE_EXPORTED")


if __name__ == "__main__":
    main()
