from __future__ import annotations

"""Real model carrier + synthetic valid TESSA tokens: typed expert court.

No test asserts source truth, source contact, editable drawable, or learned
unseen quality. Corrupt proposals must not be laundered via re-hashed summaries.
"""

from copy import deepcopy
from dataclasses import replace

import pytest

from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.rss_vnext_cross_expert_capability_court_v1 import (
    compile_rss_vnext_cross_expert_capability_court_v1,
)
from compiler.realsas_compiler_core.rss_vnext_surface_decisions_v1 import (
    compile_rss_vnext_surface_candidates_v1,
)
from compiler.realsas_compiler_core.rss_vnext_face_relation_schedule_v1 import (
    compile_rss_vnext_sparse_face_relations_v1,
)
from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from tests.models.test_rss_vnext_surface_decisions_v1 import _real_surface_fixture, _fanout


def _ready():
    cfg, _state, _rgba, _camera, evidence, carrier = _real_surface_fixture(two_faces=True)
    reader = _fanout(cfg, evidence, carrier)
    surface = compile_rss_vnext_surface_candidates_v1(reader)
    relations = compile_rss_vnext_sparse_face_relations_v1(
        reader, surface, requested_face_pairs=[(0,1), (1,0)],
    )
    return cfg, evidence, surface, relations


def _compile(cfg, evidence, surface, relations):
    return compile_rss_vnext_cross_expert_capability_court_v1(
        evidence=evidence, surface=surface, model_config=cfg, relation=relations,
    )


def test_all_specialists_preserved_but_no_convenient_capability_promoted():
    cfg, evidence, surface, relations = _ready()
    court = _compile(cfg, evidence, surface, relations)
    assert court.state == "TYPED_CROSS_EXPERT_PROPOSALS_PRESERVED"
    assert court.face_count == 2
    assert court.computed_surface_sha256 == content_sha256(surface.__dict__)
    assert court.sparse_relation_sha256 == content_sha256(relations.__dict__)
    assert court.min_refined_base_area_ratio > 0
    assert court.min_refined_base_normal_alignment > 0
    assert court.max_semantic_unknown_probability > 0
    assert court.directed_attachment_pair_count == 2
    claims = {claim.capability: claim for claim in court.claims}
    assert set(claims) == {
        "INDEXED_SURFACE", "SEMANTIC_REGION", "GRAPHICS_2D", "RIG_CONTROL",
        "SKIN_DEFORMATION", "DIRECTED_ATTACHMENT",
        "INDEPENDENT_DRAWABLE", "ANIMATION_READY_MESH",
    }
    assert claims["SEMANTIC_REGION"].state == "TYPED_PROPOSAL_ONLY"
    for cap in ("DIRECTED_ATTACHMENT", "GRAPHICS_2D", "ANIMATION_READY_MESH"):
        assert claims[cap].state == "ABSTAIN_MISSING_EXECUTABLE_WITNESS"
    assert all(not c.executable_authority and not c.source_truth_authority
               for c in court.claims)
    assert not court.canonical_mesh_authority
    assert not court.source_visibility_authority
    assert not court.executable_attachment_authority


def test_forged_semantic_graphics_source_and_contact_never_promote():
    cfg, evidence, surface, relations = _ready()
    wrong = replace(surface.faces[0], source_visibility_qualified=True)
    with pytest.raises(QualificationError, match="FACE_SEMANTIC_TOPOLOGY_ADDRESS_OR_AUTHORITY_DRIFT"):
        _compile(cfg, evidence, replace(surface, faces=(wrong, *surface.faces[1:])), relations)
    wrong = replace(surface.faces[0], graphics_logits_by_corner=(
        (999., *surface.faces[0].graphics_logits_by_corner[0][1:]),
        *surface.faces[0].graphics_logits_by_corner[1:],
    ))
    with pytest.raises(QualificationError, match="GRAPHICS_MECHANICS_SOURCE_ADDRESS_DRIFT"):
        _compile(cfg, evidence, replace(surface, faces=(wrong, *surface.faces[1:])), relations)
    with pytest.raises(QualificationError, match="SPARSE_RELATION_AUTHORITY_OR_PARENT_DRIFT"):
        _compile(cfg, evidence, surface, replace(relations, attachment_authority=True))
    wrong = replace(relations.face_relations[0], mechanical_parent_anchor_qualified=True)
    with pytest.raises(QualificationError, match="RELATION_CANNOT_PROMOTE_ATTACHMENT"):
        _compile(cfg, evidence, surface, replace(relations,
            face_relations=(wrong, *relations.face_relations[1:])))


def test_refinement_inverted_face_is_rejected_even_if_rehashed_and_bound():
    cfg, evidence, surface, relations = _ready()
    mesh = deepcopy(dict(evidence.mesh))
    mesh["rest_xyz"] = deepcopy(mesh["rest_xyz"])
    mesh["rest_xyz"][0], mesh["rest_xyz"][1] = (
        mesh["rest_xyz"][1], mesh["rest_xyz"][0],
    )
    changed = seal_inference_evidence_v1(replace(
        evidence, mesh=mesh, evidence_hash="",
    ))
    rebased = replace(
        surface, source_inference_evidence_hash=changed.evidence_hash,
        candidate_mesh_sha256=content_sha256(mesh),
    )
    re_relations = replace(
        relations, inference_evidence_hash=changed.evidence_hash,
        candidate_mesh_sha256=content_sha256(mesh),
    )
    with pytest.raises(QualificationError, match="REFINED_TOPOLOGY_FACE_ORIENTATION_INVERTED"):
        _compile(cfg, changed, rebased, re_relations)


def test_parent_model_config_and_face_slot_cannot_be_swapped():
    cfg, evidence, surface, relations = _ready()
    with pytest.raises(QualificationError, match="SOURCE_OR_AUTHORITY_DRIFT"):
        _compile(cfg, evidence, replace(surface, unified_model_config_sha256="e"*64), relations)
    face = surface.faces[0]
    wrong = replace(face, region_probabilities=tuple(reversed(face.region_probabilities)))
    with pytest.raises(QualificationError, match="FACE_SEMANTIC_TOPOLOGY_ADDRESS_OR_AUTHORITY_DRIFT"):
        _compile(cfg, evidence, replace(surface, faces=(wrong, *surface.faces[1:])), relations)
