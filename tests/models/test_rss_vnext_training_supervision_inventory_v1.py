from __future__ import annotations

"""Ensure the source/output/loss inventory cannot silently omit a model loss.

This is a static ownership audit, not an assertion that any training corpus
or GPU checkpoint passed. It intentionally imports neither torch nor data.
"""

import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
INVENTORY=ROOT/"models/rss_vnext/training_supervision_inventory_v1.json"


def _declared_class_field_names(relative:str,klass:str)->set[str]:
    tree=ast.parse((ROOT/relative).read_text(encoding="utf8"))
    node=next(
        node for node in tree.body
        if isinstance(node,ast.ClassDef) and node.name==klass
    )
    return {
        node.target.id for node in node.body
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name)
    }


def _declared_default_weights(relative: str, klass: str) -> dict[str, float]:
    tree=ast.parse((ROOT/relative).read_text(encoding="utf8"))
    node=next(
        node for node in tree.body
        if isinstance(node,ast.ClassDef) and node.name==klass
    )
    return {
        child.target.id: float(ast.literal_eval(child.value))
        for child in node.body
        if isinstance(child,ast.AnnAssign)
        and isinstance(child.target,ast.Name) and child.value is not None
    }


def _phase_loss_names()->set[str]:
    tree=ast.parse(
        (ROOT/"models/rss_vnext/training_phase_contract_v1.py").read_text(
            encoding="utf8"
        )
    )
    field=next(
        node for node in tree.body
        if isinstance(node,ast.Assign)
        and any(isinstance(x,ast.Name) and x.id=="LOSS_FIELDS"
                for x in node.targets)
    )
    return set(ast.literal_eval(field.value))


def _document():
    return json.loads(INVENTORY.read_text(encoding="utf8"))


def test_structural_inventory_exhaustively_matches_actual_weight_and_phase_fields():
    inventory=_document()
    ids=[x["id"] for x in inventory["structural"]]
    weights=_declared_class_field_names(
        "models/rss_vnext/training_engine.py","RSSvNextTrainingWeights"
    )
    assert len(ids)==len(set(ids))==20
    assert set(ids)==weights-{"clip_gradient_norm"}
    assert set(ids)==_phase_loss_names()


def test_rendered_slrm_inventory_matches_the_actual_stage_loss_weights():
    inventory=_document()
    ids=[x["id"] for x in inventory["slrm"]]
    weights=_declared_class_field_names(
        "models/rss_vnext/realsas_slrm_training_driver_v1.py",
        "RealSaSSLRMTrainingLossWeightsV1",
    )
    assert len(ids)==len(set(ids))==9
    assert set(ids)==weights


def test_mira_codec_inventory_covers_every_source_supervised_result_family():
    inventory=_document()
    ids={x["id"] for x in inventory["mira"]}
    fields=_declared_class_field_names(
        "models/rss_vnext/mira_embedded_codec_source_training_v1.py",
        "RSSvNextEmbeddedMIRACodecResultV1",
    )
    assert ids=={
        "reconstruction_loss","quantization_loss",
        "source_pose_deformation_loss","source_area_skin_loss",
        "source_area_skin_simplex_kl_loss",
    }
    assert ids.issubset(fields)


def test_each_loss_binds_model_output_target_mask_receipt_and_preparation():
    inventory=_document()
    assert inventory["schema"]=="RealSaS.RSSvNext.OutputLossTruthPreparationInventory.v1"
    assert inventory["source_corpus_verified"] is False
    assert inventory["product_unseen_quality_proven"] is False
    for group in ("structural","slrm","mira","diffusion"):
        assert inventory[group]
        for entry in inventory[group]:
            for name in (
                "owner","prediction","objective","teacher_target",
                "qualification_mask","trusted_receipt","preparation",
                "preparation_route","source_truth_status",
            ):
                assert isinstance(entry[name],str) and entry[name].strip(), (
                    group,entry["id"],name
                )
            assert entry["real_corpus_verified"] is False
            assert entry["gpu_verified"] is False
            assert "STUDENT_PREDICTION_IS_TRUTH" not in entry["trusted_receipt"]


def test_untrained_outputs_remain_visible_as_gaps_not_counted_as_trained_heads():
    inventory=_document()
    uncovered={x["id"] for x in inventory["unsupervised_outputs"]}
    assert {
        "dynamic_refine.quality_head","RSSvNext deformation_quality",
        "Graphics editable hidden drawable texture",
        "TESSA face topology versus high-res FC surface",
        "IRIS log_uncertainty",
    }.issubset(uncovered)
    assert not any(
        x["real_corpus_verified"]
        for x in inventory["unsupervised_outputs"]
    )


def test_stage_a_and_b_have_distinct_paired_source_roles_and_teacher_domains():
    inventory=_document()
    stages={x["id"]:x for x in inventory["diffusion"]}
    assert set(stages)=={
        "STAGE_CANONICAL","STAGE_MULTIVIEW","ALPHA_HEAD",
    }
    a,b=stages["STAGE_CANONICAL"],stages["STAGE_MULTIVIEW"]
    assert "Two arbitrary" in a["preparation"]
    assert "Two independently rendered canonical" in b["preparation"]
    assert "canonical_target_rgb_sha256[2]" in a["teacher_target"]
    assert "multiview_target_normal_sha256[6]" in b["teacher_target"]


def test_source_nonsigned_geometry_remains_distinct_from_closed_sdf():
    x={row["id"]:row for row in _document()["structural"]}
    assert x["source_udf"]["source_truth_status"]!="CLOSED_SOLID_PROOF_REQUIRED"
    assert x["source_signed_sdf"]["source_truth_status"]=="CLOSED_SOLID_PROOF_REQUIRED"
    assert x["source_inside_sign"]["source_truth_status"]=="CLOSED_SOLID_PROOF_REQUIRED"
    assert "closed" in x["source_signed_sdf"]["preparation"].lower()
    assert "INTERVENTION"==x["typed_relations"]["preparation_route"]



def test_inventory_default_weight_values_are_exactly_mainline_values():
    inventory=_document()
    for group,source,klass in (
        ("structural","models/rss_vnext/training_engine.py",
         "RSSvNextTrainingWeights"),
        ("slrm","models/rss_vnext/realsas_slrm_training_driver_v1.py",
         "RealSaSSLRMTrainingLossWeightsV1"),
    ):
        actual=_declared_default_weights(source,klass)
        if group=="structural":
            actual.pop("clip_gradient_norm")
        recorded={x["id"]:x["default_weight"] for x in inventory[group]}
        assert actual==recorded
