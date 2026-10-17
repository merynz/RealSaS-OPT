from __future__ import annotations

"""IRIS vNext Fit8 trainer v6: cumulative retention curriculum.

V6 preserves the repaired G2 early-stop, safe FP32 probability BCE semantics,
and resume-safe final-binding bootstrap from V5.  Its only training-policy change
is to stop later single-objective stages from overwriting previously learned
shared scene evidence.  From A1 onward, each stage carries the existing J1
retention envelope for prior objectives while giving the newly introduced
objective unit weight.
"""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile

from .cohort_v1 import VNextCheckpointStoreV1, VNextFit8PolicyV1, _sha, _vh
from .cohort_v5 import IRISVNextFit8TrainerV5, _source_identity_v5
from .cohort_v4 import vnext_fit8_policy_v3
from .contracts_v1 import IRISVNextConfigV1


def vnext_fit8_policy_v4() -> VNextFit8PolicyV1:
    """Cumulative curriculum; stage lengths and total 12k updates are unchanged."""
    base = vnext_fit8_policy_v3()
    stages = list(base.stages)

    a1 = stages[3]
    r1 = stages[4]
    t1 = stages[5]
    j1 = stages[6]

    if a1.stage_id != "A1_APPEARANCE_EVIDENCE":
        raise RuntimeError("VNEXT_V6_A1_BASE_POLICY_DRIFT")
    if r1.stage_id != "R1_ROLE_FREE_RELATIONS":
        raise RuntimeError("VNEXT_V6_R1_BASE_POLICY_DRIFT")
    if t1.stage_id != "T1_MECHANICAL_STRUCTURE":
        raise RuntimeError("VNEXT_V6_T1_BASE_POLICY_DRIFT")
    if j1.stage_id != "J1_JOINT_FINETUNE":
        raise RuntimeError("VNEXT_V6_J1_BASE_POLICY_DRIFT")

    # Reuse the already-authored J1 weights as the retention envelope rather
    # than inventing a new family of preservation coefficients.
    retain_field = float(j1.field_weight)
    retain_extraction = float(j1.extraction_weight)
    retain_mechanical = float(j1.mechanical_weight)
    retain_appearance = float(j1.appearance_weight)
    retain_relation = float(j1.relation_weight)

    stages[3] = replace(
        a1,
        scene_lr_scale=0.25,
        evidence_lr_scale=1.0,
        field_weight=retain_field,
        extraction_weight=retain_extraction,
        mechanical_weight=retain_mechanical,
        appearance_weight=1.0,
    )

    stages[4] = replace(
        r1,
        scene_lr_scale=0.25,
        evidence_lr_scale=1.0,
        field_weight=retain_field,
        extraction_weight=retain_extraction,
        mechanical_weight=retain_mechanical,
        appearance_weight=retain_appearance,
        relation_weight=1.0,
    )

    stages[5] = replace(
        t1,
        scene_lr_scale=0.25,
        evidence_lr_scale=0.5,
        structure_lr_scale=1.0,
        field_weight=retain_field,
        extraction_weight=retain_extraction,
        mechanical_weight=retain_mechanical,
        appearance_weight=retain_appearance,
        relation_weight=retain_relation,
        topology_weight=1.0,
    )

    policy = replace(base, stages=tuple(stages))
    policy.validate()
    if policy.total_steps != 12000:
        raise RuntimeError("VNEXT_V6_TOTAL_STEP_DRIFT")
    return policy


def _source_identity_v6(root: Path):
    value = dict(_source_identity_v5(root))
    value["models/iris/vnext/cohort_v6.py"] = _sha(root / "models/iris/vnext/cohort_v6.py")
    return value


class IRISVNextFit8TrainerV6(IRISVNextFit8TrainerV5):
    """V5 trainer mechanics with cumulative retention objectives from A1 onward."""

    architecture_id = "RealSaS.IRISVNextFit8Trainer.v6"

    def __init__(
        self,
        *,
        data_root,
        manifest,
        checkpoint_root,
        device,
        model_config=IRISVNextConfigV1(),
        policy=None,
        repo_root=None,
        flexicubes_root=None,
    ):
        policy = vnext_fit8_policy_v4() if policy is None else policy
        policy.validate()
        target_root = Path(checkpoint_root)

        # Preserve V5's final-binding-first guarantee: no ancestor binding may
        # inspect the final V8 checkpoint directory during construction.
        with tempfile.TemporaryDirectory(prefix="iris_vnext_v6_bootstrap_") as bootstrap:
            super().__init__(
                data_root=data_root,
                manifest=manifest,
                checkpoint_root=bootstrap,
                device=device,
                model_config=model_config,
                policy=policy,
                repo_root=repo_root,
                flexicubes_root=flexicubes_root,
            )

        self.binding = {
            **self.binding,
            "schema": "RealSaS.IRISVNextFit8Binding.v8",
            "source": _source_identity_v6(self.repo_root),
            "policy": asdict(policy),
            "curriculum_semantics": (
                "G1__G2_FIELD_ANCHORED_FLEXICUBES_EARLY_STOP_400__"
                "M1__A1_CUMULATIVE_RETENTION__R1_CUMULATIVE_RETENTION__"
                "T1_CUMULATIVE_RETENTION__J1"
            ),
            "retention_semantics": {
                "source": "EXISTING_J1_RETENTION_ENVELOPE",
                "field_weight": float(policy.stages[-1].field_weight),
                "extraction_weight": float(policy.stages[-1].extraction_weight),
                "mechanical_weight": float(policy.stages[-1].mechanical_weight),
                "appearance_weight": float(policy.stages[-1].appearance_weight),
                "relation_weight": float(policy.stages[-1].relation_weight),
                "a1_shared_scene_lr_scale": float(policy.stages[3].scene_lr_scale),
                "r1_shared_scene_lr_scale": float(policy.stages[4].scene_lr_scale),
                "t1_shared_scene_lr_scale": float(policy.stages[5].scene_lr_scale),
            },
            "failed_v5_forgetting_lineage_reused": False,
        }
        self.binding_hash = _vh(self.binding)
        self.store = VNextCheckpointStoreV1(target_root, self.binding_hash)

        previous = self.store.load()
        if previous:
            self._restore(previous)
        else:
            self.step = 0
            self.stage_index = 0
            self.stage_step = 0
            self._apply_stage_lrs()
