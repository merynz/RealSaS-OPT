from __future__ import annotations

"""IRIS vNext Fit8 trainer v8: seal shared scene core after M1.

The V6 authority run proved that cumulative loss retention preserves mechanics
but still allows A1 to rewrite shared geometry/extraction features.  V8 uses the
model's existing modular boundary directly: after M1, scene_core is frozen and
head-specific stages train only the head that owns the new evidence.  J1 then
jointly fine-tunes evidence/structure heads while scene_core remains sealed.

The V7 TESSA witness-coordinate repair is retained unchanged.
"""

from dataclasses import asdict, replace
from pathlib import Path
import tempfile

import torch

from .cohort_v1 import VNextCheckpointStoreV1, VNextFit8PolicyV1, _sha, _vh
from .cohort_v7 import IRISVNextFit8TrainerV7, _source_identity_v7
from .cohort_v6 import vnext_fit8_policy_v4
from .contracts_v1 import IRISVNextConfigV1


V6_M1_END_STEP = 3600
V6_M1_END_SHA256 = "ff2341a6c15626714e2b01f5c65d4043f560cf088550b0015905133ce4a0e8aa"
V6_BINDING_SHA256 = "0a17710c6b5cda7bcd635d99a38f9556a4243d2a729baf5738898a77033cece5"


def vnext_fit8_policy_v5() -> VNextFit8PolicyV1:
    """Head-isolated post-M1 curriculum; total remains exactly 12k updates."""
    base = vnext_fit8_policy_v4()
    stages = list(base.stages)
    a1, r1, t1, j1 = stages[3], stages[4], stages[5], stages[6]

    if a1.stage_id != "A1_APPEARANCE_EVIDENCE":
        raise RuntimeError("VNEXT_V8_A1_BASE_POLICY_DRIFT")
    if r1.stage_id != "R1_ROLE_FREE_RELATIONS":
        raise RuntimeError("VNEXT_V8_R1_BASE_POLICY_DRIFT")
    if t1.stage_id != "T1_MECHANICAL_STRUCTURE":
        raise RuntimeError("VNEXT_V8_T1_BASE_POLICY_DRIFT")
    if j1.stage_id != "J1_JOINT_FINETUNE":
        raise RuntimeError("VNEXT_V8_J1_BASE_POLICY_DRIFT")

    # A1/R1: scene_core frozen, active evidence head learns from sealed H.
    stages[3] = replace(
        a1,
        scene_lr_scale=0.0,
        evidence_lr_scale=1.0,
        field_weight=0.0,
        extraction_weight=0.0,
        mechanical_weight=0.0,
        appearance_weight=1.0,
        relation_weight=0.0,
        topology_weight=0.0,
    )
    stages[4] = replace(
        r1,
        scene_lr_scale=0.0,
        evidence_lr_scale=1.0,
        field_weight=0.0,
        extraction_weight=0.0,
        mechanical_weight=0.0,
        appearance_weight=0.0,
        relation_weight=1.0,
        topology_weight=0.0,
    )

    # T1: both scene_core and evidence heads are sealed; only structure learns.
    stages[5] = replace(
        t1,
        scene_lr_scale=0.0,
        evidence_lr_scale=0.0,
        structure_lr_scale=1.0,
        field_weight=0.0,
        extraction_weight=0.0,
        mechanical_weight=0.0,
        appearance_weight=0.0,
        relation_weight=0.0,
        topology_weight=1.0,
    )

    # J1 jointly reconciles learned heads/structure but keeps scene_core sealed,
    # so geometry/extraction authority cannot be overwritten after M1.
    stages[6] = replace(
        j1,
        scene_lr_scale=0.0,
        evidence_lr_scale=0.5,
        structure_lr_scale=0.5,
        field_weight=0.0,
        extraction_weight=0.0,
        mechanical_weight=0.5,
        appearance_weight=0.25,
        relation_weight=0.5,
        topology_weight=0.75,
    )

    policy = replace(base, stages=tuple(stages))
    policy.validate()
    if policy.total_steps != 12000:
        raise RuntimeError("VNEXT_V8_TOTAL_STEP_DRIFT")
    return policy


def _source_identity_v8(root: Path):
    value = dict(_source_identity_v7(root))
    value["models/iris/vnext/cohort_v8.py"] = _sha(root / "models/iris/vnext/cohort_v8.py")
    return value


class IRISVNextFit8TrainerV8(IRISVNextFit8TrainerV7):
    architecture_id = "RealSaS.IRISVNextFit8Trainer.v8"

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
        policy = vnext_fit8_policy_v5() if policy is None else policy
        policy.validate()
        target_root = Path(checkpoint_root)

        with tempfile.TemporaryDirectory(prefix="iris_vnext_v8_bootstrap_") as bootstrap:
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
            "schema": "RealSaS.IRISVNextFit8Binding.v10",
            "source": _source_identity_v8(self.repo_root),
            "policy": asdict(policy),
            "post_m1_parameter_authority": {
                "scene_core": "SEALED_LR_ZERO",
                "A1": "APPEARANCE_HEAD_ONLY",
                "R1": "RELATION_HEAD_ONLY",
                "T1": "STRUCTURE_HEAD_ONLY",
                "J1": "EVIDENCE_AND_STRUCTURE_HEADS_ONLY",
            },
            "geometry_retention_semantics": "STRUCTURAL_FREEZE_NOT_LOSS_BALANCING",
            "migration_source": {
                "binding_sha256": V6_BINDING_SHA256,
                "checkpoint_step": V6_M1_END_STEP,
                "checkpoint_sha256": V6_M1_END_SHA256,
            },
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

    def migrate_v6_m1_end(self, payload: dict, *, source_checkpoint_sha256: str):
        if self.step != 0 or self.stage_index != 0 or self.stage_step != 0:
            raise RuntimeError("VNEXT_V8_MIGRATION_TARGET_NOT_FRESH")
        if source_checkpoint_sha256 != V6_M1_END_SHA256:
            raise RuntimeError("VNEXT_V8_MIGRATION_SOURCE_SHA_DRIFT")
        if int(payload.get("step", -1)) != V6_M1_END_STEP:
            raise RuntimeError("VNEXT_V8_MIGRATION_SOURCE_STEP_DRIFT")
        if int(payload.get("stage_index", -1)) != 2 or int(payload.get("stage_step", -1)) != 1600:
            raise RuntimeError("VNEXT_V8_MIGRATION_SOURCE_STAGE_DRIFT")
        if payload.get("binding_sha256") != V6_BINDING_SHA256:
            raise RuntimeError("VNEXT_V8_MIGRATION_SOURCE_BINDING_DRIFT")
        if payload.get("schema") != "RealSaS.IRISVNextFit8Checkpoint.v4":
            raise RuntimeError("VNEXT_V8_MIGRATION_SOURCE_SCHEMA_DRIFT")

        self.model.load_state_dict(payload["model"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["torch_rng"])
        if payload.get("cuda_rng"):
            torch.cuda.set_rng_state_all(payload["cuda_rng"])

        self.step = V6_M1_END_STEP
        self.stage_index = 2
        self.stage_step = 1600
        self._apply_stage_lrs()
        self.optimizer.zero_grad(set_to_none=True)

        receipt = {
            "schema": "RealSaS.IRISVNextFit8V6M1ToV8MigrationReceipt.v1",
            "status": "PASS",
            "migration": "V6_M1_END_STEP3600__HEAD_ISOLATED_POST_M1_CURRICULUM",
            "source_checkpoint_step": V6_M1_END_STEP,
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "source_binding_sha256": payload.get("binding_sha256"),
            "target_binding_sha256": self.binding_hash,
            "target_binding_schema": self.binding["schema"],
            "model_parameter_state_exactly_loaded": True,
            "optimizer_state_exactly_loaded": True,
            "rng_state_exactly_loaded": True,
            "source_scene_core_preserved": True,
            "post_m1_scene_core_lr_zero": True,
            "coordinate_contract_repair_retained": True,
        }
        pointer = self.save()
        receipt["migrated_checkpoint_pointer"] = pointer
        return receipt
