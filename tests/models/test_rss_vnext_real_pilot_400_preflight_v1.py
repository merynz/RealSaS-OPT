from __future__ import annotations

"""Pure regression courts. These synthetic 300 records are NOT a real pilot."""

from dataclasses import replace
from hashlib import sha256
import json

import pytest
import torch
from torch import nn

from models.rss_vnext.product_real_pilot_cohort_v1 import (
    EVIDENCE_SCHEMA, REQUIRED_CORE, RealPilotAssetV1,
    qualify_real_pilot_cohort_v1, verify_pilot_source_evidence_v1,
    assert_train_only_source_v1,
)
from models.rss_vnext.product_real_pilot_execution_v1 import (
    PilotCompletedStageV1, empty_pilot_ledger_v1,
    load_pilot_ledger_v1, save_pilot_ledger_v1,
    RealPilotTrainingEnvelopeV1, run_source_qualified_pilot_updates_v1,
)
from models.rss_vnext.product_seeded_phase_ownership_v1 import (
    ProductSeededPhaseOwnerPolicyV1, PHASES,
)


def sha(value):
    return sha256(value.encode()).hexdigest()


def fixture_assets(tmp_path, n=300):
    assets=[]
    root=tmp_path/"receipts"
    for i in range(n):
        aid="SK190" if i==n-1 else f"P{i:04d}"
        # A synthetic example of *user-authored* split input: Pieta is TRAIN.
        split=("TRAIN" if i == n-1 or i < int(.8*n)
               else "VAL" if i < int(.9*n) else "UNSEEN_TEST")
        asset_sha=sha("mesh:"+aid)
        teacher=tuple(sorted(REQUIRED_CORE))
        hashes={}
        for kind in ("license","native","render"):
            content={
                "schema":EVIDENCE_SCHEMA,"kind":kind.upper(),
                "source_id":aid,"source_asset_sha256":asset_sha,
                "decision":"APPROVED_SOURCE_VERIFICATION",
                "reviewer":"test-fixture-not-a-real-approval",
                "reviewed_at":"2026-11-20",
            }
            if kind=="render":
                content["qualified_teacher_families"]=list(teacher)
            path=root/aid/(kind+".json")
            path.parent.mkdir(parents=True,exist_ok=True)
            data=json.dumps(content,sort_keys=True).encode()
            path.write_bytes(data)
            hashes[kind]=sha256(data).hexdigest()
        assets.append(RealPilotAssetV1(
            source_id=aid,source_asset_sha256=asset_sha,
            original_front_sha256=sha("front:"+aid),
            original_back_sha256=sha("back:"+aid),
            lineage_group="source-original-scene:"+aid,split=split,
            license_receipt_sha256=hashes["license"],
            native_receipt_sha256=hashes["native"],
            render_receipt_sha256=hashes["render"],
            teacher_families=teacher,
        ))
    return tuple(assets),root


@pytest.fixture
def admitted(tmp_path):
    assets,root=fixture_assets(tmp_path)
    cohort=qualify_real_pilot_cohort_v1(assets)
    verified=verify_pilot_source_evidence_v1(assets,root,cohort)
    return assets,root,verified


def test_300_400_subjects_respect_user_split_and_allow_pieta_in_train(admitted):
    assets,root,receipt=admitted
    assert receipt.total_assets==300
    assert (receipt.train_assets,receipt.val_assets,
            receipt.unseen_test_assets)==(241,30,29)
    assert assets[-1].source_id=="SK190" and assets[-1].split=="TRAIN"
    assert receipt.source_evidence_bytes_verified
    assert not receipt.full_pilot_launch_qualified
    assert_train_only_source_v1(assets[0],receipt)
    assert_train_only_source_v1(assets[-1],receipt)
    for item in (assets[250],assets[-2]):
        with pytest.raises(ValueError,match="TRAIN_SOURCE_NOT_ADMITTED"):
            assert_train_only_source_v1(item,receipt)
    assert receipt.teacher_family_coverage_train["NATIVE_MESH"]==241


def test_300_is_minimum_and_license_noai_core_truth_cannot_be_invented(admitted):
    assets,root,receipt=admitted
    with pytest.raises(ValueError,match="300_TO_400"):
        qualify_real_pilot_cohort_v1(assets[:299])
    for updated in (
        replace(assets[0],noai_review="PENDING"),
        replace(assets[0],license_decision="RESEARCH_ONLY"),
        replace(assets[0],teacher_families=()),
        replace(assets[0],source_id="../other"),
    ):
        with pytest.raises(ValueError,match="SOURCE_LICENSE_NATIVE"):
            qualify_real_pilot_cohort_v1((updated,*assets[1:]))
    # Arbitrary user-authored split proportions are accepted; no 80/10/10 rule.
    manual=tuple(replace(a,split=("TRAIN" if i<285 else
                                    "VAL" if i<295 else "UNSEEN_TEST"))
                 for i,a in enumerate(assets))
    result=qualify_real_pilot_cohort_v1(manual)
    assert (result.train_assets,result.val_assets,result.unseen_test_assets)==(285,10,5)
    with pytest.raises(ValueError,match="SPLIT_REQUIRED_NONEMPTY"):
        qualify_real_pilot_cohort_v1(tuple(replace(a,split="TRAIN") for a in assets))


def test_duplicate_and_lineage_leaks_rejected(admitted):
    assets,root,receipt=admitted
    with pytest.raises(ValueError,match="LINEAGE_LEAK"):
        qualify_real_pilot_cohort_v1((*assets[:298],replace(
            assets[-2],lineage_group=assets[0].lineage_group,),assets[-1],
        ))
    with pytest.raises(ValueError,match="DUPLICATE_SOURCE_OR_IMAGE"):
        qualify_real_pilot_cohort_v1((*assets[:299],replace(
            assets[-1],original_front_sha256=assets[0].original_front_sha256,
        )))


def test_source_evidence_sha_and_fields_are_rechecked(admitted):
    assets,root,cohort=admitted
    path=root/assets[0].source_id/"render.json"
    original=path.read_bytes()
    path.write_bytes(original+b"\n")
    with pytest.raises(ValueError,match="SOURCE_EVIDENCE_SHA_DRIFT"):
        verify_pilot_source_evidence_v1(assets,root,replace(
            cohort,source_evidence_bytes_verified=False,
        ))
    path.write_bytes(original)


def test_phase_order_and_byte_qualified_gpu_validation(admitted,tmp_path):
    assets,root,cohort=admitted
    ledger=empty_pilot_ledger_v1(cohort)
    assert ledger.next_stage=="A_CANONICAL_SDXL"
    model_file=tmp_path/"stage-a.safetensors"
    model_file.write_bytes(b"synthetic-checkpoint-for-court-only")
    validation=tmp_path/"val.json"
    validation.write_text(json.dumps({
        "stage":"A_CANONICAL_SDXL",
        "cohort_sha256":cohort.cohort_sha256,
        "evaluation_split":"VAL",
        "status":"PASS","val_examples":30,
        "gradient_audit_sha256":sha("gradient-audit"),
    }))
    gpu=tmp_path/"gpu.json"
    gpu.write_text(json.dumps({
        "device_type":"cuda","device_name":"synthetic-gpu-test-only",
        "amp_dtype":"BF16","peak_allocated_mib":1000,
        "status":"PASS",
    }))
    receipt=PilotCompletedStageV1(
        stage="A_CANONICAL_SDXL",
        cohort_sha256=cohort.cohort_sha256,
        input_weights_sha256=sha("true-pretrained-origin-pin"),
        previous_stage_weights_sha256=None,
        output_checkpoint_path=str(model_file),
        output_checkpoint_sha256=sha256(model_file.read_bytes()).hexdigest(),
        objective_policy_sha256=sha("objective"),
        optimizer_ownership_sha256=sha("owner"),
        validation_report_path=str(validation),
        validation_report_sha256=sha256(validation.read_bytes()).hexdigest(),
        gpu_evidence_path=str(gpu),
        gpu_evidence_sha256=sha256(gpu.read_bytes()).hexdigest(),
        audited_gradient_report_sha256=sha("gradient-audit"),
        trained_steps=10,trained_examples=240,val_examples=30,
        gpu_device_name="synthetic-gpu-test-only",amp_dtype="BF16",
    )
    ledger=ledger.advance(receipt,cohort=cohort)
    assert ledger.next_stage=="B_PAIRED_RGB_NORMAL_MVADAPTER"
    dest=tmp_path/"sealed-ledger.json"
    save_pilot_ledger_v1(ledger,dest)
    loaded=load_pilot_ledger_v1(dest)
    assert loaded.digest==ledger.digest
    with pytest.raises(RuntimeError,match="OUT_OF_ORDER"):
        loaded.advance(receipt,cohort=cohort)
    wrong=replace(receipt,stage="B_PAIRED_RGB_NORMAL_MVADAPTER",
                  previous_stage_weights_sha256=receipt.output_checkpoint_sha256,
                  input_weights_sha256=sha("wrong-input"))
    # B starts with its own pretrained MV-Adapter (not A UNet weights).
    # The dependency on A is independently pinned above.
    bval=tmp_path/"val-b.json"
    bval.write_text(json.dumps({
        "stage":"B_PAIRED_RGB_NORMAL_MVADAPTER",
        "cohort_sha256":cohort.cohort_sha256,
        "evaluation_split":"VAL",
        "status":"PASS","val_examples":30,
        "gradient_audit_sha256":sha("gradient-audit"),
    }))
    b_receipt=replace(
        wrong,validation_report_path=str(bval),
        validation_report_sha256=sha256(bval.read_bytes()).hexdigest(),
    )
    stage_b=loaded.advance(b_receipt,cohort=cohort)
    assert stage_b.next_stage=="C_SLRM"
    # An incorrect stage B validation report still fails.
    with pytest.raises(RuntimeError,match="REAL_GPU_OR_VALIDATION_MISSING"):
        loaded.advance(wrong,cohort=cohort)
    gpu.write_text(json.dumps({"device_type":"cpu","status":"PASS"}))
    with pytest.raises(RuntimeError,match="ARTIFACT_BYTES_DRIFT"):
        empty_pilot_ledger_v1(cohort).advance(receipt,cohort=cohort)


def test_train_runner_never_uses_validation_or_unseen_examples(admitted):
    assets,root,cohort=admitted
    class Trainer:
        dataset_manifest_sha256=cohort.cohort_sha256
        def __init__(self):
            self.calls=0
            self.step=0
        def train_step(self,batch):
            self.calls+=1
            self.step+=1
            return type("Loss",(),{"total_loss":torch.tensor(float(batch))})()
    runner=Trainer()
    source_index={a.source_id:a for a in assets}
    train=[
        RealPilotTrainingEnvelopeV1(
            a.source_id,a.source_asset_sha256,float(i+1),
        )
        for i,a in enumerate(assets[:2])
    ]
    results=run_source_qualified_pilot_updates_v1(
        trainer=runner,staged_examples=train,source_index=source_index,
        cohort=cohort,max_steps=2,
    )
    assert runner.calls==2
    assert len(results)==2
    assert [r["step"] for r in results]==[1,2]
    with pytest.raises(ValueError,match="SOURCE_INDEX_COHORT_DRIFT"):
        run_source_qualified_pilot_updates_v1(
            trainer=runner,staged_examples=train,cohort=cohort,
            source_index={**source_index,assets[0].source_id:replace(
                assets[0],lineage_group="fabricated-family",
            )},max_steps=1,
        )
    with pytest.raises(ValueError,match="300_TO_400"):
        run_source_qualified_pilot_updates_v1(
            trainer=runner,staged_examples=train,cohort=cohort,
            source_index={a.source_id:a for a in assets[:2]},max_steps=1,
        )
    class NoCounter:
        dataset_manifest_sha256=cohort.cohort_sha256
        def train_step(self,batch):
            return {"total":torch.tensor(1.)}
    with pytest.raises(ValueError,match="PROGRESS_COUNTER_REQUIRED"):
        run_source_qualified_pilot_updates_v1(
            trainer=NoCounter(),staged_examples=train,source_index=source_index,
            cohort=cohort,max_steps=1,
        )
    class SilentCounter:
        dataset_manifest_sha256=cohort.cohort_sha256
        step=0
        def train_step(self,batch):
            return {"total":torch.tensor(1.)}
    with pytest.raises(RuntimeError,match="OPTIMIZER_STEP_NOT_RECORDED"):
        run_source_qualified_pilot_updates_v1(
            trainer=SilentCounter(),staged_examples=train,source_index=source_index,
            cohort=cohort,max_steps=1,
        )
    with pytest.raises(ValueError,match="TRAIN_SOURCE_NOT_ADMITTED"):
        run_source_qualified_pilot_updates_v1(
            trainer=runner,staged_examples=[
                RealPilotTrainingEnvelopeV1(
                    assets[-2].source_id,assets[-2].source_asset_sha256,1.,
                ),
            ],source_index=source_index,cohort=cohort,max_steps=1,
        )


class TinyProduct(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision=nn.Linear(2,2)
        self.iris=nn.Linear(2,2)
        self.tessa_seeded_head=nn.Linear(2,2)
        self.graphics=nn.Linear(2,2)
        self.axis=nn.Linear(2,2)
        self.mira=nn.Module()
        self.mira.codec=nn.Linear(2,2)
        self.mira.skin=nn.Linear(2,2)
    def forward(self,x):
        return (self.iris(x)+self.tessa_seeded_head(x)
                +self.graphics(x)+self.axis(x)+self.mira.skin(x)).sum()


def test_structural_phase_ownership_freezes_vision_and_embedded_codec():
    model=TinyProduct()
    policy=ProductSeededPhaseOwnerPolicyV1.standard("FULL_SHARED_H_FINETUNE")
    names=policy.activate_before_optimizer_v1(model)
    assert "iris.weight" in names and "graphics.weight" in names
    assert not any(name.startswith("vision.") for name in names)
    assert not any(name.startswith("mira.codec.") for name in names)
    opt=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],lr=1e-4,
    )
    assert len(policy.assert_qualified_optimizer(model,opt))==64
    model(torch.randn(2,2)).backward()
    policy.assert_gradient_owners_after_backward(model)
    opt.step()
    with pytest.raises(ValueError,match="MOMENTS_NOT_FRESH"):
        policy.assert_qualified_optimizer(model,opt)


def test_bad_gradient_route_and_stage_mismatch_fail_closed():
    model=TinyProduct()
    phase=ProductSeededPhaseOwnerPolicyV1.standard("MSTAR_GEOMETRY_BOOTSTRAP")
    phase.activate_before_optimizer_v1(model)
    assert not model.axis.weight.requires_grad
    assert not model.graphics.weight.requires_grad
    with pytest.raises(RuntimeError,match="EXPECTED_GRADIENT_MISSING"):
        phase.assert_gradient_owners_after_backward(model)
    with pytest.raises(ValueError,match="OWNER_POLICY_INVALID"):
        replace(phase,allowed_prefixes=("iris.",)).validate()
