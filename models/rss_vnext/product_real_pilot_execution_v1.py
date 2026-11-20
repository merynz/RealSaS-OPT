from __future__ import annotations

"""Immutable progression and real trainer-step runner for a 300–400 source pilot.

No import side effects, no downloading assets, no invented run/quality receipt.
Stage transitions are a TRAINING dependency graph, not a product inference
sequence. External stage-specific constructors must provide real pretrained
A/B task adapters and actual production source-backed trainer objects.
"""

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import math
import os
import tempfile

from .product_real_pilot_cohort_v1 import (
    RealPilotAssetV1, RealPilotCohortReceiptV1,
    assert_train_only_source_v1,
)

SCHEMA = "RealSaS.RSSvNext.Real400PilotPhaseLedger.v1"
STAGES = (
    "A_CANONICAL_SDXL",
    "B_PAIRED_RGB_NORMAL_MVADAPTER",
    "C_SLRM",
    "D_EMBEDDED_MIRA_CODEC",
    "E_MSTAR_GEOMETRY_BOOTSTRAP",
    "E_IRIS_GEOMETRY_SLR_GRAPHICS",
    "E_AXIS_MIRA_MSTAR_BINDING",
    "E_SOURCE_POSED_CONSEQUENCE",
    "E_FULL_SHARED_H_FINETUNE",
)
FINAL_TEST_STAGE = "UNSEEN_TEACHER_FREE_COMPILE"
STAGE_KIND = {
    "C_SLRM":"SLRM",
    "D_EMBEDDED_MIRA_CODEC":"MIRA_CODEC",
    **{x:"STRUCTURAL" for x in STAGES if x.startswith("E_")},
}
E_CURRICULUM = {
    name:name[2:] for name in STAGES if name.startswith("E_")
}


def _hex(v: object) -> bool:
    return type(v) is str and len(v)==64 and all(
        c in "0123456789abcdef" for c in v
    )


def _file_digest(path: str | Path) -> str:
    file=Path(path).expanduser()
    if not file.is_file() or file.is_symlink():
        raise ValueError("RSS_VNEXT_PILOT_REQUIRED_CHECKPOINT_FILE_MISSING")
    h=sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _digest_json(payload: object) -> str:
    return sha256(json.dumps(
        payload,sort_keys=True,separators=(",",":"),allow_nan=False,
    ).encode()).hexdigest()


@dataclass(frozen=True)
class PilotCompletedStageV1:
    stage: str
    cohort_sha256: str
    # The input source pretrained checkpoint and upstream TRAINED A/B
    # dependencies are independently recorded; do not collapse them.
    input_weights_sha256: str
    previous_stage_weights_sha256: str | None
    output_checkpoint_path: str
    output_checkpoint_sha256: str
    objective_policy_sha256: str
    optimizer_ownership_sha256: str
    validation_report_path: str
    validation_report_sha256: str
    gpu_evidence_path: str
    gpu_evidence_sha256: str
    audited_gradient_report_sha256: str
    trained_steps: int
    trained_examples: int
    val_examples: int
    gpu_device_name: str
    amp_dtype: str

    def validate(self) -> None:
        fields=(
            self.cohort_sha256,self.input_weights_sha256,
            self.output_checkpoint_sha256,self.objective_policy_sha256,
            self.optimizer_ownership_sha256,self.validation_report_sha256,
            self.gpu_evidence_sha256,self.audited_gradient_report_sha256,
        )
        if (
            self.stage not in STAGES
            or any(not _hex(x) for x in fields)
            or (self.previous_stage_weights_sha256 is not None
                and not _hex(self.previous_stage_weights_sha256))
            or type(self.trained_steps) is not int or self.trained_steps<1
            or type(self.trained_examples) is not int or self.trained_examples<1
            or type(self.val_examples) is not int or self.val_examples<1
            or type(self.gpu_device_name) is not str or not self.gpu_device_name
            or self.amp_dtype not in ("BF16","FP16")
            or any(type(x) is not str or not x for x in (
                self.output_checkpoint_path,self.validation_report_path,
                self.gpu_evidence_path,
            ))
        ):
            raise ValueError("RSS_VNEXT_PILOT_STAGE_RECEIPT_INCOMPLETE")


@dataclass(frozen=True)
class PilotLedgerV1:
    schema: str
    cohort_sha256: str
    completed: tuple[PilotCompletedStageV1,...]

    @property
    def next_stage(self) -> str:
        return (STAGES[len(self.completed)]
                if len(self.completed)<len(STAGES) else FINAL_TEST_STAGE)

    @property
    def digest(self) -> str:
        return _digest_json(asdict(self))

    def validate(self) -> None:
        if (
            self.schema!=SCHEMA or not _hex(self.cohort_sha256)
            or type(self.completed) is not tuple
            or len(self.completed)>len(STAGES)
        ):
            raise ValueError("RSS_VNEXT_PILOT_STAGE_LEDGER_INVALID")
        prev=None
        for expected,receipt in zip(STAGES,self.completed):
            if type(receipt) is not PilotCompletedStageV1:
                raise TypeError("RSS_VNEXT_PILOT_STAGE_RECEIPT_TYPE")
            receipt.validate()
            if (
                receipt.stage!=expected
                or receipt.cohort_sha256!=self.cohort_sha256
                or receipt.previous_stage_weights_sha256!=prev
            ):
                raise ValueError("RSS_VNEXT_PILOT_STAGE_ORDER_OR_DEPENDENCY_DRIFT")
            prev=receipt.output_checkpoint_sha256

    def advance(
        self,receipt:PilotCompletedStageV1,*,
        cohort:RealPilotCohortReceiptV1,
    ) -> "PilotLedgerV1":
        self.validate()
        if (
            not cohort.source_evidence_bytes_verified
            or cohort.cohort_sha256!=self.cohort_sha256
        ):
            raise RuntimeError("RSS_VNEXT_PILOT_COHORT_SOURCE_BYTES_NOT_VERIFIED")
        receipt.validate()
        if receipt.stage!=self.next_stage:
            raise RuntimeError("RSS_VNEXT_PILOT_OUT_OF_ORDER_STAGE")
        previous=(
            self.completed[-1].output_checkpoint_sha256
            if self.completed else None
        )
        if receipt.previous_stage_weights_sha256!=previous:
            raise RuntimeError("RSS_VNEXT_PILOT_STAGE_PREVIOUS_WEIGHTS_DRIFT")
        if previous is not None and receipt.input_weights_sha256!=previous:
            # Real stage B has its *own* pretrained MV-Adapter weights and
            # depends on A; use the separate B sidecar pretrained pin in
            # validation evidence, not as a replacement for A's digest.
            raise RuntimeError("RSS_VNEXT_PILOT_STAGE_INPUT_CHECKPOINT_DRIFT")
        if (
            _file_digest(receipt.output_checkpoint_path)
                !=receipt.output_checkpoint_sha256
            or _file_digest(receipt.validation_report_path)
                !=receipt.validation_report_sha256
            or _file_digest(receipt.gpu_evidence_path)
                !=receipt.gpu_evidence_sha256
        ):
            raise RuntimeError("RSS_VNEXT_PILOT_STAGE_ARTIFACT_BYTES_DRIFT")
        validation=json.loads(Path(receipt.validation_report_path).read_text())
        gpu=json.loads(Path(receipt.gpu_evidence_path).read_text())
        if (
            type(validation) is not dict
            or validation.get("stage")!=receipt.stage
            or validation.get("cohort_sha256")!=self.cohort_sha256
            or validation.get("status")!="PASS"
            or validation.get("evaluation_split")!="VAL"
            or validation.get("val_examples")!=receipt.val_examples
            or validation.get("gradient_audit_sha256")
                !=receipt.audited_gradient_report_sha256
            or type(gpu) is not dict
            or gpu.get("device_type")!="cuda"
            or gpu.get("device_name")!=receipt.gpu_device_name
            or gpu.get("amp_dtype")!=receipt.amp_dtype
            or type(gpu.get("peak_allocated_mib")) not in (float,int)
            or not math.isfinite(gpu["peak_allocated_mib"])
            or gpu["peak_allocated_mib"]<=0
            or gpu.get("status")!="PASS"
        ):
            raise RuntimeError("RSS_VNEXT_PILOT_STAGE_REAL_GPU_OR_VALIDATION_MISSING")
        new=PilotLedgerV1(
            schema=SCHEMA,cohort_sha256=self.cohort_sha256,
            completed=(*self.completed,receipt),
        )
        new.validate()
        return new


def empty_pilot_ledger_v1(cohort:RealPilotCohortReceiptV1)->PilotLedgerV1:
    if not cohort.source_evidence_bytes_verified:
        raise ValueError("RSS_VNEXT_PILOT_COHORT_BYTES_REQUIRED")
    result=PilotLedgerV1(SCHEMA,cohort.cohort_sha256,())
    result.validate()
    return result


def save_pilot_ledger_v1(ledger:PilotLedgerV1,path:str|Path)->None:
    ledger.validate()
    destination=Path(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",encoding="utf-8",dir=destination.parent,
        prefix="rss-pilot-ledger-",suffix=".json",delete=False,
    ) as stream:
        temporary=Path(stream.name)
        json.dump(asdict(ledger),stream,sort_keys=True,indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary,destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_pilot_ledger_v1(path:str|Path)->PilotLedgerV1:
    raw=json.loads(Path(path).read_text())
    if (
        type(raw) is not dict or raw.get("schema")!=SCHEMA
        or type(raw.get("completed")) is not list
    ):
        raise ValueError("RSS_VNEXT_PILOT_LEDGER_STORAGE_INVALID")
    entries=tuple(PilotCompletedStageV1(**row) for row in raw["completed"])
    ledger=PilotLedgerV1(raw["schema"],raw["cohort_sha256"],entries)
    ledger.validate()
    return ledger


@dataclass(frozen=True)
class RealPilotTrainingEnvelopeV1:
    """Caller must supply an independently source-admitted trainer batch."""
    source_id: str
    source_asset_sha256: str
    batch: Any


def run_source_qualified_pilot_updates_v1(
    *, trainer:Any, staged_examples:Iterable[RealPilotTrainingEnvelopeV1],
    source_index:Mapping[str,RealPilotAssetV1],
    cohort:RealPilotCohortReceiptV1,
    max_steps:int,
) -> list[dict[str,object]]:
    """Execute actual trainer train_step; never train VAL or UNSEEN_TEST.

    Adapters must perform their *own* source-tensor/teacher receipt validation;
    this outer court prevents source and identity leakage in the pilot loader.
    True A/B stage adapters are still required; importing a pretrained
    pipeline alone does not implement a task trainer.
    """
    if (
        not cohort.source_evidence_bytes_verified
        or type(max_steps) is not int or max_steps<1
        or max_steps>1_000_000
        or not callable(getattr(trainer,"train_step",None))
    ):
        raise ValueError("RSS_VNEXT_PILOT_TRAINER_NOT_ADMITTED")
    manifest=getattr(trainer,"dataset_manifest_sha256",
             getattr(trainer,"manifest_sha256",None))
    if manifest is not None and manifest!=cohort.cohort_sha256:
        raise ValueError("RSS_VNEXT_PILOT_TRAINER_MANIFEST_IDENTITY_DRIFT")
    if not source_index:
        raise ValueError("RSS_VNEXT_PILOT_TRAIN_SOURCE_INDEX_MISSING")
    reports=[]
    for envelope in staged_examples:
        if len(reports)>=max_steps:
            break
        if type(envelope) is not RealPilotTrainingEnvelopeV1:
            raise TypeError("RSS_VNEXT_PILOT_BATCH_ENVELOPE_INVALID")
        item=source_index.get(envelope.source_id)
        if (
            item is None or item.source_asset_sha256
                !=envelope.source_asset_sha256
        ):
            raise ValueError("RSS_VNEXT_PILOT_BATCH_ASSET_BINDING_DRIFT")
        assert_train_only_source_v1(item,cohort)
        result=trainer.train_step(envelope.batch)
        scalar=(
            result.get("total_loss",result.get("loss",result.get("total")))
            if isinstance(result,dict) else
            getattr(result,"total_loss",getattr(result,"loss",None))
        )
        try:
            value=float(scalar.detach().item())
        except (TypeError,AttributeError,ValueError,RuntimeError) as exc:
            raise RuntimeError("RSS_VNEXT_PILOT_TRAIN_LOSS_TENSOR_REQUIRED") from exc
        if not math.isfinite(value):
            raise RuntimeError("RSS_VNEXT_PILOT_TRAIN_NONFINITE_LOSS")
        reports.append({
            "source_id":envelope.source_id,"step":len(reports),
            "total_loss":value,
        })
    if len(reports)!=max_steps:
        raise RuntimeError("RSS_VNEXT_PILOT_TRAIN_STREAM_EARLY_END")
    return reports


__all__=[
    "SCHEMA","STAGES","STAGE_KIND","E_CURRICULUM","FINAL_TEST_STAGE",
    "PilotCompletedStageV1","PilotLedgerV1","empty_pilot_ledger_v1",
    "save_pilot_ledger_v1","load_pilot_ledger_v1",
    "RealPilotTrainingEnvelopeV1","run_source_qualified_pilot_updates_v1",
]
