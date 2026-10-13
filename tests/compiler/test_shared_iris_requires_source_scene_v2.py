import pytest

from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_services.orchestrator.adapters.iris_observation_binding_v1 import (
    SHARED_PROFILE,
    SOURCE_SCENE_PROFILE,
    iris_observation_binding,
)


def test_shared_iris_requires_strong_source_scene_profile_before_proof_binding():
    ctx={"run_manifest":{"iris_fit":{"training_profile":SHARED_PROFILE},"source_audit":{}}}
    with pytest.raises(QualificationError,match="IRIS_SHARED_COHORT_SOURCE_SCENE_PROFILE_REQUIRED"):
        iris_observation_binding(ctx,set(),{}, {})


def test_nonshared_legacy_path_is_not_retroactively_forced():
    ctx={"run_manifest":{"iris_fit":{},"source_audit":{}}}
    assert iris_observation_binding(ctx,set(),{}, {}) is None


def test_shared_profile_accepts_source_scene_profile_then_requires_its_proof():
    ctx={"run_manifest":{"iris_fit":{"training_profile":SHARED_PROFILE},
                         "source_audit":{"profile":SOURCE_SCENE_PROFILE,
                                         "iris_observation_admission_required":True}}}
    with pytest.raises(QualificationError,match="IRIS_OBSERVATION_ADMISSION_REQUIRED"):
        iris_observation_binding(ctx,set(),{}, {})
