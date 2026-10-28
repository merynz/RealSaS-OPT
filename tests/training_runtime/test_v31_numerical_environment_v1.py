from __future__ import annotations

from models.iris.v31.numerical_environment_v1 import (
    SCHEMA_V1,
    numerical_environment_sha256_v1,
    numerical_environment_v1,
)


def test_cpu_numerical_environment_identity_is_versioned_and_stable_shape():
    value = numerical_environment_v1("cpu")
    assert value["schema"] == SCHEMA_V1
    assert value["device_type"] == "cpu"
    assert value["device_index"] is None
    assert value["device_name"] is None
    assert value["device_capability"] is None
    assert value["authority_class"] == "NUMERICAL_LINEAGE_ONLY"
    assert value["torch_version"]
    assert value["torchvision_version"]
    digest = numerical_environment_sha256_v1("cpu")
    assert len(digest) == 64
    assert digest == numerical_environment_sha256_v1("cpu")
