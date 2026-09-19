from __future__ import annotations

from dataclasses import replace
import pytest

from compiler.realsas_compiler_core.observation_authority_v1 import (
    QualifiedObservationViewIR,
    build_qualified_observation_set,
    validate_qualified_observation_set,
)
from compiler.realsas_compiler_core.types import QualificationError


def _views():
    return tuple(
        QualifiedObservationViewIR(
            i,8,8,f"obs-{i}",f"{i+16:064x}",f"{i:064x}",f"cam-{i}","PASS",(f"evidence-{i}",)
        )
        for i in range(8)
    )


def test_observation_set_requires_exact_eight_qualified_camera_bound_views():
    value=build_qualified_observation_set(_views())
    assert len(value.views)==8
    assert len(value.observation_set_hash)==64
    validate_qualified_observation_set(value)


def test_observation_set_rejects_camera_aliasing():
    rows=list(_views())
    rows[1]=replace(rows[1],camera_binding_hash=rows[0].camera_binding_hash)
    with pytest.raises(QualificationError,match="DUPLICATE_CAMERA"):
        build_qualified_observation_set(tuple(rows))


def test_source_raster_byte_identity_is_first_class_and_hash_sensitive():
    rows=list(_views())
    value=build_qualified_observation_set(tuple(rows))
    changed=replace(rows[0],source_raster_sha256="f"*64)
    other=build_qualified_observation_set((changed,*rows[1:]))
    assert value.observation_set_hash!=other.observation_set_hash
