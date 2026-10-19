from __future__ import annotations

from tools.iris_vnext.preflight_fit8_direct_surface_a100_v1 import (
    engineering_admission,
    parser,
)


def test_a100_class_resource_envelope_is_admitted():
    ok, blockers = engineering_admission(
        cuda_memory_bytes=40 * 1024**3,
        host_memory=64 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        compute_capability_major=8,
    )
    assert ok is True
    assert blockers == []


def test_consumer_gpu_is_not_silently_promoted_to_full_surface_lane():
    ok, blockers = engineering_admission(
        cuda_memory_bytes=24 * 1024**3,
        host_memory=64 * 1024**3,
        free_disk_bytes=100 * 1024**3,
        compute_capability_major=8,
    )
    assert ok is False
    assert "CUDA_MEMORY_BELOW_39_GIB" in blockers


def test_preflight_has_no_reduced_mesh_resolution_switch():
    destinations = {action.dest for action in parser()._actions}
    assert "base_cells" not in destinations
    assert "fine_cells" not in destinations
    assert "resolution" not in destinations
    assert "model_query_chunk" in destinations
